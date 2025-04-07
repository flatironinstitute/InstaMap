"""Utility script to generate pseudo-data from a relion reconstructed volume.

This script reads in information from a relion reconstructed volume,
and produces simulated images from the volume and shot information.

"""

from __future__ import annotations

import dataclasses
import functools
import math
import logging

from typing import Callable, Dict, Optional, Tuple

import hydra
import hydra.core.config_store
import numpy as np
import omegaconf
import pytorch_lightning.utilities
import torch
import tqdm

from . import electron_image, relion
from . import optics, pipeline, scattering, sensor


@dataclasses.dataclass
class SimulationNoiseConfig:
    model: str = 'none'


@dataclasses.dataclass
class SimulationGaussianNoiseConfig(SimulationNoiseConfig):
    model: str = 'gaussian'
    sigma: Optional[float] = None
    relative: bool = False


@dataclasses.dataclass
class SimulationBioemNoiseConfig(SimulationNoiseConfig):
    model: str = 'bioem'
    sigma: Optional[float] = None
    N_hi: float = 1.0
    N_lo: float = 0.1
    mu_hi: float = +10.0
    mu_lo: float = -10.0
    method: str = 'saddle-approx'


@dataclasses.dataclass
class SimulatedRenderingConfig(electron_image.CryoEmRenderingConfig):
    """Configuration for rendering from simulation.

    Attributes
    ----------
    augment : scattering.ViewAugmentationConfig
        Configuration for augmenting the pose parameters of the simulated images.
        Allows for rendering with different pose parameters than the ones recorded.
    noise_ratio : float
        If not None, adds gaussian noise to the simulated images, with this parameter
        defining the standard deviation of the noise.
    noise_ratio_relative : bool
        If `True`, the noise ratio is interpreted as a fraction of the signal variance.
        Otherwise it is interpreted as the absolute standard deviation of the noise.
    normalize : bool
        If `True`, normalizes the simulated images to have mean 0 and standard deviation 1.
    upsample : float
        Upsampling factor to apply to the simulated images when scattering, for higher
        resolution and less aliasing. Note that this is applied after the CTF but before
        noise and normalization.
    """
    augment: scattering.ViewAugmentationConfig = scattering.ViewAugmentationConfig()
    noise: SimulationNoiseConfig = omegaconf.MISSING
    normalize: bool = True
    upsample: float = 1


def _render_for_shot(
    shot_info: electron_image.CryoEmShotInfo,
    renderer,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    with torch.inference_mode():
        if device is not None:
            shot_info = pytorch_lightning.utilities.move_data_to_device(shot_info, device)
        images, _ = renderer(shot_info, generator=generator)

    return images


def render_loop(
    render: Callable[[electron_image.CryoEmShotInfo], torch.Tensor],
    output: torch.Tensor,
    shot_info: electron_image.CryoEmShotInfo,
    batch_size: int,
):
    """Inner rendering loop, parametrized over a generic render function.

    This loop takes care of the batching and collating of the rendered images.
    It is implemented as a generator, yielding the number of images rendered in each batch,
    in order to allow for progress monitoring.

    The output data is written to the given tensor, which must be allocated to the correct size.

    Parameters
    ----------
    render : Callable[[electron_image.CryoEmShotInfo], torch.Tensor]
        The render function to use. Must take a shot info object as input, and return a tensor
        of shape (B, H, W) containing the rendered images.
    output : torch.Tensor
        An allocated tensor to write the rendered images to. Must have shape (N, H, W),
        where N is the total number of images to render (inferred from the length of tensors
        contained in the shot info object).

        This tensor is modified in-place, and may be allocated on any device.
    shot_info : electron_image.CryoEmShotInfo
        The shot info object containing the information about the shots to render.
    batch_size : int
        The batch size to use for rendering.
    """
    shot_batches = {
        k: torch.chunk(v, (len(v) + batch_size - 1) // batch_size)
        for k, v in shot_info.items()
    }

    processed = 0

    with torch.inference_mode():
        for batch_idx in range(len(shot_batches['view_phi'])):
            batch_shot_info = {k: v[batch_idx] for k, v in shot_batches.items()}
            images = render(batch_shot_info)

            output.narrow(0, processed, len(images)).copy_(images, non_blocking=True)
            num_batch = len(images)
            processed += num_batch
            yield num_batch


def make_sensor_from_config(image: electron_image.ImageConfig, noise: SimulationNoiseConfig):
    if noise.model == 'gaussian':
        if noise.relative:
            raise NotImplementedError('Relative noise not implemented yet')
        else:
            return sensor.GaussianSensor(image, noise.sigma or 0)
    elif noise.model == 'bioem':
        return sensor.BioemSensor(image, noise.sigma or 0, 
                                  method=noise.method, 
                                  N_hi=noise.N_hi,  
                                  N_lo=noise.N_lo,  
                                  mu_hi=noise.mu_hi,  
                                  mu_lo=noise.mu_lo,  
                                  )
    elif noise.model == 'poisson':
        return sensor.NormalizingPoissonSensor(
            image, noise.electrons_per_angstrom, noise.signal_variance
        )
    elif noise.model == 'ice_poisson':
        base_sensor = sensor.NormalizingPoissonSensor(
            image, noise.electrons_per_angstrom, noise.signal_variance
        )

        ice_path = hydra.utils.to_absolute_path(noise.ice_path)
        ice_data = np.load(ice_path)
        ice_data_slices = torch.from_numpy(ice_data['image']).flatten(end_dim=-3)
        slice_thickness = float(ice_data['slice_depth'])

        return sensor.LinearMultiSliceNoise(
            image,
            base_sensor,
            ice_samples=ice_data_slices,
            pixel_size=float(ice_data['pixel_size']),
            slice_thickness=slice_thickness,
            num_slices=int(math.ceil(noise.ice_depth / slice_thickness))
        )
    else:
        raise ValueError(f'Unknown noise model {noise.model}')


def register_noise_configs(store: hydra.core.config_store.ConfigStore, group: str):
    store.store('gaussian_base', node=SimulationGaussianNoiseConfig, group=group)
    store.store('bioem_base', node=SimulationBioemNoiseConfig, group=group)


def make_pipeline_from_field(field: torch.nn.Module, config: SimulatedRenderingConfig):
    """Create a standard pipeline for rendering from a given field.

    Parameters
    ----------
    field : torch.nn.Module
        The field to query when rendering. Must be a callable which maps
        batches of tensor in R^3 to values in R.
    config : SimulatedRenderingConfig
        General configuration for rendering
    """
    rendering_config = scattering.RenderingConfig(
        int(config.image_height * config.upsample),
        int(config.image_width * config.upsample),
        pixel_size=config.pixel_size / config.upsample,
        depth_samples=int(config.depth_samples * config.upsample)
    )

    scattering_base = scattering.DirectSphereProjection(field, rendering_config, config.extent)
    scattering_full = scattering.ViewAugmentingAdapter(scattering_base, config.augment)

    if config.use_ctf:
        pipeline_optics = optics.FourierContrastTransferOptics(rendering_config)
    else:
        pipeline_optics = optics.NullOptics()

    image_config = electron_image.ImageConfig(
        config.image_height, config.image_width, pixel_size=config.pixel_size
    )
    pipeline_sensor = make_sensor_from_config(image_config, config.noise)

    if config.upsample != 1:
        pipeline_sensor = sensor.DownsamplingSensorAdapter(
            pipeline_sensor, (config.image_height, config.image_width)
        )

    if config.normalize:
        radius = math.sqrt(image_config.height**2 + image_config.width**2)
        pipeline_sensor = sensor.BackgroundNormalizingSensorAdapter(
            image_config, pipeline_sensor, mask_radius=radius / config.pixel_size
        )

    return pipeline.RenderingPipeline(scattering_full, pipeline_optics, pipeline_sensor)


def render_all_images(
    renderer: Callable[[electron_image.CryoEmShotInfo], Tuple[torch.Tensor, Dict[str,
                                                                                 torch.Tensor]]],
    shot_info: electron_image.CryoEmShotInfo,
    image_size: Tuple[int, int],
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    progress: bool = True
) -> torch.Tensor:
    """
    """

    if device is None:
        device = torch.device('cpu')

    shot_info = {k: v.pin_memory() for k, v in shot_info.items()}

    generator = torch.Generator(device=device).manual_seed(42)

    image_results = torch.empty(
        (len(shot_info['view_phi']), ) + image_size, dtype=torch.float32, device='cpu'
    )
    image_results.pin_memory()

    render = functools.partial(
        _render_for_shot, renderer=renderer, generator=generator, device=device
    )

    with tqdm.tqdm(total=len(shot_info['view_phi']), disable=not progress) as pbar:
        processed = 0

        for num_processed in render_loop(render, image_results, shot_info, batch_size):
            pbar.update(num_processed)
            processed += num_processed

    return image_results


def load_shot_info(shot_info_path: str, default_image_size: Optional[int] = None):
    logger = logging.getLogger(__name__)
    logger.info(f'Loading shot information from path: {shot_info_path}')
    return relion.load_shot_info_from_file(shot_info_path, default_image_size)


def get_shot_subset(shot_info, num_shots: Optional[int] = None):
    if num_shots is not None:
        num_shots_original = len(shot_info['view_phi'])
        rng = np.random.Generator(np.random.PCG64(seed=42))
        subset = torch.from_numpy(rng.choice(num_shots_original, num_shots, replace=False))
        subset, _ = subset.sort()
        shot_info = {k: v[subset] for k, v in shot_info.items()}
        original_shot_index = subset
    else:
        original_shot_index = torch.arange(len(shot_info['view_phi']))

    return shot_info, original_shot_index


def save_preview(path: str, data: np.ndarray):
    try:
        import imageio
    except:
        logger = logging.getLogger(__name__)
        logger.warning('Could not import imageio, skipping preview image generation', exc_info=True)
        return

    image = data[0]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = (image * 255).astype(np.uint8)

    imageio.imsave(path, image)

