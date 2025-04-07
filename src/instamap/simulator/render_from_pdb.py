"""Utility script to generate pseudo-data from a pdb file.
"""

from __future__ import annotations

import dataclasses
import functools
import gzip
import io
import logging
import os
from typing import Dict, List, Mapping, Optional, TextIO, Tuple


import hydra
import importlib_resources
import omegaconf
import numpy as np
import torch

import instamap.simulator.interpolation
import instamap.simulator.optics
import instamap.simulator.pipeline
import instamap.simulator.scattering
import instamap.simulator.sensor

from . import build_lobato_approximation, integration, render_from_volume
from . import relion


@dataclasses.dataclass
class PdbSetupConfig:
    """Configuration for setting up a pdb file for rendering

    Attributes
    ----------
    path : str
        Path to the pdb file to load
    offset : Optional[List[float]]
        If not `None`, apply this offset to the coordinates of the atoms
        after loading the pdb file. This may be used to center the molecule
        if required.
    """
    path: str = omegaconf.MISSING
    offset: Optional[List[float]] = None
    zero_center_of_mass: bool = False


@dataclasses.dataclass
class SimulatedRenderingConfig(render_from_volume.SimulatedRenderingConfig):
    """Configuration for rendering from a pdb file and poses.

    Attributes
    ----------
    use_gaussian_scattering : bool
        If `True`, uses scattering which directly integrates the 2d projections
        as gaussians. Otherwise, use the 3d volume to render the images.
    """
    use_gaussian_scattering: bool = True


@dataclasses.dataclass
class PdbRelionRenderingConfig:
    """Configuration for rendering from a relion volume and estimated poses.

    Parameters
    ----------
    device : str
        The torch device name to use for rendering
    batch_size : int
        The batch size to use for rendering
    volume_path : str
        Path to the mrc file containing the volume to render
    shot_info_path : str
        Path to the star file containing the shot information
    render : RenderingConfig
        Configuration for rendering the electron images
    include_reference : bool
        Whether to additionally render reference images for each shot
        (i.e. render without any CTF or noise added), and save those renders.
    noise_ratio : Optional[float]
        Amount of gaussian noise to add to the rendered images, specified as the
        variance as a fraction of the signal variance.
    normalize : bool
        If `True`, normalizes the rendered images to have zero mean and unit
        variance (as is commonly the case for experimental data).
    num_shots : int, optional
        If not `None`, only render this many shots (chosen at random).
        Otherwise, renders all shots as described in the star file.
    output_mrcs : bool
        If `True`, indicates that a mrcs file containing the rendered images
        should also be saved.
    volume_extent : Tuple[int, int, int]
        The extent of the volume in pixels to create
    volume_pixel_size : float
        The size of a single voxel in the volume in Angstroms
    volume_only : bool
        If `True`, only renders the volume and saves it to a file.
        Otherwise, also renders the simulated electron images.
    """
    device: str = 'cpu'
    batch_size: int = 16
    pdb: PdbSetupConfig = PdbSetupConfig()
    shot_info_path: str = omegaconf.MISSING
    render: SimulatedRenderingConfig = SimulatedRenderingConfig()
    include_reference: bool = False
    num_shots: Optional[int] = None
    output_mrcs: bool = False
    volume_shape: Tuple[int, int, int] = (256, 256, 256)
    volume_pixel_size: float = 1.0
    volume_only: bool = False


@functools.lru_cache()
def load_potential():
    data_root = importlib_resources.files('instamap.simulator')
    data = (data_root / 'info' / 'lobato_approximation.json').read_text()
    return build_lobato_approximation.load_lobato_approximation(
        io.StringIO(data), adjust_scaling_to_density=True
    )


def coordinates_to_gaussians_split(
    coordinates: Mapping[int, torch.Tensor],
    potential: Mapping[int, build_lobato_approximation.BubbleGaussianApproximation]
) -> Dict[str, torch.Tensor]:
    """Converts the given atom coordinates to the corresponding gaussian approximation.
    This version splits the coordinates and (weights / scales) portion, by only storing
    a single coordinate for each stack of gaussians. The weights and scales are padded
    if the number of components in each approximation is not the same.
    """
    num_centers = []
    centers: List[torch.Tensor] = []
    weights: List[torch.Tensor] = []
    scales: List[torch.Tensor] = []
    bubbles = []

    for k, c in coordinates.items():
        p = potential[k]

        p_s = p['s'].to(c)
        p_a = p['a'].to(c)

        num_centers.append(len(c))
        centers.append(c)
        scales.append(p_s)
        weights.append(p_a)
        bubbles.append(p['bubble'])

    num_centers = centers[0].new_tensor(num_centers, dtype=torch.int64)

    all_centers = torch.cat(centers)
    max_components = max(len(w) for w in weights)

    weights = [torch.cat([w, w.new_zeros(max_components - len(w))]) for w in weights]
    scales = [torch.cat([s, s.new_ones(max_components - len(s))]) for s in scales]

    all_weights = torch.stack(weights, dim=0).repeat_interleave(num_centers, dim=0)
    all_scales = torch.stack(scales, dim=0).repeat_interleave(num_centers, dim=0)
    all_bubbles = all_weights.new_tensor(bubbles).repeat_interleave(num_centers, dim=0)

    return {
        'center': all_centers,
        'weight': all_weights,
        'scale': all_scales,
        'bubble_weight': all_bubbles,
    }


def coordinates_to_gaussians(
    coordinates: Mapping[int, torch.Tensor],
    potential: Mapping[int, build_lobato_approximation.BubbleGaussianApproximation]
) -> Dict[str, torch.Tensor]:
    """Converts the given atom coordinates to the corresponding gaussian approximation.

    Parameters
    ----------
    coordinates : Mapping[int, torch.Tensor]
        A dictionary mapping atom types to the coordinates of the atoms of that type
    potential : Mapping[int, build_lobato_approximation.BubbleGaussianApproximation]
        A dictionary mapping atom types to the potential of that type
    """
    gaussian_centers = {}
    gaussian_weights = {}
    gaussian_scales = {}

    bubble_weights = {}

    for k, c in coordinates.items():
        p = potential[k]

        p_s = p['s'].to(c)
        p_a = p['a'].to(c)

        gaussian_centers[k] = torch.repeat_interleave(c, len(p_a), dim=0)
        gaussian_scales[k] = p_s.repeat(len(c))
        gaussian_weights[k] = p_a.repeat(len(c))
        bubble_weights[k] = c.new_full((len(c), ), p['bubble'])

    atom_types = list(coordinates.keys())

    all_centers = torch.cat([gaussian_centers[k] for k in atom_types])
    all_weights = torch.cat([gaussian_weights[k] for k in atom_types])
    all_scales = torch.cat([gaussian_scales[k] for k in atom_types])
    all_bubble_weights = torch.cat([bubble_weights[k] for k in atom_types])

    return {
        'center': all_centers,
        'weight': all_weights,
        'scale': all_scales,
        'bubble_weight': all_bubble_weights,
    }


def _make_pipeline_from_gaussians(
    gaussians: Dict[str, torch.Tensor], config: SimulatedRenderingConfig
):

    rendering_config = instamap.simulator.scattering.ImageConfig(
        int(config.image_height * config.upsample), int(config.image_width * config.upsample),
        config.pixel_size / config.upsample
    )

    scattering_base = instamap.simulator.scattering.GaussianProjectionIntegration(
        rendering_config, gaussians['center'], gaussians['scale'], gaussians['weight'],
        gaussians['bubble_weight']
    )
    scattering_full = instamap.simulator.scattering.ViewAugmentingAdapter(
        scattering_base, config.augment
    )

    if config.use_ctf:
        pipeline_optics = instamap.simulator.optics.FourierContrastTransferOptics(rendering_config)
    else:
        pipeline_optics = instamap.simulator.optics.NullOptics()


    image_config = instamap.simulator.scattering.ImageConfig(
        config.image_height,
        config.image_width,
        pixel_size=config.pixel_size)
    pipeline_sensor = render_from_volume.make_sensor_from_config(image_config, config.noise)

    if config.upsample != 1:
        pipeline_sensor = instamap.simulator.sensor.DownsamplingSensorAdapter(
            pipeline_sensor, (config.image_height, config.image_width)
        )

    if config.normalize:
        radius = min(config.image_height, config.image_width) / 2
        pipeline_sensor = instamap.simulator.sensor.BackgroundNormalizingSensorAdapter(
            image_config, pipeline_sensor, mask_radius=radius / config.pixel_size
        )

    return instamap.simulator.pipeline.RenderingPipeline(
        scattering_full, pipeline_optics, pipeline_sensor
    )


def _make_pipeline(
    coordinates: Mapping[int, torch.Tensor],
    potential: Mapping[int, build_lobato_approximation.BubbleGaussianApproximation],
    volume: Optional[torch.Tensor],
    config: SimulatedRenderingConfig,
):
    """Creates a rendering pipeline according to the given configuration.

    This function switches between the volume rendering pipeline and the
    direct gaussian integration pipeline depending on the configuration.
    """
    if config.use_gaussian_scattering:
        gaussians = coordinates_to_gaussians_split(coordinates, potential)
        return _make_pipeline_from_gaussians(gaussians, config)
    else:
        if volume is None:
            raise ValueError('Volume rendering requested but no volume given')

        field = instamap.simulator.interpolation.TrilinearScalarField(volume, config.extent)
        return render_from_volume.make_pipeline_from_field(field, config)


def raster_pdb(
    coordinates: Mapping[int, torch.Tensor],
    potential: Mapping[int, build_lobato_approximation.BubbleGaussianApproximation],
    shape: Tuple[int, int, int], pixel_size: float
) -> torch.Tensor:

    g_info = coordinates_to_gaussians(coordinates, potential)

    field_gaussian = integration.integrate_gaussian_nd_batch(
        shape, [64, 64, 64], g_info['center'].mul(1 / pixel_size),
        g_info['scale'].mul(1 / pixel_size), g_info['weight']
    )

    all_coordinates = torch.cat([c for c in coordinates.values()])
    all_coordinates.mul_(1 / pixel_size)

    field = integration.scatter_add_point_masses(
        field_gaussian, all_coordinates, g_info['bubble_weight']
    )

    volume_element = pixel_size**3

    return field.mul_(1 / volume_element)


def open_with_compression(filename: os.PathLike, mode='rt') -> TextIO:
    if os.path.splitext(filename)[1] == '.gz':
        return gzip.open(filename, mode)
    else:
        return open(filename, mode)


def read_pdb_file(filename: os.PathLike,
                  dtype: torch.dtype = torch.float32,
                  zero_center_of_mass: bool = False,
                  ) -> Dict[int, torch.Tensor]:
    """Read a PDB file.

    This function uses a custom implementation of PDB file reading to handle pdb
    files with missing atom types. When the atom type is missing, it is assumed
    to be a hydrogen atom. This ensures compatibility with a wider range of MD
    software.
    """
    import ase.atoms
    from ase.io import proteindatabank as pdb_io

    logger = logging.getLogger(__name__)

    with open_with_compression(filename) as f:
        lines = f.readlines()

    symbols = []
    coords = []

    for i, l in enumerate(lines):
        if not l.startswith('ATOM'):
            continue

        try:
            l_data = pdb_io.read_atom_line(l)
        except:
            logger.exception('Failed to parse line %d: %s', i, l)
            raise

        symbol = l_data[0]
        name = l_data[1]
        coord = l_data[4]


        if symbol == '':
            symbol = name[0]

        symbols.append(symbol)
        coords.append(coord)

    atoms = ase.atoms.Atoms(symbols, positions=np.stack(coords, axis=0))

    atomic_numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()
    if zero_center_of_mass: positions -= atoms.get_center_of_mass()


    return {
        z: torch.from_numpy(positions[atomic_numbers == z]).to(dtype=dtype)
        for z in np.unique(atomic_numbers)
    }


def read_gro_file(
    filename: os.PathLike,
    dtype: torch.dtype = torch.float32
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    """Read a gromacs file.

    Custom line-by-line parsing of gromacs files.
    This extracts only the main atom type and position information.
    """
    import ase.atoms

    with open_with_compression(filename) as f:
        lines = iter(f)
        next(lines)
        num_atoms = int(next(lines))

        positions = np.zeros((num_atoms, 3), dtype=np.float32)
        symbols = []

        for i in range(num_atoms):
            l = next(lines)

            symbols.append(l[10:15].strip()[0])
            positions[i, 0] = float(l[20:28]) * 10
            positions[i, 1] = float(l[28:36]) * 10
            positions[i, 2] = float(l[36:44]) * 10

        box_vector_line = next(lines).split()
        box = torch.tensor([float(x) * 10 for x in box_vector_line[:3]], dtype=dtype)

    atoms = ase.atoms.Atoms(symbols, positions=positions)
    atomic_numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()

    return {
        z: torch.from_numpy(positions[atomic_numbers == z]).to(dtype)
        for z in np.unique(atomic_numbers)
    }, box


@hydra.main(config_name='render_from_pdb', config_path='conf', version_base="1.2")
def main(config: PdbRelionRenderingConfig):
    config = omegaconf.OmegaConf.to_object(config)

    logger = logging.getLogger(__name__)

    pdb_path = hydra.utils.to_absolute_path(config.pdb.path)
    logger.info(f'Loading PDB file from {pdb_path}...')
    coordinates = read_pdb_file(pdb_path, zero_center_of_mass=config.pdb.zero_center_of_mass)

    logger.info(f'Creating volume from PDB...')
    coordinates = {
        k: torch.as_tensor(v, dtype=torch.float32, device=config.device)
        for k, v in coordinates.items()
    }

    if config.pdb.offset is not None:
        coordinates = {k: v + v.new_tensor(config.pdb.offset) for k, v in coordinates.items()}

    potential = load_potential()

    for k in list(coordinates.keys()):
        if k not in potential:
            del coordinates[k]
            logger.warning(f'No potential available for atomic number {k}. Removing from PDB.')


    field = raster_pdb(coordinates, potential, config.volume_shape, config.volume_pixel_size)

    # Save the reference volume
    volume_path = os.path.abspath('volume.npy')
    logger.info(f'Saving volume to {volume_path}...')
    np.save(volume_path, field.cpu().numpy())

    if config.volume_only:
        logger.info('Not rendering images as volume_only is requested. Exiting now.')
        return

    image_size = (config.render.image_height, config.render.image_width)
    shot_info_path = hydra.utils.to_absolute_path(config.shot_info_path)
    logger.info(f'Loading shot information from {shot_info_path}...')
    pixel_size, shot_info = render_from_volume.load_shot_info(
        shot_info_path, default_image_size=image_size[0]
    )

    if config.render.extent is None:
        config.render.extent = tuple([v * config.volume_pixel_size for v in config.volume_shape])

    shot_info, original_shot_index = render_from_volume.get_shot_subset(shot_info, config.num_shots)

    save_data = {'shot_info_' + k: np.asarray(v) for k, v in shot_info.items()}
    save_data['original_shot_index'] = original_shot_index
    save_data['pixel_size'] = pixel_size
    save_data['config'] = omegaconf.OmegaConf.to_yaml(config)

    device = torch.device(config.device)

    logger.info('Rendering images...')
    pipeline = _make_pipeline(coordinates, potential, field, config.render)
    images = render_from_volume.render_all_images(
        pipeline.to(device=device),
        shot_info,
        image_size,
        batch_size=config.batch_size,
        device=device
    )
    images = images.cpu().numpy()
    save_data['image'] = images

    if config.include_reference:
        logger.info('Rendering reference images')

        reference_render_config = dataclasses.replace(
            config.render,
            use_ctf=False,
            noise=render_from_volume.SimulationGaussianNoiseConfig()
        )
        pipeline = _make_pipeline(coordinates, potential, field, reference_render_config)

        images_reference = render_from_volume.render_all_images(
            pipeline.to(device=device),
            shot_info,
            image_size,
            batch_size=config.batch_size,
            device=device
        )

        images_reference = images_reference.cpu().numpy()
        save_data['image_reference'] = images_reference

    data_path = os.path.abspath('rendered.npz')
    logger.info(f'Saving rendered images to path: {data_path}')
    np.savez(data_path, **save_data)

    render_from_volume.save_preview('preview.png', save_data['image'])
    if config.include_reference:
        render_from_volume.save_preview('preview_reference.png', save_data['image_reference'])

    if config.output_mrcs:
        output_mrcs_path = os.path.abspath('rendered.mrcs')
        logger.info(f'Saving images to mrcs file: {output_mrcs_path}')
        relion.save_stack_to_mrcs(images, output_mrcs_path)


if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store(name='render_from_pdb_base', node=PdbRelionRenderingConfig)
    render_from_volume.register_noise_configs(cs, group='render/noise')
    main()
