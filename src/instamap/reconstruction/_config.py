"""Configuration classes for training.

Note: stored in separate file from train module to ensure they have
stable names for pickling and unpickling.

"""

from __future__ import annotations

import dataclasses
from typing import Optional, overload, Tuple, Dict, List

import omegaconf

from instamap.nn.lightning import TrainerConfiguration
from instamap.simulator import electron_image
from . import field


@dataclasses.dataclass
class InstaMapDataConfig:
    """Configuration for the data loading.

    Attributes
    ----------
    path : str
        Path to the data file containing the observed images and shot information.
    image : ImageDataConfig
        Basic description of the images in the dataset.
    flip_contrast_images : bool
        If `True`, indicates that images are stored inverted, and should be flipped.
        This is useful for raw cryo-EM images, for which we would like to represent
        the molecule with positive density.
    make_local_copy : bool
        If `True`, indicates that the training process will make a local copy of the data
        file before attempting to run the training. This may be helpful in multi-gpu
        configurations with data over the network.
    proportion_train : float
        Proportion of the data to use for training.
    num_data_workers : int
        Number of workers to use for data loading.
    limit: Optional[int]
        If not `None`, the number of images to use. Otherwise, all images are used.
    reference_volume_path: Optional[str]
        If not `None`, the path to a reference volume to use for evaluation during training.
    """
    path: str = omegaconf.MISSING
    image: electron_image.ImageConfig = electron_image.ImageConfig()
    flip_contrast_images: bool = False
    make_local_copy: bool = False
    proportion_train: float = 0.95
    num_data_workers: int = 4
    limit: Optional[int] = None
    reference_volume_path: Optional[str] = None
    reference_volume_paths: Optional[List[str]] = None
    


@dataclasses.dataclass
class OptimConfig:
    """Configuration for the optimizer.

    Attributes
    ----------
    learning_rate : float
        The learning rate to use for the optimizer.
    weight_decay : float
        The weight decay to use for the optimizer.
    gradient_accumulation_steps : int, optional
        If not `None`, the number of steps to accumulate gradients before
        before updating the model.
    """
    learning_rate: float = 1e-3
    weight_decay: float = 0
    gradient_accumulation_steps: Optional[int] = None
    optimizer: Optional[str] = 'AdamW'


@dataclasses.dataclass
class InstaMapCriterionConfig:
    """Parameters for the loss function used for training.

    Attributes
    ----------
    normalize : bool
        If True, normalize the images to have mean 0 and standard deviation 1
        before computing the loss.
    normalization_penalty : float, optional
        If not `None`, adds a small penalty to encourage the model to output
        already normalized images.
    noise_ratio : float, optional
        If not `None`, additionally rescales the images by a factor computed
        such that the noisy image would have standard deviation 1. Otherwise,
        if set to `None` but normalization is requested, we use a scale free
        MSE loss which deduces the best scale by optimization.
    regularize_tv : float, optional
        If not `None`, adds a total variation regularization term to the loss
        with the given weight.
    """
    normalize: bool = True
    normalization_penalty: Optional[float] = 0.1
    noise_ratio: Optional[float] = None
    regularize_tv: Optional[float] = None
    sensor_type: Optional[str] = None


@dataclasses.dataclass
class RendererConfig:
    """Cnofiguration for the cryo-EM renderer.

    Attributes
    -----------
    spatial_sampling_rate : float
        The spatial sampling rate to use for the renderer, relative to the images' native resolution.
        Values greater than 1.0 will result in a lower resolution rendering, while values less than
        will result in a higher resolution rendering.
    depth_samples : int
        The number of samples in the depth dimension for each pixel.
    integration_depth : Optional[float]
        If not `None`, the depth to integrate to. Otherwise, the depth is determined by the
        extent of the volume.
    use_ctf : bool
        If `True`, uses the CTF to modulate the rendering. Otherwise, the CTF is ignored.
    extent : Optional[Tuple[float, float, float]]
        If not `None`, the extent of the volume in Angstroms. Otherwise, the extent is determined
        as a cube with side length given by the image size.
    jitter_scale : Optional[float]
        How much to scale the jitter augmentation
    """
    spatial_sampling_rate: float = 1.0
    depth_samples: int = 128
    integration_depth: Optional[float] = None
    use_ctf : bool = True
    extent: Optional[Tuple[float, float, float]] = None
    jitter_scale: Optional[float] = 1.0


@dataclasses.dataclass
class InstaMapTrainingConfig(TrainerConfiguration):
    """Configuration for training a InstaMap model.

    Attributes
    ----------
    rendering : InstaMapRenderingConfig
        Configuration for the rendering.
    batch_size : int
        Batch size per GPU.
    debug : bool
        If `True`, enables additional diagnostics during training (may slow down training).
    num_epochs : int
        Number of epochs to train for.
    num_gpus : int
        Number of GPUs to use for training.
    """
    augmentation: electron_image.InstaMapAugmentConfig = electron_image.InstaMapAugmentConfig()
    data: InstaMapDataConfig = InstaMapDataConfig()
    model: field.HashedImageFieldConfig = field.HashedImageFieldConfig()
    renderer: RendererConfig = RendererConfig()
    criterion: InstaMapCriterionConfig = InstaMapCriterionConfig()
    optim: OptimConfig = OptimConfig()
    batch_size: int = 2
    debug: bool = False
    num_epochs: int = 10
    num_gpus: int = 1
    log_every_n_steps: int = 50
    val_check_interval: float = 1.0
    volume_render_steps: tuple = ()
    regularization: Dict[str, float] = dataclasses.field(default_factory=dict)
    heterogeneity: electron_image.HeterogeneityConfig = electron_image.HeterogeneityConfig()
    pose: electron_image.PoseConfig = electron_image.PoseConfig()
    mask: electron_image.MaskConfig = electron_image.MaskConfig()



@overload
def fixup_config(config: InstaMapTrainingConfig) -> InstaMapTrainingConfig: ...


def fixup_config(config: InstaMapTrainingConfig):
    """Fixup configuration to handle default / missing values.

    This post-processes the configuration as necessary to handle
    more complex default values which are derived from other values.
    """
    if config.renderer.extent is None:
        if config.data.image.height != config.data.image.width:
            raise ValueError("Must specify extent if image is not square")

        box_size = config.data.image.height * config.data.image.pixel_size
        config.renderer.extent = (box_size, box_size, box_size)

    return config


def make_rendering_config(image: electron_image.ImageConfig, renderer: RendererConfig) -> electron_image.CryoEmRenderingConfig:
    """Create the rendering configuration from the given image / renderer configuration.
    """
    return electron_image.CryoEmRenderingConfig(
        image_height=int(image.height * renderer.spatial_sampling_rate),
        image_width=int(image.width * renderer.spatial_sampling_rate),
        depth_samples=renderer.depth_samples,
        pixel_size=image.pixel_size / renderer.spatial_sampling_rate,
        integration_depth=renderer.integration_depth,
        use_ctf=renderer.use_ctf,
        extent=renderer.extent)
