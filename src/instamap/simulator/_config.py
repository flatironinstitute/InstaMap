from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass
class ImageConfig:
    """
    Attributes
    ----------
    height : int
        The height of the rendered image in pixels
    image_width : int
        The width of the rendered image in pixels
    pixel_size : float
        The size of a pixel in Angstroms
    """
    height: int = 160
    width: int = 160
    pixel_size: float = 1.2


@dataclasses.dataclass
class HeterogeneityConfig:
    enabled: bool = False
    downsample: float = 4.0
    multi_stage_training: bool = False
    num_epochs: int = 1
    optimizer: str = 'Adam'
    learning_rate: float = 1e-4
    fourier_crop: bool = False
    fourier_crop_to: int = 64
    num_hidden: int = 128

@dataclasses.dataclass
class PoseConfig:
    inference_enabled: bool = False
    n_output_pose_params: int = 5
    num_hidden: int = 16

@dataclasses.dataclass
class MaskConfig:
    masking_enabled: bool = False
    mask_path: Optional[str] = None