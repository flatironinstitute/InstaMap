from __future__ import annotations

import copy
import dataclasses
from typing import Callable, Dict, Optional, Tuple, TypedDict

import torch

import instamap.nn.fft
import instamap.nn.transform as transform
from . import ctf, crop, scattering
from ._config import ImageConfig, HeterogeneityConfig, PoseConfig, MaskConfig


@dataclasses.dataclass
class RenderingConfig:
    """Configuration for rendering from a given field.

    Attributes
    ----------
    image_height : int
        The height of the rendered image in pixels
    image_width : int
        The width of the rendered image in pixels
    depth_samples : int
        The number of samples along the view ray to use for rendering
    pixel_size : float
        The size of a pixel in Angstroms
    integration_depth : float, optional
        The depth of integration along the view ray in Angstroms.
        If `None`, the integration depth is automatically computed for each shot
        according to the bounding box of the volume.
    """
    image_height: int = 160
    image_width: int = 160
    depth_samples: int = 128
    pixel_size: float = 1.2
    integration_depth: Optional[float] = None

    
@dataclasses.dataclass
class CryoEmRenderingConfig(RenderingConfig):
    """Configuration for rendering cryo-em images.

    Attributes
    ----------
    use_ctf : bool
        Whether to use the CTF when rendering the image
    extent : Optional[Tuple[float, float, float]]
        If not `None`, the extent of the volume in Angstroms.
    """
    use_ctf: bool = True
    extent: Optional[Tuple[float, float, float]] = None
    heterogeneity: Optional[HeterogeneityConfig] = None
    pose: Optional[PoseConfig] = None
    mask: Optional[MaskConfig] = None


@dataclasses.dataclass
class InstaMapAugmentConfig:
    """Parameters for data augmentation during training.

    Attributes
    ----------
    angle : float, optional
        If not `None`, the maximum angle to perturb the rotation matrix by,
        in degrees.
    offset : float, optional
        If not `None`, the maximum offset to perturb the translation vector by,
        in Angstroms.
    """
    angle: Optional[float] = None
    offset: Optional[float] = None


class CryoEmShotInfo(TypedDict):
    view_phi: torch.Tensor
    view_theta: torch.Tensor
    view_psi: torch.Tensor

    offset_x: torch.Tensor
    offset_y: torch.Tensor

    defocus_u: torch.Tensor
    defocus_v: torch.Tensor
    defocus_angle: torch.Tensor
    spherical_aberration: torch.Tensor

    voltage: torch.Tensor
    amplitude_contrast_ratio: torch.Tensor

    phase_shift: torch.Tensor
    b_factor: torch.Tensor


def make_base_query_points_from_config(
    config: CryoEmRenderingConfig,
    integration_depth: Optional[float] = None,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    return scattering.make_base_query_points(
        config.image_height,
        config.image_width,
        config.depth_samples,
        config.pixel_size,
        integration_depth if integration_depth is not None else config.integration_depth,
        device=device,
        dtype=dtype
    )


def _make_transform_matrix(
    phi: torch.Tensor,
    theta: torch.Tensor,
    psi: torch.Tensor,
    t_x: Optional[torch.Tensor],
    t_y: Optional[torch.Tensor],
    augment_angle: Optional[float],
    augment_translation: Optional[float],
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Make transformation matrix from ZYZ rotation angles and in-plane translation."""
    transform_v = transform.make_rotation_from_angles_zyz(
        phi, theta, psi, degrees=True, intrinsic=True, homogeneous=True
    )

    if augment_angle is not None:
        transform_v = transform.apply_random_small_rotation(
            transform_v, augment_angle, degrees=True, generator=generator
        )

    if t_x is not None and t_y is not None:
        if augment_translation is not None:
            t_x = t_x + torch.rand(
                t_x.shape, generator=generator, device=t_x.device, dtype=t_x.dtype
            ).mul_(2).sub_(0.5).mul_(augment_translation)
            t_y = t_y + torch.rand(
                t_x.shape, generator=generator, device=t_x.device, dtype=t_x.dtype
            ).mul_(2).sub_(0.5).mul_(augment_translation)

        translation = transform.make_translation(t_x, t_y)
        transform_v = transform_v @ translation

    return transform_v


def make_transform_matrix(
    shot_info: CryoEmShotInfo,
    augment_angle: Optional[float] = None,
    augment_translation: Optional[float] = None,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Make transformation matrix from given shot information."""
    return _make_transform_matrix(
        shot_info['view_phi'], shot_info['view_theta'], shot_info['view_psi'],
        shot_info['offset_x'], shot_info['offset_y'], augment_angle, augment_translation, generator
    )


def compute_ctf_from_shot_info(
    shot_info: CryoEmShotInfo,
    image_shape: Tuple[int, int],
    pixel_size: float,
    freqs: torch.Tensor = None,
    device: Optional[torch.device] = None,
    normalize: bool = True
) -> torch.Tensor:
    if freqs is None:
        fx = torch.fft.fftfreq(image_shape[0], pixel_size, device=device)
        fy = torch.fft.fftfreq(image_shape[1], pixel_size, device=device)
        freqs = torch.stack(torch.meshgrid(fx, fy, indexing='ij'), -1)

    return ctf.compute_ctf_power(
        freqs,
        shot_info['defocus_u'],
        shot_info['defocus_v'],
        shot_info['defocus_angle'],
        shot_info['voltage'],
        shot_info['spherical_aberration'],
        shot_info['amplitude_contrast_ratio'],
        shot_info['phase_shift'],
        normalize=normalize
    )
