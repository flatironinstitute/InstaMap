"""Utilities for cropping views and computing effective transformation
"""

from __future__ import annotations
from typing import Tuple, Sequence

import torch

from . import electron_image


def adjust_shot_info_from_crop(
    shot_info: electron_image.CryoEmShotInfo, input_shape: Sequence[int], offset_x: int,
    offset_y: int, config: electron_image.ImageConfig
):
    """Adjusts the given shot information to account for a given crop.

    Parameters
    ----------
    shot_info : CryoEmShotInfo
        The shot information of the original image.
    input_shape : Sequence[int]
        The shape of the original image.
    offset_x : int
        Offset of the crop in the x direction.
    offset_y : int
        Offset of the crop in the y direction.
    config : RenderingConfig
        Configuration for the cropped image.
    """
    new_center_x = offset_x + config.width / 2
    new_center_y = offset_y + config.height / 2

    offset_center_x = new_center_x - input_shape[-1] / 2
    offset_center_y = new_center_y - input_shape[-2] / 2

    shot_info = {
        **shot_info,
        'offset_x': shot_info['offset_x'] + offset_center_x * config.pixel_size,
        'offset_y': shot_info['offset_y'] + offset_center_y * config.pixel_size,
    }

    return shot_info


def crop_view(x: torch.Tensor, config: electron_image.ImageConfig, offset_x: int, offset_y: int) -> torch.Tensor:
    """Crops a view to a smaller size, according to the given offset.
    """
    return x.narrow(-1, offset_x, config.width).narrow(-2, offset_y, config.height)


def make_random_offset(input_shape: Sequence[int], config: electron_image.ImageConfig, generator: torch.Generator = None, shape=()):
    """Makes a random offset for cropping a view.

    This ensures that the offsets are compatible with the bounds.
    """
    offset_x = torch.randint(0, input_shape[-1] - config.width, shape, generator=generator)
    offset_y = torch.randint(0, input_shape[-2] - config.height, shape, generator=generator)
    return offset_x, offset_y


def crop_view_random(
    x: torch.Tensor,
    shot_info: electron_image.CryoEmShotInfo,
    config: electron_image.ImageConfig,
    generator: torch.Generator = None
) -> Tuple[torch.Tensor, electron_image.CryoEmShotInfo]:
    """Crop a view to a smaller size randomly, and adjust the shot info accordingly.

    Parameters
    ----------
    x : torch.Tensor
        The view tensor to crop.
    shot_info : torch.Tensor
        The original shot info.
    config : RenderingConfig
        Rendering configuration for the output image
    """
    if x.shape[-2] == config.height and x.shape[-1] == config.width:
        return x, shot_info

    offset_x, offset_y = make_random_offset(x.shape, config, generator=generator)

    input_shape = x.shape

    shot_info = adjust_shot_info_from_crop(shot_info, input_shape, offset_x, offset_y, config)
    x = crop_view(x, config, offset_x, offset_y)

    return x, shot_info


def compute_extremal_point(
    extent: torch.Tensor,
    direction: torch.Tensor,
    origin: torch.Tensor,
    return_coefficient: bool = False
):
    """Computes extremal points of a ray intersecting with an axis-aligned box centered at the origin.

    Note that if the ray does not intersect with the box, the returned point will also
    not be inside the box, and the specific point returned is unspecified.

    Parameters
    ----------
    extent : torch.Tensor
        Tensor of shape (d,) containing the extent of the box in each dimension.
        The box is assumed to be centered at the origin, and aligned with the coordinate axes,
        and thus has extent [-extent / 2, extent / 2] in each dimension.
    direction : torch.Tensor
        Tensor of shape (d,) containing the direction of the ray.
    origin : torch.Tensor
        Tensor of shape (d,) containing the origin of the ray.
    return_coefficient : bool, optional
        If `True`, indicates that the coefficient of the point along the ray
        should be returned in addition to the point itself.

    Returns
    -------
    torch.Tensor
        Tensor of shape (2, d) containing the two extremal points of the ray intersecting with the box.
        The first point is in the negative direction of the ray, and the second point is in the positive direction.
    """
    extent = 0.5 * extent

    extreme_max_a, _ = (torch.copysign(extent, direction) - origin).div(direction).min(dim=-1)
    extreme_min_a, _ = (torch.copysign(extent, direction).neg() - origin).div(direction).max(dim=-1)

    extreme_a = torch.stack([extreme_min_a, extreme_max_a], dim=-1)
    points = torch.addcmul(origin.unsqueeze(-2), direction.unsqueeze(-2), extreme_a.unsqueeze(-1))

    if return_coefficient:
        return points, extreme_a
    else:
        return points
