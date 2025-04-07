"""Adapters modify existing sensors to add new functionality.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

import instamap.nn.affine
from .base import NullSensor
from ..electron_image import ImageConfig


class DownsamplingSensorAdapter(torch.nn.Module):
    """Adapts the given sensor model to be downsampled to the given size.

    This module allows for higher-resolution incoming wavefronts to be
    downsampled to the given observed image size. This may be useful to
    reduce aliasing issues in the rendering process.
    """
    def __init__(self, sensor: torch.nn.Module, size: Tuple[int, int], intensity: bool = True):
        """Create a new downsampling sensor adapter.

        Parameters
        ----------
        sensor : torch.nn.Module
            Underlying sensor model to use for likelihood evaluation.
        size : Tuple[int, int]
            Size to downsample the simulated image to.
        intensity : bool
            If `True`, the resulting image will be scaled to preserve intensity
            (i.e. illumination / area). Otherwise, the resulting image will be scaled
            to preserve total illumination.
        """
        super().__init__()

        self.size = size
        self.sensor = sensor
        self.intensity = intensity

    def _downsample(self, img: torch.Tensor):
        if img.shape[-2:] == self.size:
            return img

        scaling_factor = img.shape[-2] * img.shape[-1] / (self.size[0] * self.size[1])

        img = torch.nn.functional.interpolate(
            img.unsqueeze(1), self.size, mode='bilinear', align_corners=False, antialias=True
        ).squeeze_(1)

        if not self.intensity:
            img = img.mul_(scaling_factor)

        return img

    def forward(
        self,
        shot_info: Dict[str, torch.Tensor],
        simulated: torch.Tensor,
        observed: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        simulated = self._downsample(simulated)

        if observed is not None:
            observed = self._downsample(observed)

        return self.sensor(shot_info, simulated, observed, generator=generator)


class BackgroundNormalizingSensorAdapter(torch.nn.Module):
    """Adapter which normalizes the background image to mean 0 and variance 1.

    If provided, this adapter may choose to normalize based on the statistics of the
    entire image, or only the statistics outside of a circular mask of the given radius.
    """

    mask: Optional[torch.Tensor]

    def __init__(
        self, image: ImageConfig, sensor: torch.nn.Module, mask_radius: Optional[float] = None
    ):
        """Create a new normalizing sensor adapter.

        Parameters
        ----------
        image : ImageConfig
            Configuration of the image to normalize.
        sensor : torch.nn.Module
            Underlying sensor model to use for likelihood evaluation and simulation.
        mask_radius : Optional[float]
            If provided, the image will be normalized based on the statistics of a
            circular region of the given radius in Angstrom.
            Otherwise, the image will be normalized based on the statistics of the
            entire image.
        """

        super().__init__()

        self.sensor = sensor

        if mask_radius is not None:
            mask = instamap.nn.affine.make_circular_mask(
                (image.height, image.width), mask_radius * image.pixel_size / image.height
            ).sub_(1.0).neg_()

            if mask.sum() == 0:
                raise ValueError('No pixels are left after masking')

            self.register_buffer('mask', mask)
        else:
            self.mask = None

    def forward(
        self,
        shot_info: Dict[str, torch.Tensor],
        simulated: torch.Tensor,
        observed: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        simulated, info = self.sensor(shot_info, simulated, observed, generator=generator)

        if observed is not None:
            return simulated, info

        if self.mask is not None:
            n = self.mask.sum(dim=(-1, -2), keepdim=True)
            mu = simulated.mul(self.mask).sum(dim=(-1, -2), keepdim=True).div_(n)
            sigma = simulated.sub(mu).square_().mul_(self.mask).sum(dim=(-1, -2),
                                                                    keepdim=True).div_(n).sqrt_()
        else:
            mu = simulated.mean(dim=(-1, -2), keepdim=True)
            sigma = simulated.std(dim=(-1, -2), keepdim=True, unbiased=False)

        info['normalization_mu'] = mu
        info['normalization_sigma'] = sigma

        return simulated.sub(mu).div_(sigma), info


def normalize_background(img: torch.Tensor, mask_radius: float = 1.0) -> torch.Tensor:
    """Normalizes the background of the given image to mean 0 and variance 1.

    The background is defined as the pixels outside of a circular mask of the given radius.
    If no mask radius is provided, the entire image is used.

    Parameters
    ----------
    img : torch.Tensor
        The image to normalize.
    mask_radius : Optional[float]
        The radius of the circular mask to use, as a proportion of the half image size.
    """
    sensor = BackgroundNormalizingSensorAdapter(
        ImageConfig(img.shape[-2], img.shape[-1], 1.0), NullSensor(), mask_radius * img.shape[-2]
    )
    return sensor({}, img)[0]
