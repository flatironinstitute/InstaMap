"""Models for the optics of the microscope (mostly dealing with the CTF)
"""

from __future__ import annotations

from typing import Optional, Tuple, TypedDict

import dataclasses
import torch

import instamap.nn.fft

from . import ctf
from ._config import ImageConfig
from .scattering import filters


class OpticsShotInfo(TypedDict):
    defocus_u: torch.Tensor
    defocus_v: torch.Tensor
    defocus_angle: torch.Tensor
    spherical_aberration: torch.Tensor
    voltage: torch.Tensor
    amplitude_contrast_ratio: torch.Tensor
    phase_shift: torch.Tensor
    b_factor: torch.Tensor


class FourierContrastTransferOptics(torch.nn.Module):
    """Optics model applying the contrast transfer function in Fourier space."""

    rfreqs: torch.Tensor

    def __init__(self, image_config: ImageConfig):
        super().__init__()

        self.register_buffer(
            'rfreqs',
            instamap.nn.fft.make_rfftn_freqs((image_config.height, image_config.width), image_config.pixel_size)
        )

        self.antialias = filters.AntiAliasingFilter(image_config, cutoff=1.0, real=True)


    def forward(
        self,
        scattered_image: torch.Tensor,
        info: OpticsShotInfo,
        generator: Optional[torch.Generator] = None
    ):
        ctf_power = ctf.compute_ctf_power(
            self.rfreqs, info['defocus_u'], info['defocus_v'], info['defocus_angle'],
            info['voltage'], info['spherical_aberration'], info['amplitude_contrast_ratio'],
            info['phase_shift'],
            info.get('b_factor'))

        image_ft = torch.fft.rfft2(scattered_image)
        image_ft *= ctf_power
        image_ft = self.antialias(image_ft, fourier=True)
        diffraction_image = torch.fft.irfft2(image_ft) 

        return diffraction_image, {}


class NullOptics(torch.nn.Module):
    """Null optics model which propagates the scattered image unchanged."""
    def forward(
        self,
        scattered_image: torch.Tensor,
        info: OpticsShotInfo,
        generator: Optional[torch.Generator] = None
    ):
        return scattered_image, {}

