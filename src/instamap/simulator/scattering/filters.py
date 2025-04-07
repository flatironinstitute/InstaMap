"""Useful filters for propagation
"""

from __future__ import annotations

import math
from typing import Optional

import torch

import instamap.nn.fft

from .._config import ImageConfig


def make_anti_aliasing_filter(
    image: ImageConfig,
    cutoff: float = 0.667,
    rolloff: float = 0.05,
    real: bool = False,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Create an anti-aliasing filter.

    Parameters
    ----------
    image : ImageConfig
        The configuration of the image to create the filter for.
    cutoff : float, optional
        The cutoff frequency as a fraction of the Nyquist frequency, by default 0.667
    rolloff : float, optional
        The rolloff width as a fraction of the Nyquist frequency, by default 0.05
    real : bool, optional
        If `True`, indicates that the filter should be for a real transform.
        Otherwise, the filter will be for a complex transform.

    Returns
    -------
    torch.Tensor
        A tensor representing the anti-aliasing filter.
    """
    freqs = instamap.nn.fft.make_fftn_freqs(
        (image.height, image.width), image.pixel_size, real=real, dtype=dtype, device=device
    )

    k_max = 1 / (2 * image.pixel_size)
    k_cut = cutoff * k_max

    freqs_norm = freqs.norm(dim=-1)

    frequencies_cut = freqs_norm > k_cut

    if rolloff > 0:
        rolloff_width = rolloff * k_max
        mask = 0.5 * (1 + torch.cos((freqs_norm - k_cut - rolloff_width) / rolloff_width * math.pi))
        mask[frequencies_cut] = 0
        mask[freqs_norm <= k_cut - rolloff_width] = 1
    else:
        mask = 1 - frequencies_cut.to(torch.float32)

    return mask


class AntiAliasingFilter(torch.nn.Module):
    """Simple anti-aliasing filter.

    This filter cuts off all frequencies above a 2/3 Nyquist frequency.
    """

    mask: torch.Tensor

    def __init__(
        self, image: ImageConfig, cutoff: float = 0.667, rolloff: float = 0.1, real: bool = False
    ):
        super().__init__()

        mask = make_anti_aliasing_filter(image, cutoff=cutoff, rolloff=rolloff, real=real)
        self.register_buffer('mask', mask)
        self.real = real

    def forward(self, image: torch.Tensor, fourier: bool = False):
        if fourier:
            return image * self.mask
        else:
            if self.real:
                return torch.fft.irfft2(torch.fft.rfft2(image) * self.mask)
            else:
                result = torch.fft.ifft2(torch.fft.fft2(image) * self.mask)

                if not torch.is_complex(image):
                    result = result.abs()

                return result


    def __call__(self, image: torch.Tensor, fourier: bool = False):
        return super().__call__(image, fourier=fourier)


def low_pass_image(image: torch.Tensor, cutoff: float = 0.667, rolloff: float = 0.1):
    """Applies a low-pass filter to the given image.

    The low-pass filter is Fourier filter that cuts off all frequencies above the given
    multiple of the Nyquist frequency. Additionally, a rolloff width can be specified
    to smoothly transition between the pass and stop bands.
    """
    image_config = ImageConfig(image.shape[-2], image.shape[-1], pixel_size=1.0)
    f = AntiAliasingFilter(image_config, cutoff=cutoff, rolloff=rolloff, real=not torch.is_complex(image))
    return f(image)
