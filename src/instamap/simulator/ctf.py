"""Contrast Transfer Function (CTF) computation
"""

from __future__ import annotations

from typing import Optional

import dataclasses
import math

import torch


def _broadcast(t: torch.Tensor):
    return t.unsqueeze(-1).unsqueeze_(-1)


@torch.jit.script
def compute_ctf_power(
    freqs: torch.Tensor,
    defocus_u: torch.Tensor,
    defocus_v: torch.Tensor,
    defocus_angle: torch.Tensor,
    voltage: torch.Tensor,
    spherical_aberration: torch.Tensor,
    amplitude_contrast_ratio: torch.Tensor,
    phase_shift: torch.Tensor,
    b_factor: Optional[torch.Tensor] = None,
    normalize: bool = True
) -> torch.Tensor:
    """Computes CTF with given parameters.

    The CTF is computed at the given set of 2d Fourier frequencies.

    Parameters
    ----------
    freqs : torch.Tensor
        A 2d tensor of shape (n_freq, 2) containing the 2d Fourier frequencies at which to compute the CTF.
    defocus_u : torch.Tensor
        The defocus in the major axis in Angstroms.
    defocus_v : torch.Tensor
        The defocus in the minor axis in Angstroms.
    defocus_angle : torch.Tensor
        The defocus angle in degree.
    voltage : torch.Tensor
        The accelerating voltage in kV.
    spherical_aberration : torch.Tensor
        The spherical aberration in mm.
    amplitude_contrast_ratio : torch.Tensor
        The amplitude contrast ratio.
    phase_shift : torch.Tensor
        The phase shift in degrees.
    b_factor : torch.Tensor, optional
        The B factor in A^2. If not provided, the B factor is assumed to be 0.
    normalize : bool, optional
        Whether to normalize the CTF so that it has norm 1 in real space. Default is True.
    """
    defocus_u = _broadcast(defocus_u)
    defocus_v = _broadcast(defocus_v)
    defocus_angle = _broadcast(torch.deg2rad(defocus_angle))
    voltage = _broadcast(voltage * 1000)
    spherical_aberration = _broadcast(spherical_aberration * 1e7) 
    amplitude_contrast_ratio = _broadcast(amplitude_contrast_ratio)
    phase_shift = _broadcast(torch.deg2rad(phase_shift))
    b_factor = _broadcast(b_factor) if b_factor is not None else None

    lam = 12.2643 / (voltage + 0.97845e-6 * voltage * voltage)**0.5
    kx, ky = torch.unbind(freqs, dim=-1)

    angle = torch.atan2(ky, kx)
    power = torch.square(kx) + torch.square(ky)

    defocus = 0.5 * (
        defocus_u + defocus_v + (defocus_u - defocus_v) * torch.cos(2 * (angle - defocus_angle))
    )

    gamma_defocus = -0.5 * defocus * lam * power
    gamma_sph = .25 * spherical_aberration * (lam**3) * (power**2)

    gamma = (2 * math.pi) * (gamma_defocus + gamma_sph) - phase_shift
    ctf = (1 - amplitude_contrast_ratio**
           2)**.5 * torch.sin(gamma) - amplitude_contrast_ratio * torch.cos(gamma)

    if normalize:
        ctf = ctf / (
            torch.norm(ctf, p=2, dim=(-2, -1), keepdim=True) /
            math.sqrt(ctf.shape[-2] * ctf.shape[-1])
        )


    if b_factor is not None:
        ctf *= torch.exp(-0.25 * b_factor * power)

    return ctf
