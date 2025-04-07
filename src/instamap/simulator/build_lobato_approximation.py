"""Script to approximate family of potential with gaussian mixture.

For performance reasons, it may be necessary to approximate potentials by Gaussian mixtures.
This script performs the necessary non-linear fitting.

"""

from __future__ import annotations


import json
import math
from typing import Any, TypedDict
import torch

class BubbleGaussianApproximation(TypedDict):
    a: torch.Tensor
    s: torch.Tensor
    bubble: torch.Tensor


def adjust_scaling_to_pdf(s: torch.Tensor, d: int = 3) -> torch.Tensor:
    return math.pow(2 * math.pi, d / 2) * (s**d)


def load_lobato_approximation(fp,
                              adjust_scaling_to_density: bool = False
                              ) -> dict[int, BubbleGaussianApproximation]:
    """Loads saved approximation from JSON file.

    Parameters
    ----------
    fp : file-like object
        File-like object to read from
    adjust_scaling_to_density : bool
        If `True`, adjusts the scaling factor after loading to correspond to a
        weighted sum of normalized Gaussian PDFs. Otherwise, the scaling factor
        corresponds to a weighted sum of exp(-r^2 / s^2) kernels.
    """
    data = json.load(fp)

    parameters = data['potential']
    parameters = {
        int(k): BubbleGaussianApproximation(
            a=torch.tensor(v['a'], dtype=torch.float64),
            s=torch.tensor(v['s'], dtype=torch.float64),
            bubble=torch.tensor(v['bubble'], dtype=torch.float64)
        )
        for k, v in parameters.items()
    }

    if adjust_scaling_to_density:
        parameters = {
            k: BubbleGaussianApproximation(
                a=adjust_scaling_to_pdf(v['s'], d=3) * v['a'], s=v['s'], bubble=v['bubble']
            )
            for k, v in parameters.items()
        }

    return parameters
