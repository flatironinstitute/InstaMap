"""Utilities to interpolate between fields.
"""

from __future__ import annotations

from typing import Union, Sequence

import torch

import instamap.nn.interpolation


class TrilinearScalarField(torch.nn.Module):
    """Utility module which represents a field from a sampled grid using trilinear interpolation.
    """
    field: torch.Tensor
    extent: torch.Tensor

    def __init__(
        self,
        field: torch.Tensor,
        extent: Union[float, Sequence[float]] = 2.0,
        padding_mode: str = 'border'
    ):
        """Create a new field from the given tensor.

        The tensor is assumed to represent the field within the given extent centered
        at the origin, i.e. [-extent/2, extent/2]^3.

        Parameters
        ----------
        field : torch.Tensor
            A 3d tensor of shape (H, W, D) or a 4d tensor of shape (C, H, W, D)
            containing the grid values to interpolate.
        extent : float or Sequence[float]
            The extent of the field in each dimension. If a single float is given, the extent is assumed to be the same
            in all dimensions. It is assumed that the field is centered at the origin.
        """
        super().__init__()

        extent = torch.as_tensor(extent, dtype=torch.float32)

        self.register_buffer("field", field)
        self.register_buffer("extent", extent)
        self.padding_mode = padding_mode

    def forward(self, x: torch.Tensor):
        return instamap.nn.interpolation.sample_3d(
            self.field, x.mul(2 / self.extent), padding_mode=self.padding_mode
        )

    def __repr__(self):
        return "TrilinearScalarField(field.shape={}, extent={})".format(
            list(self.field.shape), self.extent
        )
