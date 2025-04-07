"""Integration of kernels for rasterization onto regular grids.
"""

from __future__ import annotations

import itertools

import math

from typing import Sequence

import torch


def integrate_gaussian_1d(boundaries: torch.Tensor, centers: torch.Tensor, scales: torch.Tensor):
    """Integrate a Gaussian density over a set of intervals given by boundaries.

    Parameters
    ----------
    boundaries : torch.Tensor
        Boundaries of intervals to integrate over, shape (n + 1,).
    centers : torch.Tensor
        Centers of Gaussian densities, shape (m,).
    scales : torch.Tensor
        Scales of Gaussian densities, shape (m,).

    Returns
    -------
    torch.Tensor
        Integrals of Gaussian densities over intervals, shape (m, n).
    """
    centers = centers.unsqueeze(-1)
    scales = scales.unsqueeze(-1)

    z = (boundaries - centers) / (scales * math.sqrt(2))
    return 0.5 * torch.diff(torch.special.erf(z))


def integrate_gaussian_nd(
    grid_shape: Sequence[int], centers: torch.Tensor, scales: torch.Tensor, weights: torch.Tensor
):
    """Integrate a sum of Gaussians over a grid given by the given shape.

    Computes the integral of a function which is represented as a weighted sum of Gaussians
    with the given centers and scales. The coordinate system correspond to the grid with
    unit spacing and with the origin at the center of the grid.

    Parameters
    ----------
    grid_shape : Sequence[int]
        Shape of grid to integrate over. The number of dimensions is inferred from the length of this sequence.
    centers : torch.Tensor
        Centers of Gaussian densities, shape (m, d).
    scales : torch.Tensor
        Scales of Gaussian densities, shape (m,) or (m, d). If specified per-dimension,
        these correspond to the standard deviations of the Gaussian densities along each dimension.
        Otherwise, the same scale is used for all dimensions.
    weights : torch.Tensor
        Weights of Gaussian densities, shape (m,).

    Returns
    -------
    torch.Tensor
        Integrals of Gaussian densities over grid, shape given by ``grid_shape``.
    """
    if scales.ndim == 1:
        scales = scales.unsqueeze(-1).expand_as(centers)

    if len(centers) == 0:
        return centers.new_zeros(grid_shape)

    marginals = [
        integrate_gaussian_1d(
            torch.arange(0, s + 1, dtype=centers.dtype, device=centers.device).sub_(s / 2),
            centers[:, i], scales[:, i]
        ) for i, s in enumerate(grid_shape)
    ]

    batch = len(marginals)



    if len(grid_shape) == 2:
        mx, my = marginals
        mx = mx.mul_(weights.unsqueeze(-1))
        result = torch.matmul(mx.mT, my)
    else:
        result = torch.einsum(
            *itertools.chain.from_iterable([(m, [batch, i]) for i, m in enumerate(marginals)]), weights,
            [batch], list(range(len(marginals)))
        )

    return result


def integrate_gaussian_nd_batch(
    grid: Sequence[int],
    subgrid: Sequence[int],
    centers: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor,
    cutoff_scale: float = 4
):
    """Integrates a weighted sum of Gaussians over a grid given by the given shape.
    Splits the grid by batch into subgrids to reduce memory usage and improve performance.

    Parameters
    ----------
    grid : Sequence[int]
        Shape of the grid to rasterize onto. The number of dimensions is inferred from the length of this sequence.
    subgrid : Sequence[int]
        Shape of the subgrids to use for integration. Must be the same length as ``grid``.
    centers : torch.Tensor
        Centers of Gaussian densities, shape (m, d). Units are in pixels.
    scales : torch.Tensor
        Scales of Gaussian densities, shape (m,) or (m, d). Units are in pixels.
    weights : torch.Tensor
        Weights of Gaussian densities, shape (m,).
    cutoff_scale : float
        Number of standard deviations after which density is considered zero.
    """
    if scales.ndim == 1:
        scales = scales.unsqueeze(-1).expand_as(centers)

    output = centers.new_empty(grid)

    num_subgrids = [int(math.ceil(g / s)) for g, s in zip(grid, subgrid)]

    grid_f = centers.new_tensor(grid)
    subgrid_f = centers.new_tensor(subgrid)

    for subgrid_idx in itertools.product(*[range(n) for n in num_subgrids]):
        subgrid_center = centers.new_tensor(subgrid_idx).mul_(subgrid_f
                                                              ).add_(subgrid_f / 2 - grid_f / 2)

        centers_subgrid = centers - subgrid_center

        mask = (centers_subgrid.abs().sub(subgrid_f / 2) < cutoff_scale * scales).all(dim=-1)

        centers_subgrid = centers_subgrid[mask]
        scales_subgrid = scales[mask]
        weights_subgrid = weights[mask]

        subgrid_output = integrate_gaussian_nd(
            subgrid, centers_subgrid, scales_subgrid, weights_subgrid
        )

        output_slice = output[tuple(
            slice(s * g, (s + 1) * g) for s, g in zip(subgrid_idx, subgrid)
        )]

        output_slice.copy_(subgrid_output[tuple(slice(g) for g in output_slice.shape)])

    return output


def scatter_add_point_masses(field: torch.Tensor, coordinates: torch.Tensor, weights: torch.Tensor):
    """Scatter-accumulate point masses onto a field.

    Parameters
    ----------
    field : torch.Tensor
        Field to scatter-accumulate onto.
    coordinates : torch.Tensor
        Coordinates of point masses, shape (m, d), in pixel units.
    weights : torch.Tensor
        Weights of point masses, shape (m,).
    """
    grid_shape = coordinates.new_tensor(field.shape)
    coordinates_pixel = coordinates.add(0.5 * grid_shape).floor_().long()
    stride = coordinates_pixel.new_tensor(field.stride(), dtype=torch.long)

    offset_linear = (coordinates_pixel * stride).sum(dim=-1)
    field.view(-1).scatter_add_(0, offset_linear, weights)
    return field


