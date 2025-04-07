"""Implementations of model to compute scattered potential for a given field.

These implementations are parametrized by the given field and can be used to compute the scattered
potential for a (batch of) viewpoints using different approximations.

"""

from __future__ import annotations

import dataclasses

from typing import Dict, Optional, Tuple, TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

from instamap.nn import transform
from instamap.simulator.fourier import fourier_crop

from .. import integration
from .._config import ImageConfig, HeterogeneityConfig, PoseConfig, MaskConfig

try:
    import torch_scatter
except ImportError:
    torch_scatter = None


@dataclasses.dataclass
class RenderingConfig(ImageConfig):
    """Configuration for rendering from a given field.

    Attributes
    ----------
    depth_samples : int
        The number of samples along the view ray to use for rendering
    """
    depth_samples: int = 128
    heterogeneity: Optional[HeterogeneityConfig] = None
    pose: Optional[PoseConfig] = None
    mask: Optional[MaskConfig] = None

@torch.jit.script
def make_axis_aligned_grid(
    grid: Tuple[int, int, int],
    spacing: torch.Tensor,
    homogeneous: bool = True,
    center_grid_origin: bool = True,
) -> torch.Tensor:
    """Creates a tensor representing points in an axis-aligned grid, with the given number
    of samples in each direction, and with the given spacing between each grid point.
    The device and dtype of the output tensor will match the device and dtype of the
    the `spacing` tensor.

    Parameters
    ----------
    grid : Tuple[int, int, int]
        The number of samples in each dimension
    spacing : torch.Tensor
        The spacing between each grid point in each dimension
    homogeneous : bool, optional
        If `True`, the output tensor will represent points in homogeneous coordinates.
        Otherwise, the output tensor will represent points in affine coordinates.
    center_grid_origin : bool, optional
        If `True`, the grid will be centered at the origin (i.e. (0, 0, 0)).
        Otherwise, it will be offset by a half grid spacing in each dimension.
    """
    device = spacing.device
    dtype = spacing.dtype

    half_pixel_offset = 0.5 if center_grid_origin else 0.0

    gx, gy, gz = grid
    px = torch.arange(0, gx, device=device,
                      dtype=dtype).sub_(gx / 2 - half_pixel_offset).mul_(spacing[0])
    py = torch.arange(0, gy, device=device,
                      dtype=dtype).sub_(gy / 2 - half_pixel_offset).mul_(spacing[1])
    pz = torch.arange(0, gz, device=device,
                      dtype=dtype).sub_(gz / 2 - half_pixel_offset).mul_(spacing[2])

    if homogeneous:
        grid_points = torch.cartesian_prod(px, py, pz, px.new_ones(1))
    else:
        grid_points = torch.cartesian_prod(px, py, pz)
    grid_points = grid_points.view(gx, gy, gz, -1)
    return grid_points


@torch.jit.script
def make_base_query_points(
    height: int,
    width: int,
    depth_samples: int,
    pixel_size: float,
    integration_depth: float,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Make a tensor representing points in an axis-aligned grid, with the given number
    of samples in each dimension. Returns a tensor of shape `[height, width, depth, 4]`,
    representing points in homogeneous xyz pixel space.

    The output represents a cube, centered at the origin.
    """
    return make_axis_aligned_grid(
        (height, width, depth_samples),
        torch.as_tensor(
            [pixel_size, pixel_size, integration_depth / depth_samples], device=device, dtype=dtype
        ),
        homogeneous=True,
        center_grid_origin=True,
    )


@torch.jit.script
def _transform_query_points_for_shot(
    transform_matrix: torch.Tensor, query_points: torch.Tensor
) -> torch.Tensor:
    transform_matrix = transform_matrix[..., None, None, None, :, :]

    query_points = torch.matmul(transform_matrix, query_points.unsqueeze(-1)).squeeze(-1)

    return query_points[..., :3] / torch.unsqueeze(query_points[..., 3], -1)


def sparse_distance_matrix(points, distance_cutoff, n_points_per_chunk):
    ''' Notes: some error in mask.nonzero when points_per_chunk ~> 900'''

    sparse_matrices = []
    index_offset = 0
    for indices in torch.chunk(torch.arange(len(points)), len(points)//n_points_per_chunk):
        
        dist = torch.cdist(points[indices,:3], points[:,:3])


        mask = dist <= distance_cutoff

        indices = mask.nonzero(as_tuple=False).t()

        values = dist[mask]

        indices[0] += index_offset
        sparse_dist_matrix = torch.sparse_coo_tensor(indices, values, dist.size())
        sparse_dist_matrix
        sparse_matrices.append(sparse_dist_matrix)
        index_offset += len(indices[0])

    sparse_distance = torch.concat(sparse_matrices)
    return sparse_distance


class ScatteringShotInfo(TypedDict):
    """Parameters specifying the rendering for a shot of a given field.
    """
    view_phi: torch.Tensor
    view_theta: torch.Tensor
    view_psi: torch.Tensor
    offset_x: torch.Tensor
    offset_y: torch.Tensor


class GridJitter(torch.nn.Module):
    enabled: bool
    scale: torch.Tensor

    def __init__(self, scale, enabled: bool = True):
        super().__init__()

        self.register_buffer('scale', torch.as_tensor(scale))
        self.enabled = enabled

    def forward(self, points: torch.Tensor, generator=None):
        if not self.enabled or not self.training:
            return points

        jitter = torch.rand(
            points.shape, dtype=points.dtype, device=points.device, generator=generator
        )
        jitter.sub_(0.5).mul_(self.scale).add_(points)
        return jitter



class FullyConnectedMLP(nn.Module):
    def __init__(self, input_dim, num_points, output_dim, num_hidden):
        super(FullyConnectedMLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_points * output_dim)
        
        self.num_points = num_points
        self.output_dim = output_dim

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        x = x.view(-1, self.num_points, self.output_dim)
        
        return x
    

class GridDisplace(torch.nn.Module):

    def __init__(self, image_shape: Tuple[int,int], num_points: int, num_hidden: int):
        super().__init__()
        output_dim=3
        height, width = image_shape
        assert height == width
        self.model = FullyConnectedMLP(input_dim=height*width, num_points=num_points, output_dim=output_dim, num_hidden=num_hidden)

    def forward(self, image: torch.Tensor):
        vector_field = self.model(image)
        return vector_field


class DirectRectangularProjection(torch.nn.Module):
    """Projects the field onto the given view direction, using a rectangular
    sampling scheme.
    """
    extent: torch.Tensor
    jitter_scale: torch.Tensor
    grid: torch.Tensor

    def __init__(
        self,
        field: torch.nn.Module,
        config: RenderingConfig,
        extent: Tuple[float, float, float],
        jitter: bool = False,
    ):
        """Create a new renderer with the give parameters.

        Parameters
        ----------
        field : torch.nn.Module
            The underlying field to render from
        config : RenderingConfig
            General configuration for rendering
        extent : Tuple[float, float, float]
            The extent of the field in each dimension, in Angstroms.
            This is used to determine the size of the sampling grid.
        jitter : bool, optional
            If `True`, the sampling points will be jittered randomly within their
            boxes in training mode. Otherwise, they will be sampled at the center
            of the box.
        """

        super().__init__()

        self.config = config

        self.jitter = jitter
        self.field = field

        self.register_buffer('extent', torch.as_tensor(extent, dtype=torch.float32))

        self.integration_depth = extent[2]

        grid = make_base_query_points(
            config.height, config.width, config.depth_samples, config.pixel_size,
            self.integration_depth
        )
        self.register_buffer('grid', grid.contiguous(), persistent=False)

        self.jitter = GridJitter(
            [
                config.pixel_size, config.pixel_size, self.integration_depth / config.depth_samples,
                0.0
            ],
            enabled=jitter
        )

    def forward(
        self,
        transform_matrix: torch.Tensor,
        generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        grid = self.jitter(self.grid, generator=generator)
        query_points = _transform_query_points_for_shot(transform_matrix, grid)

        potential: torch.Tensor = self.field(query_points).float()

        dz = self.integration_depth / self.config.depth_samples
        diffraction_image = potential.sum(dim=-1).mul_(dz)

        return diffraction_image, {}

    def rasterize(self, size: Tuple[int, int, int]) -> torch.Tensor:
        grid = make_axis_aligned_grid(
            size,
            self.extent / self.extent.new_tensor(size),
            homogeneous=False)

        return self.field(grid)


def _make_sphere_grid(
    grid: Tuple[int, int, int], spacing: torch.Tensor, radius: float, homogeneous: bool = True
):
    """Compute grid of points restricted to those within a sphere of the given radius.

    This function additionally computes a pointer array which may be used to project the sphere
    points into a two-dimensional image.
    """
    base_grid = make_axis_aligned_grid(
        grid, spacing, homogeneous=homogeneous, center_grid_origin=True
    )
    mask = torch.norm(base_grid[..., :3], dim=-1) <= radius

    bin_counts = mask.to(torch.int64).sum(dim=-1).flatten()

    pointers = bin_counts.new_zeros(bin_counts.numel() + 1)
    torch.cumsum(bin_counts, dim=0, out=pointers[1:])

    return base_grid, base_grid[mask], pointers


def _make_masked_grid(
    grid: Tuple[int, int, int], spacing: torch.Tensor, radius: float, homogeneous: bool = True
):
    """Compute grid of points restricted to those within a sphere of the given radius.

    This function additionally computes a pointer array which may be used to project the sphere
    points into a two-dimensional image.
    """
    base_grid = make_axis_aligned_grid(
        grid, spacing, homogeneous=homogeneous, center_grid_origin=True
    )
    mask = base_grid[:,:,:,-2] < 0

    bin_counts = mask.to(torch.int64).sum(dim=-1).flatten()

    pointers = bin_counts.new_zeros(bin_counts.numel() + 1)
    torch.cumsum(bin_counts, dim=0, out=pointers[1:])

    return base_grid, base_grid[mask], pointers


class DirectMaskedProjection(torch.nn.Module):
    """Projects the field onto a given view direction, using a masked sampling scheme.

    """
    grid: torch.Tensor
    pointers: torch.Tensor
    extent: torch.Tensor

    def __init__(
        self,
        field: torch.nn.Module,
        config: RenderingConfig,
        extent: Tuple[float, float, float],
        jitter: bool = False,
        force_sparse_matrix: bool = False,
        jitter_scale: float = 1.0,
    ):
        super().__init__()

        self.field = field
        self.radius = max(config.height, config.width, config.depth_samples) * config.pixel_size / 2
        base_grid, grid, _ = _make_sphere_grid(
            (config.height, config.width, config.depth_samples),
            torch.as_tensor(
                [config.pixel_size, config.pixel_size, extent[2] / config.depth_samples]
            ), self.radius
        )
        base_grid = make_axis_aligned_grid(
        (config.height, config.width, config.depth_samples), 
        torch.as_tensor(
                [config.pixel_size, config.pixel_size, extent[2] / config.depth_samples]
            ), 
            homogeneous=True, center_grid_origin=True
    )

        self.register_buffer('base_grid', base_grid)
        self.register_buffer('grid', grid)

        self.register_buffer('extent', torch.as_tensor(extent, dtype=torch.float32))
        self.image_shape = (config.height, config.width)
        self.dz = extent[2] / config.depth_samples

        self.config = config 

        self.jitter = GridJitter(
            [jitter_scale * config.pixel_size, jitter_scale * config.pixel_size, jitter_scale * self.dz, 0.0], enabled=jitter
        )

        self.use_sparse_matrix = force_sparse_matrix or torch_scatter is None


        if config.mask.masking_enabled:
            mask = torch.from_numpy(np.load(config.mask.mask_path)).transpose(0,-1) 
            mask = (mask > 1e-6).float()
            self.register_buffer('mask_array', mask)
            self.mask_shape = mask.shape


    def setup_masked_buffers(self,
                  transform_matrix: torch.Tensor,
                  config: RenderingConfig,
                  extent: Tuple[float, float, float],
                  generator: Optional[torch.Generator] = None,
                  ):

        jittered_base_grid = self.jitter(self.base_grid, generator=generator)
        x_dim, y_dim, z_dim = self.base_grid.shape[:3]
        query_points = torch.matmul(transform_matrix.unsqueeze(-3), self.base_grid.reshape(x_dim*y_dim*z_dim,4,1)).squeeze_(-1) 

        assert len(query_points) == 1, 'in current implementation, batch size must be one'
        neg_one_to_one_query_points = torch.empty(x_dim*y_dim*z_dim,4).to(query_points.device)
        neg_one_to_one_query_points[:,3] = 1
        neg_one_to_one_query_points[:,:3] = query_points.squeeze(0)[:,:3] / self.base_grid.reshape(x_dim*y_dim*z_dim,4)[:,:3].abs().max()
        n_batch = 1
        n_dim, n_channels = 3, 1
        n_points = len(neg_one_to_one_query_points)
        mask_linear_interpolate = F.grid_sample(self.mask_array.reshape((n_batch,n_channels,) + self.mask_shape),
                              neg_one_to_one_query_points[:,:3].reshape(n_batch,1,n_points,1,n_dim),
                                                                    mode='bilinear', padding_mode='zeros', align_corners=True).reshape(n_channels,n_points).transpose(1,0).reshape(x_dim,y_dim,z_dim,n_channels)
        mask = mask_linear_interpolate.squeeze(-1).to(torch.bool)
        bin_counts = mask.to(torch.int64).sum(dim=-1).flatten() 

        pointers = bin_counts.new_zeros(bin_counts.numel() + 1)
        torch.cumsum(bin_counts, dim=0, out=pointers[1:])
        
        self.register_buffer('grid', jittered_base_grid[mask])
        self.register_buffer('pointers', pointers)

        if self.use_sparse_matrix:
            self.register_buffer('sparse_col_indices',torch.arange(self.pointers[-1].item(), dtype=self.pointers.dtype))
            self.register_buffer('sparse_values', torch.ones(self.pointers[-1].item(), dtype=torch.float32))


    def forward(
        self,
        transform_matrix: torch.Tensor,
        do_het: Optional[bool] = False,
        image: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        self.setup_masked_buffers(transform_matrix, self.config, self.extent, generator=generator)
        grid = self.grid
        query_points = torch.matmul(transform_matrix.unsqueeze(-3), grid.unsqueeze(-1)).squeeze_(-1)
        query_points = query_points[..., :3] / query_points[..., 3].unsqueeze(-1)

        potential: torch.Tensor = self.field(query_points).float()

        batch_shape = potential.shape[:-1]
        batch_prefix = (1, ) * len(batch_shape)

        if self.use_sparse_matrix:
            m = torch.sparse_csr_tensor(
                self.pointers,
                self.sparse_col_indices.to('cuda'),
                self.sparse_values.to('cuda'),
                size=(self.pointers.numel() - 1, len(self.grid))
            )
            diffraction_image = m.matmul(potential.unsqueeze_(-1)).squeeze_(-1)
        else:
            diffraction_image = torch_scatter.segment_sum_csr(
                potential, self.pointers.view(*batch_prefix, -1)
            )

        diffraction_image = diffraction_image.view(*diffraction_image.shape[:-1], *self.image_shape)
        diffraction_image.mul_(self.dz)

        return diffraction_image, {}

    def rasterize(self, size: Tuple[int, int, int]) -> torch.Tensor:
        grid = make_axis_aligned_grid(
            size,
            self.extent / self.extent.new_tensor(size),
            homogeneous=False,
        )

        result = self.field(grid)

        mask = torch.norm(grid, dim=-1) <= self.radius
        result *= mask.to(result.dtype)

        return result

class DirectSphereProjection(torch.nn.Module):
    """Projects the field onto a given view direction, using a spherical sampling scheme.

    Unlike `DirectRectangularProjection`, this class only samples within the sphere inscribed
    within the box defined by the given extent. It behaves as if the field were zero outside
    of this sphere.

    This presents a substantial computational saving, as the number of samples is reduced
    by about a factor of 2. It may also interact favourably with learnable fields which can
    allocate parameters towards the interior of the sphere.

    The accumulation of the field is performed using a segmented reduction with pre-computed
    indices according to the which projected pixel each sampling point belongs to.
    """
    grid: torch.Tensor
    pointers: torch.Tensor
    extent: torch.Tensor
    heterogeneity: Dict

    def __init__(
        self,
        field: torch.nn.Module,
        config: RenderingConfig,
        extent: Tuple[float, float, float],
        jitter: bool = False,
        force_sparse_matrix: bool = False,
        jitter_scale: float = 1.0,
    ):
        super().__init__()

        self.field = field
        self.radius = max(config.height, config.width, config.depth_samples) * config.pixel_size / 2

        _, grid, pointers = _make_sphere_grid(
            (config.height, config.width, config.depth_samples),
            torch.as_tensor(
                [config.pixel_size, config.pixel_size, extent[2] / config.depth_samples]
            ), self.radius
        ) 

        self.register_buffer('grid', grid)
        self.register_buffer('pointers', pointers)

        self.register_buffer('extent', torch.as_tensor(extent, dtype=torch.float32))
        self.image_shape = (config.height, config.width)
        self.dz = extent[2] / config.depth_samples

        
        self.jitter = GridJitter(
            [jitter_scale * config.pixel_size, jitter_scale * config.pixel_size, jitter_scale * self.dz, 0.0], enabled=jitter
        )

        self.use_sparse_matrix = force_sparse_matrix or torch_scatter is None

        if self.use_sparse_matrix:

            self.register_buffer(
                'sparse_col_indices',
                torch.arange(self.pointers[-1].item(), dtype=self.pointers.dtype)
            )
            self.register_buffer(
                'sparse_values', torch.ones(self.pointers[-1].item(), dtype=torch.float32)
            )

        self.heterogeneity = config.heterogeneity
        if config.heterogeneity.enabled:
            height, width = self.image_shape
            assert height == width

            self.num_points_3root = round(width / config.heterogeneity.downsample)
            self.num_points = self.num_points_3root**3

            if self.heterogeneity.fourier_crop:
                height = width = self.heterogeneity.fourier_crop_to

            self.displace = GridDisplace((height, width),self.num_points, num_hidden=self.heterogeneity.num_hidden)

        self.heterogeneity = config.heterogeneity



    def total_variation(self, deformation_gradient: torch.Tensor) -> torch.Tensor:
        tv = torch.nanmean(deformation_gradient.abs().mean())
        return tv

    

    def forward(
        self,
        transform_matrix: torch.Tensor,
        do_het: Optional[bool] = False,
        image: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        grid = self.jitter(self.grid, generator=generator)


        grid_interpolate = False
        n_batch = len(transform_matrix.reshape(-1, 4, 4))
        if image is not None:
            assert len(image) == n_batch
        vector_field_on_regular_array = None

        query_points = torch.matmul(transform_matrix.unsqueeze(-3), grid.unsqueeze(-1)).squeeze_(-1)
        query_points = query_points[..., :3] / query_points[..., 3].unsqueeze(-1)


        rotated_vector_field = 0
        if do_het and len(image) == 1 and self.heterogeneity.enabled:
            grid_interpolate = True
            assert image is not None
            if grid_interpolate:
                n_dim = 3
                if self.heterogeneity.fourier_crop:
                    image = fourier_crop(image, self.heterogeneity.fourier_crop_to)
                vector_field_on_regular_array = self.displace(image.reshape(-1, image.numel())).reshape(
                    n_batch, n_dim, self.num_points_3root, self.num_points_3root, self.num_points_3root)

                n_points = len(grid)
                neg_one_to_one_query_points = torch.matmul(transform_matrix.unsqueeze(-3), grid.unsqueeze(-1)).squeeze(-1).squeeze_(0)
                neg_one_to_one_query_points[:, :3] = query_points.squeeze(0)[:, :3] / grid[:, :3].abs().max()
                rotated_vector_field = vector_field_on_query_points = F.grid_sample(
                    vector_field_on_regular_array,
                    neg_one_to_one_query_points[:, :3].reshape(n_batch, 1, n_points, 1, n_dim),
                    mode='bilinear', padding_mode='zeros', align_corners=True).reshape(n_dim, n_points).transpose(1, 0)

                rotated_vector_field = rotated_vector_field.reshape(query_points.shape)

        potential: torch.Tensor = self.field(query_points + rotated_vector_field).float()


        batch_shape = potential.shape[:-1]
        batch_prefix = (1,) * len(batch_shape)


        if self.use_sparse_matrix:
            m = torch.sparse_csr_tensor(
                self.pointers,
                self.sparse_col_indices,
                self.sparse_values,
                size=(self.pointers.numel() - 1, len(self.grid))
            )

            diffraction_image = m.matmul(potential.unsqueeze_(-1)).squeeze_(-1)

        else:
            diffraction_image = torch_scatter.segment_sum_csr(
                potential, self.pointers.view(*batch_prefix, -1)
            )

        diffraction_image = diffraction_image.view(*diffraction_image.shape[:-1], *self.image_shape)
        diffraction_image.mul_(self.dz)


        return_dict = {
            'vector_field_on_regular_array': vector_field_on_regular_array,
            'rotated_vector_field': rotated_vector_field
        }


        return diffraction_image, return_dict

    def rasterize(self, size: Tuple[int, int, int]) -> torch.Tensor:
        grid = make_axis_aligned_grid(
            size,
            self.extent / self.extent.new_tensor(size),
            homogeneous=False,
        )

        result = self.field(grid)

        # Apply mask to remove points outside of sphere
        mask = torch.norm(grid, dim=-1) <= self.radius
        result *= mask.to(result.dtype)

        return result


class GaussianProjectionIntegration(torch.nn.Module):
    """Projects a set of gaussians onto a given view direction, and directly integrates
    the result to obtain the scattered wave in the projection approximation.

    Unlike sampled integration from a numerical field representation, this method directly
    integrates gaussians which allows the efficient processing of very high resolution maps
    (> 2048^2) in a reasonably efficient manner.

    Attributes
    ----------
    angstrom_to_pixel : float
        Conversion factor from angstroms units to pixel units.
    centers : torch.Tensor
        Tensor of shape (N, 3) containing the centers of the gaussians, in angstroms.
    scales : torch.Tensor
        Tensor of shape (N, F) containing the scales of the gaussians, in pixels.
    weights : torch.Tensor
        Tensor of shape (N, F) containing the weights of the gaussians.
    """
    angstrom_to_pixel: float
    centers: torch.Tensor
    scales: torch.Tensor
    weights: torch.Tensor

    def __init__(
        self,
        config: ImageConfig,
        centers: torch.Tensor,
        scales: torch.Tensor,
        weights: torch.Tensor,
        bubble_weights: Optional[torch.Tensor] = None
    ):
        """Create a new field from a set of gaussians.

        Parameters
        ----------
        centers : torch.Tensor
            A tensor of shape (N, 3) containing the centers of the gaussians.
        scales : torch.Tensor
            A tensor of shape (N, F) containing a set of scales for each gaussian location.
            Having multiple scales allows the representation of non-gaussian shapes as
            superpositions of gaussians with different scales.
        weights : torch.Tensor
            A tensor of shape (N, F) containing the weights of each gaussian.
        bubble_weights : Optional[torch.Tensor]
            A tensor of sahpe (N,) containing a bubble weight at each center.
            This corresponds to assigning a weight to a gaussian of zero scale.
        """
        super().__init__()
        self.config = config

        centers = torch.as_tensor(centers)
        if centers.shape[-1] == 3:
            centers = torch.cat([centers, torch.ones_like(centers[..., :1])], dim=-1)

        self.angstrom_to_pixel = 1 / config.pixel_size
        self.inverse_area_element = 1 / config.pixel_size ** 2

        scales = torch.as_tensor(scales).mul(self.angstrom_to_pixel)
        weights = torch.as_tensor(weights)

        if scales.ndim == 1:
            scales = scales.unsqueeze(-1)

        self.register_buffer('centers', centers)
        self.register_buffer('scales', scales)
        self.register_buffer('weights', weights)

        if bubble_weights is not None:
            self.register_buffer('bubble_weights', bubble_weights)
        else:
            self.bubble_weights = None

    def forward(
        self,
        transform_matrix: torch.Tensor,
        do_het: Optional[bool] = False,
        image: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Computes the scattered wave in the projection approximation.

        Parameters
        ----------
        transform_matrix : torch.Tensor
            A tensor of shape (..., 4, 4) containing the transformation matrices.
        """
        centers_projected, _ = torch.linalg.solve_ex(
            transform_matrix.unsqueeze(-3), self.centers.unsqueeze(-1),
            check_errors=False,
        )
        centers_projected: torch.Tensor = centers_projected.squeeze(-1)

        centers_projected = centers_projected[..., :2] / centers_projected[..., -1].unsqueeze(-1)
        centers_projected = centers_projected.mul_(self.angstrom_to_pixel)

        batch_dimensions = centers_projected.shape[:-2]

        centers_projected = centers_projected.view(-1, centers_projected.shape[-2], 2)

        result = centers_projected.new_empty(
            batch_dimensions + (self.config.height, self.config.width)
        )
        result_flat = result.view(-1, result.shape[-2], result.shape[-1])

        for i in range(centers_projected.shape[0]):
            output = integration.integrate_gaussian_nd_batch(
                (self.config.height, self.config.width),
                (128, 128),
                centers=centers_projected[i].unsqueeze(1).expand(-1, self.scales.shape[1],
                                                                     -1).reshape(-1, 2),
                scales=self.scales.reshape(-1),
                weights=self.weights.reshape(-1)
            )
            if self.bubble_weights is not None:
                integration.scatter_add_point_masses(
                    output, centers_projected[i], self.bubble_weights
                )
            result_flat[i] = output

        result.mul_(self.inverse_area_element)

        return result, {}


def _make_transform_matrix(
    phi: torch.Tensor,
    theta: torch.Tensor,
    psi: torch.Tensor,
    t_x: torch.Tensor,
    t_y: torch.Tensor,
    augment_angle: Optional[float],
    augment_translation: Optional[float],
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Make transformation matrix from ZYZ rotation angles and in-plane translation.

    Parameters
    ----------
    augment_angle : float, optional
        If not `None`, the rotation is augmented by randomly sampling a small rotation
        constrained by the given angle in degrees.
    augment_translation : float, optional
        If not `None`, the translation is augmented by randomly sampling a small translation
        constrained by the given value in angstroms.
    """
    transform_v = transform.make_rotation_from_angles_zyz(
        phi, theta, psi, degrees=True, intrinsic=True, homogeneous=True
    )

    if augment_angle is not None:
        transform_v = transform.apply_random_small_rotation(
            transform_v, augment_angle, degrees=True, generator=generator
        )

    if augment_translation is not None:
        t_x = t_x + torch.rand(t_x.shape, generator=generator, device=t_x.device,
                               dtype=t_x.dtype).mul_(2).sub_(0.5).mul_(augment_translation)
        t_y = t_y + torch.rand(t_x.shape, generator=generator, device=t_x.device,
                               dtype=t_x.dtype).mul_(2).sub_(0.5).mul_(augment_translation)

    translation = transform.make_translation(-t_y, -t_x)
    transform_v = transform_v @ translation

    return transform_v


@dataclasses.dataclass
class ViewAugmentationConfig:
    """Configuration for random augmentation of the pose (view angle and offset)

    Attributes
    ----------
    angle : float, optional
        If not `None`, the amount of random rotation to apply to the view angle, in degrees.
    offset : float, optional
        If not `None`, the amount of random translation to apply to the view offset, in angstroms.
    """
    angle: Optional[float] = None
    offset: Optional[float] = None

class PoseInference(torch.nn.Module):

    def __init__(self, image_shape: Tuple[int,int], n_output_pose_params: int, num_hidden: int):
        super().__init__()
        height, width = image_shape
        assert height == width
        n_pose_params_input = 16
        self.n_output_pose_params = n_output_pose_params
        self.model = FullyConnectedMLP(input_dim=height*width + n_pose_params_input, num_points=1, output_dim=n_output_pose_params, num_hidden=num_hidden)

    def forward(self, image: torch.Tensor, pose: torch.Tensor):
        assert len(image) == len(pose)
        assert len(image.shape) == len(pose.shape) == 2
        pose = self.model(torch.concat([image, pose],dim=-1)).reshape(-1,self.n_output_pose_params)
        return pose

class ViewAugmentingAdapter(torch.nn.Module):
    """Adapts a renderer to render from the given `ScatteringShotInfo` object,
    optionally applying random view angle and offset perturbations.
    """
    def __init__(self, renderer: torch.nn.Module, 
                 config: Optional[ViewAugmentationConfig] = None, 
                 image_config: Optional[RenderingConfig] = None,
                 ):
        super().__init__()

        if config is None:
            config = ViewAugmentationConfig()

        self.renderer = renderer
        self.augment_angle = config.angle
        self.augment_offset = config.offset

        if image_config is not None and image_config.pose.inference_enabled: 
            self.image_shape = image_config.height, image_config.width
            self.pose_inference = PoseInference(self.image_shape, n_output_pose_params=image_config.pose.n_output_pose_params, num_hidden=image_config.pose.num_hidden)
            self.pose_config = image_config.pose

    def forward(self, 
                view_info: ScatteringShotInfo, 
                do_het: Optional[bool] = False,
                image: Optional[torch.Tensor] = None,
                generator: Optional[torch.Generator] = None):
        transform_matrix = _make_transform_matrix(
            view_info['view_phi'],
            view_info['view_theta'],
            view_info['view_psi'],
            view_info['offset_x'],
            view_info['offset_y'],
            self.augment_angle,
            self.augment_offset,
            generator=generator
        )

        if hasattr(self,'pose_config') and self.pose_config.inference_enabled:
            inferred_pose_delta = self.pose_inference(image.reshape(len(image),-1),
                                                                   transform_matrix.reshape(len(image.reshape((-1,)+self.image_shape)), -1))
            d_phi, d_theta, d_psi, d_x, d_y = inferred_pose_delta.transpose(0,1)
            transform_matrix = _make_transform_matrix(
                view_info['view_phi'] + d_phi,
                view_info['view_theta'] + d_theta,
                view_info['view_psi'] + d_psi,
                view_info['offset_x'] + d_x,
                view_info['offset_y'] + d_y,
                self.augment_angle,
                self.augment_offset,
                generator=generator
            )
        projection = self.renderer(transform_matrix, do_het=do_het, image=image, generator=generator)

        return projection

    def rasterize(self, size: Tuple[int, int, int]) -> torch.Tensor:
        return self.renderer.rasterize(size)
