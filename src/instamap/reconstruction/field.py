"""Main implicit field model classes
"""

import dataclasses
import math
import time

from typing import Callable, Tuple

import torch

from instamap.simulator.electron_image import CryoEmShotInfo as CryoEmShotInfo

@dataclasses.dataclass
class FieldConfig:
    _target_: str = ''
    size: int = 160


@dataclasses.dataclass
class HashedImageFieldConfig(FieldConfig):
    """Configuration for a hashed image field.

    Note that many of these parameters control the tuning
    of an instant-ngp like field model, please see the
    paper for details of their effects.

    Attributes
    ----------
    base_size: int
        The size of the base (coarsest) grid
    size: int
        The size of the finest grid
    levels: int
        The number of levels of grids to use
    hash_size_log2: int
        Log2 of the size of the hash table at each level (number of entries).
    num_hidden: int
        Number of hidden units in the readout MLP
    num_layers : int
        Number of layers in the readout MLP
    n_frequencies : int
        Number of frequencies for Frequency based encoding (non Grid): https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md#frequency
    """
    _target_: str = 'instamap.reconstruction.field.HashedImageField'
    encoding_config_otype : str = 'Grid'
    base_size: int = 8
    levels: int = 8
    hash_size_log2: int = 19
    num_hidden: int = 64
    num_layers: int = 2
    n_frequencies: int = 32
    points_batch_size_log2 : int = 22
    n_bins: int = 128
    degree: int = 8


class HashedImageField(torch.nn.Module):
    """Basic field built from a multi-level hashed grid structure (instant-ngp).
    """
    extent: torch.Tensor

    def __init__(self, config: HashedImageFieldConfig, extent: torch.Tensor, boundary: str = 'periodic'):
        """Create a new HashedImageField.

        Parameters
        ----------
        config : HashedImageFieldConfig
            The configuration for the field.
        extent : torch.Tensor
            The extent of the field in each dimension.
        boundary : str
            The boundary condition to use for the field.
        points_batch_size : int
            The maximum number of points to process at once.
        """
        super().__init__()

        if not torch.cuda.is_available():
            raise ValueError("HashedImageField requires CUDA")


        self.register_buffer('extent', torch.as_tensor(extent, dtype=torch.float32).max())

        compute_major, compute_minor = torch.cuda.get_device_capability()
        compute_level = compute_major * 10 + compute_minor
        mlp_type = 'FullyFusedMLP' if compute_level > 70 else 'CutlassMLP'

        import tinycudann
        if config.encoding_config_otype == 'Grid':
            encoding_config = {
                'otype':
                    'Grid',
                'type':
                    'Hash',
                'n_levels':
                    config.levels,
                'log2_hashmap_size':
                    config.hash_size_log2,
                'base_resolution':
                    config.base_size,
                'per_level_scale':
                    math.exp(math.log(config.size / config.base_size) / (config.levels - 1)),
                'interpolation':
                    'linear',
            }
        elif config.encoding_config_otype == "Frequency":
            encoding_config = {"otype": "Frequency", "n_frequencies": config.n_frequencies}

        elif config.encoding_config_otype == "OneBlob":
            encoding_config = {"otype": "OneBlob", "n_bins": config.n_bins}

        elif config.encoding_config_otype == "SphericalHarmonics":
            encoding_config = {"otype": "SphericalHarmonics", "degree": config.degree, }

        elif config.encoding_config_otype == "TriangleWave":
            encoding_config = {"otype": "TriangleWave", "n_frequencies": config.n_frequencies}      
            
        self.encoding = tinycudann.NetworkWithInputEncoding(
            3, 1, encoding_config, {
                'otype': mlp_type,
                'activation': 'ReLU',
                'output_activation': 'None',
                'n_neurons': config.num_hidden,
                'n_hidden_layers': config.num_layers,
            }
        )

        self.boundary = boundary
        self.points_batch_size = 2**config.points_batch_size_log2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape

        x = x.view(-1, 3)
        x = x.div(self.extent)

        if self.points_batch_size is not None:
            x_batch = torch.split(x, self.points_batch_size, dim=0)
        else:
            x_batch = [x]
        result_batch = [self.encoding(x) for x in x_batch]
        result = torch.cat(result_batch, dim=0)

        return result.view(*x_shape[:-1])


@dataclasses.dataclass
class DirectImageFieldConfig(FieldConfig):
    """Configuration for a direct image field.
    """
    _target_: str = 'instamap.reconstruction.field.DirectImageField'


class DirectImageField(torch.nn.Module):
    """Direct sampled field, with simple trilinear filtering.
    """
    data: torch.Tensor
    half_extent: torch.Tensor

    def __init__(self, config: DirectImageFieldConfig, extent: torch.Tensor):
        super().__init__()

        self.register_parameter(
            'data',
            torch.nn.Parameter(torch.randn((1, 1, config.size, config.size, config.size), dtype=torch.float32)))
        self.register_buffer('half_extent', torch.as_tensor(extent, dtype=torch.float32).max() / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape

        x = x.view(-1, 3)
        x = x.div(self.half_extent)

        result = torch.nn.functional.grid_sample(
            self.data,
            x.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            mode='bilinear',
            padding_mode='zeros')

        return result.view(*x_shape[:-1])


def make_field_from_config(config: FieldConfig, extent: torch.Tensor) -> torch.nn.Module:
    """Create a field from the given configuration.
    """

    if config._target_ == 'instamap.reconstruction.field.HashedImageField':
        return HashedImageField(config, extent)
    elif config._target_ == 'instamap.reconstruction.field.DirectImageField':
        return DirectImageField(config, extent)
    else:
        raise ValueError(f"Unknown field type: {config._target_}")

