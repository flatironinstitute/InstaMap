"""Data-loading facility for InstaMap.
"""

from __future__ import annotations

from typing import Dict, TypedDict
from typing_extensions import NotRequired

import numpy as np
import torch
import torch.utils.data

from instamap.simulator import crop, electron_image


class ElectronImageObservation(TypedDict):
    """A dictionary containing the observation of an electron image.

    Attributes
    ----------
    image : torch.Tensor
        The observed image (or a stack thereof).
    info : electron_image.CryoEmShotInfo
        The shot information for each image.
    image_reference : torch.Tensor
        If present, a reference image (potentially representing an image
        captured from ideal conditions, such as lack of noise or other
        optical abberations).
    """
    image: torch.Tensor
    info: electron_image.CryoEmShotInfo
    image_reference: NotRequired[torch.Tensor]


class ElectronImageDataset(torch.utils.data.Dataset[ElectronImageObservation]):
    """Dataset to load electron images from a .npz file

    The .npz file should contain the following keys:
    - images: a (N, H, W) array of electron images, corresponding to the observations
    - shot_info_*: an array for each entry in electron_image.CryoEmShotInfo, containing
        the corresponding shot information for each image
    - images_reference: (optional) a (N, H, W) array of reference images, corresponding
        to a "reference" image for each observation, when it is available (e.g. in
        simulated data).
    """

    images: torch.Tensor
    info: electron_image.CryoEmShotInfo

    def __init__(self, path: str, normalize: bool = True, flip: bool = False):
        with np.load(path) as data:
            self.images = torch.as_tensor(data['image'], dtype=torch.float32)
            self.info = {
                k[len('shot_info_'):]: torch.as_tensor(v, dtype=torch.float32)
                for k, v in data.items() if k.startswith('shot_info_')
            }


            if 'image_reference' in data:
                self.images_reference = torch.as_tensor(
                    data['image_reference'], dtype=torch.float32
                )
            else:
                self.images_reference = None

        if normalize:
            a = torch.min(self.images)
            b = torch.max(self.images)
            self.images.sub_(a).div_(b - a)

        if flip:
            self.images.mul_(torch.tensor(-1, dtype=self.images.dtype))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> ElectronImageObservation:
        result = {
            'image': self.images[index],
            'info': {k: v[index]
                     for k, v in self.info.items()},
        }

        if self.images_reference is not None:
            result['image_reference'] = self.images_reference[index]

        return result
