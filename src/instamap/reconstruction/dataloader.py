"""Utilities for loading data into the training pipeline.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import shutil
import tempfile

import hydra
import pytorch_lightning
import torch
import torch.utils.data

from instamap.simulator import electron_image

from . import dataset
from ._config import InstaMapDataConfig


class InstaMapDataModule(pytorch_lightning.LightningDataModule):
    _ds: torch.utils.data.Dataset[dataset.ElectronImageObservation]

    def __init__(
        self,
        config: InstaMapDataConfig,
        batch_size: int,
        include_index: bool = False,
        image_config: electron_image.ImageConfig = None,
    ):
        """Create a new datamodule processing observations from a cryo-EM dataset.

        Parameters
        ----------
        config : InstaMapDataConfig
            Configuration for creating the dataset and data loaders
        batch_size : int
            Batch size to use for the dataloaders
        include_index : bool, optional
            If `True`, the index of the observation in the dataset is included in the batch
        rendering : Optional[electron_image.RenderingConfig]
            If not `None`, the dataset is cropped according to the rendering configuration
        """
        super().__init__()

        self.config = config
        self.batch_size = batch_size

        self.image_config = image_config


        self._ds = dataset.ElectronImageDataset(
            hydra.utils.to_absolute_path(self.config.path),
            normalize=False,
            flip=self.config.flip_contrast_images,
        )

        self.image_shape = self._ds.images.shape[1:]
        self.crop_required = image_config is not None and (
            image_config.height != self.image_shape[-2]
            or image_config.width != self.image_shape[-1]
        )

        if include_index:
            self._ds = dataset.IndexDataset(self._ds)

        self._ds_train = None
        self._ds_val = None

    def setup(self, stage: str = None):
        if self.config.limit is not None:
            limit = min(self.config.limit, len(self._ds))
            self._ds, _ = torch.utils.data.random_split(self._ds, [limit, len(self._ds) - limit])

        num_train_samples = int(len(self._ds) * self.config.proportion_train)

        self._ds_train, self._ds_val = torch.utils.data.random_split(
            self._ds, [num_train_samples, len(self._ds) - num_train_samples],
            torch.Generator().manual_seed(42)
        )

        if self.crop_required:
            original_pixels = self.image_shape[-2] * self.image_shape[-1]
            cropped_pixels = self.image_config.height * self.image_config.width
            expansion_factor = int(original_pixels / cropped_pixels)

            self._ds_train = dataset.CroppingDataset(
                self._ds_train, self.image_config, expansion_factor
            )
            self._ds_val = dataset.CroppingDataset(self._ds_val, self.image_config, 1)

    def _make_dataloader(
        self, ds: torch.utils.data.Dataset[dataset.ElectronImageObservation], shuffle: bool = True
    ):
        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_data_workers,
            pin_memory=True,
            persistent_workers=self.config.num_data_workers > 0
        )

    def train_dataloader(self):
        return self._make_dataloader(self._ds_train, shuffle=True)

    def val_dataloader(self):
        return self._make_dataloader(self._ds_val, shuffle=False)


def load_data_locally_if_requested(config: InstaMapDataConfig) -> InstaMapDataConfig:
    """Copies the dataset into a local temporary directory if requested.

    This can be helpful when training in a multi-process setting with data
    in a distributed file system. By copying the data locally, we may speed
    up the loading time for child processes.

    This function constructs a unique name for the local copy of the dataset
    based on the path to the original dataset. It then creates a copy
    of the original dataset in the temporary directory.
    """
    if not config.make_local_copy:
        return config

    logger = logging.getLogger(__name__)
    logger.info('Requested local copy of dataset.')

    path = hydra.utils.to_absolute_path(config.path)
    ext = os.path.splitext(path)[1]
    temporary_name = hashlib.sha256(path.encode('utf-8')).hexdigest()
    tmpdir = tempfile.gettempdir()

    local_file = os.path.join(tmpdir, f'{temporary_name}{ext}')

    new_config = copy.copy(config)
    new_config.path = local_file

    if os.path.exists(local_file) and os.stat(local_file).st_size == os.stat(path).st_size:
        logger.info(f'Using local copy of {path} at {local_file}.')
        return new_config

    logger.info(f'Copying {path} to {local_file}.')
    shutil.copyfile(path, local_file)

    return new_config
