"""Training script for field reconstruction
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import pathlib
import warnings

from typing import Any, Callable, Dict, List, Tuple, Optional

import hydra
import omegaconf
import numpy as np
import torch
import torch.utils.data
import torchmetrics
import yaml
import mrcfile

import pytorch_lightning
import pytorch_lightning.callbacks
import pytorch_lightning.strategies

import instamap.nn.lightning
import instamap.nn.lr_scheduler
import instamap.simulator.crop

import instamap.simulator.optics
import instamap.simulator.pipeline
import instamap.simulator.scattering
import instamap.simulator.sensor

from instamap import nn
from instamap.simulator import electron_image
from . import dataset, dataloader, field
from ._config import *


def _normalize(x: torch.Tensor):
    x_min = torch.min(x)
    x_max = torch.max(x)
    return x.sub(x_min).div_(x_max - x_min)


def _get_extent(config: electron_image.CryoEmRenderingConfig) -> torch.Tensor:
    if config.extent is not None:
        return torch.tensor(config.extent, dtype=torch.float32)

    if config.image_height == config.image_width:
        return torch.tensor(
            [config.image_height, config.image_height, config.image_height], dtype=torch.float32
        ).mul_(config.pixel_size)
    else:
        raise ValueError('Cannot infer extent from image size for non-square images.')


def compute_and_plot_fsc(x: torch.Tensor, y: torch.Tensor, pixel_size: float):
    """Plots Fourier Shell Correlation between two given volumes.
    """
    small_number = 1e-7
    x_noise = x + small_number*torch.randn_like(x)
    fsc_pos = nn.fourier_shell_correlation(x_noise, y)
    fsc_neg = nn.fourier_shell_correlation(x_noise, -y)
    if fsc_pos.sum() > fsc_neg.sum():
        fsc = fsc_pos
    else:
        fsc = fsc_neg

    freq = nn.fourier_shell_frequency(x, d=pixel_size)

    try:
        idxs_half = np.nonzero(fsc < 0.5)[0]
        if len(idxs_half) > 0:
            resolution = 1 / freq[idxs_half[0]].item()
        else:
            resolution = 0.0
    except:
        warnings.warn("FSC poorly defined", UserWarning)
        resolution = np.nan

    info = {
        'fsc': fsc,
        'freq': freq,
        'resolution': resolution,
    }

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(freq.cpu().numpy(), fsc.cpu().numpy())
        ax.xaxis.set_major_formatter(lambda x, _: f'{1/x:.1f}' if x > 0 else '')
        ax.set_xlabel('Resolution (Å)')
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warn(f'Failed to plot FSC: {e}', exc_info=True)
        fig = None

    return fig, info


def make_pipeline(field: torch.nn.Module, config: InstaMapTrainingConfig):
    """Create a rendering pipeline from the given training configuration.
    """

    def pose_patch(config):
        '''Notes: Useful for loading old checkpoint'''
        if 'pose' not in config.keys():
            config['pose'] = {'inference_enabled': False, 'n_output_pose_params': -1, 'num_hidden': -1}
    pose_patch(config)
    def mask_patch(config):
        '''Notes: Useful for loading old checkpoint'''
        if 'mask' not in config.keys():
            config['mask'] = {'masking_enabled': False, 'mask_path': '/tmp/mask.txt'}
    mask_patch(config)


    image_config = instamap.simulator.scattering.RenderingConfig(
        height=int(config.data.image.height * config.renderer.spatial_sampling_rate),
        width=int(config.data.image.width * config.renderer.spatial_sampling_rate),
        depth_samples=config.renderer.depth_samples,
        pixel_size=config.data.image.pixel_size / config.renderer.spatial_sampling_rate,
        heterogeneity=config.heterogeneity,
        pose=config.pose,
        mask=config.mask,
    ) 

    if config.mask.masking_enabled:
        scattering_base = instamap.simulator.scattering.DirectMaskedProjection(
            field, image_config, config.renderer.extent, jitter=True, jitter_scale=config.renderer.jitter_scale,
        )
    else:
        scattering_base = instamap.simulator.scattering.DirectSphereProjection(
            field, image_config, config.renderer.extent, jitter=True, jitter_scale=config.renderer.jitter_scale,
        )

    scattering = instamap.simulator.scattering.ViewAugmentingAdapter(
        scattering_base, config.augmentation, image_config
    )

    if config.renderer.use_ctf:
        optics = instamap.simulator.optics.FourierContrastTransferOptics(image_config)
    else:
        optics = instamap.simulator.optics.NullOptics()

    if config.criterion.normalize:
        raise NotImplementedError('Normalization not implemented for this pipeline.')

    if config.criterion.sensor_type == 'GaussianSensor':
        sensor = instamap.simulator.sensor.GaussianSensor(image_config, config.criterion.noise_ratio, mask_radius=1.15)
    elif config.criterion.sensor_type == 'CorrelationSensor':
        sensor = instamap.simulator.sensor.CorrelationSensor(image_config, config.criterion.noise_ratio, mask_radius=1.15)
    elif config.criterion.sensor_type == 'BioemSensor-saddle-approx':
        sensor = instamap.simulator.sensor.BioemSensor(image_config, config.criterion.noise_ratio, mask_radius=1.15, method='saddle-approx')
    elif config.criterion.sensor_type == 'BioemSensor-N-mu':
        sensor = instamap.simulator.sensor.BioemSensor(image_config, config.criterion.noise_ratio, mask_radius=1.15, method='N-mu')
    elif config.criterion.sensor_type == 'BioemSensor-N-mu-gaussian-prior-N':
        sensor = instamap.simulator.sensor.BioemSensor(image_config, config.criterion.noise_ratio, mask_radius=1.15, method='N-mu-gaussian-prior-N')
    else:
        raise NotImplementedError('specify an impelemnted sensor in config.sensor')
    
    if config.renderer.spatial_sampling_rate != 1:
        sensor = instamap.simulator.sensor.DownsamplingSensorAdapter(
            sensor, (config.data.image.height, config.data.image.width)
        )

    return instamap.simulator.pipeline.RenderingPipeline(scattering, optics, sensor)


class InstaMapTrainer(pytorch_lightning.LightningModule):
    """Main trainer module for InstaMap reconstruction.

    This module keeps track of the state of the reconstruction (i.e. parameters of the field),
    and additionally caches temporary tensors used for rendering (e.g. initial query points).

    """
    hparams: InstaMapTrainingConfig

    def __init__(self, config: InstaMapTrainingConfig):
        super().__init__()

        if not omegaconf.OmegaConf.is_config(config):
            config = omegaconf.OmegaConf.structured(config)

        self.save_hyperparameters(config)
        rendering_config = make_rendering_config(config.data.image, config.renderer)
        if 'regularization' in config: 
            self.regularization = config.regularization
        else:
            self.regularization = None
            Warning('No regularization specified in config, using None')

        self.extent = _get_extent(rendering_config) 
        self.render_npix = config.data.image.height
        assert config.data.image.height == config.data.image.height
        
        if config.model.encoding_config_otype == 'Voxel':
            field = field.DirectImageField(config.model, self.extent)
        elif config.model.encoding_config_otype in ['Grid', 'Frequency', 'OneBlob', 'SphericalHarmonics', 'TriangleWave']:
            field = field.HashedImageField(config.model, self.extent)
        else:
            assert False, 'use either voxel, Grid, Frequency, OneBlob, or SphericalHarmonics representation'
        
        self.pipeline = make_pipeline(field, config)

        self.corr = nn.PerInstanceCorrelation(dim=(-1, -2))
        self.val_corr = nn.PerInstanceCorrelation(dim=(-1, -2))
        self.val_psnr = torchmetrics.PeakSignalNoiseRatio()


        self.reference_volume = None
        self.reference_volumes = []
        if config.data.reference_volume_path is not None:
            if config.data.reference_volume_paths is None:
                Warning('use either reference_volume_path or reference_volume_paths')
        if config.data.reference_volume_paths is None and config.data.reference_volume_path is not None:
            config.data.reference_volume_paths = []
            config.data.reference_volume_paths.append(config.data.reference_volume_path)

        if self.global_rank == 0 and self.hparams.data.reference_volume_paths is not None:
            try:
                for path in self.hparams.data.reference_volume_paths:
                    reference_volume_path = hydra.utils.to_absolute_path(path)
                    self.reference_volumes.append(torch.as_tensor(np.load(reference_volume_path)))
                

            except FileNotFoundError:
                logging.getLogger(__name__).warning(
                    'Reference volume not found at %s, skipping reference volume metrics.',
                    reference_volume_path
                )
        self.volume_render_steps = set(config.volume_render_steps)
        self.epoch_start_heterogeneity = config.epoch_start_heterogeneity



    def forward(
        self,
        shot: electron_image.CryoEmShotInfo,
        do_het: Optional[bool] = False,
        observed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_or_loss, info = self.pipeline(shot, do_het, observed, regularization_scales=self.regularization) 
        return image_or_loss, info

    def __call__(
        self,
        shot: electron_image.CryoEmShotInfo,
        do_het: Optional[bool] = False,
        observed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return super().__call__(shot, do_het, observed) 

    def _get_image_info_from_batch(self, batch: dataset.ElectronImageObservation):
        image = batch['image']
        shot_info = batch['info']
        return image, shot_info

    def _debug_rendered(self, batch, t: torch.Tensor):
        """Checks for NaNs in the rendered image. If any are found,
        dumps current batch and parameters to file for debugging.
        """
        if not self.hparams.debug:
            return

        if not torch.isnan(t).any().item():
            return

        logger = logging.getLogger(__name__)
        logger.error('NaNs in rendered image')

        has_nan_parameter = False
        for n, p in self.named_parameters():
            if torch.isnan(p).any().item():
                logger.error(f'NaNs in parameter {n}')
                has_nan_parameter = True

        if not has_nan_parameter:
            logger.info('No NaN parameters found')

        weights_debug_path = os.path.abspath('debug_model.pt')
        logger.info(f'Saving model to {weights_debug_path}')
        self.trainer.save_checkpoint(weights_debug_path)
        torch.save(batch, 'debug_batch.pt')

        raise RuntimeError('NaNs in rendered image')

    def _save_batch_reproduction(self, batch):
        """Saves the current batch and model parameters to file for debugging
        """
        if self.global_rank != 0:
            return

        logger = logging.getLogger(__name__)
        save_dir = os.path.abspath(os.path.join(self.trainer.log_dir, 'debug', f'step_{self.trainer.global_step}'))
        logger.info(f'Saving batch reproduction to {save_dir}')

        self.trainer.save_checkpoint(os.path.join(save_dir, 'model.pt'))
        torch.save(batch, os.path.join(save_dir, 'batch.pt'))


    def run_training_step(
        self, batch, loss: torch.Tensor, info: Dict[str, torch.Tensor], reference: torch.Tensor,
        batch_idx: int
    ) -> torch.Tensor:
        """Runs the training step given rendered and reference image.

        This method performs checks, computes the loss, and logs relevant metrics.
        """
        rendered = info['optics_image']
        self._debug_rendered(batch, rendered)

        if batch_idx % 7 == 0:
            self.logger.experiment.add_images(
                'train/rendered',
                torch.stack(
                    [
                        _normalize(info['scattering_image'].detach()[0]),
                        _normalize(info['optics_image'].detach()[0]),
                        _normalize(reference[0])
                    ],
                    dim=0
                ).unsqueeze_(-1),
                self.global_step,
                dataformats='NHWC'
            )

        with torch.no_grad():
            if self.global_step in self.volume_render_steps:
                n_pix = self.render_npix
                logging.getLogger(__name__).warning(f'Beware of GPU memory use from large volume rendering for n_pix={n_pix}')
                volume_from_field = self.pipeline.scattering.rasterize((n_pix,n_pix,n_pix)) 
                memory_gb = volume_from_field.storage().nbytes()*10**-9
                logging.getLogger(__name__).warning('Rasterized (high res?) volume is {:1.1e} GB'.format(memory_gb))
                with mrcfile.new('val_rendering_vol_epoch{:d}_globalstep{:04d}.mrc'.format(self.current_epoch,self.global_step)) as mrc:
                    mrc.set_data(volume_from_field.type(torch.float32).detach().cpu().numpy())
                del volume_from_field 

        if batch_idx == 10 and self.hparams.debug:
            self._save_batch_reproduction(batch)

        with torch.no_grad():
            corr = self.corr(rendered, reference)

        loss = loss.mean()

        self.log('loss', loss, prog_bar=True)
        logging_fields = ['sensor_logging_image_loss']
        for field in logging_fields:
            if field in info and info[field] is not None:
                self.log(f'train/{field}', info[field].mean(), prog_bar=True)
        if batch_idx % 7 == 0:
            self.log('train/loss*', loss)
        self.log('train/snr', nn.snr_from_correlation(corr))
        self.log('train/corr', corr)

        return loss

    def het_logic(self, epoch_start_heterogeneity: int):
        if self.current_epoch >= epoch_start_heterogeneity:
            do_het = True
        else:
            do_het = False
        return do_het
    
    def training_step(self, batch: dataset.ElectronImageObservation, batch_idx):
        image, shot_info = self._get_image_info_from_batch(batch)
        loss, info = self(shot_info, self.het_logic(self.epoch_start_heterogeneity), image) 
        return self.run_training_step(batch, loss, info, image, batch_idx) 

    def validation_step(self, batch: dataset.ElectronImageObservation, batch_idx):
        image, shot = self._get_image_info_from_batch(batch)
        loss, info = self(shot, self.het_logic(self.epoch_start_heterogeneity), image) 

        rendered = info['optics_image']
        rendered_no_ctf = info['scattering_image']
        rendered_ctf = rendered

        self.log('val/loss', loss.mean(), sync_dist=True)
        logging_fields = ['scattering_l2_norm', 'scattering_tv']
        for field in logging_fields:
            if field in info and info[field] is not None:
                self.log(f'val/{field}', info[field].mean(), sync_dist=True)

        self.val_corr(rendered, image)
        self.log('val/corr', self.val_corr, sync_dist=True)
        self.log('val/snr', nn.snr_from_correlation(self.val_corr.compute()), sync_dist=True)

        tb = self.logger.experiment


        renderings = [rendered_no_ctf[0], rendered_ctf[0], image[0]]

        if 'image_reference' in batch and batch['image_reference'].shape == image.shape:
            image_reference = batch['image_reference']
            renderings.append(image_reference[0])

            if self.hparams.criterion.normalize:
                image_reference, _ = nn.normalize_mean_variance(image_reference)
                rendered_no_ctf, _ = nn.normalize_mean_variance(rendered_no_ctf)

            loss_reference = torch.nn.functional.mse_loss(rendered_no_ctf, image_reference)
            self.val_psnr(rendered_no_ctf, image_reference)

            self.log('val/loss_reference', loss_reference, sync_dist=True)
            self.log('val/psnr_reference', self.val_psnr)

        tb.add_images(
            f'images_val/{batch_idx}',
            torch.stack([_normalize(im) for im in renderings], dim=0).unsqueeze(-1),
            self.global_step,
            dataformats='NHWC'
        )

        if batch_idx == 0:
            with open(f'val_rendering_{self.current_epoch}.pt', 'wb') as f:
                torch.save({
                    'renderings': torch.stack(renderings, 0),
                    'info': batch['info'],
                }, f)

    def rasterize(self, shape: Tuple[int, int, int]) -> torch.Tensor:
        return self.pipeline.scattering.rasterize(shape)


    def compute_current_volume_metrics(self):
        """Computes the volume metrics for the current fitted volume.
        """

        reference = self.reference_volume

        def one_volume(reference):
            if reference is None:
                raise ValueError(
                    'No reference volume available. Either pass it as an argument or set it as an attribute of the trainer.'
                )

            rasterized_volume = self.rasterize(reference.shape)

            if rasterized_volume.dtype == torch.float16:
                rasterized_volume = rasterized_volume.to(torch.float32)

            reference_volume = reference.to(rasterized_volume)
            return self.compute_volume_metrics(reference_volume, rasterized_volume)
        
        return [one_volume(reference) for reference in self.reference_volumes]

    def compute_volume_metrics(
        self, reference_volume: torch.Tensor, rasterized_volume: torch.Tensor
    ):
        """Computes a number of metrics from a given reference and estimated volume.
        """
        mask = instamap.simulator.scattering.make_axis_aligned_grid(
            reference_volume.shape,
            1 / reference_volume.new_tensor(reference_volume.shape),
            homogeneous=False
        ).norm(dim=-1) < 0.95

        r2_volume = nn.mask_corr(rasterized_volume, reference_volume, mask)**2

        fsc_plot, fsc_info = compute_and_plot_fsc(
            rasterized_volume * mask, reference_volume * mask,
            (self.extent[0] / reference_volume.shape[0]).item()
        )

        return {
            'rasterized_volume': rasterized_volume,
            'reference_volume': reference_volume,
            'mask': mask,
            'r2': r2_volume,
            'fsc_plot': fsc_plot,
            'fsc_resolution': fsc_info['resolution'],
        }

    def on_validation_epoch_end(self) -> None:
        if self.reference_volume is None and len(self.reference_volumes) == 0:
            return

        volume_metrics_list = self.compute_current_volume_metrics()

        for i, volume_metrics in enumerate(volume_metrics_list):
            self.log(f'val/volume_{i}_r2', volume_metrics['r2'], rank_zero_only=True)
            self.log(
                f'val/volume_{i}_fsc_resolution',
                volume_metrics['fsc_resolution'],
                rank_zero_only=True,
                prog_bar=True
            )

            if self.global_rank == 0:
                fsc_plot = volume_metrics['fsc_plot']
                if fsc_plot is not None:
                    self.trainer.loggers[0].experiment.add_figure(
                        f'val/volume_{i}_fsc', volume_metrics['fsc_plot'], self.global_step
                    )

    def configure_optimizers(self):

        if self.hparams.optim.optimizer == 'Adam':
            optim = torch.optim.Adam(self.parameters(), lr=self.hparams.optim.learning_rate)
            return {'optimizer': optim}
        elif self.hparams.optim.optimizer == 'AdamW':
        
            total_batch_size = self.hparams.batch_size * self.hparams.num_gpus

            base_lr = self.hparams.optim.learning_rate
            lr = base_lr * total_batch_size

            optim = torch.optim.AdamW(
                self.parameters(), lr=lr, weight_decay=self.hparams.optim.weight_decay
            )

            batches_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs

            scheduler_decay = torch.optim.lr_scheduler.StepLR(
                optim, step_size=batches_per_epoch, gamma=0.7
            )
            scheduler = instamap.nn.lr_scheduler.LinearWarmupScheduler(
                min(batches_per_epoch, 2000 // self.hparams.batch_size), scheduler_decay
            )

            return {
                'optimizer': optim,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step'
                },
            }
        else:
            raise ValueError(f'Unknown optimizer {self.hparams.optim.optimizer}')


def train_with_config(
    config: InstaMapTrainingConfig,
    dm: pytorch_lightning.LightningDataModule,
    callbacks: Optional[List[pytorch_lightning.Callback]] = None,
    trainer_kwargs: Optional[Dict[str, Any]] = None
):
    """Trains a InstaMap model with the given configuration and dataset."""
    if callbacks is None:
        callbacks = []

    if not any(isinstance(cb, pytorch_lightning.callbacks.LearningRateMonitor) for cb in callbacks):
        callbacks.append(pytorch_lightning.callbacks.LearningRateMonitor())

    if config.heterogeneity.multi_stage_training:
        config.heterogeneity.enabled = True
        config.epoch_start_heterogeneity = config.num_epochs + 1
        model = InstaMapTrainer(config) 
        trainer = instamap.nn.lightning.make_trainer(
            config, callbacks=callbacks, trainer_kwargs=trainer_kwargs
        )
        for param in model.pipeline.scattering.renderer.displace.parameters(): 
            param.requires_grad = False
        trainer.fit(model, dm)
        
        model.epoch_start_heterogeneity = 0
        config.optim.optimizer = config.heterogeneity.optimizer
        config.optim.learning_rate = config.heterogeneity.learning_rate
        config.num_epochs = config.heterogeneity.num_epochs
        config.data.reference_volume_paths = None
        for param in model.pipeline.scattering.renderer.field.parameters(): 
            param.requires_grad = False
        for param in model.pipeline.scattering.renderer.displace.parameters(): 
            param.requires_grad = True
        trainer = instamap.nn.lightning.make_trainer(
            config, callbacks=callbacks, trainer_kwargs=trainer_kwargs
        )
        trainer.fit(model, dm)

    else:    
        model = InstaMapTrainer(config)
        trainer = instamap.nn.lightning.make_trainer(
            config, callbacks=callbacks, trainer_kwargs=trainer_kwargs
        )    
        trainer.fit(model, dm)


    return trainer


@contextlib.contextmanager
def chdir(dir):
    """Context manager for executing code in the given working directory.

    This is an implementation of Python 3.11+ `contextlib.chdir`.
    """

    curdir = os.getcwd()
    try:
        os.chdir(dir)
        yield
    finally:
        os.chdir(curdir)


def load_trained_model(run_directory, klass=InstaMapTrainer, device='cpu'):
    """Loads a trained model from the given run directory.
    """
    run_directory = pathlib.Path(run_directory)
    checkpoints_dir = next((run_directory / 'lightning_logs').iterdir()) / 'checkpoints'

    checkpoint_path = next((f for f in checkpoints_dir.glob('*.ckpt') if f.name != 'last'))

    yaml_path = run_directory / '.hydra' / 'hydra.yaml'
    train_cwd = yaml.safe_load(yaml_path.read_text())['hydra']['runtime']['cwd']

    checkpoint_path = checkpoint_path.resolve()

    with chdir(train_cwd):
        model = klass.load_from_checkpoint(checkpoint_path, map_location=device)

    return model


def load_model_datasets(run_directory: str, device: Optional[torch.device] = None):
    """Loads the dataset used when training the model found in the given run directory."""
    from instamap.nn.dataset import MovingCollatingDataset

    run_directory = pathlib.Path(run_directory)

    config = omegaconf.OmegaConf.load(run_directory / '.hydra' / 'config.yaml')
    config: InstaMapTrainingConfig = omegaconf.OmegaConf.merge(
        omegaconf.OmegaConf.structured(InstaMapTrainingConfig), config
    )

    yaml_path = run_directory / '.hydra' / 'hydra.yaml'
    train_cwd = yaml.safe_load(yaml_path.read_text())['hydra']['runtime']['cwd']

    with chdir(train_cwd):
        dm = dataloader.InstaMapDataModule(
            config.data, batch_size=config.batch_size, image_config=config.data.image
        )
        dm.setup()
        return {
            'train': MovingCollatingDataset(dm._ds_train, device=device),
            'val': MovingCollatingDataset(dm._ds_val, device=device),
        }


@hydra.main(config_path='conf/reconstruction', config_name='train', version_base="1.2")
def main(config: InstaMapTrainingConfig):
    config = fixup_config(config)

    logger = logging.getLogger(__name__)
    data_config = dataloader.load_data_locally_if_requested(config.data)
    dm = dataloader.InstaMapDataModule(
        data_config, batch_size=config.batch_size, image_config=config.data.image
    )

    if dm.crop_required:
        logger.info(
            f'Cropping dataset from size {dm.image_shape} to match rendering configuration '
            f'{(dm.image_config.height, dm.image_config.width)}.'
        )

    callbacks = []
    trainer = train_with_config(config, dm, callbacks=callbacks)

    if trainer.is_global_zero:
        logger.info(f'Finished training')
        if 'val/corr' in trainer.logged_metrics:
            logger.info(f'Final correlation: {trainer.logged_metrics["val/corr"]}')
        if 'val/volume_r2' in trainer.logged_metrics:
            logger.info(f'Final volume R2: {trainer.logged_metrics["val/volume_r2"]}')
        if 'val/volume_fsc_resolution' in trainer.logged_metrics:
            logger.info(
                f'Final volume FSC resolution: {trainer.logged_metrics["val/volume_fsc_resolution"]}'
            )


if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store('train_base', node=InstaMapTrainingConfig)
    main()
