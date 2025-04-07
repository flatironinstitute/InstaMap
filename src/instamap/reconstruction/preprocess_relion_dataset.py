"""Simple script to pre-process a relion output into a format which can be fed into the reconstruction.

"""

from __future__ import annotations

import collections
import dataclasses
import logging
import os

from typing import Optional

import hydra
import omegaconf
import mrcfile
import numpy as np
import pandas as pd
import starfile
import scipy
import scipy.spatial.transform
import torch

import instamap.simulator.electron_image
import instamap.simulator.relion


@dataclasses.dataclass
class PerturbationConfig:
    """Additional configuration for introducing perturbations to the data.

    Attributes
    ----------
    angle : float
        Scale of the random angular perturbation to apply to the rotation matrix, in degrees.
    offset : float
        Scale of the random offset perturbation to apply to the translation vector, in Angstroms.
    seed : int
        Seed to use for the random number generator.
    """
    angle: Optional[float] = None
    offset: Optional[float] = None
    seed: int = 0


@dataclasses.dataclass
class PreprocessRelionDatasetConfig:
    image_path: Optional[str] = omegaconf.MISSING
    info_path: str = omegaconf.MISSING
    read_image_from_starfile: bool = False
    subset: Optional[int] = None
    limit: Optional[int] = None
    perturbation: PerturbationConfig = PerturbationConfig()
    downsample: int = 1


def read_info(path):
    """Read relion *.star file to obtain shot information.

    Note that there may be some variety in how the info is saved
    exactly, and we attempt to handle that here.
    """
    info_all = starfile.read(path)

    if isinstance(info_all, collections.OrderedDict):
        info = info_all["particles"]
        info_optics = info_all["optics"]
        info = info.join(info_optics.set_index("rlnOpticsGroup"), on="rlnOpticsGroup")
    else:
        info = info_all

    if 'rlnPhaseShift' not in info:
        info['rlnPhaseShift'] = 0

    if 'rlnCtfBfactor' not in info:
        info['rlnCtfBfactor'] = 0

    return info


def process(images: np.ndarray, config: PreprocessRelionDatasetConfig):
    logger = logging.getLogger(__name__)

    if config.downsample != 1:
        logger.info(f"Downsampling images by a factor of {config.downsample}")
        images = torch.nn.functional.avg_pool2d(torch.from_numpy(images), config.downsample).numpy()
    return images


def perturb_angles(
    shot_info: instamap.simulator.electron_image.CryoEmShotInfo,
    config: PreprocessRelionDatasetConfig
) -> instamap.simulator.electron_image.CryoEmShotInfo:
    """Randomly perturbs the angles of the rotation in the given shot information.
    """
    generator = torch.Generator().manual_seed(config.perturbation.seed)

    view_phi_orig = shot_info['view_phi']
    view_theta_orig = shot_info['view_theta']
    view_psi_orig = shot_info['view_psi']

    angles = torch.stack([
        view_phi_orig,
        view_theta_orig,
        view_psi_orig,
    ], dim=-1)


    angles_perturb = torch.randn(*angles.shape, generator=generator)
    angles_perturb.clamp_(-3, 3).mul_(config.perturbation.angle)

    rotations = scipy.spatial.transform.Rotation.from_euler('ZYZ', angles, degrees=True)
    rot_perturb = scipy.spatial.transform.Rotation.from_euler('xyz', angles_perturb, degrees=True)
    rot_total = rotations * rot_perturb
    angles = torch.from_numpy(rot_total.as_euler('ZYZ', degrees=True))

    view_phi, view_psi, view_theta = angles.unbind(dim=-1)

    return {
        **shot_info,
        'view_phi': view_phi,
        'view_theta': view_theta,
        'view_psi': view_psi,
        'view_phi_orig': view_phi_orig,
        'view_theta_orig': view_theta_orig,
        'view_psi_orig': view_psi_orig,
    }


def process_info(
    shot_info: instamap.simulator.electron_image.CryoEmShotInfo,
    config: PreprocessRelionDatasetConfig
):
    log = logging.getLogger(__name__)
    if config.perturbation.angle is not None:
        log.info(f"Applying random angular perturbation of {config.perturbation.angle} degrees")
        shot_info = perturb_angles(shot_info, config)

    if config.perturbation.offset is not None:
        raise NotImplementedError("Offset perturbation is not implemented yet.")

    return shot_info


def _get_image_size(images: np.ndarray, info):
    logger = logging.getLogger(__name__)
    if 'rlnImageSize' in info:
        image_size = info['rlnImageSize'][0]
    else:
        image_size = images.shape[-1]

    if images.shape[1] != image_size:
        logger.warning(
            f"Image size in data file {images.shape[1]} does not match info size {image_size}."
        )
        logger.warning("Using image size from info file.")

    return image_size


@hydra.main(config_path='conf/preprocessing', config_name="preprocessing", version_base="1.2")
def main(config: PreprocessRelionDatasetConfig):
    logger = logging.getLogger(__name__)

    info_path = hydra.utils.to_absolute_path(config.info_path)
    logger.info(f"Reading info from {info_path}")
    info = read_info(info_path)

    if config.image_path is not None:
        image_path = hydra.utils.to_absolute_path(config.image_path)
        logger.info(f"Reading images from {image_path}")
        image_file_data = mrcfile.open(image_path, 
                                mode='r', 
                                permissive=True
                                ).data

        image_size = _get_image_size(image_file_data, info)

    elif config.read_image_from_starfile:
        pixel_size = instamap.simulator.relion.get_pixel_size(info)
        
        running_len = 0
        image_sizes = []
        initialize_flag = True
        new_info_list = []
        for image_path in np.unique(info["rlnImageName"].apply(lambda x: x.split('@')[-1])):
             
            good_idx = info["rlnImageName"].apply(lambda x: x.endswith(image_path))
            info_subset = info[good_idx]
            new_info_list.append(info_subset)
            frames_idx = info_subset["rlnImageName"].apply(lambda x: int(x.split('@')[0])) - 1 
            logger.info(f"Reading images from {image_path}")
            image_file = mrcfile.open(image_path, 
                                    mode='r', 
                                    permissive=True
                                    )
            image_size = _get_image_size(image_file.data[frames_idx], info_subset)
            if initialize_flag:
                image_file_data = np.zeros((len(info), image_size, image_size), dtype=np.float32)
                initialize_flag = False
            image_file_data[running_len:running_len + len(frames_idx)] = image_file.data[frames_idx]
            running_len += len(frames_idx)
            image_sizes.append(image_size)
        assert len(np.unique(image_sizes)) == 1, "Image sizes are not consistent across the dataset."
        image_size = image_sizes[0]
        info = pd.concat(new_info_list, ignore_index=True)

    else:
        raise ValueError("Either image_path or read_from_starfile must be set.")
    
    pixel_size = instamap.simulator.relion.get_pixel_size(info)

    shot_info = instamap.simulator.relion.make_shot_info(info, image_size)
    shot_info = process_info(shot_info, config)

    if config.subset is not None:
        idx = np.flatnonzero(info["rlnRandomSubset"] == config.subset)
        logger.info(f"Using subset {config.subset} with {len(idx)} images")
    else:
        idx = np.arange(len(info))

    if config.limit is not None:
        rng = np.random.Generator(np.random.PCG64(0))
        idx = rng.choice(idx, config.limit, replace=None, shuffle=False)

    shot_info = {k: v[idx].numpy() for k, v in shot_info.items()}

    images = image_file_data[idx].astype(np.float32)
    images = process(images, config)

    data = {
        'image': images,
        'pixel_size': pixel_size * config.downsample,
        'original_shot_index': idx,
    }
    data.update({'shot_info_' + k: v for k, v in shot_info.items()})


    data['config'] = omegaconf.OmegaConf.to_yaml(config, resolve=True)

    save_path = os.path.abspath('dataset.npz')
    logger.info(f"Saving dataset to {save_path}")
    np.savez(save_path, **data)


if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store(name="base_preprocessing_config", node=PreprocessRelionDatasetConfig)
    main()
