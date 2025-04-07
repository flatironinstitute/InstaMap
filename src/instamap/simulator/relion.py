"""Compatibility with Relion information
"""

from __future__ import annotations

import logging

from typing import Any, Mapping, Optional, Tuple

import mrcfile
import numpy as np
import torch
import starfile
import scipy.spatial.transform

from . import electron_image


def _handle_offset(offset: torch.Tensor, size: int, pixel_size: float) -> torch.Tensor:
    if size % 2 == 0:
        offset = offset + 0.5 * pixel_size
    return offset


def _cast(v):
    if isinstance(v, torch.Tensor):
        return v

    if hasattr(v, 'to_numpy'):
        v = v.to_numpy()
    return torch.as_tensor(v, dtype=torch.float32)


def get_pixel_size(info: Mapping[str, Any]) -> float:
    """Compute pixel size in Angstrom from relion star file information.
    """
    if 'rlnImagePixelSize' in info:
        pixel_size_angstrom = _cast(info['rlnImagePixelSize'])
    elif 'rlnDetectorPixelSize' in info:
        detector_pixel_size = _cast(info['rlnDetectorPixelSize'])
        magnification = _cast(info['rlnMagnification'])
        pixel_size_angstrom = detector_pixel_size / magnification * 1e4

    pixel_size_angstrom = np.unique(pixel_size_angstrom)
    if len(pixel_size_angstrom) != 1:
        raise ValueError('Pixel size is not constant across the dataset.')

    return float(pixel_size_angstrom[0])


def make_shot_info(
    info: Mapping[str, Any],
    image_size: Optional[int] = None,
    pixel_size_angstrom: Optional[float] = None
) -> electron_image.CryoEmShotInfo:
    """Create shot info from a mapping of relion star file information.
    """

    if image_size is None:
        image_size = _cast(info['rlnImageSize'])

    if pixel_size_angstrom is None:
        pixel_size_angstrom = get_pixel_size(info)

    if 'rlnOriginXAngst' in info:
        offset_x = _cast(info['rlnOriginXAngst'])
    elif 'rlnOriginX' in info:
        offset_x = _cast(info['rlnOriginX']) * pixel_size_angstrom
    else:
        offset_x = torch.zeros(len(info))
    offset_x = _handle_offset(offset_x, image_size, pixel_size_angstrom)

    if 'rlnOriginYAngst' in info:
        offset_y = _cast(info['rlnOriginYAngst'])
    elif 'rlnOriginY' in info:
        offset_y = _cast(info['rlnOriginY']) * pixel_size_angstrom
    else:
        offset_y = torch.zeros(len(info))
    offset_y = _handle_offset(offset_y, image_size, pixel_size_angstrom)

    if 'rlnCtfBfactor' not in info:
        info['rlnCtfBfactor'] = 0.0

    shot_info = electron_image.CryoEmShotInfo(
        view_phi=info['rlnAngleRot'],
        view_theta=info['rlnAngleTilt'],
        view_psi=info['rlnAnglePsi'],
        offset_x=offset_x,
        offset_y=offset_y,
        defocus_u=info['rlnDefocusU'],
        defocus_v=info['rlnDefocusV'],
        defocus_angle=info['rlnDefocusAngle'],
        spherical_aberration=info['rlnSphericalAberration'],
        voltage=info['rlnVoltage'],
        amplitude_contrast_ratio=info['rlnAmplitudeContrast'],
        phase_shift=info['rlnPhaseShift'],
        b_factor=info['rlnCtfBfactor'],
    )

    return {k: _cast(v) for k, v in shot_info.items()}


def get_rotation_scipy(
    shot_info: electron_image.CryoEmShotInfo
) -> scipy.spatial.transform.Rotation:
    """Obtain the rotation from the shot info, as a scipy Rotation object.

    This is useful for converting to/from the Relion convention, which creates
    the rotation from Euler ZYZ angles applied intrinsically.
    """
    angles = torch.stack(
        [shot_info['view_phi'], shot_info['view_theta'], shot_info['view_psi']], dim=-1
    )
    return scipy.spatial.transform.Rotation.from_euler('ZYZ', angles, degrees=True)


def load_shot_info_from_file(star_path: str, default_image_size: Optional[int] = None) -> Tuple[float, electron_image.CryoEmShotInfo]:
    """Loads shot information from a relion star file.

    Parameters
    ----------
    star_path : str
        Path to the star file.
    default_image_size : int, optional
        Default image size to use if the star file does not contain the image size.
    """
    logger = logging.getLogger(__name__)
    info = starfile.read(star_path, always_dict=True)

    if len(info) == 1:
        particles = next(iter(info.values()))
    else:
        particles = info['particles']
        optics = info['optics']
        particles = particles.join(optics.set_index('rlnOpticsGroup'), on='rlnOpticsGroup')

    if 'rlnImageSize' in particles.columns:
        image_sizes = particles['rlnImageSize'].unique()
        if len(image_sizes) > 1:
            raise ValueError('Found multiple image sizes in the star file')
        image_size = image_sizes[0]
    elif default_image_size is not None:
        image_size = default_image_size
        logger.warning(f'No image size found in star file, assuming {image_size}')
    else:
        raise ValueError('No image size found in star file and no default image size provided')

    pixel_size = get_pixel_size(particles)
    shot_info = make_shot_info(
        particles, image_size=image_size, pixel_size_angstrom=pixel_size
    )

    return pixel_size, shot_info


def save_stack_to_mrcs(data: torch.Tensor, path: str):
    """Saves an image stack to the given MRC file.

    This function handles converting the image stack to the correct convention,
    and saving it to the given file.

    Parameters
    ----------
    data : torch.Tensor
        The batch of images to save. Must be of shape (batch, height, width).
    path : str
        The path to save the MRC file to.
    """
    with mrcfile.new(path, overwrite=True) as mrc:
        data = torch.rot90(data, dims=(-2, -1)).neg_().to(dtype=torch.float32, device='cpu')
        mrc.set_data(data.numpy())
