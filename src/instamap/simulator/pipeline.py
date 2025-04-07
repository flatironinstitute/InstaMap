"""Rendering pipeline implementation."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple
from copy import deepcopy
import time

import torch
import torch.nn

from . import electron_image


def _merge_by_key_exclude_loss(
    info: Dict[str, torch.Tensor], other: Dict[str, torch.Tensor], prefix: str
):
    for key, value in other.items():
        if key.startswith('loss_'):
            info[key] = value
        else:
            info[prefix + '_' + key] = value
    return info


class RenderingPipeline(torch.nn.Module):
    """We model the cryo-EM formation process at three main steps:

    1. The scattering problem, which computes the image formed after the sample
    2. The optics, which computes the CTF and observed image before the sensor
    3. The sensor, which computes the observed response from the wavefunction at the sensor
    """
    def __init__(
        self, scattering: torch.nn.Module, optics: torch.nn.Module, sensor: torch.nn.Module
    ):
        super().__init__()

        self.scattering = scattering
        self.optics = optics
        self.sensor = sensor

    def compute_scattering_image(
        self,
        shot_info: Mapping[str, torch.Tensor],
        do_het: Optional[bool] = False,
        image: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        scattered_image, info = self.scattering(shot_info, do_het=do_het, image=image, generator=generator) 
        return scattered_image, info

    def compute_optics_image(
        self,
        shot_info: Dict[str, torch.Tensor],
        scattered_image: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if scattered_image is None:
            
            scattered_image, _ = self.compute_scattering_image(shot_info, do_het=False, image=scattered_image, generator=generator) 

        sensor_image, info = self.optics(scattered_image, shot_info, generator=generator)
        return sensor_image, info

    def build_info_dictionary(
        self, info_scatter: Dict[str, torch.Tensor], info_optics: Dict[str, torch.Tensor],
        info_sensor: Dict[str,
                          torch.Tensor], scattered_image: torch.Tensor, sensor_image: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        info = {}
        _merge_by_key_exclude_loss(info, info_scatter, 'scattering')
        _merge_by_key_exclude_loss(info, info_optics, 'optics')
        _merge_by_key_exclude_loss(info, info_sensor, 'sensor')
        info['scattering_image'] = scattered_image
        info['optics_image'] = sensor_image
        return info

    def forward(
        self,
        shot_info: electron_image.CryoEmShotInfo,
        do_het: Optional[bool] = False,
        observed: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        regularization_scales: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        scattered_image, info_scatter = self.compute_scattering_image(
            shot_info, do_het=do_het, image=observed, generator=generator
        ) 

        sensor_image, info_optics = self.compute_optics_image(
            shot_info, scattered_image, generator=generator
        )

        observed_image_or_loss, info_sensor = self.sensor(shot_info, sensor_image, observed, generator=generator)

        if observed is not None:
            info_sensor['logging_image_loss'] = deepcopy(info_sensor['image_loss'].detach())

            if regularization_scales is not None:
                for key, value in regularization_scales.items():
                    if key in ['tv', 'l2_norm', 'rigidity']:
                        observed_image_or_loss += value*info_scatter[key]
                    else:
                        Warning(f"Unknown regularization key {key}")

        info = self.build_info_dictionary(
            info_scatter, info_optics, info_sensor, scattered_image, sensor_image
        )


        return observed_image_or_loss, info

    def __call__(
        self,
        shot_info: electron_image.CryoEmShotInfo,
        do_het: Optional[bool] = False,
        observed: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        regularization_scales: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return super().__call__(shot_info, do_het=do_het, observed=observed, generator=generator, regularization_scales=regularization_scales) 
