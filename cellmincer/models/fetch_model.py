import numpy as np
import torch
from torch import nn

import logging
from typing import Tuple

from .denoising_model import DenoisingModel

# import model subclasses here...
from .spatial_unet_2d_temporal_denoiser import SpatialUnet2dTemporalDenoiser
from .spatial_unet_2d_multiframe import SpatialUnet2dMultiframe
from .spatiotemporal_unet_3d import SpatiotemporalUnet3d


# ...and add models to this lookup dictionary
_MODEL_DICT = {
    'spatial-unet-2d-temporal-denoiser': SpatialUnet2dTemporalDenoiser,
    'spatial-unet-2d-multiframe': SpatialUnet2dMultiframe,
    'spatiotemporal-unet-3d': SpatiotemporalUnet3d
}


def init_model(
        model_config: dict,
        device: torch.device = torch.device('cuda'),
        dtype: torch.dtype = torch.float32) -> DenoisingModel:
    try:
        denoising_model = _MODEL_DICT[model_config['type']](model_config, device, dtype)
    except KeyError:
        logging.warning(f'Unrecognized model type; options are {", ".join([name for name in _MODEL_DICT])}')
        exit(0)
        
    return denoising_model


def get_best_window_padding(
        model_config: dict,
        output_min_size_lo: int,
        output_min_size_hi: int) -> Tuple[int, int, int, int]:
    try:
        return _MODEL_DICT[model_config['type']].get_best_window_padding(
            config=model_config,
            output_min_size_lo=output_min_size_lo,
            output_min_size_hi=output_min_size_hi)
    except KeyError:
        logging.warning(f'Unrecognized model type; options are {", ".join([name for name in _MODEL_DICT])}')
        exit(0)