import numpy as np
import torch
from torch import nn

import logging

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
        model_state_path: str = None,
        device: torch.device = torch.device('cuda'),
        dtype: torch.dtype = torch.float32) -> DenoisingModel:
    try:
        denoising_model = _MODEL_DICT[model_config['type']](model_config, device, dtype)
    except KeyError:
        logging.warning('Unrecognized model type; see recognized options:')
        exit(0)
    if model_state_path is not None:
        denoising_model.load_state_dict(torch.load(model_state_path))
        
    return denoising_model
