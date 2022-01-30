import numpy as np
import torch
from torch import nn

import logging
from typing import Tuple, Union

from .denoising_model import DenoisingModel

# import model subclasses here...
from .spatial_unet_2d_temporal_denoiser import SpatialUnet2dTemporalDenoiser, PlSpatialUnet2dTemporalDenoiser

# ...and add models to this lookup dictionary
_MODEL_DICT = {
    'spatial-unet-2d-temporal-denoiser': {'model': SpatialUnet2dTemporalDenoiser, 'wrapper': PlSpatialUnet2dTemporalDenoiser}
}


def init_model(
        model_config: dict,
        train_config: dict) -> DenoisingModel:
    if model_config['type'] in _MODEL_DICT:
        denoising_model = _MODEL_DICT[model_config['type']]['wrapper'](
            model_config=model_config,
            train_config=train_config)
    else:
        logging.warning(f'Unrecognized model type; options are {", ".join([name for name in _MODEL_DICT])}')
        exit(0)
        
    return denoising_model

def load_model_from_checkpoint(
        model_type: str,
        ckpt_path: str) -> DenoisingModel:
    if model_type in _MODEL_DICT:
        denoising_model = _MODEL_DICT[model_type]['wrapper'].load_from_checkpoint(ckpt_path)
    else:
        logging.warning(f'Unrecognized model type; options are {", ".join([name for name in _MODEL_DICT])}')
        exit(0)
        
    return denoising_model

def get_temporal_order_from_config(model_config: dict) -> int:
    if model_config['type'] in _MODEL_DICT:
        return _MODEL_DICT[model_config['type']]['model'].get_temporal_order_from_config(config=model_config)
    else:
        logging.warning(f'Unrecognized model type; options are {", ".join([name for name in _MODEL_DICT])}')
        exit(0)

def get_window_padding_from_config(
        model_config: dict,
        output_min_size: Union[int, list, np.ndarray]) -> np.ndarray:
    if model_config['type'] in _MODEL_DICT:
        return _MODEL_DICT[model_config['type']]['model'].get_window_padding_from_config(
            config=model_config,
            output_min_size=output_min_size)
    else:
        logging.warning(f'Unrecognized model type; options are {", ".join([name for name in _MODEL_DICT])}')
        exit(0)