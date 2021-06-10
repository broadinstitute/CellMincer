import numpy as np
import torch
from torch import nn
from typing import List, Tuple, Dict

from .denoising_model import DenoisingModel

# import model subclasses here...
from .spatial_unet_2d_temporal_denoiser import SpatialUnet2dTemporalDenoiser


# ...and add models to this lookup dictionary
MODEL_DICT = {
    'spatial-unet-2d-temporal-denoiser': SpatialUnet2dTemporalDenoiser
}

def initialize_model(
        model_config: dict,
        model_state_path: str = None,
        device: torch.device = torch.device('cuda'),
        dtype: torch.dtype = torch.float32) -> DenoisingModel:
    try:
        denoising_model = MODEL_DICT[model_config['type']](model_config, device, dtype)
    except KeyError:
        print('Unrecognized model type; see recognized options:')
        for key in MODEL_DICT:
            print(f'\t{key}')
        exit(0)
    if model_state_path is not None:
        denoising_model.load_state_dict(torch.load(model_state_path))
        
    return denoising_model
    
def get_minimum_padding(
            model_config: dict,
            training_x_window: int,
            training_y_window: int) -> Tuple[int, int]:
    if model_config['type'] == 'spatial-unet-2d-temporal-denoiser':
        return SpatialUnet2dTemporalDenoiser.get_minimum_padding(
            model_config,
            training_x_window,
            training_y_window)
