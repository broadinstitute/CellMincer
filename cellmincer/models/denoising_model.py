import numpy as np
import torch
from torch import nn
from typing import List, Tuple, Dict

import sys

from cellmincer.util.ws import OptopatchBaseWorkspace, OptopatchDenoisingWorkspace


class DenoisingModel(nn.Module):
    def __init__(
            self,
            name: str,
            t_order: int,
            device: torch.device,
            dtype: torch.dtype):
        
        assert t_order & 1 == 1
        
        super(DenoisingModel, self).__init__()
        
        self.name = name
        self.t_order = t_order
        self.device = device
        self.dtype = dtype

    '''
    Denoises the 'diff' movie segment in ws_denoising,
    bounded by [t_begin, t_end) and windowed by (x0, y0, x_window, y_window).
    
    Returns a CPU tensor containing the denoised movie.
    '''
    def denoise_movie(
            self,
            ws_denoising: OptopatchDenoisingWorkspace,
            t_begin: int = 0,
            t_end: int = None,
            x0: int = 0,
            y0: int = 0,
            x_window: int = None,
            y_window: int = None
        ) -> torch.Tensor:
        
        raise NotImplementedError

    '''
    Computes the minimum xy-padding on the input tensor that the model needs.
    '''
    @staticmethod
    def get_minimum_padding(
            config: dict,
            training_x_window: int,
            training_y_window: int) -> Tuple[int, int]:
        raise NotImplementedError
