import numpy as np
import torch
from torch import nn

import logging
from typing import Tuple

from cellmincer.util.ws import OptopatchBaseWorkspace, OptopatchDenoisingWorkspace


class DenoisingModel(nn.Module):
    def __init__(
            self,
            name: str,
            t_order: int,
            device: torch.device,
            dtype: torch.dtype):
        
        if t_order is None:
            logging.warning('t_order assignment deferred; ensure this behavior is intended')
            logging.info(f'name: {name}')
        else:
            assert t_order & 1 == 1
            logging.info(f'name: {name} | t_order: {t_order}')
        
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

    def get_best_input_size(
            self,
            output_min_lo: int,
            output_min_hi: int) -> Tuple[int, int]:
        
        raise NotImplementedError