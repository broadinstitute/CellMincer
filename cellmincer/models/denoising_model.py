import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl

import logging
from typing import Tuple, Union

from cellmincer.util.ws import OptopatchBaseWorkspace, OptopatchDenoisingWorkspace


class DenoisingModel(pl.LightningModule):
    def __init__(
            self,
            name: str,
            t_order: int):
        if t_order is None:
            logging.warning('t_order assignment deferred; ensure this behavior is intended')
            logging.info(f'name: {name}')
        else:
            assert t_order & 1 == 1
            logging.info(f'name: {name} | t_order: {t_order}')
        
        super(DenoisingModel, self).__init__()
        self.name = name
        self.t_order = t_order

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
            y_window: int = None) -> torch.Tensor:
        raise NotImplementedError

    '''
    Prints summary of network architecture.
    '''
    def summary(self, ws_denoising: OptopatchDenoisingWorkspace):
        raise NotImplementedError

    '''
    Returns the padding size for an image dimension with given minimum output size.
    '''
    @staticmethod
    def get_window_padding(
            output_min_size: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        raise NotImplementedError

    '''
    Returns the temporal order of a model with given configuration.
    '''
    @staticmethod
    def get_temporal_order_from_config(config: dict) -> int:
        raise NotImplementedError

    '''
    Returns the padding size for an image dimension
    with given minimum output size and model configuration.
    '''
    @staticmethod
    def get_window_padding_from_config(
            config: dict,
            output_min_size: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        raise NotImplementedError
