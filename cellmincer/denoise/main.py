import os
import logging
import pprint
import time

import json
import pickle

import matplotlib.pylab as plt
import numpy as np
import torch
from typing import List

from torch.optim.lr_scheduler import LambdaLR
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from cellmincer import consts
from cellmincer.containers import Noise2Self
from cellmincer.models import DenoisingModel
from cellmincer.util import \
    OptopatchBaseWorkspace, \
    OptopatchDenoisingWorkspace, \
    crop_center, \
    get_tagged_dir
    
class Denoise:
    def __init__(
            self,
            params: dict):
        self.params = params
        
        self.ws_denoising_list, self.denoising_model = Noise2Self(params).get_resources()
    
    def run(self):
        denoise_dir = get_tagged_dir(
            name=self.params['model']['type'],
            config_tag=self.params['model_tag'],
            root_dir=self.params['root_denoise_dir'])

        if not os.path.exists(denoise_dir):
            os.mkdir(denoise_dir)

        logging.info('Denoising movies...')
        for i_dataset, (name, ws_denoising) in enumerate(zip(self.params['datasets'], self.ws_denoising_list)):
            start = time.time()
            denoised_movie_txy = self.denoising_model.denoise_movie(ws_denoising).numpy()
            
            x_window = min(denoised_movie_txy.shape[-2], ws_denoising.width)
            y_window = min(denoised_movie_txy.shape[-1], ws_denoising.height)
            
            cropped_denoised_movie_txy = crop_center(
                denoised_movie_txy,
                target_width=x_window,
                target_height=y_window)
            
            cropped_base_movie_txy = crop_center(
                ws_denoising.ws_base_bg.movie_txy,
                target_width=x_window,
                target_height=y_window)

            cropped_denoised_movie_txy *= ws_denoising.cached_features.norm_scale
            cropped_denoised_movie_txy += cropped_base_movie_txy

            np.save(
                os.path.join(denoise_dir, f'{name}__denoised_tyx.npy'),
                cropped_denoised_movie_txy.transpose((0, 2, 1)))
            
            elapsed = time.time() - start
            logging.info(f'({i_dataset + 1}/{len(self.ws_denoising_list)}) {name} -- {elapsed:.2f} s')
