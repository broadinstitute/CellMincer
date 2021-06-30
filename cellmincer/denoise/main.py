import os
import logging
import pprint
import time

import json
import pickle

from skvideo import io as skio
from matplotlib.colors import Normalize
import matplotlib.pylab as plt
import numpy as np
import torch
from typing import List

from torch.optim.lr_scheduler import LambdaLR
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from cellmincer.containers import Noise2Self
from cellmincer.models import DenoisingModel
from cellmincer.util import \
    OptopatchBaseWorkspace, \
    OptopatchDenoisingWorkspace, \
    crop_center
    
class Denoise:
    def __init__(
            self,
            params: dict):
        
        self.name = params['name']
        self.root_denoise_dir = params['root_denoise_dir']
        self.datasets = params['datasets']
        self.write_avi = params['write_avi']
        
        self.ws_denoising_list, self.denoising_model = Noise2Self(params).get_resources()
        self.denoising_model.load_state_dict(torch.load(params['model_state_path']))
    
    def run(self):
        denoise_dir = os.path.join(self.root_denoise_dir, self.name)
        if not os.path.exists(denoise_dir):
            os.mkdir(denoise_dir)

        logging.info('Denoising movies...')
        self.denoising_model.eval()
        
        for i_dataset, (name, ws_denoising) in enumerate(zip(self.datasets, self.ws_denoising_list)):
            start = time.time()
            denoised_movie_txy = crop_center(
                self.denoising_model.denoise_movie(ws_denoising).numpy(),
                target_width=ws_denoising.width,
                target_height=ws_denoising.height)

            denoised_movie_txy *= ws_denoising.cached_features.norm_scale
            denoised_movie_txy += ws_denoising.ws_base_bg.movie_txy

            np.save(
                os.path.join(denoise_dir, f'{name}__denoised_tyx.npy'),
                denoised_movie_txy.transpose((0, 2, 1)))
            
            elapsed = time.time() - start
            logging.info(f'({i_dataset + 1}/{len(self.ws_denoising_list)}) {name} -- {elapsed:.2f} s')
            
            if self.write_avi:
                denoised_movie_norm_txy = self.normalize_movie(
                    denoised_movie_txy - ws_denoising.ws_base_bg.movie_txy,
                    n_sigmas=10)
                
                writer = skio.FFmpegWriter(
                    os.path.join(denoise_dir, f'{name}__denoised.avi'),
                    outputdict={'-vcodec': 'rawvideo', '-pix_fmt': 'yuv420p', '-r': '60'})
                
                for i in range(len(denoised_movie_norm_txy)):
                    writer.writeFrame(denoised_movie_norm_txy[i].T[None, ...])
                writer.close()

    
    def normalize_movie(
            self,
            movie_txy: np.ndarray,
            n_sigmas: float,
            mean=None,
            std=None,
            max_intensity=255):
        if mean is None:
            mean = movie_txy.mean(axis=0)
        if std is None:
            std = movie_txy.std(axis=0)
        z_movie_txy  = (movie_txy - mean) / std
        norm = Normalize(vmin=0, vmax=n_sigmas, clip=True)
        return max_intensity * norm(z_movie_txy)
