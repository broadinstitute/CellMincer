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
from typing import List, Optional

from cellmincer.containers import Noise2Self
from cellmincer.util import crop_center
    
class Denoise:
    def __init__(
            self,
            input_dir: str,
            output_dir: str,
            model_state: str,
            config: dict):
        self.output_dir = output_dir
        self.avi = config['avi']
        
        ws_denoising_list, self.denoising_model = Noise2Self(
            datasets=[input_dir],
            config=config).get_resources()
        self.ws_denoising = ws_denoising_list[0]
        self.denoising_model.load_state_dict(torch.load(model_state))
    
    def run(self):
        logging.info('Denoising movie...')
        self.denoising_model.eval()

        denoised_movie_txy = crop_center(
            self.denoising_model.denoise_movie(self.ws_denoising).numpy(),
            target_width=self.ws_denoising.width,
            target_height=self.ws_denoising.height)

        if self.avi['enabled']:
            assert len(self.avi['sigma_range']) == 2
            
            denoised_movie_norm_txy = self.normalize_movie(
                denoised_movie_txy,
                sigma_lo=self.avi['sigma_range'][0],
                sigma_hi=self.avi['sigma_range'][1])

            writer = skio.FFmpegWriter(
                os.path.join(self.output_dir, f'denoised_movie.avi'),
                outputdict={'-vcodec': 'rawvideo', '-pix_fmt': 'yuv420p', '-r': '60'})
            
            i_start, i_end = self.avi['range'] if 'range' in self.avi else (0, len(denoised_movie_norm_txy))
            
            logging.info(f'Writing .avi with sigma range={self.avi["sigma_range"]}; frames=[{i_start}, {i_end}]')

            for i_frame in range(i_start, i_end):
                writer.writeFrame(denoised_movie_norm_txy[i_frame].T[None, ...])
            writer.close()

        denoised_movie_txy *= self.ws_denoising.cached_features.norm_scale
        denoised_movie_txy += self.ws_denoising.ws_base_bg.movie_txy

        np.save(
            os.path.join(self.output_dir, f'denoised_movie_tyx.npy'),
            denoised_movie_txy.transpose((0, 2, 1)))
        logging.info('Done.')

    def normalize_movie(
            self,
            movie_txy: np.ndarray,
            sigma_lo: float,
            sigma_hi: float,
            mean=None,
            std=None,
            max_intensity=255):
        if mean is None:
            mean = movie_txy.mean()
        if std is None:
            std = movie_txy.std()
        z_movie_txy  = (movie_txy - mean) / std
        norm = Normalize(vmin=sigma_lo, vmax=sigma_hi, clip=True)
        return max_intensity * norm(z_movie_txy)