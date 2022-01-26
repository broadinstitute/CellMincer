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

from cellmincer.util import crop_center
    
class Denoise:
    def __init__(
            self,
            input_dir: str,
            output_dir: str,
            model_state: str,
            config: dict,
            peak: Optional[int] = 65535):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        self.avi = config['avi']
        self.window = config.get('window') # None if not provided
        
        clean_path = os.path.join(input_dir, 'clean.npy')
        self.clean = np.load(clean_path) if os.path.exists(clean_path) else None
        if self.window:
            self.clean = self.clean[...,
                self.window['x0']:self.window['x0'] + self.window['x_window'],
                self.window['y0']:self.window['y0'] + self.window['y_window']]
        self.peak = peak
        
        ws_denoising_list, self.denoising_model = None, None #Noise2Self(
#             datasets=[input_dir],
#             config=config).get_resources()
        self.ws_denoising = ws_denoising_list[0]
        self.denoising_model.load_state_dict(torch.load(model_state))
    
    def run(self):
        logging.info('Denoising movie...')
        self.denoising_model.eval()

        if self.window:
            denoised_txy = self.denoising_model.denoise_movie(
                self.ws_denoising,
                x0=self.window['x0'],
                y0=self.window['y0'],
                x_window=self.window['x_window'],
                y_window=self.window['y_window']).numpy()
        else:
            denoised_txy = self.denoising_model.denoise_movie(self.ws_denoising).numpy()

        if self.avi['enabled']:
            assert len(self.avi['sigma_range']) == 2
            
            denoised_norm_txy = self.normalize_movie(
                denoised_txy,
                sigma_lo=self.avi['sigma_range'][0],
                sigma_hi=self.avi['sigma_range'][1])

            writer = skio.FFmpegWriter(
                os.path.join(self.output_dir, f'denoised.avi'),
                outputdict={'-vcodec': 'rawvideo', '-pix_fmt': 'yuv420p', '-r': '60'})
            
            i_start, i_end = self.avi.get('range', (0, len(denoised_norm_txy)))
            
            logging.info(f'Writing .avi with sigma range={self.avi["sigma_range"]}; frames=[{i_start}, {i_end}]')

            for i_frame in range(i_start, i_end):
                writer.writeFrame(denoised_norm_txy[i_frame].T[None, ...])
            writer.close()

        denoised_txy *= self.ws_denoising.cached_features.norm_scale
        if self.window:
            denoised_txy += self.ws_denoising.bg_movie_txy[...,
                self.window['x0']:self.window['x0'] + self.window['x_window'],
                self.window['y0']:self.window['y0'] + self.window['y_window']]
        else:
            denoised_txy += self.ws_denoising.bg_movie_txy
        
        if self.clean is not None:
            mse_t = np.mean(np.square(self.clean - denoised_txy), axis=tuple(range(1, self.clean.ndim)))
            psnr_t = 10 * np.log10(self.peak * self.peak / mse_t)
            np.save(
                os.path.join(self.output_dir, 'psnr_t.npy'),
                psnr_t)

        np.save(
            os.path.join(self.output_dir, f'denoised_tyx.npy'),
            denoised_txy.transpose((0, 2, 1)))
        logging.info('Denoising done.')

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
