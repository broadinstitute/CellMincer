import os
import logging
import pprint
import time

import json
import pickle
import tifffile

from skvideo import io as skio
from matplotlib.colors import Normalize
import matplotlib.pylab as plt
import numpy as np
import torch
from typing import List, Optional

from cellmincer.datasets import build_ws_denoising

from cellmincer.models import load_model_from_checkpoint

from cellmincer.util import const, crop_center
    
class Denoise:
    def __init__(
            self,
            input_dir: str,
            output_dir: str,
            model_ckpt: str,
            model_type: str,
            avi_enabled: bool,
            avi_frames: Optional[List[int]],
            avi_sigma: Optional[List[int]],
            peak: Optional[int] = 65535):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        
        clean_path = os.path.join(input_dir, 'clean.npy')
        self.clean = np.load(clean_path) if os.path.exists(clean_path) else None
        
        self.model = load_model_from_checkpoint(
            model_type=model_type,
            ckpt_path=model_ckpt)
        self.model.hparams.model_config['occlude_padding'] = False
        self.model.to(const.DEFAULT_DEVICE)
        
        self.ws_denoising = build_ws_denoising(
            dataset=input_dir,
            model_config=self.model.hparams.model_config,
            device=torch.device('cuda'),
            use_memmap=False)

        self.avi_enabled = avi_enabled
        self.avi_frames = avi_frames if avi_frames is not None else [0, self.ws_denoising.n_frames]
        self.avi_sigma = avi_sigma if avi_sigma is not None else [0, 10]
        self.peak = peak
    
    def run(self):
        logging.info('Denoising movie...')
        self.model.eval()

        denoised_txy = self.model.denoising_model.denoise_movie(self.ws_denoising).numpy()

        denoised_txy *= self.ws_denoising.cached_features.norm_scale
        
        tifffile.imwrite(
            os.path.join(self.output_dir, f'denoised_detrended_tyx.tif'),
            denoised_txy.transpose((0, 2, 1)))

        if self.avi_enabled:
            denoised_norm_txy = self.normalize_movie(
                denoised_txy,
                sigma_lo=self.avi_sigma[0],
                sigma_hi=self.avi_sigma[1])

            writer = skio.FFmpegWriter(
                os.path.join(self.output_dir, f'denoised.avi'),
                outputdict={'-vcodec': 'rawvideo', '-pix_fmt': 'yuv420p', '-r': '60'})
            
            i_start, i_end = self.avi_frames
            if i_start < 0:
                i_start += self.ws_denoising.n_frames
            if i_end < 0:
                i_end += self.ws_denoising.n_frames
            
            logging.info(f'Writing .avi with sigma range={self.avi_sigma}; frames=[{i_start}, {i_end}]')

            for i_frame in range(i_start, i_end):
                writer.writeFrame(denoised_norm_txy[i_frame].T[None, ...])
            writer.close()

        denoised_txy += self.ws_denoising.bg_movie_txy
        
        if self.clean is not None:
            mse_t = np.mean(np.square(self.clean - denoised_txy), axis=tuple(range(1, self.clean.ndim)))
            psnr_t = 10 * np.log10(self.peak * self.peak / mse_t)
            np.save(
                os.path.join(self.output_dir, 'psnr_t.npy'),
                psnr_t)

        tifffile.imwrite(
            os.path.join(self.output_dir, f'denoised_tyx.tif'),
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
