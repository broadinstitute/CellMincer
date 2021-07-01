import os
import logging
import pprint
import time

import numpy as np
import pandas as pd
import skimage
    
class Insight:
    def __init__(
            self,
            params: dict):
        
        self.clean = np.load(params['clean_path'])
        self.denoised = np.load(params['denoised_path'])
        self.insight_dir = params['insight_dir']
        
        assert self.clean.shape == self.denoised.shape, f'{self.clean.shape} != {self.denoised.shape}'
        
        self.peak = params['peak']
    
    def run(self):
        logging.info('Computing per-frame PSNR...')
        psnr_t = self.psnr()
        psnr_stats = {
            'mean': [np.mean(psnr_t)],
            'var': [np.var(psnr_t)],
            'median': [np.median(psnr_t)],
            'q1': [np.quantile(psnr_t, 0.25)],
            'q3': [np.quantile(psnr_t, 0.75)]
        }
        
        logging.info('Computing per-frame SSIM...')
        mssim_t = []
        S_accumulate = np.zeros(self.clean.shape[1:])
        for clean, denoised in zip(self.clean, self.denoised):
            mssim, S = skimage.metrics.structural_similarity(clean, denoised, gaussian_weights=True, full=True)
            mssim_t.append(mssim)
            S_accumulate += S
        mssim_stats = {
            'mean': [np.mean(mssim_t)],
            'var': [np.var(mssim_t)],
            'median': [np.median(mssim_t)],
            'q1': [np.quantile(mssim_t, 0.25)],
            'q3': [np.quantile(mssim_t, 0.75)]
        }
        S_mean = S_accumulate / len(mssim_t)
        
        logging.info('Saving results...')
        pd.DataFrame(psnr_stats).to_csv(os.path.join(self.insight_dir, 'psnr_stats.csv'), index=False)
        pd.DataFrame(mssim_stats).to_csv(os.path.join(self.insight_dir, 'mssim_stats.csv'), index=False)
        np.save(os.path.join(self.insight_dir, 'ssim_map.npy'), S_mean)

    def psnr(self):
        mse_t = np.mean(np.square(self.clean - self.denoised), axis=tuple(range(1, self.clean.ndim)))
        psnr_t = 10 * np.log10(self.peak * self.peak / mse_t)

        return psnr_t
