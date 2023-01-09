import os
import logging
import pprint
import time
from datetime import timedelta

import json
import pickle
import tarfile

import matplotlib.pylab as plt
import numpy as np
import skimage
import torch
import pandas as pd
from typing import List, Optional, Tuple

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

from cellmincer.datasets import build_datamodule

from cellmincer.models import \
    init_model, \
    get_window_padding_from_config, \
    load_model_from_checkpoint

from cellmincer.util import \
    const, \
    crop_center

import neptune.new as neptune
    
class Train:
    def __init__(
            self,
            inputs: List[str],
            output_dir: str,
            config: dict,
            gpus: int,
            use_memmap: bool,
            pretrain: Optional[str] = None,
            checkpoint: Optional[str] = None,
            checkpoint_start: Optional[str] = None):
        
        self.model = None
        if pretrain:
            try:
                self.model = load_model_from_checkpoint(
                    model_type=config['model']['type'],
                    ckpt_path=pretrain)
                logging.info('Loaded pretrained model state.')
            except EOFError:
                logging.warning('Bad checkpoint; ignoring pretrained state.')
        
        # if continuing from checkpoint
        self.ckpt_path = None
        if checkpoint_start is not None and os.path.exists(checkpoint_start):
            try:
                resume_model = load_model_from_checkpoint(
                    model_type=config['model']['type'],
                    ckpt_path=checkpoint_start)
                self.model = resume_model
                self.ckpt_path = checkpoint_start
                logging.info('Successfully loaded restarted checkpoint.')
            except EOFError:
                logging.warning('Bad checkpoint; ignoring checkpointed restart state.')
                pass
        
        # if continuing from checkpoint
        if checkpoint is not None and os.path.exists(checkpoint):
            try:
                resume_model = load_model_from_checkpoint(
                    model_type=config['model']['type'],
                    ckpt_path=checkpoint)
                self.model = resume_model
                self.ckpt_path = checkpoint
                logging.info('Successfully loaded checkpoint.')
            except EOFError:
                logging.warning('Bad checkpoint; ignoring checkpointed state. Expected behavior when Terra first initializes training.')
                pass
        
        if self.model:
            train_config = self.model.hparams.train_config
        else:
            train_config = config['train']
            
            # compute training padding with maximal output/input receptive field ratio
            output_min_size = np.arange(config['train']['output_min_size_lo'], config['train']['output_min_size_hi'] + 1)
            output_min_size = output_min_size[(output_min_size % 2) == 0]
            train_padding = get_window_padding_from_config(
                model_config=config['model'],
                output_min_size=output_min_size)
            best_train = np.argmax(output_min_size / (output_min_size + 2 * train_padding))
            train_config['x_window'] = train_config['y_window'] = output_min_size[best_train]
            train_config['x_padding'] = train_config['y_padding'] = train_padding[best_train]
            
        self.movie_dm = build_datamodule(
            datasets=inputs,
            model_config=config['model'],
            train_config=train_config,
            gpus=gpus,
            use_memmap=use_memmap)
        
        if self.model is None:
            logging.info('Initializing new model.')
            model_config = config['model']
            model_config['n_global_features'] = self.movie_dm.n_global_features
            self.model = init_model(
                model_config=model_config,
                train_config=train_config)
        
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        
        self.neptune_enabled = config['neptune']['enabled']
        pl_logger = True
        
        if self.neptune_enabled:
            if self.model.neptune_run_id is not None:
                logging.info('Reinitializing existing Neptune run...')
                neptune_run = neptune.init(
                    api_token=config['neptune']['api_token'],
                    project=config['neptune']['project'],
                    run=self.model.neptune_run_id,
                    tags=config['neptune']['tags'],
                    capture_hardware_metrics=False)
                pl_logger = NeptuneLogger(run=neptune_run)
            else:
                logging.info('Initializing new Neptune run...')
                pl_logger = NeptuneLogger(
                    api_key=config['neptune']['api_token'],
                    project=config['neptune']['project'],
                    tags=config['neptune']['tags'])
                pl_logger.experiment['datasets'] = inputs
        else:
            logging.info('Skipping Neptune initialization.')

        self.trainer = Trainer(
            strategy='ddp',
            gpus=gpus,
            max_epochs=train_config['n_iters'],
            default_root_dir=self.output_dir,
            # TODO experiment with these settings because docs are ambiguous
            callbacks=[ModelCheckpoint(dirpath=self.output_dir, save_last=True)],
            logger=pl_logger)
        
        self.insight = config['insight']
        if self.insight['enabled']:
            self.bg_paths = [os.path.join(dataset, 'trend.npy') for dataset in inputs]
            self.clean_paths = [os.path.join(dataset, 'clean.npy') for dataset in inputs]

    def evaluate_insight(self, i_iter: int):
        self.denoising_model.eval()
        for i_dataset, (ws_denoising, bg_path, clean_path) in enumerate(zip(self.ws_denoising_list, self.bg_paths, self.clean_paths)):
            denoised = crop_center(
                self.denoising_model.denoise_movie(ws_denoising).numpy(),
                target_width=ws_denoising.width,
                target_height=ws_denoising.height)
            
            denoised *= ws_denoising.cached_features.norm_scale
            denoised += np.load(bg_path)

            clean = np.load(clean_path)
            
            # compute psnr
            mse_t = np.mean(np.square(clean - denoised), axis=tuple(range(1, clean.ndim)))
            psnr_t = 10 * np.log10(self.insight['peak'] * self.insight['peak'] / mse_t)
            
            # compute ssim
            mssim_t = []
            S_accumulate = np.zeros(clean.shape[1:])
            for clean_frame, denoised_frame in zip(clean, denoised):
                mssim, S = skimage.metrics.structural_similarity(
                    clean_frame,
                    denoised_frame,
                    gaussian_weights=True,
                    full=True,
                    data_range=self.insight['peak'])
                mssim_t.append(mssim)
                S_accumulate += (S + 1) / 2
            
            if self.neptune_enabled:
                self.neptune_run['metrics/iter'].log(i_iter + 1)
                
                self.neptune_run[f'metrics/{i_dataset}/psnr/mean'].log(np.mean(psnr_t))
                self.neptune_run[f'metrics/{i_dataset}/psnr/var'].log(np.var(psnr_t))
                self.neptune_run[f'metrics/{i_dataset}/psnr/median'].log(np.median(psnr_t))
                self.neptune_run[f'metrics/{i_dataset}/psnr/q1'].log(np.quantile(psnr_t, 0.25))
                self.neptune_run[f'metrics/{i_dataset}/psnr/q3'].log(np.quantile(psnr_t, 0.75))
                
                self.neptune_run[f'metrics/{i_dataset}/ssim/mean'].log(np.mean(mssim_t))
                self.neptune_run[f'metrics/{i_dataset}/ssim/var'].log(np.var(mssim_t))
                self.neptune_run[f'metrics/{i_dataset}/ssim/median'].log(np.median(mssim_t))
                self.neptune_run[f'metrics/{i_dataset}/ssim/q1'].log(np.quantile(mssim_t, 0.25))
                self.neptune_run[f'metrics/{i_dataset}/ssim/q3'].log(np.quantile(mssim_t, 0.75))
                self.neptune_run[f'metrics/{i_dataset}/ssim/map'].log(neptune.types.File.as_image(S_accumulate / len(mssim_t)))

    def run(self):
        logging.info('Training model...')
        
        self.trainer.fit(
            model=self.model,
            train_dataloaders=self.movie_dm,
            ckpt_path=self.ckpt_path)

        # save trained model
        logging.info('Training complete; saving model...')
        if self.neptune_enabled:
            self.model.logger.experiment['final'].upload(os.path.join(self.output_dir, 'last.ckpt'))
