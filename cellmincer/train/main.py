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
    DenoisingModel, \
    init_model, \
    load_model_from_checkpoint, \
    get_window_padding_from_config

from cellmincer.util import \
    const, \
    crop_center, \
    generate_optimizer, \
    generate_lr_scheduler, \
    generate_batch_indices, \
    generate_occluded_training_data, \
    get_noise2self_loss

import neptune.new as neptune
    
class Train:
    def __init__(
            self,
            inputs: List[str],
            output_dir: str,
            config: dict,
            gpus: int,
            pretrain: Optional[str] = None,
            checkpoint: Optional[str] = None):
        
        self.model = None
        if pretrain:
            self.model = load_model_from_checkpoint(
                model_config=config['model'],
                ckpt_path=pretrain)
        
        # if continuing from checkpoint
        if checkpoint is not None and os.path.exists(checkpoint):
            # TODO figure out what error this raises when checkpoint is bad
            resume_model = load_model_from_checkpoint(
                model_config=config['model'],
                ckpt_path=checkpoint)

            self.model = resume_model
        
        if self.model:
            train_config = self.model.train_config
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
            gpus=gpus)
        
        if self.model is None:
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
            neptune_run = None
            if self.model.neptune_run_id is not None:
                logging.info('Reinitializing existing Neptune run...')
                neptune_run = neptune.init(
                    api_token=config['neptune']['api_token'],
                    project=config['neptune']['project'],
                    run=self.model.neptune_run_id,
                    tags=config['neptune']['tags'])
            else:
                logging.info('Initializing new Neptune run...')

            pl_logger = NeptuneLogger(
                api_key=config['neptune']['api_token'],
                project=config['neptune']['project'],
                run=neptune_run,
                tags=config['neptune']['tags'])

        self.trainer = Trainer(
            strategy='ddp',
            gpus=gpus,
            max_steps=train_config['n_iters'],
            default_root_dir=self.output_dir,
            # TODO experiment with these settings because docs are ambiguous
            callbacks=[ModelCheckpoint(dirpath=self.output_dir, save_last=True, train_time_interval=timedelta(seconds=3600))],
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
                
    def save_final(self):
        torch.save(
            self.model.state_dict(),
            os.path.join(self.output_dir, 'model.pt'))
        
        if self.neptune_enabled:
            self.neptune_run['final'].upload(os.path.join(self.output_dir, 'model.pt'))

    def run(self):
        logging.info('Training model...')
        
        # select validation frames and shape into batches
#         assert self.train_config['n_frames_validation'] % self.train_config['n_batch_validation'] == 0
        
#         val_dataset_indices, val_frame_indices = generate_batch_indices(
#             self.ws_denoising_list,
#             n_batch=self.train_config['n_frames_validation'],
#             t_mid=self.denoising_model.t_order,
#             dataset_selection='balanced')
#         val_batch_shape = (
#             self.train_config['n_frames_validation'] // self.train_config['n_batch_validation'],
#             self.train_config['n_batch_validation'])
#         val_dataset_indices = val_dataset_indices.reshape(val_batch_shape)
#         val_frame_indices = val_frame_indices.reshape(val_batch_shape)

        self.trainer.fit(self.model, self.movie_dm, None)
        
#         last_train_loss = None
#         last_val_loss = None

#         update_time = True
#         for i_iter in range(self.start_iter, self.train_config['n_iters']):
#             if update_time:
#                 start = time.time()
#                 update_time = False

#             norm_p = self.train_config['norm_p']
#             # anneal L0 loss
#             if norm_p == 0:
#                 norm_p = (self.train_config['n_iters'] - i_iter) / self.train_config['n_iters']

#             c_total_loss_hist = []
#             c_rec_loss_hist = []
#             c_reg_loss_hist = []

#             self.denoising_model.train()
#             self.optim.zero_grad()
            
#             # aggregate gradients
#             for i_loop in range(self.train_config['n_loop']):
#                 batch_data = generate_occluded_training_data(
#                     ws_denoising_list=self.ws_denoising_list,
#                     t_order=self.denoising_model.t_order,
#                     t_tandem=self.train_config['t_tandem'],
#                     n_batch=self.train_config['n_batch_per_loop'],
#                     x_window=self.x_train_window,
#                     y_window=self.y_train_window,
#                     x_padding=self.x_train_padding,
#                     y_padding=self.y_train_padding,
#                     include_bg=self.include_bg,
#                     occlusion_prob=self.train_config['occlusion_prob'],
#                     occlusion_radius=self.train_config['occlusion_radius'],
#                     occlusion_strategy=self.train_config['occlusion_strategy'],
#                     device=self.device,
#                     dtype=const.DEFAULT_DTYPE)

#                 loss_dict = get_noise2self_loss(
#                     batch_data=batch_data,
#                     ws_denoising_list=self.ws_denoising_list,
#                     denoising_model=self.denoising_model,
#                     norm_p=norm_p,
#                     loss_type=self.train_config['loss_type'],
#                     enable_continuity_reg=self.enable_continuity_reg,
#                     reg_func=self.train_config['reg_func'],
#                     continuity_reg_strength=self.train_config['continuity_reg_strength'],
#                     noise_threshold_to_std=self.train_config['noise_threshold_to_std'])

#                 # calculate gradient
#                 if self.enable_continuity_reg:
#                     total_loss = (loss_dict['rec_loss'] + loss_dict['reg_loss']) / self.train_config['n_loop']
#                 else:
#                     total_loss = loss_dict['rec_loss'] / self.train_config['n_loop']

#                 total_loss.backward()
                
#                 c_total_loss_hist.append(total_loss.item() * self.train_config['n_loop'])
#                 c_rec_loss_hist.append(loss_dict['rec_loss'].item())

#             current_lr = self.sched.get_lr()[0]

#             self.optim.step()
#             self.sched.step()
            
#             last_train_loss = np.mean(c_total_loss_hist)
#             if self.neptune_enabled:
#                 self.neptune_run['train/iter'].log(i_iter + 1)
#                 self.neptune_run['train/total_loss'].log(last_train_loss)
#                 self.neptune_run['train/rec_loss'].log(np.mean(c_rec_loss_hist))
#                 self.neptune_run['train/lr'].log(current_lr)

#             # validate with n2s loss on select frames
#             if (i_iter + 1) % self.train_config['validate_every'] == 0:
#                 self.denoising_model.eval()
                
#                 x_window_full = self.ws_denoising_list[0].width
#                 y_window_full = self.ws_denoising_list[0].height
#                 x_padding_full, y_padding_full = self.denoising_model.get_window_padding([x_window_full, y_window_full])
                
#                 c_val_loss = []
#                 for val_dataset_batch, val_frame_batch in zip(val_dataset_indices, val_frame_indices):
#                     batch_data = generate_occluded_training_data(
#                         ws_denoising_list=self.ws_denoising_list,
#                         t_order=self.denoising_model.t_order,
#                         t_tandem=0,
#                         n_batch=self.train_config['n_batch_validation'],
#                         # TODO assumes all datasets are compatible with the same full window
#                         x_window=x_window_full,
#                         y_window=y_window_full,
#                         x_padding=x_padding_full,
#                         y_padding=y_padding_full,
#                         include_bg=self.include_bg,
#                         occlusion_prob=1,
#                         occlusion_radius=0,
#                         occlusion_strategy='validation',
#                         dataset_indices=val_dataset_batch,
#                         frame_indices=val_frame_batch,
#                         device=self.device,
#                         dtype=const.DEFAULT_DTYPE)

#                     with torch.no_grad():
#                         loss_dict = get_noise2self_loss(
#                             batch_data=batch_data,
#                             ws_denoising_list=self.ws_denoising_list,
#                             denoising_model=self.denoising_model,
#                             norm_p=norm_p,
#                             loss_type=self.train_config['loss_type'],
#                             enable_continuity_reg=self.enable_continuity_reg,
#                             reg_func=self.train_config['reg_func'],
#                             continuity_reg_strength=self.train_config['continuity_reg_strength'],
#                             noise_threshold_to_std=self.train_config['noise_threshold_to_std'])
                
#                     if self.enable_continuity_reg:
#                         c_val_loss.append((loss_dict['rec_loss'] + loss_dict['reg_loss']).item())
                        
#                     else:
#                         c_val_loss.append(loss_dict['rec_loss'].item())

#                 last_val_loss = np.mean(c_rec_loss_hist)
#                 if self.neptune_enabled:
#                     self.neptune_run['val/iter'].log(i_iter + 1)
#                     self.neptune_run['val/loss'].log(last_val_loss)

#             # compute performance metrics with clean reference
#             if self.insight['enabled'] and (i_iter + 1) % self.insight['evaluate_every'] == 0:
#                 logging.info(f'Evaluating model output against clean reference')
#                 self.evaluate_insight(i_iter)

#             # log training status
#             if (i_iter + 1) % self.train_config['log_every'] == 0:
#                 elapsed = time.time() - start
#                 update_time = True
#                 val_loss_str = f'{last_val_loss:.4f}' if last_val_loss is not None else 'n/a'
#                 logging.info(
#                     f'iter {i_iter + 1}/{self.train_config["n_iters"]} | '
#                     f'train loss={last_train_loss:.4f} | '
#                     f'val loss={val_loss_str} | '
#                     f'{self.train_config["log_every"] / elapsed:.2f} iter/s')

#             # write checkpoint
#             if (i_iter + 1) % self.train_config['checkpoint_every'] == 0:
#                 if 'checkpoint_path' in self.train_config:
#                     index = (i_iter + 1) // self.train_config['checkpoint_every']
#                     logging.info(f'Updating and gzipping checkpoint at index {index}')
#                     self.save_checkpoint(self.train_config['checkpoint_path'], index)
#                 else:
#                     logging.info(f'No checkpoint path specified; skipping.')

        # save trained model
        logging.info('Training complete; saving model...')
        self.save_final()
