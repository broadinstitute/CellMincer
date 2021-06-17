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

from cellmincer.containers import Noise2Self
from cellmincer.models import DenoisingModel
from cellmincer.util import \
    OptopatchBaseWorkspace, \
    OptopatchDenoisingWorkspace, \
    consts, \
    crop_center, \
    get_nn_spatio_temporal_mean, \
    get_nn_spatial_mean, \
    generate_optimizer, \
    generate_lr_scheduler, \
    generate_batch_indices, \
    generate_occluded_training_data, \
    get_noise2self_loss, \
    save_model_state
    
class Train:
    def __init__(
            self,
            params: dict):
        self.params = params
        
        self.ws_denoising_list, self.denoising_model = Noise2Self(params).get_resources()

        if 'state_index' in self.params:
            model_dir = os.path.join(self.params['root_model_dir'], self.params['model']['type'])
            self.optim = torch.load(
                os.path.join(model_dir, f'{self.params["state_index"]:06d}', 'optim_state.pt'))
            self.sched = torch.load(
                os.path.join(model_dir, f'{self.params["state_index"]:06d}', 'sched_state.pt'))
        else:
            self.optim = None
            self.sched = None
        
    def run(self):
        model_dir = os.path.join(self.params['root_model_dir'], self.params['model']['type'])
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        logging.info('Training model...')
        
        # get xy training window
        x_window, y_window, x_padding, y_padding = self.denoising_model.get_best_input_size(
            self.params['training']['output_min_size_lo'],
            self.params['training']['output_min_size_hi'])
        
        if 'seed' in self.params['training']:
            torch.manual_seed(self.params['training']['seed'])
        self.denoising_model.train()

        n_iters = self.params['training']['n_iters']
        n_loop = self.params['training']['n_loop']

        # initialize optimizer and scheduler if not loaded from save state
        if self.optim is None:
            self.optim = generate_optimizer(
                denoising_model=self.denoising_model,
                optim_params=self.params['training']['optim_params'],
                lr=self.params['training']['lr_params']['max'])
        if self.sched is None:
            self.sched = generate_lr_scheduler(
                optim=self.optim,
                lr_params=self.params['training']['lr_params'],
                n_iters=n_iters)

        enable_continuity_reg = self.params['training']['enable_continuity_reg']
        
        start_iter = (
            self.params['state_index'] * self.params['training']['save_every']
            if 'state_index' in self.params else 0)
        
        val_dataset_indices, val_frame_indices = generate_batch_indices(
            self.ws_denoising_list,
            n_batch=self.params['training']['n_batch_validation'],
            t_mid=self.denoising_model.t_order,
            dataset_selection='balanced')
        last_val_loss = None

        rec_loss_hist = []
        reg_loss_hist = []

        update_time = True
        for i_iter in range(start_iter, n_iters):
            if update_time:
                start = time.time()
                update_time = False
            
            c_rec_loss_hist = []
            c_reg_loss_hist = []

            self.optim.zero_grad()
            
            # aggregate gradients
            for i_loop in range(n_loop):
                batch_data = generate_occluded_training_data(
                    ws_denoising_list=self.ws_denoising_list,
                    t_order=self.denoising_model.t_order,
                    t_tandem=self.params['training']['t_tandem'],
                    n_batch=self.params['training']['n_batch_per_loop'],
                    x_window=x_window,
                    y_window=y_window,
                    occlusion_prob=self.params['training']['occlusion_prob'],
                    occlusion_radius=self.params['training']['occlusion_radius'],
                    occlusion_strategy=self.params['training']['occlusion_strategy'],
                    device=self.params['device'],
                    dtype=consts.DEFAULT_DTYPE)

                loss_dict = get_noise2self_loss(
                    batch_data=batch_data,
                    ws_denoising_list=self.ws_denoising_list,
                    denoising_model=self.denoising_model,
                    norm_p=self.params['training']['norm_p'],
                    loss_type=self.params['training']['loss_type'],
                    enable_continuity_reg=enable_continuity_reg,
                    reg_func=self.params['training']['reg_func'],
                    continuity_reg_strength=self.params['training']['continuity_reg_strength'],
                    noise_threshold_to_std=self.params['training']['noise_threshold_to_std'])

                # calculate gradient
                if enable_continuity_reg:
                    ((loss_dict['rec_loss'] + loss_dict['reg_loss']) / n_loop).backward()
                else:
                    (loss_dict['rec_loss'] / n_loop).backward()

                c_rec_loss_hist.append(loss_dict['rec_loss'].item())
                if enable_continuity_reg:
                    c_reg_loss_hist.append(loss_dict['reg_loss'].item())

            # stochastic update
            self.optim.step()
            self.sched.step()
            
            if (i_iter + 1) % self.params['training']['validate_every'] == 0:
                batch_data = generate_occluded_training_data(
                    ws_denoising_list=self.ws_denoising_list,
                    t_order=self.denoising_model.t_order,
                    t_tandem=0,
                    n_batch=self.params['training']['n_batch_validation'],
                    # TODO assumes all datasets are compatible with the same full window
                    x_window=self.ws_denoising_list[0].width,
                    y_window=self.ws_denoising_list[0].height,
                    occlusion_prob=1,
                    occlusion_radius=0,
                    occlusion_strategy='validation',
                    dataset_indices=val_dataset_indices,
                    frame_indices=val_frame_indices,
                    device=self.params['device'],
                    dtype=consts.DEFAULT_DTYPE)
                
                with torch.no_grad():
                    loss_dict = get_noise2self_loss(
                        batch_data=batch_data,
                        ws_denoising_list=self.ws_denoising_list,
                        denoising_model=self.denoising_model,
                        norm_p=self.params['training']['norm_p'],
                        loss_type=self.params['training']['loss_type'],
                        enable_continuity_reg=enable_continuity_reg,
                        reg_func=self.params['training']['reg_func'],
                        continuity_reg_strength=self.params['training']['continuity_reg_strength'],
                        noise_threshold_to_std=self.params['training']['noise_threshold_to_std'])
                
                if enable_continuity_reg:
                    last_val_loss = (loss_dict['rec_loss'] + loss_dict['reg_loss']).item()
                else:
                    last_val_loss = loss_dict['rec_loss'].item()
                
                # TODO record val loss somewhere

            if (i_iter + 1) % self.params['training']['log_every'] == 0:
                elapsed = time.time() - start
                update_time = True
                logging.info(
                    f'iter {i_iter + 1}/{n_iters} | '
                    f'val loss={last_val_loss:.1f} | '
                    f'{self.params["training"]["log_every"] / elapsed:.2f} iter/s')

            if (i_iter + 1) % self.params['training']['save_every'] == 0:
                index = (i_iter + 1) // self.params['training']['save_every']
                logging.info(f'Saving checkpoint at index {index}')
                save_model_state(
                    denoising_model=self.denoising_model,
                    model_dir=model_dir,
                    index=index,
                    optim=self.optim,
                    sched=self.sched,
                    save_train_state=True)


        # save trained model
        logging.info('Training complete; saving model...')
        save_model_state(
            denoising_model=self.denoising_model,
            model_dir=model_dir,
            index=0)
