import os
import logging
import pprint
import time

import json
import pickle

import matplotlib.pylab as plt
import numpy as np
import torch
import pandas as pd
from typing import List

from cellmincer.containers import Noise2Self
from cellmincer.models import DenoisingModel, get_best_window_padding
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
        
        self.x_window, x_padding = get_best_window_padding(
            model_config=self.params['model'],
            output_min_size_lo=self.params['training']['output_min_size_lo'],
            output_min_size_hi=self.params['training']['output_min_size_hi'])
        self.y_window, y_padding = self.x_window, x_padding
        
        n2s = Noise2Self(
            params=params,
            x_padding=x_padding,
            y_padding=y_padding)
        
        self.ws_denoising_list, self.denoising_model = n2s.get_resources()

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
        training_config = self.params['training']
        
        if 'seed' in training_config:
            torch.manual_seed(training_config['seed'])
        self.denoising_model.train()

        n_iters = training_config['n_iters']
        n_loop = training_config['n_loop']

        # initialize optimizer and scheduler if not loaded from save state
        if self.optim is None:
            self.optim = generate_optimizer(
                denoising_model=self.denoising_model,
                optim_params=training_config['optim_params'],
                lr=training_config['lr_params']['max'])
        if self.sched is None:
            self.sched = generate_lr_scheduler(
                optim=self.optim,
                lr_params=training_config['lr_params'],
                n_iters=n_iters)

        enable_continuity_reg = training_config['enable_continuity_reg']
        
        start_iter = (
            self.params['state_index'] * training_config['save_every']
            if 'state_index' in self.params else 0)
        
        # select validation frames and shape into batches
        assert training_config['n_frames_validation'] % training_config['n_batch_validation'] == 0
        
        val_dataset_indices, val_frame_indices = generate_batch_indices(
            self.ws_denoising_list,
            n_batch=training_config['n_frames_validation'],
            t_mid=self.denoising_model.t_order,
            dataset_selection='balanced')
        val_batch_shape = (
            training_config['n_frames_validation'] // training_config['n_batch_validation'],
            training_config['n_batch_validation'])
        val_dataset_indices = val_dataset_indices.reshape(val_batch_shape)
        val_frame_indices = val_frame_indices.reshape(val_batch_shape)
        
        last_val_loss = None

        total_loss_hist = []
        rec_loss_hist = []
        reg_loss_hist = []
        val_loss_hist = []

        update_time = True
        for i_iter in range(start_iter, n_iters):
            if update_time:
                start = time.time()
                update_time = False
            
            c_total_loss_hist = []
            c_rec_loss_hist = []
            c_reg_loss_hist = []

            self.optim.zero_grad()
            
            # aggregate gradients
            for i_loop in range(n_loop):
                batch_data = generate_occluded_training_data(
                    ws_denoising_list=self.ws_denoising_list,
                    t_order=self.denoising_model.t_order,
                    t_tandem=training_config['t_tandem'],
                    n_batch=training_config['n_batch_per_loop'],
                    x_window=self.x_window,
                    y_window=self.y_window,
                    occlusion_prob=training_config['occlusion_prob'],
                    occlusion_radius=training_config['occlusion_radius'],
                    occlusion_strategy=training_config['occlusion_strategy'],
                    device=self.params['device'],
                    dtype=consts.DEFAULT_DTYPE)

                loss_dict = get_noise2self_loss(
                    batch_data=batch_data,
                    ws_denoising_list=self.ws_denoising_list,
                    denoising_model=self.denoising_model,
                    norm_p=training_config['norm_p'],
                    loss_type=training_config['loss_type'],
                    enable_continuity_reg=enable_continuity_reg,
                    reg_func=training_config['reg_func'],
                    continuity_reg_strength=training_config['continuity_reg_strength'],
                    noise_threshold_to_std=training_config['noise_threshold_to_std'])

                # calculate gradient
                if enable_continuity_reg:
                    total_loss = (loss_dict['rec_loss'] + loss_dict['reg_loss']) / n_loop
                else:
                    total_loss = loss_dict['rec_loss'] / n_loop

                total_loss.backward()
                
                c_total_loss_hist.append(total_loss.item() * n_loop)
                c_rec_loss_hist.append(loss_dict['rec_loss'].item())
                if enable_continuity_reg:
                    c_reg_loss_hist.append(loss_dict['reg_loss'].item())

            # stochastic update
            self.optim.step()
            self.sched.step()
            
            total_loss_hist.append(np.mean(c_total_loss_hist))
            rec_loss_hist.append(np.mean(c_rec_loss_hist))
            if enable_continuity_reg:
                reg_loss_hist.append(np.mean(c_reg_loss_hist))
            
            if (i_iter + 1) % training_config['validate_every'] == 0:
                c_val_loss = []
                for val_dataset_batch, val_frame_batch in zip(val_dataset_indices, val_frame_indices):
                    batch_data = generate_occluded_training_data(
                        ws_denoising_list=self.ws_denoising_list,
                        t_order=self.denoising_model.t_order,
                        t_tandem=0,
                        n_batch=training_config['n_batch_validation'],
                        # TODO assumes all datasets are compatible with the same full window
                        x_window=self.ws_denoising_list[0].width,
                        y_window=self.ws_denoising_list[0].height,
                        occlusion_prob=1,
                        occlusion_radius=0,
                        occlusion_strategy='validation',
                        dataset_indices=val_dataset_batch,
                        frame_indices=val_frame_batch,
                        device=self.params['device'],
                        dtype=consts.DEFAULT_DTYPE)

                    with torch.no_grad():
                        loss_dict = get_noise2self_loss(
                            batch_data=batch_data,
                            ws_denoising_list=self.ws_denoising_list,
                            denoising_model=self.denoising_model,
                            norm_p=training_config['norm_p'],
                            loss_type=training_config['loss_type'],
                            enable_continuity_reg=enable_continuity_reg,
                            reg_func=training_config['reg_func'],
                            continuity_reg_strength=training_config['continuity_reg_strength'],
                            noise_threshold_to_std=training_config['noise_threshold_to_std'])
                
                    if enable_continuity_reg:
                        c_val_loss.append((loss_dict['rec_loss'] + loss_dict['reg_loss']).item())
                        
                    else:
                        c_val_loss.append(loss_dict['rec_loss'].item())

                last_val_loss = np.mean(c_val_loss)
                val_loss_hist.append(last_val_loss)

            if (i_iter + 1) % training_config['log_every'] == 0:
                elapsed = time.time() - start
                update_time = True
                logging.info(
                    f'iter {i_iter + 1}/{n_iters} | '
                    f'val loss={last_val_loss:.1f} | '
                    f'{training_config["log_every"] / elapsed:.2f} iter/s')

            if (i_iter + 1) % training_config['save_every'] == 0:
                index = (i_iter + 1) // training_config['save_every']
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

        train_loss_dict = {
            'iter': np.arange(1, n_iters + 1),
            'total_loss': total_loss_hist,
            'rec_loss': rec_loss_hist}
        if enable_continuity_reg:
            train_loss_dict['reg_loss'] = reg_loss_hist
        train_loss_df = pd.DataFrame(train_loss_dict)
        
        val_loss_df = pd.DataFrame({
            'iter': np.arange(training_config['validate_every'], n_iters + 1, training_config['validate_every']),
            'val_loss': val_loss_hist})
        
        train_loss_df.to_csv(os.path.join(model_dir, 'train_loss.csv'), index=False)
        val_loss_df.to_csv(os.path.join(model_dir, 'val_loss.csv'), index=False)
