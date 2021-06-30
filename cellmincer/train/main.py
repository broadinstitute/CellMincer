import os
import logging
import pprint
import time

import json
import pickle
import tarfile

import matplotlib.pylab as plt
import numpy as np
import torch
import pandas as pd
from typing import List, Optional, Tuple

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
    get_noise2self_loss
    
class Train:
    def __init__(
            self,
            params: dict):
        
        self.x_window, x_padding = get_best_window_padding(
            model_config=params['model'],
            output_min_size_lo=params['training']['output_min_size_lo'],
            output_min_size_hi=params['training']['output_min_size_hi'])
        self.y_window, y_padding = self.x_window, x_padding
        
        n2s = Noise2Self(
            params=params,
            x_padding=x_padding,
            y_padding=y_padding)
        
        self.ws_denoising_list, self.denoising_model = n2s.get_resources()
        
        self.model_dir = os.path.join(params['root_model_dir'], params['name'])
        self.training_config = params['training']
        self.start_iter = 0
        self.device = torch.device(params['device'])
        
        self.enable_continuity_reg = self.training_config['enable_continuity_reg']
        
        # initialize optimizer and scheduler
        self.optim = generate_optimizer(
                denoising_model=self.denoising_model,
                optim_params=self.training_config['optim_params'],
                lr=self.training_config['lr_params']['max'])
        self.sched = generate_lr_scheduler(
                optim=self.optim,
                lr_params=self.training_config['lr_params'],
                n_iters=self.training_config['n_iters'])
        
        self.train_loss_names = ['iter', 'total_loss', 'rec_loss', 'reg_loss'] \
            if self.enable_continuity_reg \
            else ['iter', 'total_loss', 'rec_loss']
        self.val_loss_names = ['iter', 'val_loss']
        self.train_loss_hist = []
        self.val_loss_hist = []
        
        # if continuing from checkpoint
        if os.path.exists(self.training_config['checkpoint_path']):
            logging.info('Checkpoint found; extracting...')
            
            with tarfile.open(self.training_config['checkpoint_path']) as tar:
                tar.extractall(self.model_dir)
            
            with open(os.path.join(self.model_dir, 'latest/ckpt_index.txt'), 'r') as f:
                self.start_iter = int(f.read()) * self.training_config['train_checkpoint_every']

            logging.info(f'Restarting training from iteration {self.start_iter}.')
            
            # load training state
            self.denoising_model.load_state_dict(torch.load(
                os.path.join(self.model_dir, 'latest/model_state.pt')))
            self.optim.load_state_dict(torch.load(
                os.path.join(self.model_dir, 'latest/optim_state.pt')))
            self.sched.load_state_dict(torch.load(
                os.path.join(self.model_dir, 'latest/sched_state.pt')))

            # load previous loss history
            train_loss_df = pd.read_csv(os.path.join(self.model_dir, 'latest/train_loss.csv'))
            self.train_loss_hist = list(tuple(x) for x in train_loss_df[self.train_loss_names].to_records(index=False))
            val_loss_df = pd.read_csv(os.path.join(self.model_dir, 'latest/val_loss.csv'))
            self.val_loss_hist = list(tuple(x) for x in val_loss_df[self.val_loss_names].to_records(index=False))
        
        else:
            if not os.path.exists(os.path.join(self.model_dir, 'states')):
                os.mkdir(os.path.join(self.model_dir, 'states'))
            if not os.path.exists(os.path.join(self.model_dir, 'latest')):
                os.mkdir(os.path.join(self.model_dir, 'latest'))
            

    def save_model_state(self, index: int):
        torch.save(
            self.denoising_model.state_dict(),
            os.path.join(self.model_dir, f'states/model_state__{index:06d}.pt'))
        
    def save_checkpoint(self, index: int):
        # update latest
        with open(os.path.join(self.model_dir, 'latest/ckpt_index.txt'), 'w') as f:
            f.write(str(index))
            
        torch.save(
            self.denoising_model.state_dict(),
            os.path.join(self.model_dir, 'latest/model_state.pt'))
        torch.save(
            self.optim.state_dict(),
            os.path.join(self.model_dir, 'latest/optim_state.pt'))
        torch.save(
            self.sched.state_dict(),
            os.path.join(self.model_dir, 'latest/sched_state.pt'))
        
        self.save_loss(os.path.join(self.model_dir, 'latest'))
        
        # tarball states, latest
        checkpoint_path_tmp = os.path.join(self.model_dir, 'checkpoint_tmp.tar.gz')
        checkpoint_path = os.path.join(self.model_dir, 'checkpoint.tar.gz')
        with tarfile.open(checkpoint_path_tmp, 'w:gz') as tar:
            tar.add(os.path.join(self.model_dir, 'states'), arcname='states')
            tar.add(os.path.join(self.model_dir, 'latest'), arcname='latest')
        os.replace(checkpoint_path_tmp, checkpoint_path)
            

    def save_final(self):
        if not os.path.exists(os.path.join(self.model_dir, 'final')):
            os.mkdir(os.path.join(self.model_dir, 'final'))
        torch.save(
            self.denoising_model.state_dict(),
            os.path.join(self.model_dir, f'final/model_state.pt'))
        self.save_loss(os.path.join(self.model_dir, 'final'))
        
    def save_loss(self, path: str):
        train_loss_df = pd.DataFrame(self.train_loss_hist, columns=self.train_loss_names)
        train_loss_df.to_csv(os.path.join(path, 'train_loss.csv'), index=False)
        val_loss_df = pd.DataFrame(self.val_loss_hist, columns=self.val_loss_names)
        val_loss_df.to_csv(os.path.join(path, 'val_loss.csv'), index=False)
        

    def run(self):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        logging.info('Training model...')
        
        # select validation frames and shape into batches
        assert self.training_config['n_frames_validation'] % self.training_config['n_batch_validation'] == 0
        
        val_dataset_indices, val_frame_indices = generate_batch_indices(
            self.ws_denoising_list,
            n_batch=self.training_config['n_frames_validation'],
            t_mid=self.denoising_model.t_order,
            dataset_selection='balanced')
        val_batch_shape = (
            self.training_config['n_frames_validation'] // self.training_config['n_batch_validation'],
            self.training_config['n_batch_validation'])
        val_dataset_indices = val_dataset_indices.reshape(val_batch_shape)
        val_frame_indices = val_frame_indices.reshape(val_batch_shape)

        update_time = True
        for i_iter in range(self.start_iter, self.training_config['n_iters']):
            if update_time:
                start = time.time()
                update_time = False
            
            c_total_loss_hist = []
            c_rec_loss_hist = []
            c_reg_loss_hist = []

            self.denoising_model.train()
            self.optim.zero_grad()
            
            # aggregate gradients
            for i_loop in range(self.training_config['n_loop']):
                batch_data = generate_occluded_training_data(
                    ws_denoising_list=self.ws_denoising_list,
                    t_order=self.denoising_model.t_order,
                    t_tandem=self.training_config['t_tandem'],
                    n_batch=self.training_config['n_batch_per_loop'],
                    x_window=self.x_window,
                    y_window=self.y_window,
                    occlusion_prob=self.training_config['occlusion_prob'],
                    occlusion_radius=self.training_config['occlusion_radius'],
                    occlusion_strategy=self.training_config['occlusion_strategy'],
                    device=self.device,
                    dtype=consts.DEFAULT_DTYPE)

                loss_dict = get_noise2self_loss(
                    batch_data=batch_data,
                    ws_denoising_list=self.ws_denoising_list,
                    denoising_model=self.denoising_model,
                    norm_p=self.training_config['norm_p'],
                    loss_type=self.training_config['loss_type'],
                    enable_continuity_reg=self.enable_continuity_reg,
                    reg_func=self.training_config['reg_func'],
                    continuity_reg_strength=self.training_config['continuity_reg_strength'],
                    noise_threshold_to_std=self.training_config['noise_threshold_to_std'])

                # calculate gradient
                if self.enable_continuity_reg:
                    total_loss = (loss_dict['rec_loss'] + loss_dict['reg_loss']) / self.training_config['n_loop']
                else:
                    total_loss = loss_dict['rec_loss'] / self.training_config['n_loop']

                total_loss.backward()
                
                c_total_loss_hist.append(total_loss.item() * self.training_config['n_loop'])
                c_rec_loss_hist.append(loss_dict['rec_loss'].item())
                if self.enable_continuity_reg:
                    c_reg_loss_hist.append(loss_dict['reg_loss'].item())

            # stochastic update
            self.optim.step()
            self.sched.step()
            
            train_loss = (i_iter + 1, np.mean(c_total_loss_hist), np.mean(c_rec_loss_hist))
            if self.enable_continuity_reg:
                train_loss += (np.mean(c_reg_loss_hist),)
            self.train_loss_hist.append(train_loss)
            
            if (i_iter + 1) % self.training_config['validate_every'] == 0:
                self.denoising_model.eval()
                
                c_val_loss = []
                for val_dataset_batch, val_frame_batch in zip(val_dataset_indices, val_frame_indices):
                    batch_data = generate_occluded_training_data(
                        ws_denoising_list=self.ws_denoising_list,
                        t_order=self.denoising_model.t_order,
                        t_tandem=0,
                        n_batch=self.training_config['n_batch_validation'],
                        # TODO assumes all datasets are compatible with the same full window
                        x_window=self.ws_denoising_list[0].width,
                        y_window=self.ws_denoising_list[0].height,
                        occlusion_prob=1,
                        occlusion_radius=0,
                        occlusion_strategy='validation',
                        dataset_indices=val_dataset_batch,
                        frame_indices=val_frame_batch,
                        device=self.device,
                        dtype=consts.DEFAULT_DTYPE)

                    with torch.no_grad():
                        loss_dict = get_noise2self_loss(
                            batch_data=batch_data,
                            ws_denoising_list=self.ws_denoising_list,
                            denoising_model=self.denoising_model,
                            norm_p=self.training_config['norm_p'],
                            loss_type=self.training_config['loss_type'],
                            enable_continuity_reg=self.enable_continuity_reg,
                            reg_func=self.training_config['reg_func'],
                            continuity_reg_strength=self.training_config['continuity_reg_strength'],
                            noise_threshold_to_std=self.training_config['noise_threshold_to_std'])
                
                    if self.enable_continuity_reg:
                        c_val_loss.append((loss_dict['rec_loss'] + loss_dict['reg_loss']).item())
                        
                    else:
                        c_val_loss.append(loss_dict['rec_loss'].item())

                self.val_loss_hist.append((i_iter + 1, np.mean(c_val_loss)))

            if (i_iter + 1) % self.training_config['log_every'] == 0:
                elapsed = time.time() - start
                update_time = True
                val_loss_str = f'{self.val_loss_hist[-1][1]:.4f}' if self.val_loss_hist else 'n/a'
                logging.info(
                    f'iter {i_iter + 1}/{self.training_config["n_iters"]} | '
                    f'train loss={self.train_loss_hist[-1][1]:.4f} | '
                    f'val loss={val_loss_str} | '
                    f'{self.training_config["log_every"] / elapsed:.2f} iter/s')

            if (i_iter + 1) % self.training_config['model_save_every'] == 0:
                index = (i_iter + 1) // self.training_config['model_save_every']
                logging.info(f'Saving model state at index {index}')
                self.save_model_state(index)

            if (i_iter + 1) % self.training_config['train_checkpoint_every'] == 0:
                index = (i_iter + 1) // self.training_config['train_checkpoint_every']
                logging.info(f'Updating and gzipping checkpoint at index {index}')
                self.save_checkpoint(index)


        # save trained model
        logging.info('Training complete; saving model...')
        self.save_final()
