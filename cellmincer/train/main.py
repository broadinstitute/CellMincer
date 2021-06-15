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

from torch.optim.lr_scheduler import LambdaLR
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from cellmincer import consts
from cellmincer.containers import Noise2Self
from cellmincer.models import DenoisingModel
from cellmincer.util import \
    OptopatchBaseWorkspace, \
    OptopatchDenoisingWorkspace, \
    crop_center, \
    get_nn_spatio_temporal_mean, \
    get_nn_spatial_mean, \
    get_tagged_dir, \
    generate_occluded_training_data, \
    get_noise2self_loss
    
class Train:
    def __init__(
            self,
            params: dict):
        self.params = params
        
        self.ws_denoising_list, self.denoising_model = Noise2Self(params).get_resources()

        if self.params["state_index"]:
            model_dir = get_tagged_dir(
                name=self.params['model']['type'],
                config_tag=self.params['model_tag'],
                root_dir=self.params['root_model_dir'])
            self.optim = torch.load(
                os.path.join(model_dir, f'{self.params["state_index"]:06d}', 'optim_state.pt'))
            self.sched = torch.load(
                os.path.join(model_dir, f'{self.params["state_index"]:06d}', 'sched_state.pt'))
        else:
            self.optim = None
            self.sched = None
        
    def generate_lr_scheduler(
            self,
            optim: torch.optim.Optimizer,
            lr_params: dict,
            n_iters: int):
        if lr_params['type'] == 'const':
            sched = LambdaLR(optim, lr_lambda=lambda it: 1)
        elif lr_params['type'] == 'cosine-annealing-warmup':
            sched = CosineAnnealingWarmupRestarts(
                optim=optim,
                first_cycle_steps=lr_params['cycle_len'],
                cycle_mult=1.0,
                max_lr=lr_params['max'],
                min_lr=0.001,
                warmup_steps=lr_params['warmup'],
                gamma=1.0)
        else:
            raise ValueError('Unrecognized learning rate type.')
        return sched

    def save_model_state(
            self,
            denoising_model: DenoisingModel,
            model_dir: str,
            index: int,
            optim = None,
            sched = None,
            save_train_state: bool = False):
        
        model_ckpt_dir = os.path.join(model_dir, f'{index:06d}')
        if not os.path.exists(model_ckpt_dir):
            os.mkdir(model_ckpt_dir)

        torch.save(
            self.denoising_model.state_dict(),
            os.path.join(model_ckpt_dir, 'model_state.pt'))
        if save_train_state:
            torch.save(
                self.optim.state_dict(),
                os.path.join(model_ckpt_dir, 'optim_state.pt'))
            torch.save(
                self.sched.state_dict(),
                os.path.join(model_ckpt_dir, 'sched_state.pt'))
        
    def run(self):
        model_dir = get_tagged_dir(
            name=self.params['model']['type'],
            config_tag=self.params['model_tag'],
            root_dir=self.params['root_model_dir'])
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        # train model
        logging.info('Training model...')
        x_window, y_window = self.denoising_model.get_best_input_size(
            self.params['training']['output_min_size_lo'],
            self.params['training']['output_min_size_hi'])
        
        if self.params['training']['seed'] is not None:
            torch.manual_seed(self.params['training']['seed'])
        self.denoising_model.train()

        n_iters = self.params['training']['n_iters']
        n_loop = self.params['training']['n_loop']

        if self.optim is None:
            # TODO: change to SGD for annealed cosine warmup
            self.optim = \
                torch.optim.SGD(self.denoising_model.parameters(), lr=self.params['training']['lr_params']['max'], momentum=0.9)
#                 torch.optim.Adam(self.denoising_model.parameters(), lr=self.params['training']['lr_params']['max'], betas=(0.9, 0.999))
        if self.sched is None:
            self.sched = self.generate_lr_scheduler(self.optim, self.params['training']['lr_params'], n_iters)

        enable_continuity_reg = self.params['training']['enable_continuity_reg']
        start_iter = (
            self.params["state_index"] * self.params['training']['save_every']
            if self.params["state_index"] else 0)

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

            # weight decay
            for group in self.optim.param_groups:
                for param in group['params']:
                    param.data *= (1 - self.params['training']['weight_decay'] * group['lr'])

            # stochastic update
            self.optim.step()
            self.sched.step()

            if (i_iter + 1) % self.params['training']['log_every'] == 0:
                elapsed = time.time() - start
                update_time = True
                logging.info(f'iter {i_iter + 1}/{n_iters} -- {elapsed / self.params["training"]["log_every"]:.2f} s/iter')

            if (i_iter + 1) % self.params['training']['save_every'] == 0:
                index = (i_iter + 1) // self.params['training']['save_every']
                logging.info(f'Saving checkpoint at index {index}')
                self.save_model_state(
                    denoising_model=self.denoising_model,
                    model_dir=model_dir,
                    index=index,
                    optim=optim,
                    sched=sched,
                    save_train_state=True)


        # save trained model
        logging.info('Training complete; saving model...')
        self.save_model_state(
            denoising_model=self.denoising_model,
            model_dir=model_dir,
            index=0)
