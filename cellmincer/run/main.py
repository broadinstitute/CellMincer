import os
import yaml
import logging
import pself.log_info
import time
import math

import numpy as np
import torch
from typing import Dict, Any, Callable, Optional

from cellmincer import consts

from mighty_codes.ptpsa import \
    BatchedStateManipulator, \
    BatchedStateEnergyCalculator, \
    MCMCAcceptanceRatePredictor, \
    PyTorchBatchedSimulatedAnnealing, \
    SimulatedAnnealingExitCode

from mighty_codes.torch_utils import \
    to_np, \
    to_torch, \
    to_one_hot_encoded

from mighty_codes.nn_utils import \
    generate_dense_nnet

from mighty_codes.channel_utils import \
    calculate_bac_standard_metric_dict, \
    calculate_bac_f1_reject_auc_metric_dict

from mighty_codes.experiments import \
    ChannelModelSpecification, \
    ExperimentSpecification

from mighty_codes.channels import \
    BinaryChannelSpecification, \
    BinaryAsymmetricChannelModel

from mighty_codes.ptpsa import \
    PyTorchBatchedSimulatedAnnealing, \
    SimulatedAnnealingExitCode, \
    estimate_energy_scale

from mighty_codes.plot_utils import \
    plot_sa_trajectory, \
    plot_sa_codebook, \
    plot_sa_resampling_buffer

log_info = logging.info

class Setup:
    def __init__(self, params, logger = None):
        self.params = params
        
        if logger is None:
            logger = log_info
        self.log_info = logger
        
    def load_datasets(self) -> List[OptopatchDenoisingWorkspace]:
        self.log_info('Loading datasets...')
        datasets = self.params['datasets']
        dataset_dirs = [get_tagged_dir(
                name=dataset,
                config_tag=self.params['tag'],
                root_dir=self.params['root_data_dir'])
            for dataset in datasets]

        assert all([os.path.exists(dataset_dir) for dataset_dir in dataset_dirs])

        x_padding, y_padding = get_minimum_padding(
            model_config=self.params['model'],
            training_x_window=self.params['training']['training_x_window'],
            training_y_window=self.params['training']['training_y_window'])

        ws_denoising_list = []
        for i_dataset in range(len(datasets)):
            base_diff_path = os.path.join(dataset_dirs[i_dataset], 'trend_subtracted.npy')
            ws_base_diff = OptopatchBaseWorkspace.from_npy(base_diff_path)

            base_bg_path = os.path.join(dataset_dirs[i_dataset], 'trend.npy')
            ws_base_bg = OptopatchBaseWorkspace.from_npy(base_bg_path)

            opto_noise_params_path = os.path.join(dataset_dirs[i_dataset], 'noise_params.json')
            with open(opto_noise_params_path, 'r') as f:
                noise_params = json.load(f)

            opto_feature_path = os.path.join(dataset_dirs[i_dataset], 'features.pkl')
            with open(opto_feature_path, 'rb') as f:
                feature_container = pickle.Unpickler(f).load()

            ws_denoising_list.append(
                OptopatchDenoisingWorkspace(
                    ws_base_diff=ws_base_diff,
                    ws_base_bg=ws_base_bg,
                    noise_params=noise_params,
                    features=feature_container,
                    x_padding=x_padding,
                    y_padding=y_padding,
                    device=self.params['device'],
                    dtype=consts.DEFAULT_DTYPE
                )
            )

        return ws_denoising_list
    
    def instance_model(
        self,
        n_global_features: int):
    
        self.params['model']['n_global_features'] = n_global_features
        model_dir = get_tagged_dir(
            name=self.params['model']['type'],
            config_tag=self.params['tag'],
            root_dir=self.params['root_model_dir'])

        if self.params['state_index'] is None:
            denoising_model = initialize_model(
                self.params['model'],
                device=self.params['device'],
                dtype=consts.DEFAULT_DTYPE)
        else:
            model_state_path = os.path.join(model_dir, f'model_state__{self.params['state_index']:06d}.pt')
            denoising_model = initialize_model(
                self.params['model'],
                model_state_path=model_state_path,
                device=self.params['device'],
                dtype=consts.DEFAULT_DTYPE)
    
    return denoising_model

    def run(self):
        ws_denoising_list = load_datasets(config)

        denoising_model = instance_model(
            config=config,
            n_global_features=ws_denoising_list[0].n_global_features)
        
        return ws_denoising_list, denoising_model
    
class Train:
    def __init__(self, params, ws_denoising_list, denoising_model, logger = None):
        self.params = params
        self.ws_denoising_list = ws_denoising_list
        self.denoising_model = denoising_model
        
        if logger is None:
            logger = log_info
        self.log_info = logger
        
    def generate_lr_scheduler(optim, lr_params, n_iters):
        if lr_params['type'] == 'const':
            scheduler = LambdaLR(optim, lr_lambda=lambda it: 1)
        elif lr_params['type'] == 'cosine-annealing-warmup':
            scheduler = CosineAnnealingWarmupRestarts(
                optim=optim,
                first_cycle_steps=lr_params['cycle_len'],
                cycle_mult=1.0,
                max_lr=lr_params['max'],
                min_lr=0.001,
                warmup_steps=lr_params['warmup'],
                gamma=1.0)
        else:
            raise ValueError('Unrecognized learning rate type.')
        return scheduler

    def save_model_state(
            denoising_model,
            model_dir: str,
            index: int,
            optim = None,
            scheduler = None,
            save_train_state: bool = False):

        torch.save(
            denoising_model.state_dict(),
            os.path.join(model_dir, f'model_state__{index:06d}.pt'))
        if save_train_state:
            torch.save(
                optim.state_dict(),
                os.path.join(model_dir, f'optim_state__{full_model_prefix}.pt'))
            torch.save(
                scheduler.state_dict(),
                os.path.join(model_dir, f'sched_state__{full_model_prefix}.pt'))
            torch.save(
                torch.get_rng_state(),
                os.path.join(model_dir, f'rng_state__{full_model_prefix}.pt'))
        
    def run(self):
        model_dir = get_tagged_dir(
            name=self.params['model']['type'],
            config_tag=self.params['tag'],
            root_dir=self.params['root_model_dir'])
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        # train model
        self.log_info('Training model...')
        if self.params['training']['seed'] is not None:
            torch.manual_seed(self.params['training']['seed'])
        denoising_model.train()

        optim = torch.optim.Adam(denoising_model.parameters(), lr=self.params['training']['lr_params']['max'], betas=(0.9, 0.999))
        scheduler = self.generate_lr_scheduler(optim, self.params['training']['lr_params'], )

        enable_continuity_reg = self.params['training']['enable_continuity_reg']

        rec_loss_hist = []
        reg_loss_hist = []

        n_iters = self.params['training']['n_iters']
        n_loop = self.params['training']['n_loop']

        for i_iter in range(n_iters):
            c_rec_loss_hist = []
            c_reg_loss_hist = []

            optim.zero_grad()

            # aggregate gradients
            for i_loop in range(n_loop):
                batch_data = generate_occluded_training_data(
                    ws_denoising_list=ws_denoising_list,
                    t_order=denoising_model.t_order,
                    t_tandem=self.params['training']['t_tandem'],
                    n_batch=self.params['training']['n_batch_per_loop'],
                    x_window=self.params['training']['training_x_window'],
                    y_window=self.params['training']['training_y_window'],
                    occlusion_prob=self.params['training']['occlusion_prob'],
                    occlusion_radius=self.params['training']['occlusion_radius'],
                    occlusion_strategy=self.params['training']['occlusion_strategy'],
                    device=self.params['device'],
                    dtype=consts.DEFAULT_DTYPE)

                loss_dict = get_noise2self_loss(
                    batch_data=batch_data,
                    ws_denoising_list=ws_denoising_list,
                    denoising_model=denoising_model,
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
            for group in optim.param_groups:
                for param in group['params']:
                    # TODO check that this still works
                    param.data *= (1 - self.params['training']['wd'] * group['lr'])

            # stochastic update
            optim.step()
            scheduler.step()

            if (i_iter + 1) % self.params['training']['log_every'] == 0:
                self.log_info(f'\titer {i_iter + 1}/{n_iters}')

            if (i_iter + 1) % self.params['training']['save_every'] == 0:
                index = (i_iter + 1) // self.params['training']['save_every']
                self.log_info(f'Saving checkpoint at index {index}')
                save_model_state(
                    denoising_model=denoising_model,
                    model_dir=model_dir,
                    index=index,
                    optim=optim,
                    scheduler=scheduler,
                    save_train_state=True)


        # save trained model
        self.log_info('Training complete; saving model...')
        self.save_model_state(
            denoising_model=denoising_model,
            model_dir=model_dir,
            index=0)
        
class Denoise:
    def __init__(self, params, ws_denoising_list, denoising_model, logger = None):
        self.params = params
        self.ws_denoising_list = ws_denoising_list
        self.denoising_model = denoising_model
        
        if logger is None:
            logger = log_info
        self.log_info = logger
    
    def run(self):
        denoise_dir = get_tagged_dir(
            name=config['model']['type'],
            config_tag=config['tag'],
            root_dir=config['root_denoise_dir'])

        if not os.path.exists(denoise_dir):
            os.mkdir(denoise_dir)

        print('Denoising movies...')
        for i_dataset, (name, ws_denoising) in enumerate(zip(config['datasets'], ws_denoising_list)):
            print(f'\t({i_dataset + 1}/{len(ws_denoising_list)}) {name}')
            denoised_movie_txy = denoising_model.denoise_movie(ws_denoising).numpy()

            denoised_movie_txy *= ws_denoising.cached_features.norm_scale
            denoised_movie_txy += ws_denoising.ws_base_bg.movie_txy

            np.save(
                os.path.join(denoise_dir, f'{name}__denoised_tyx.npy'),
                denoised_movie_txy.transpose((0, 2, 1)))
