import os

import matplotlib.pylab as plt
import numpy as np
from time import time
import torch
import logging
import json
import pprint
import pickle

from torch.optim.lr_scheduler import LambdaLR
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from typing import List
import yaml
import argparse

from cellmincer.opto_ws import OptopatchBaseWorkspace, OptopatchDenoisingWorkspace

from cellmincer.opto_features import OptopatchGlobalFeatureContainer

from cellmincer.opto_utils import \
    crop_center, \
    get_nn_spatio_temporal_mean, \
    get_nn_spatial_mean, \
    get_tagged_dir

from cellmincer.opto_denoise import \
    generate_occluded_training_data, \
    get_noise2self_loss

from cellmincer.models import \
    initialize_model, \
    get_minimum_padding

from cellmincer import consts


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
        

def train(
        config: dict,
        denoising_model,
        ws_denoising_list: List[OptopatchDenoisingWorkspace]):
    
    model_dir = get_tagged_dir(
        name=config['model']['type'],
        config_tag=config['tag'],
        root_dir=config['root_model_dir'])
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    # train model
    print('Training model...')
    if config['training']['seed'] is not None:
        torch.manual_seed(config['training']['seed'])
    denoising_model.train()

    optim = torch.optim.Adam(denoising_model.parameters(), lr=config['training']['lr_params']['max'], betas=(0.9, 0.999))
    scheduler = generate_lr_scheduler(optim, config['training']['lr_params'], )
    
    enable_continuity_reg = config['training']['enable_continuity_reg']

    rec_loss_hist = []
    reg_loss_hist = []
    
    n_iters = config['training']['n_iters']
    n_loop = config['training']['n_loop']
    
    for i_iter in range(n_iters):
        c_rec_loss_hist = []
        c_reg_loss_hist = []
        
        optim.zero_grad()

        # aggregate gradients
        for i_loop in range(n_loop):
            batch_data = generate_occluded_training_data(
                ws_denoising_list=ws_denoising_list,
                t_order=denoising_model.t_order,
                t_tandem=config['training']['t_tandem'],
                n_batch=config['training']['n_batch_per_loop'],
                x_window=config['training']['training_x_window'],
                y_window=config['training']['training_y_window'],
                occlusion_prob=config['training']['occlusion_prob'],
                occlusion_radius=config['training']['occlusion_radius'],
                occlusion_strategy=config['training']['occlusion_strategy'],
                device=config['device'],
                dtype=consts.DEFAULT_DTYPE)

            loss_dict = get_noise2self_loss(
                batch_data=batch_data,
                ws_denoising_list=ws_denoising_list,
                denoising_model=denoising_model,
                norm_p=config['training']['norm_p'],
                loss_type=config['training']['loss_type'],
                enable_continuity_reg=enable_continuity_reg,
                reg_func=config['training']['reg_func'],
                continuity_reg_strength=config['training']['continuity_reg_strength'],
                noise_threshold_to_std=config['training']['noise_threshold_to_std'])

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
                param.data *= (1 - config['training']['wd'] * group['lr'])

        # stochastic update
        optim.step()
        scheduler.step()
        
        if (i_iter + 1) % config['training']['log_every'] == 0:
            print(f'\titer {i_iter + 1}/{n_iters}')
            
        if (i_iter + 1) % config['training']['save_every'] == 0:
            index = (i_iter + 1) // config['training']['save_every']
            print(f'Saving checkpoint at index {index}')
            save_model_state(
                denoising_model=denoising_model,
                model_dir=model_dir,
                index=index,
                optim=optim,
                scheduler=scheduler,
                save_train_state=True)
    
    
    # save trained model
    print('Training complete; saving model...')
    save_model_state(
        denoising_model=denoising_model,
        model_dir=model_dir,
        index=0)
    
    return

def denoise(
        config: dict,
        denoising_model,
        ws_denoising_list: List[OptopatchDenoisingWorkspace]):
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
        
    return

def load_datasets(config: dict) -> List[OptopatchDenoisingWorkspace]:
    print('Loading datasets...')
    datasets = config['datasets']
    dataset_dirs = [get_tagged_dir(
            name=dataset,
            config_tag=config['tag'],
            root_dir=config['root_data_dir'])
        for dataset in datasets]
    
    assert all([os.path.exists(dataset_dir) for dataset_dir in dataset_dirs])
    
    x_padding, y_padding = get_minimum_padding(
        model_config=config['model'],
        training_x_window=config['training']['training_x_window'],
        training_y_window=config['training']['training_y_window'])
    
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
                device=config['device'],
                dtype=consts.DEFAULT_DTYPE
            )
        )
        
    return ws_denoising_list

def instance_model(
        config: dict,
        n_global_features: int,
        model_dir: str):
    
    config['model']['n_global_features'] = n_global_features
    
    if config['state_index'] is None:
        denoising_model = initialize_model(
            config['model'],
            device=config['device'],
            dtype=consts.DEFAULT_DTYPE)
    else:
        model_state_path = os.path.join(model_dir, f'model_state__{config["state_index"]:06d}.pt')
        denoising_model = initialize_model(
            config['model'],
            model_state_path=model_state_path,
            device=config['device'],
            dtype=consts.DEFAULT_DTYPE)
    
    return denoising_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model training and denoising.')
    parser.add_argument('--train', help='invokes training routine', action="store_true")
    parser.add_argument('--denoise', help='invokes denoising routine', action="store_true")
    parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()
    
    with open(args.configfile, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    if not(args.train or args.denoise):
        print('Neither training [--train] or denoising [--denoise] was selected...exiting.')
        exit(0)
    
    # load datasets
    ws_denoising_list = load_datasets(config)
    
    # instancing model
    model_dir = get_tagged_dir(
        name=config['model']['type'],
        config_tag=config['tag'],
        root_dir=config['root_model_dir'])
    denoising_model = instance_model(
        config=config,
        n_global_features=ws_denoising_list[0].n_global_features,
        model_dir=model_dir)
    
    if args.train:
        train(
            config=config,
            denoising_model=denoising_model,
            ws_denoising_list=ws_denoising_list)
    if args.denoise:
        denoise(
            config=config,
            denoising_model=denoising_model,
            ws_denoising_list=ws_denoising_list)
