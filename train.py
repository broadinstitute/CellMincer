import os

import matplotlib.pylab as plt
import numpy as np
from time import time
import torch
import logging
import json
import pprint
import pickle

from typing import List
import yaml
import argparse

from cellmincer.opto_ws import OptopatchBaseWorkspace, OptopatchDenoisingWorkspace

from cellmincer.opto_features import OptopatchGlobalFeatureContainer

from cellmincer.opto_utils import \
    crop_center, \
    get_nn_spatio_temporal_mean, \
    get_nn_spatial_mean

from cellmincer.opto_denoise import \
    get_minimum_spatial_padding, \
    generate_occluded_training_data

dtype = torch.float32


def train(config_file):
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    device = torch.device(config['device'])
    
    
    # create model
    denoising_model = initialize_model(
        config['model'],
        device=device,
        dtype=dtype)
    
    
    # load datasets
    dataset_names = config['datasets']['dataset_names']
    # TODO: directory of processed input should come directly from name
    dataset_paths = config['datasets']['dataset_paths']
    
    assert len(dataset_names) == len(dataset_paths)
    assert all([os.path.exists(dataset_path) for dataset_path in dataset_paths])
    
    n_datasets = len(dataset_names)
    
    ws_denoising_list = []
    for i_dataset in range(n_datasets):
        base_diff_path = os.path.join(dataset_paths[i_dataset], 'trend_subtracted.npy')
        ws_base_diff = OptopatchBaseWorkspace.from_npy(base_diff_path)
        
        base_bg_path = os.path.join(dataset_paths[i_dataset], 'trend.npy')
        ws_base_bg = OptopatchBaseWorkspace.from_npy(base_bg_path)
        
        opto_noise_params_path = os.path.join(dataset_paths[i_dataset], 'noise_params.json')
        with open(opto_noise_params_path, 'r') as f:
            noise_params = json.load(f)
            
        opto_feature_path = os.path.join(dataset_paths[i_dataset], 'features.pkl')
        with open(opto_feature_path, 'rb') as f:
            feature_container = pickle.Unpickler(f).load()
        
        ws_denoising_list.append(
            OptopatchDenoisingWorkspace(
                ws_base_diff=ws_base_diff,
                ws_base_bg=ws_base_bg,
                noise_params=noise_params,
                features=feature_container,
                x_padding=x_padding, # TODO: use opto_denoise.get_minimum_spatial_padding() to compute this
                y_padding=y_padding,
                device=device,
                dtype=dtype
            )
        )
        
        
    # train model
    if 'seed' in config['training']:
        torch.manual_seed(config['training']['seed'])
    denoising_model.train()

    optim = torch.optim.Adam(denoising_model.params(), lr=config['training']['lr'], betas=(0.9, 0.999))
    
    total_loss_hist = []
    unet_rec_loss_hist = []
    unet_reg_loss_hist = []
    temporal_rec_loss_hist = []
    temporal_reg_loss_hist = []
    
    for i_iter in config['training']['n_iters']:
        c_total_loss_hist = []
        c_unet_rec_loss_hist = []
        c_unet_reg_loss_hist = []
        c_temporal_rec_loss_hist = []
        c_temporal_reg_loss_hist = []
        
        optim.zero_grad()

        # aggregate gradients
        for i_loop in range(n_loop):

            batch_data = generate_occluded_training_data(
                ws_denoising_list=ws_denoising_list,
                t_order=denoiser_model.t_order,
                t_tandem=config['training']['t_tandem'],
                n_batch=config['training']['n_batch_per_loop'],
                x_window=config['training']['training_x_window'],
                y_window=training_y_window,
                occlusion_prob=config['training']['occlusion_prob'],
                occlusion_radius=config['training']['occlusion_radius'],
                occlusion_strategy=config['training']['occlusion_strategy'],
                device=device,
                dtype=dtype)

            loss_dict = denoising_model.get_noise2self_loss(
                batch_data=batch_data,
                ws_denoising_list=ws_denoising_list,
                norm_p=config['training']['norm_p'],
                loss_type=config['training']['loss_type'],
                enable_continuity_reg=config['training']['enable_continuity_reg'],
                reg_func=config['training']['reg_func'],
                continuity_reg_strength=config['training']['continuity_reg_strength'],
                noise_threshold_to_std=config['training']['noise_threshold_to_std'])

            # calculate gradient
            (loss_dict['total_loss'] / n_loop).backward()

            # history
            c_total_loss_hist.append(loss_dict['total_loss'].item())
            c_unet_rec_loss_hist.append(loss_dict['unet_endpoint_rec_loss'].item())
            c_temporal_rec_loss_hist.append(loss_dict['temporal_endpoint_rec_loss'].item())
            if config['training']['enable_continuity_reg']:
                c_unet_reg_loss_hist.append(loss_dict['unet_endpoint_reg_loss'].item())
                c_temporal_reg_loss_hist.append(loss_dict['temporal_endpoint_reg_loss'].item())

        # weight decay
        for group in optim.param_groups:
            for param in group['params']:
                param.data = param.data.add(-wd * group['lr'], param.data)        

        # stochastic update
        optim.step()
    
    
    # save trained model
    pass


def get_noise2self_loss(
        self,
        batch_data,
        ws_denoising_list: List[OptopatchDenoisingWorkspace],
        denoising_model: DenoisingModel,
        loss_type: str,
        norm_p: int,
        enable_continuity_reg: bool,
        reg_func: str,
        continuity_reg_strength: float,
        noise_threshold_to_std: float):
    """Calculates the loss of a Noise2Self predictor on a given minibatch."""

    assert reg_func in {'clamped_linear', 'tanh'}
    assert loss_type in {'lp', 'poisson_gaussian'}

    device = batch_data['padded_sliced_diff_movie_ntxy'].device
    dtype = batch_data['padded_sliced_diff_movie_ntxy'].dtype
    x_window = batch_data['x_window']
    y_window = batch_data['y_window']
    padded_x_window = batch_data['padded_x_window']
    padded_y_window = batch_data['padded_y_window']
    n_batch, t_total = batch_data['padded_sliced_diff_movie_ntxy'].shape[:2]
    n_global_features = batch_data['padded_global_features_nfxy'].shape[-3]
    t_tandem = batch_data['padded_occlusion_masks_ntxy'].shape[-3] - 1
    t_mid = (t_total - 1) // 2
    total_pixels = x_window * y_window
    
    assert self.t_order == t_total - t_tandem

    # iterate over the middle frames and accumulate loss
    def add_lp_to_loss(_loss, _err, _norm_p=norm_p, _scale=1.):
        _new_loss = (_scale * ((_err.abs() + eps).pow(_norm_p))).sum()
        if _loss is None:
            return _new_loss
        else:
            return _loss + _new_loss

    def add_factor_to_loss(_loss, _factor):
        if _loss is None:
            return _factor
        else:
            return _loss + _factor

    # fetch and crop the dataset std (for regularization)
    if enable_continuity_reg:
        cropped_movie_t_std_nxy = crop_center(
            batch_data['padded_global_features_nfxy'][:, batch_data['detrended_std_feature_index'], ...],
            target_width=x_window,
            target_height=y_window)

    unet_endpoint_rec_loss = None
    unet_endpoint_reg_loss = None
    temporal_endpoint_rec_loss = None
    temporal_endpoint_reg_loss = None
    prev_cropped_unet_endpoint_nxy = None
    prev_cropped_temporal_endpoint_nxy = None

    # calculate processed features
#     unet_output_list = [
#         self.spatial_unet(
#             batch_data['padded_sliced_diff_movie_ntxy'][:, i_t:i_t+1, :, :],
#             batch_data['padded_global_features_nfxy'])
#         for i_t in range(t_total)]
#     unet_features_nctxy = torch.stack([output['features_ncxy'] for output in unet_output_list], dim=-3)
#     unet_readout_n1xy_list = [output['readout_ncxy'] for output in unet_output_list]
#     unet_features_width = unet_features_nctxy.shape[-2]
#     unet_features_height = unet_features_nctxy.shape[-1]
#     unet_cropped_global_features_nfxy = crop_center(
#         batch_data['padded_global_features_nfxy'],
#         target_width=unet_features_width,
#         target_height=unet_features_height)

    temporal_endpoint_txy = denoising_model(
        
    )

    # calculate the loss on occluded points of the middle frames
    # and total variation loss between frames (if enabled)
    for i_t in range(t_tandem + 1):  # i_t denotes the index of the middle frames, starting from 0

        # unet readout
#         cropped_unet_endpoint_nxy = crop_center(
#             unet_readout_n1xy_list[(t_order - 1) // 2 + i_t][:, 0, :, :],
#             target_width=x_window,
#             target_height=y_window)

#         # get the temporal denoiser output
#         cropped_temporal_endpoint_nxy = crop_center(
#             self.temporal_denoiser(
#                 unet_features_nctxy[:, :, i_t:(i_t + t_order), :, :],
#                 unet_cropped_global_features_nfxy),
#             target_width=x_window,
#             target_height=y_window)

        # crop the occlusion mask
        cropped_mask_nxy = crop_center(
            batch_data['padded_occlusion_masks_ntxy'][:, i_t, ...],
            target_width=x_window,
            target_height=y_window)

        # crop expected output
        expected_output_nxy = crop_center(
            batch_data['padded_middle_frames_ntxy'][:, i_t, ...],
            target_width=x_window,
            target_height=y_window)

        # reconstruction losses
        total_masked_pixels = cropped_mask_nxy.sum().type(dtype)
        loss_scale = 1. / ((t_tandem + 1) * (eps + total_masked_pixels))

        if loss_type == 'poisson_gaussian':

            var_nxy = torch.cat([
                ws_denoising_list[i_dataset].get_modeled_variance(
                    scaled_bg_movie_ntxy=crop_center(
                        batch_data['padded_sliced_bg_movie_ntxy'][i_dataset, t_mid + i_t, :, :],
                        target_width=x_window,
                        target_height=y_window)[None, None, :, :],
                    scaled_diff_movie_ntxy=cropped_unet_endpoint_nxy[i_dataset, :, :][None, None, :, :])
                for i_dataset in batch_data['dataset_indices']], dim=0)[:, 0, :, :]

            c_unet_endpoint_rec_loss = get_poisson_gaussian_nll(
                var_nxy=var_nxy,
                pred_nxy=cropped_unet_endpoint_nxy,
                obs_nxy=expected_output_nxy,
                mask_nxy=cropped_mask_nxy).sum()
            unet_endpoint_rec_loss = add_factor_to_loss(
                unet_endpoint_rec_loss, loss_scale * c_unet_endpoint_rec_loss)

            var_nxy = torch.cat([
                ws_denoising_list[i_dataset].get_modeled_variance(
                    scaled_bg_movie_ntxy=crop_center(
                        batch_data['padded_sliced_bg_movie_ntxy'][i_dataset, t_mid + i_t, :, :],
                        target_width=x_window,
                        target_height=y_window)[None, None, :, :],
                    scaled_diff_movie_ntxy=cropped_temporal_endpoint_nxy[i_dataset, :, :][None, None, :, :])
                for i_dataset in batch_data['dataset_indices']], dim=0)[:, 0, :, :]

            c_temporal_endpoint_rec_loss = get_poisson_gaussian_nll(
                var_nxy=var_nxy,
                pred_nxy=cropped_temporal_endpoint_nxy,
                obs_nxy=expected_output_nxy,
                mask_nxy=cropped_mask_nxy).sum()
            temporal_endpoint_rec_loss = add_factor_to_loss(
                temporal_endpoint_rec_loss, loss_scale * c_temporal_endpoint_rec_loss)

        elif loss_type == 'lp':

            unet_endpoint_rec_loss = add_lp_to_loss(
                unet_endpoint_rec_loss,
                cropped_mask_nxy * (cropped_unet_endpoint_nxy - expected_output_nxy),
                _norm_p=norm_p,
                _scale=loss_scale)

            temporal_endpoint_rec_loss = add_lp_to_loss(
                temporal_endpoint_rec_loss,
                cropped_mask_nxy * (cropped_temporal_endpoint_nxy - expected_output_nxy),
                _norm_p=norm_p,
                _scale=loss_scale)

        else:

            raise ValueError()

        # temporal continuity loss
        if enable_continuity_reg:            

            if i_t > 0:

                unet_total_variation_nxy = get_total_variation(
                    curr_frame_nxy=cropped_unet_endpoint_nxy,
                    prev_frame_nxy=prev_cropped_unet_endpoint_nxy,
                    noise_std_nxy=cropped_movie_t_std_nxy,
                    noise_threshold_to_std=noise_threshold_to_std,
                    reg_func=reg_func,
                    eps=eps)

                temporal_total_variation_nxy = get_total_variation(
                    curr_frame_nxy=cropped_temporal_endpoint_nxy,
                    prev_frame_nxy=prev_cropped_temporal_endpoint_nxy,
                    noise_std_nxy=cropped_movie_t_std_nxy,
                    noise_threshold_to_std=noise_threshold_to_std,
                    reg_func=reg_func,
                    eps=eps)

                unet_endpoint_reg_loss = add_lp_to_loss(
                    unet_endpoint_reg_loss,
                    unet_total_variation_nxy,
                    _norm_p=norm_p,
                    _scale=continuity_reg_strength / ((t_tandem - 1) * total_pixels))

                temporal_endpoint_reg_loss = add_lp_to_loss(
                    temporal_endpoint_reg_loss,
                    temporal_total_variation_nxy,
                    _norm_p=norm_p,
                    _scale=continuity_reg_strength / ((t_tandem - 1) * total_pixels))

            prev_cropped_unet_endpoint_nxy = cropped_unet_endpoint_nxy
            prev_cropped_temporal_endpoint_nxy = temporal_endpoint_reg_loss
            
        # total loss
        if enable_continuity_reg:
            total_temporal_loss = temporal_endpoint_rec_loss + temporal_endpoint_reg_loss
            total_unet_loss = unet_endpoint_rec_loss + unet_endpoint_reg_loss
        else:
            total_temporal_loss = temporal_endpoint_rec_loss
            total_unet_loss = unet_endpoint_rec_loss
        
        unet_admixing = self.train_params['unet_admixing']
        total_loss = (
            unet_admixing * total_unet_loss +
            (1. - unet_admixing) * total_temporal_loss) / n_loop

    return {
        'total_loss': total_loss,
        'unet_endpoint_rec_loss': unet_endpoint_rec_loss,
        'unet_endpoint_reg_loss': unet_endpoint_reg_loss,
        'temporal_endpoint_rec_loss': temporal_endpoint_rec_loss,
        'temporal_endpoint_reg_loss': temporal_endpoint_reg_loss}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()
    train(args.configfile)
