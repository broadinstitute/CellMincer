import numpy as np
from skimage.filters import threshold_otsu
import torch
import logging
from typing import List, Tuple, Optional, Union, Dict

from .ws import OptopatchBaseWorkspace, OptopatchDenoisingWorkspace
from .utils import crop_center, get_nn_spatio_temporal_mean, pad_images_torch

from cellmincer import consts


def generate_bernoulli_mask(
        p: float,
        n_batch: int,
        width: int,
        height: int,
        device: torch.device = consts.DEFAULT_DEVICE,
        dtype: torch.dtype = consts.DEFAULT_DTYPE) -> torch.Tensor:
    return torch.distributions.Bernoulli(
        probs=torch.tensor(p, device=device, dtype=dtype)).sample(
        [n_batch, width, height]).type(dtype)


def generate_bernoulli_mask_on_mask(
        p: float,
        in_mask: torch.Tensor,
        device: torch.device = consts.DEFAULT_DEVICE,
        dtype: torch.dtype = consts.DEFAULT_DTYPE) -> torch.Tensor:
    out_mask = torch.zeros(in_mask.shape, device=device, dtype=dtype)
    active_pixels = in_mask.sum()
    bern = torch.distributions.Bernoulli(
        probs=torch.tensor(p, device=device, dtype=dtype)).sample(
        [active_pixels.item()]).type(dtype)
    out_mask[in_mask] = bern
    return out_mask


def inflate_binary_mask(mask_ncxy: torch.Tensor, radius: int):
    assert radius >= 0
    if radius == 0:
        return mask_ncxy
    device = mask_ncxy.device
    x = torch.arange(-radius, radius + 1, dtype=torch.float, device=device)[None, :]
    y = torch.arange(-radius, radius + 1, dtype=torch.float, device=device)[:, None]
    struct = ((x.pow(2) + y.pow(2)) <= (radius ** 2)).float()
    kern = struct[None, None, ...]
    return (torch.nn.functional.conv2d(mask_ncxy, kern, padding=radius) > 0).type(mask_ncxy.dtype)

    
def generate_occluded_training_data(
        ws_denoising_list: List[OptopatchDenoisingWorkspace],
        t_order: int,
        t_tandem: int,
        n_batch: int,
        x_window: int,
        y_window: int,
        occlusion_prob: float,
        occlusion_radius: int,
        occlusion_strategy: str,
        device: torch.device = consts.DEFAULT_DEVICE,
        dtype: torch.dtype = consts.DEFAULT_DTYPE = torch.float32):
    """Generates minibatches with appropriate occlusion and padding for training a blind
    denoiser. Supports multiple datasets.
    
    The temporal length of each slice is:
    
        | (t_order - 1) // 2 | t_tandem + 1 | (t_order - 1) // 2 | 
    
    If `t_tandem == 0`, then only a single frame in the middle frame is Bernoulli-occluded and
    expected to be used for Noise2Self training. If `t_tandem > 0`, then a total number of
    `t_tandem + 1` frames sandwiched in the middle will be randomly Bernoulli-occluded.
    
    """
    
    assert t_order % 2 == 1
    assert t_tandem % 2 == 0
    assert occlusion_strategy in {'random', 'nn-average'}
    
    padded_x_window = x_window + 2 * ws_denoising_list[0].x_padding
    padded_y_window = y_window + 2 * ws_denoising_list[0].y_padding
    
    n_datasets = len(ws_denoising_list)
    n_global_features = ws_denoising_list[0].n_global_features
    t_total = t_order + t_tandem
    t_mid = (t_total - 1) // 2
    trend_mean_feature_index = ws_denoising_list[0].cached_features.get_feature_index('trend_mean_0')
    detrended_std_feature_index = ws_denoising_list[0].cached_features.get_feature_index('detrended_std_0')
    
    # sample random dataset indices
    dataset_indices = np.random.randint(0, n_datasets, size=n_batch)

    # random time slices
    time_slice_locs = np.random.rand(n_batch)
    t_begin_indices = [
        int(np.floor((ws_denoising_list[i_dataset].n_frames - t_total) * loc))
        for i_dataset, loc in zip(dataset_indices, time_slice_locs)]
    t_end_indices = [
        int(np.floor((ws_denoising_list[i_dataset].n_frames - t_total) * loc)) + t_total
        for i_dataset, loc in zip(dataset_indices, time_slice_locs)]
    
    # random space slices
    x0_list = [
        np.random.randint(0, ws_denoising_list[i_dataset].width - x_window + 1)
        for i_dataset in dataset_indices]
    y0_list = [
        np.random.randint(0, ws_denoising_list[i_dataset].height - y_window + 1)
        for i_dataset in dataset_indices]
        
    # generate a uniform bernoulli mask
    n_total_masks = n_batch * (t_tandem + 1)
    occlusion_masks_mxy = generate_bernoulli_mask(
        p=occlusion_prob,
        n_batch=n_total_masks,
        width=x_window,
        height=y_window,
        device=device,
        dtype=dtype)
    inflated_occlusion_masks_mxy = inflate_binary_mask(
        occlusion_masks_mxy[:, None, :, :], occlusion_radius)
    occlusion_masks_ntxy = occlusion_masks_mxy.view(
        n_batch, t_tandem + 1, x_window, y_window)    
    inflated_occlusion_masks_ntxy = inflated_occlusion_masks_mxy.view(
        n_batch, t_tandem + 1, x_window, y_window)    
    
    # slice the movies (w/ padding)
    movie_slice_dict_list = [
        ws_denoising_list[dataset_indices[i_batch]].get_movie_slice(
            t_begin_index=t_begin_indices[i_batch],
            t_end_index=t_end_indices[i_batch],
            x0=x0_list[i_batch],
            y0=y0_list[i_batch],
            x_window=x_window,
            y_window=y_window)
        for i_batch in range(n_batch)]
    diff_movie_slice_list = [item['diff'] for item in movie_slice_dict_list]
    bg_movie_slice_list = [item['bg'] for item in movie_slice_dict_list]
    
    # slice the features (w/ padding)
    feature_slice_list = [
        ws_denoising_list[dataset_indices[i_batch]].get_feature_slice(
            x0=x0_list[i_batch],
            y0=y0_list[i_batch],
            x_window=x_window,
            y_window=y_window)
        for i_batch in range(n_batch)]

    # stack to a batch dimension
    padded_sliced_diff_movie_ntxy = torch.cat(diff_movie_slice_list, dim=0)
    padded_sliced_bg_movie_ntxy = torch.cat(bg_movie_slice_list, dim=0)
    padded_global_features_nfxy = torch.cat(feature_slice_list, dim=0)

    # make a hard copy of the to-be-occluded frames
    padded_middle_frames_ntxy = padded_sliced_diff_movie_ntxy[
        :, (t_mid - (t_tandem // 2)):(t_mid + (t_tandem // 2) + 1), ...].clone()
    
    # pad the mask with zeros to match the padded movie
    padded_occlusion_masks_ntxy = pad_images_torch(
        images_ncxy=occlusion_masks_ntxy,
        target_width=padded_x_window,
        target_height=padded_y_window,
        pad_value_nc=torch.zeros(n_batch, t_tandem + 1, device=device, dtype=dtype))
    
    padded_inflated_occlusion_masks_ntxy = pad_images_torch(
        images_ncxy=inflated_occlusion_masks_ntxy,
        target_width=padded_x_window,
        target_height=padded_y_window,
        pad_value_nc=torch.zeros(n_batch, t_tandem + 1, device=device, dtype=dtype))
    
    if occlusion_strategy == 'nn-average':
        
        padded_sliced_diff_movie_ntxy[
            :, (t_mid - (t_tandem // 2)):(t_mid + (t_tandem // 2) + 1), :, :] *= (
                1 - padded_inflated_occlusion_masks_ntxy)
        
        for i_t in range(t_tandem + 1):
            padded_sliced_diff_movie_ntxy[:, t_mid - (t_tandem // 2) + i_t, 1:-1, 1:-1] += (
                padded_inflated_occlusion_masks_ntxy[:, i_t, 1:-1, 1:-1]
                * get_nn_spatio_temporal_mean(
                    padded_sliced_diff_movie_ntxy, t_mid - (t_tandem // 2) + i_t))
    
    elif occlusion_strategy == 'random':

        padded_sliced_diff_movie_ntxy[
            :, (t_mid - (t_tandem // 2)):(t_mid + (t_tandem // 2) + 1), :, :] = (
                (1 - padded_inflated_occlusion_masks_ntxy) * padded_sliced_diff_movie_ntxy[
                    :, (t_mid - (t_tandem // 2)):(t_mid + (t_tandem // 2) + 1), :, :]
                + padded_inflated_occlusion_masks_ntxy * torch.distributions.Normal(
                    loc=padded_global_features_nfxy[:, trend_mean_feature_index, :, :][:, None, ...].expand(
                        n_batch, t_tandem + 1, padded_x_window, padded_y_window),
                    scale=padded_global_features_nfxy[:, detrended_std_feature_index, :, :][:, None, ...].expand(
                        n_batch, t_tandem + 1, padded_x_window, padded_y_window)).sample())
    
    else:
            
        raise ValueError("Unknown occlusion strategy; valid options: 'nn-average', 'random'")
                
    return {
        'dataset_indices': dataset_indices,
        'padded_global_features_nfxy': padded_global_features_nfxy,  # global constant feature
        'padded_sliced_diff_movie_ntxy': padded_sliced_diff_movie_ntxy,  # sliced movie with occluded pixels 
        'padded_sliced_bg_movie_ntxy': padded_sliced_bg_movie_ntxy, 
        'padded_middle_frames_ntxy': padded_middle_frames_ntxy,  # original frames in the middle of the movie
        'padded_occlusion_masks_ntxy': padded_occlusion_masks_ntxy,  # occlusion masks
        'padded_inflated_occlusion_masks_ntxy': padded_inflated_occlusion_masks_ntxy,
        'x_window': x_window,
        'y_window': y_window,
        'padded_x_window': padded_x_window,
        'padded_y_window': padded_y_window,
        'trend_mean_feature_index': trend_mean_feature_index,
        'detrended_std_feature_index': detrended_std_feature_index
    }


def get_total_variation(
        dt_frame_ntxy: torch.Tensor,
        noise_std_nxy: torch.Tensor,
        noise_threshold_to_std: float,
        reg_func: str,
        eps: float = 1e-6):
    
    noise_std_ntxy = noise_std_nxy.unsqueeze(1)
    if reg_func == 'clamped_linear':
        return torch.clamp(
            dt_frame_ntxy / (eps + noise_std_ntxy),
            min=0.,
            max=noise_threshold_to_std)
    elif reg_func == 'tanh':
        eta = eps + noise_threshold_to_std
        return eta * torch.tanh(
            dt_frame_ntxy / ((eps + noise_std_ntxy) * eta))
    else:
        raise ValueError(
            f"Unknown reg_func value ({reg_func}); valid options are: 'clamped_linear', 'tanh'")        
        

def get_poisson_gaussian_nll(
        var_ntxy: torch.Tensor,
        pred_ntxy: torch.Tensor,
        obs_ntxy: torch.Tensor,
        mask_ntxy: torch.Tensor,
        scale_ntxy: torch.Tensor):
    log_var_ntxy = var_ntxy.log(dim=(0, 2, 3))[None, :, None, None]
    return 0.5 * mask_ntxy * (log_var_ntxy + (pred_ntxy - obs_ntxy).square() / var_ntxy) * scale_ntxy
    

def get_noise2self_loss(
        batch_data,
        ws_denoising_list: List[OptopatchDenoisingWorkspace],
        denoising_model,
        loss_type: str,
        norm_p: int,
        enable_continuity_reg: bool,
        reg_func: str,
        continuity_reg_strength: float,
        noise_threshold_to_std: float,
        eps: float = 1e-6):
    """Calculates the loss of a Noise2Self predictor on a given minibatch."""
    
    assert reg_func in {'clamped_linear', 'tanh'}
    assert loss_type in {'lp', 'poisson_gaussian'}
    
    x_window = batch_data['x_window']
    y_window = batch_data['y_window']
    t_total = batch_data['padded_sliced_diff_movie_ntxy'].shape[1]
    t_tandem = t_total - denoising_model.t_order
    t_mid = (denoising_model.t_order - 1) // 2
    total_pixels = x_window * y_window

    # iterate over the middle frames and accumulate loss
    def _compute_lp_loss(_err, _norm_p=norm_p, _scale=1.):
        return (_scale * (_err.abs() + eps).pow(_norm_p)).sum()
            
    # fetch and crop the dataset std (for regularization)
    if enable_continuity_reg:
        cropped_movie_t_std_nxy = crop_center(
            batch_data['padded_global_features_nfxy'][:, batch_data['detrended_std_feature_index'], ...],
            target_width=x_window,
            target_height=y_window)
        
    denoised_batch_ntxy = crop_center(
        denoising_model(batch_data),
        target_width=x_window,
        target_height=y_window)

    reg_loss = None
    rec_loss = None

    cropped_mask_ntxy = crop_center(
        batch_data['padded_occlusion_masks_ntxy'],
        target_width=x_window,
        target_height=y_window)

    expected_output_ntxy = crop_center(
        batch_data['padded_middle_frames_ntxy'],
        target_width=x_window,
        target_height=y_window)
    
    # reconstruction losses
    total_masked_pixels_t = cropped_mask_ntxy.sum(dim=1).type(denoising_model.dtype)
    loss_scale_t = 1. / ((t_tandem + 1) * (eps + total_masked_pixels_t))
    loss_scale_ntxy = loss_scale_t[None, :, None, None]
    
    # calculate the loss on occluded points of the middle frames
    # and total variation loss between frames (if enabled)
    if loss_type == 'poisson_gaussian':
        var_ntxy = torch.cat([
            ws_denoising_list[i_dataset].get_modeled_variance(
                scaled_bg_movie_txy=crop_center(
                    batch_data['padded_sliced_bg_movie_ntxy'][i_dataset, t_mid:t_mid + t_tandem + 1, ...],
                    target_width=x_window,
                    target_height=y_window),
                scaled_diff_movie_txy=denoised_batch_ntxy[i_dataset, ...])
            for i_dataset in batch_data['dataset_indices']], dim=0)
        rec_loss = get_poisson_gaussian_nll(
            var_ntxy=var_ntxy,
            pred_ntxy=denoised_batch_ntxy,
            obs_ntxy=expected_output_ntxy,
            mask_ntxy=cropped_mask_ntxy,
            scale_ntxy=loss_scale_ntxy).sum()

    elif loss_type == 'lp':
        err_ntxy = cropped_mask_ntxy * (denoised_batch_ntxy - expected_output_ntxy)
        rec_loss = _compute_lp_loss(_err=err_ntxy, _norm_p=norm_p, _scale=loss_scale_ntxy)

    else:
        raise ValueError()
        
    if enable_continuity_reg:
        total_variation_ntxy = get_total_variation(
            dt_frame_ntxy=denoised_batch_ntxy[:, 1:, ...] - denoised_batch_ntxy[:, :-1, ...],
            noise_std_nxy=cropped_movie_t_std_nxy,
            noise_threshold_to_std=noise_threshold_to_std,
            reg_func=reg_func,
            eps=eps)

        reg_loss = _compute_lp_loss(
            _err=total_variation_ntxy,
            _norm_p=norm_p,
            _scale=continuity_reg_strength / ((t_tandem - 1) * total_pixels)) # TODO ask mehrtash abou this term
            
    return {'rec_loss': rec_loss, 'reg_loss': reg_loss}
