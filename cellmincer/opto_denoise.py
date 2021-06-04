import numpy as np
from skimage.filters import threshold_otsu
import torch
import logging
from typing import List, Tuple, Optional, Union, Dict

from .opto_ws import OptopatchBaseWorkspace, OptopatchDenoisingWorkspace
from .opto_utils import pad_images_torch, crop_center, get_nn_spatio_temporal_mean

logger = logging.getLogger()


def generate_bernoulli_mask(
        p: float,
        n_batch: int,
        width: int,
        height: int,
        device: torch.device,
        dtype: torch.dtype) -> torch.Tensor:
    return torch.distributions.Bernoulli(
        probs=torch.tensor(p, device=device, dtype=dtype)).sample(
        [n_batch, width, height]).type(dtype)


def generate_bernoulli_mask_on_mask(
        p: float,
        in_mask: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype) -> torch.Tensor:
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
        device: torch.device = torch.device('cuda'),
        dtype: torch.dtype = torch.float32):
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
        curr_frame_nxy: torch.Tensor,
        prev_frame_nxy: torch.Tensor,
        noise_std_nxy: torch.Tensor,
        noise_threshold_to_std: float,
        reg_func: str,
        eps: float = 1e-6):    
    if reg_func == 'clamped_linear':
        return torch.clamp(
            (curr_frame_nxy - prev_frame_nxy) / (eps + noise_std_nxy),
            min=0.,
            max=noise_threshold_to_std)
    elif reg_func == 'tanh':
        eta = eps + noise_threshold_to_std
        return eta * torch.tanh(
            (curr_frame_nxy - prev_frame_nxy) / ((eps + noise_std_nxy) * eta))
    else:
        raise ValueError(
            f"Unknown reg_func value ({reg_func}); valid options are: 'clamped_linear', 'tanh'")        
        

def get_poisson_gaussian_nll(
        var_nxy: torch.Tensor,
        pred_nxy: torch.Tensor,
        obs_nxy: torch.Tensor,
        mask_nxy: torch.Tensor):
    return 0.5 * mask_nxy * (var_nxy.log() + (pred_nxy - obs_nxy).pow(2) / var_nxy)
    

def get_noise2self_loss(
        batch_data,
        ws_denoising_list: List[OptopatchDenoisingWorkspace],
        spatial_unet_processor: torch.nn.Module,
        temporal_denoiser: torch.nn.Module,
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
    
    device = batch_data['padded_sliced_diff_movie_ntxy'].device
    dtype = batch_data['padded_sliced_diff_movie_ntxy'].dtype
    x_window = batch_data['x_window']
    y_window = batch_data['y_window']
    padded_x_window = batch_data['padded_x_window']
    padded_y_window = batch_data['padded_y_window']
    n_batch, t_total = batch_data['padded_sliced_diff_movie_ntxy'].shape[:2]
    n_global_features = batch_data['padded_global_features_nfxy'].shape[-3]
    t_tandem = batch_data['padded_occlusion_masks_ntxy'].shape[-3] - 1
    t_order = t_total - t_tandem
    t_mid = (t_order + t_tandem - 1) // 2
    total_pixels = x_window * y_window

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
    unet_output_list = [
        spatial_unet_processor(
            batch_data['padded_sliced_diff_movie_ntxy'][:, i_t:i_t+1, :, :],
            batch_data['padded_global_features_nfxy'])
        for i_t in range(t_total)]
    unet_features_nctxy = torch.stack([output['features_ncxy'] for output in unet_output_list], dim=-3)
    unet_readout_n1xy_list = [output['readout_ncxy'] for output in unet_output_list]
#     unet_features_nctxy = torch.cat([
#         unet_features_ncxy[:, :, None, :, :]
#         for unet_features_ncxy in unet_features_ncxy_list],
#         dim=-3)
    unet_features_width = unet_features_nctxy.shape[-2]
    unet_features_height = unet_features_nctxy.shape[-1]
    unet_cropped_global_features_nfxy = crop_center(
        batch_data['padded_global_features_nfxy'],
        target_width=unet_features_width,
        target_height=unet_features_height)
    
    # calculate the loss on occluded points of the middle frames
    # and total variation loss between frames (if enabled)
    for i_t in range(t_tandem + 1):  # i_t denotes the index of the middle frames, starting from 0

        # unet readout
        cropped_unet_endpoint_nxy = crop_center(
            unet_readout_n1xy_list[(t_order - 1) // 2 + i_t][:, 0, :, :],
            target_width=x_window,
            target_height=y_window)

        # get the temporal denoiser output
        cropped_temporal_endpoint_nxy = crop_center(
            temporal_denoiser(
                unet_features_nctxy[:, :, i_t:(i_t + t_order), :, :],
                unet_cropped_global_features_nfxy),
            target_width=x_window,
            target_height=y_window)
        
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
            
    return {
        'unet_endpoint_rec_loss': unet_endpoint_rec_loss,
        'unet_endpoint_reg_loss': unet_endpoint_reg_loss,
        'temporal_endpoint_rec_loss': temporal_endpoint_rec_loss,
        'temporal_endpoint_reg_loss': temporal_endpoint_reg_loss}


def generate_input_for_single_frame_denoising(
        ws_denoising_list: List[OptopatchDenoisingWorkspace],
        i_dataset: int,
        i_t: int,
        t_order: int,
        x0: int,
        y0: int,
        x_window: int,
        y_window: int,
        device: torch.device = torch.device('cuda'),
        dtype: torch.dtype = torch.float32):
    """Prepares data for single-dataset single-frame denoising"""
    
    # slice the movies and copy to device
    t_mid = (t_order - 1) // 2
    padded_sliced_movie_txy = ws_denoising_list[i_dataset].get_movie_slice(
        t_begin_index=i_t - t_mid,
        t_end_index=i_t + t_mid + 1,
        x0=x0,
        y0=y0,
        x_window=x_window,
        y_window=y_window)['diff'][0, ...]
    
    padded_global_features_fxy = ws_denoising_list[i_dataset].get_feature_slice(
        x0=x0,
        y0=y0,
        x_window=x_window,
        y_window=y_window)[0, ...]
    
    return {
        'padded_global_features_fxy': padded_global_features_fxy,
        'padded_sliced_movie_txy': padded_sliced_movie_txy,
        'x_window': x_window,
        'y_window': y_window
    }


def denoise_single_frame(batch_data,
                         ws_denoising_list: List[OptopatchDenoisingWorkspace],
                         spatial_unet_processor: torch.nn.Module,
                         temporal_denoiser: torch.nn.Module):
    """Denoises a single frame from a single dataset"""
    device = batch_data['padded_sliced_movie_txy'].device
    dtype = batch_data['padded_sliced_movie_txy'].dtype
    t_order, padded_width, padded_height = batch_data['padded_sliced_movie_txy'].shape
    n_global_features = batch_data['padded_global_features_fxy'].shape[-3]

    with torch.no_grad():
        
        # calculate processed features
        unet_output_list = [
            spatial_unet_processor(
                batch_data['padded_sliced_movie_txy'][i_t, :, :][None, None, :, :],
                batch_data['padded_global_features_fxy'][None, :, :, :])
            for i_t in range(t_order)]
        unet_features_ncxy_list = [output['features_ncxy'] for output in unet_output_list]
        unet_readout_n1xy_list = [output['readout_ncxy'] for output in unet_output_list]
        unet_features_width = unet_features_ncxy_list[0].shape[-2]
        unet_features_height = unet_features_ncxy_list[0].shape[-1]
        
        # crop unet readout
        cropped_unet_endpoint_xy = crop_center(
            unet_readout_n1xy_list[(t_order - 1) // 2],
            target_width=batch_data['x_window'],
            target_height=batch_data['y_window'])[0, 0, :, :]
    
        # get the temporal denoiser output
        cropped_temporal_endpoint_xy = crop_center(
            temporal_denoiser(
                # features from t_order tandem frames
                torch.cat([
                    unet_features_ncxy_list[i_t][:, :, None, :, :]
                    for i_t in range(t_order)],
                dim=-3),
                # global features
                crop_center(
                    batch_data['padded_global_features_fxy'][None, :, :, :],
                    target_width=unet_features_width,
                    target_height=unet_features_height)),
            target_width=batch_data['x_window'],
            target_height=batch_data['y_window'])
        
        return {
            'unet_endpoint_xy': cropped_unet_endpoint_xy,
            'temporal_endpoint_xy': cropped_temporal_endpoint_xy
        }


def get_unet_input_size(
    output_min_size: int,
    kernel_size: int,
    n_conv_layers: int,
    depth: int):
    """Smallest input size for output size >= `output_min_size`.
    
    .. note:
        The calculated input sizes guarantee that all of the layers have even dimensions.
        This is important to prevent aliasing in downsampling (pooling) operations.
    
    """
    delta = n_conv_layers * (kernel_size - 1)
    pad = delta * sum([2 ** i for i in range(depth)])
    ds = 2 ** depth
    res = (output_min_size + pad) % ds
    bottom_size = (output_min_size + pad) // ds + res
    input_size = bottom_size
    for i in range(depth):
        input_size = 2 * (input_size + delta)
    input_size += delta
    return input_size


def get_minimum_spatial_padding(
        x_window: int,
        y_window: int,
        denoiser_config: Dict[str, Union[int, float, str]]) -> Tuple[int, int]:
    
    if not denoiser_config['spatial_unet_padding']:
        
        padded_x_window = get_unet_input_size(
            output_min_size=(
                x_window
                + (denoiser_config['spatial_unet_readout_kernel_size'] - 1)
                    * len(denoiser_config['spatial_unet_readout_hidden_layer_channels_list']) + 1),
            kernel_size=denoiser_config['spatial_unet_kernel_size'],
            n_conv_layers=denoiser_config['spatial_unet_n_conv_layers'],
            depth=denoiser_config['spatial_unet_depth'])
        padded_y_window = get_unet_input_size(
            output_min_size=(
                y_window
                + (denoiser_config['spatial_unet_readout_kernel_size'] - 1)
                    * len(denoiser_config['spatial_unet_readout_hidden_layer_channels_list']) + 1),
            kernel_size=denoiser_config['spatial_unet_kernel_size'],
            n_conv_layers=denoiser_config['spatial_unet_n_conv_layers'],
            depth=denoiser_config['spatial_unet_depth'])
    
    else:
        
        padded_x_window = get_unet_input_size(
            output_min_size=x_window,
            kernel_size=1,
            n_conv_layers=denoiser_config['spatial_unet_n_conv_layers'],
            depth=denoiser_config['spatial_unet_depth'])
        padded_y_window = get_unet_input_size(
            output_min_size=y_window,
            kernel_size=1,
            n_conv_layers=denoiser_config['spatial_unet_n_conv_layers'],
            depth=denoiser_config['spatial_unet_depth'])

    x_padding = (padded_x_window - x_window) // 2
    y_padding = (padded_y_window - y_window) // 2
    
    return x_padding, y_padding
