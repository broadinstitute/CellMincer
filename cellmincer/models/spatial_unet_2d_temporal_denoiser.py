import numpy as np
import torch
from torch import nn
from typing import Dict, List, Optional, Tuple, Union
from pytorch_lightning import LightningModule

from torchinfo import summary

from .components import \
    activation_from_str, \
    GUNet, \
    get_unet_input_size, \
    TemporalDenoiser

from .denoising_model import DenoisingModel

from cellmincer.util import \
    OptopatchDenoisingWorkspace, \
    crop_center


class SpatialUnet2dTemporalDenoiser(DenoisingModel):
    def __init__(
            self,
            config: dict,
            device: torch.device,
            dtype: torch.dtype):
        
        t_order = (1 +
            (config['temporal_denoiser_kernel_size'] - 1) *
            (config['temporal_denoiser_n_conv_layers']))
        
        super(SpatialUnet2dTemporalDenoiser, self).__init__(
            name=config['type'],
            t_order=t_order,
            device=device,
            dtype=dtype)
        
        self.feature_mode = config['spatial_unet_feature_mode']
        assert self.feature_mode in {'repeat', 'once', 'none'}
        
        self.spatial_unet = GUNet(
            in_channels=1,
            out_channels=1,
            data_dim=2,
            feature_channels=config['n_global_features'],
            noise_channels=0,
            depth=config['spatial_unet_depth'],
            first_conv_channels=config['spatial_unet_first_conv_channels'],
            ch_growth_rate=2,
            ds_rate=2,
            final_trans=lambda x: x,
            pad=config['spatial_unet_padding'],
            layer_norm=config['spatial_unet_batch_norm'],
            attention=config['spatial_unet_attention'],
            feature_mode=config['spatial_unet_feature_mode'],
            up_mode='upsample',
            pool_mode='max',
            norm_mode='batch',
            kernel_size=config['spatial_unet_kernel_size'],
            n_conv_layers=config['spatial_unet_n_conv_layers'],
            p_dropout=0.0,
            readout_hidden_layer_channels_list=[config['spatial_unet_first_conv_channels']],
            readout_kernel_size=config['spatial_unet_readout_kernel_size'],
            activation=activation_from_str(config['spatial_unet_activation']),
            device=device,
            dtype=dtype)
        
        self.temporal_denoiser = TemporalDenoiser(
            in_channels=self.spatial_unet.out_channels_before_readout,
            t_order=self.t_order,
            kernel_size=config['temporal_denoiser_kernel_size'],
            hidden_conv_channels=config['temporal_denoiser_conv_channels'],
            hidden_dense_layer_dims=config['temporal_denoiser_hidden_dense_layer_dims'],
            activation=activation_from_str(config['temporal_denoiser_activation']),
            final_trans=lambda x: x,
            device=device,
            dtype=dtype)
    
    def forward(
            self,
            x: torch.Tensor,
            features: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert not(self.feature_mode != 'none' and (features is None))
        
        t_total = x.shape[1]
        t_tandem = t_total - self.t_order

        # calculate processed features
        unet_output_list = [(
                self.spatial_unet(x[:, i_t:i_t+1, :, :], features) if self.feature_mode != 'none' else
                self.spatial_unet(x[:, i_t:i_t+1, :, :]))
            for i_t in range(t_total)]
        unet_features_nctxy = torch.stack([output['features'] for output in unet_output_list], dim=-3)
        
        # compute temporal-denoised convolutions for all t_order-length windows
        temporal_endpoint_ntxy = torch.stack([
            self.temporal_denoiser(unet_features_nctxy[:, :, i_t:(i_t + self.t_order), :, :])
            for i_t in range(t_tandem + 1)], dim=1)
            
        return temporal_endpoint_ntxy

    def denoise_movie(
            self,
            ws_denoising: OptopatchDenoisingWorkspace,
            t_begin: int = 0,
            t_end: int = None,
            x0: int = 0,
            y0: int = 0,
            x_window: int = None,
            y_window: int = None) -> torch.Tensor:
        # defaults bounds to full movie if unspecified
        if t_end is None:
            t_end = ws_denoising.n_frames
        if x_window is None:
            x_window = ws_denoising.width - x0
        if y_window is None:
            y_window = ws_denoising.height - y0
        
        assert t_end - t_begin >= self.t_order
        assert 0 <= x0 <= x0 + x_window <= ws_denoising.width
        assert 0 <= y0 <= y0 + y_window <= ws_denoising.height
        
        x_padding, y_padding = self.get_window_padding([x_window, y_window])
        
        n_frames = ws_denoising.n_frames
        t_mid = (self.t_order - 1) // 2
        mid_frame_begin = max(t_begin, t_mid)
        mid_frame_end = min(t_end, n_frames - t_mid)
        
        denoised_movie_txy_list = []
        unet_features_ncxy_list = []
        
        if self.feature_mode != 'none':
            padded_global_features_1fxy = ws_denoising.get_feature_slice(
                x0=x0,
                y0=y0,
                x_window=x_window,
                y_window=y_window,
                x_padding=x_padding,
                y_padding=y_padding)
        
        with torch.no_grad():
            for i_t in range(mid_frame_begin - t_mid, mid_frame_begin + t_mid):
                padded_sliced_movie_1txy = ws_denoising.get_movie_slice(
                    include_bg=False,
                    t_begin_index=i_t,
                    t_end_index=i_t + 1,
                    x0=x0,
                    y0=y0,
                    x_window=x_window,
                    y_window=y_window,
                    x_padding=x_padding,
                    y_padding=y_padding)['diff']

                unet_output = (
                    self.spatial_unet(padded_sliced_movie_1txy, padded_global_features_1fxy)
                        if self.feature_mode != 'none' else
                        self.spatial_unet(padded_sliced_movie_1txy))
                unet_features_ncxy_list.append(unet_output['features'])

            for i_t in range(mid_frame_begin, mid_frame_end):
                padded_sliced_movie_1txy = ws_denoising.get_movie_slice(
                    include_bg=False,
                    t_begin_index=i_t + t_mid,
                    t_end_index=i_t + t_mid + 1,
                    x0=x0,
                    y0=y0,
                    x_window=x_window,
                    y_window=y_window,
                    x_padding=x_padding,
                    y_padding=y_padding)['diff']

                unet_output = (
                    self.spatial_unet(padded_sliced_movie_1txy, padded_global_features_1fxy)
                    if self.feature_mode != 'none' else
                    self.spatial_unet(padded_sliced_movie_1txy))
                unet_features_ncxy_list.append(unet_output['features'])

                denoised_movie_txy_list.append(
                    self.temporal_denoiser(torch.stack(unet_features_ncxy_list, dim=-3)).cpu())

                unet_features_ncxy_list.pop(0)
        
        # fill in edge frames with the ends of the middle frame interval
        denoised_movie_txy_full_list = \
            [denoised_movie_txy_list[0] for i in range(mid_frame_begin - t_begin)] + \
            denoised_movie_txy_list + \
            [denoised_movie_txy_list[-1] for i in range(t_end - mid_frame_end)]
        
        denoised_movie_txy = torch.cat(denoised_movie_txy_full_list, dim=0)
        return crop_center(
            denoised_movie_txy,
            target_width=x_window,
            target_height=y_window)
    
    def summary(
            self,
            ws_denoising: OptopatchDenoisingWorkspace,
            x_window: int,
            y_window: int):
        x_padding, y_padding = self.get_window_padding([x_window, y_window])
        
        input_data = {}
        
        input_data['x'] = ws_denoising.get_movie_slice(
            include_bg=False,
            t_begin_index=0,
            t_end_index=self.t_order,
            x0=0,
            y0=0,
            x_window=x_window,
            y_window=y_window,
            x_padding=x_padding,
            y_padding=y_padding)['diff']
        
        if self.feature_mode != 'none':
            input_data['features'] = ws_denoising.get_feature_slice(
                x0=0,
                y0=0,
                x_window=x_window,
                y_window=y_window,
                x_padding=x_padding,
                y_padding=y_padding)

        return str(summary(self, input_data=input_data))

    def get_window_padding(
            self,
            output_min_size: Union[int, list, np.ndarray]) -> np.ndarray:
        input_size = get_unet_input_size(
            output_min_size=output_min_size,
            kernel_size=self.spatial_unet.kernel_size,
            n_conv_layers=self.spatial_unet.n_conv_layers,
            depth=self.spatial_unet.depth,
            ds_rate=self.spatial_unet.ds_rate)
        padding = (input_size - output_min_size) // 2
        return padding

    @staticmethod
    def get_window_padding_from_config(
            config: dict,
            output_min_size: Union[int, list, np.ndarray]) -> np.ndarray:
        input_size = get_unet_input_size(
            output_min_size=output_min_size,
            kernel_size=config['spatial_unet_kernel_size'],
            n_conv_layers=config['spatial_unet_n_conv_layers'],
            depth=config['spatial_unet_depth'],
            ds_rate=2)
        padding = (input_size - output_min_size) // 2
        return padding


# Dataloader return movie slices
# In model training step you do occlusion of pixel
#

class PlSpatialUnet2dTemporalDenoiser(LightningModule):
    def __init__(
            self,
            config_model: dict,
            config_train: dict,
            device: torch.device,
            dtype: torch.dtype):

        self.config_model = config_model
        self.config_train = config_train
        self.denoising_model = SpatialUnet2dTemporalDenoiser(config, device, dtype)

    def compute_noise2self_loss(self,
                                batch_data,
                                ws_denoising_list: List[OptopatchDenoisingWorkspace],
                                denoising_model,
                                loss_type: str,
                                norm_p: int,
                                enable_continuity_reg: bool,
                                reg_func: str,
                                continuity_reg_strength: float,
                                noise_threshold_to_std: float):
            """Calculates the loss of a Noise2Self predictor on a given minibatch."""

            assert reg_func in {'clamped_linear', 'tanh'}
            assert loss_type in {'lp', 'poisson_gaussian'}

            # iterate over the middle frames and accumulate loss
            def _compute_lp_loss(_err, _norm_p=norm_p, _scale=1.):
                return (_scale * (_err.abs() + const.EPS).pow(_norm_p)).sum()

            x_window, y_window = batch_data['x_window'], batch_data['y_window']
            total_pixels = x_window * y_window

            # TODO FIX ME

            t_total = batch_data['padded_sliced_diff_movie_ntxy'].shape[1]
            t_tandem = t_total - denoising_model.t_order
            t_mid = (denoising_model.t_order - 1) // 2

            # fetch and crop the dataset std (for regularization)
            if enable_continuity_reg:
                cropped_movie_t_std_nxy = crop_center(
                    batch_data['padded_global_features_nfxy'][:, batch_data['detrended_std_feature_index'], ...],
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
            total_masked_pixels_t = cropped_mask_ntxy.sum(dim=(0, 2, 3)).type(denoising_model.dtype)
            loss_scale_t = 1. / ((t_tandem + 1) * (const.EPS + total_masked_pixels_t))
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
                    for i_dataset in batch_data['dataset_indices']], dim=0).unsqueeze(1)
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
                raise ValueError('Unrecognized loss type.')

            if enable_continuity_reg:
                total_variation_ntxy = get_total_variation(
                    dt_frame_ntxy=denoised_batch_ntxy[:, 1:, ...] - denoised_batch_ntxy[:, :-1, ...],
                    noise_std_nxy=cropped_movie_t_std_nxy,
                    noise_threshold_to_std=noise_threshold_to_std,
                    reg_func=reg_func)

                reg_loss = _compute_lp_loss(
                    _err=total_variation_ntxy,
                    _norm_p=norm_p,
                    _scale=continuity_reg_strength / ((
                                                                  t_tandem + 1) * total_pixels))  # TODO check with mehrtash on change from (t_tandem - 1)

            return {'rec_loss': rec_loss, 'reg_loss': reg_loss}

    def forward(self,
                x: torch.Tensor,
                features: Optional[torch.Tensor] = None):

        denoised_batch_ntxy = self.denoising_model(x=x, features=features)
        denoised_batch_ntxy = crop_center(
            denoised_batch_ntxy,
            target_width=x_window,
            target_height=y_window)

        return denoised_batch_ntxy

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        # batch is what the dataloader provides -> movie slice

        occluded_movie = ......
        denoised_batch_ntxy = self(occluded_movie)
        loss_dict = self.compute_noise2self_loss(denoised_batch_ntxy, batch)

        loss = loss_dict["rec_loss"] + loss_dict["reg_loss"]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx: int = -1) -> Any:
        # If not defined. It will run the forward method
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:

        optim = generate_optimizer(
            denoising_model=self.denoising_model,
            optim_params=self.train_config['optim_params'],
            lr=self.train_config['lr_params']['max'])
        sched = generate_lr_scheduler(
            optim=self.optim,
            lr_params=self.train_config['lr_params'],
            n_iters=self.train_config['n_iters'])

        return [optim], [sched]




