import numpy as np
import torch
from torch import nn
from typing import List, Tuple, Dict

from .components import \
    activation_from_str, \
    GUNet, \
    get_best_gunet_input_size, \
    TemporalDenoiser

from .denoising_model import DenoisingModel

from cellmincer.util.utils import crop_center
from cellmincer.util.ws import OptopatchDenoisingWorkspace


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
        
        self.use_global_features = config['use_global_features']
        
        self.spatial_unet = GUNet(
            in_channels=1,
            out_channels=1,
            data_dim=2,
            feature_channels=config['n_global_features'] if self.use_global_features else 0,
            noise_channels=0,
            depth=config['spatial_unet_depth'],
            first_conv_channels=config['spatial_unet_first_conv_channels'],
            ch_growth_rate=2,
            ds_rate=2,
            final_trans=lambda x: x,
            pad=config['use_padding'],
            layer_norm=config['use_layer_norm'],
            attention=config['use_attention'],
            up_mode='upsample',
            pool_mode='max',
            norm_mode='batch',
            unet_kernel_size=config['spatial_unet_kernel_size'],
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
            batch_data
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        padded_sliced_diff_movie_ntxy = batch_data['padded_sliced_diff_movie_ntxy']
        if self.use_global_features:
            padded_global_features_nfxy = batch_data['padded_global_features_nfxy']
        
        n_batch, t_total = padded_sliced_diff_movie_ntxy.shape[:2]
        t_tandem = t_total - self.t_order
        x_window = batch_data['x_window']
        y_window = batch_data['y_window']

        cropped_unet_endpoint_nxy_list = []
        cropped_temporal_endpoint_nxy_list = []

        # calculate processed features
        unet_output_list = [(
                self.spatial_unet(
                    padded_sliced_diff_movie_ntxy[:, i_t:i_t+1, :, :],
                    padded_global_features_nfxy)
                if self.use_global_features else
                self.spatial_unet(padded_sliced_diff_movie_ntxy[:, i_t:i_t+1, :, :]))
            for i_t in range(t_total)]
        unet_features_nctxy = torch.stack([output['features_ncxy'] for output in unet_output_list], dim=-3)

        # compute temporal-denoised convolutions for all t_order-length windows
        cropped_temporal_endpoint_ntxy = torch.stack([
            crop_center(
                self.temporal_denoiser(unet_features_nctxy[:, :, i_t:(i_t + self.t_order), :, :]),
                target_width=x_window,
                target_height=y_window)
            for i_t in range(t_tandem + 1)], dim=1)
            
        return cropped_temporal_endpoint_ntxy
    
    # TODO modify to use out-of-time-window frames for denoising if they exist
    def denoise_movie(
            self,
            ws_denoising: OptopatchDenoisingWorkspace,
            t_begin: int = 0,
            t_end: int = None,
            x0: int = 0,
            y0: int = 0,
            x_window: int = None,
            y_window: int = None
        ) -> torch.Tensor:
        
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
        
        denoised_movie_txy = torch.empty(t_end - t_begin, x_window, y_window, device='cpu')
        
        unet_features_ncxy_list = []
        
        if self.use_global_features:
            padded_global_features_1fxy = ws_denoising.get_feature_slice(
                x0=x0,
                y0=y0,
                x_window=x_window,
                y_window=y_window)
        
        with torch.no_grad():
            for i_t in range(t_begin, t_begin + self.t_order - 1):
                padded_sliced_movie_1txy = ws_denoising.get_movie_slice(
                    t_begin_index=i_t,
                    t_end_index=i_t + 1,
                    x0=x0,
                    y0=y0,
                    x_window=x_window,
                    y_window=y_window)['diff']

                unet_output = (
                    self.spatial_unet(padded_sliced_movie_1txy, padded_global_features_1fxy)
                    if self.use_global_features else
                    self.spatial_unet(padded_sliced_movie_1txy))
                unet_features_ncxy_list.append(unet_output['features_ncxy'])

            t_mid = (self.t_order - 1) // 2
            for i_t in range(t_begin + t_mid, t_end - t_mid):
                padded_sliced_movie_1txy = ws_denoising.get_movie_slice(
                    t_begin_index=i_t + t_mid,
                    t_end_index=i_t + t_mid + 1,
                    x0=x0,
                    y0=y0,
                    x_window=x_window,
                    y_window=y_window)['diff']

                unet_output = (
                    self.spatial_unet(padded_sliced_movie_1txy, padded_global_features_1fxy)
                    if self.use_global_features else
                    self.spatial_unet(padded_sliced_movie_1txy))
                unet_features_ncxy_list.append(unet_output['features_ncxy'])

                denoised_movie_txy[i_t - t_begin] = crop_center(
                    self.temporal_denoiser(torch.stack(unet_features_ncxy_list, dim=-3)),
                    target_width=x_window,
                    target_height=y_window)

                unet_features_ncxy_list.pop(0)
        
        denoised_movie_txy[:t_mid] = denoised_movie_txy[t_mid]
        denoised_movie_txy[-t_mid:] = denoised_movie_txy[-t_mid - 1]

    def get_best_input_size(
            self,
            output_min_size_lo: int,
            output_min_size_hi: int) -> Tuple[int, int]:
        input_size = get_best_gunet_input_size(self.spatial_unet, output_min_size_lo, output_min_size_hi)
        return input_size, input_size