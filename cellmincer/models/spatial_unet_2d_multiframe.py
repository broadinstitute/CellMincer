import numpy as np
import torch
from torch import nn
from typing import List, Tuple, Dict

from .components import \
    activation_from_str, \
    GUNet, \
    get_best_gunet_input_output_size

from .denoising_model import DenoisingModel

from cellmincer.util.ws import OptopatchDenoisingWorkspace


class SpatialUnet2dMultiframe(DenoisingModel):
    def __init__(
            self,
            config: dict,
            device: torch.device,
            dtype: torch.dtype):
        
        super(SpatialUnet2dMultiframe, self).__init__(
            name=config['type'],
            t_order=config['t_order'],
            device=device,
            dtype=dtype)
        
        self.use_global_features = config['use_global_features']
        
        self.unet = GUNet(
            in_channels=self.t_order,
            out_channels=1,
            data_dim=2,
            feature_channels=config['n_global_features'] if self.use_global_features else 0,
            noise_channels=0,
            depth=config['unet_depth'],
            first_conv_channels=config['unet_first_conv_channels'],
            ch_growth_rate=2,
            ds_rate=2,
            final_trans=lambda x: x,
            pad=config['use_padding'],
            layer_norm=config['use_layer_norm'],
            attention=config['use_attention'],
            up_mode='upsample',
            pool_mode='max',
            norm_mode='batch',
            unet_kernel_size=config['unet_kernel_size'],
            n_conv_layers=config['unet_n_conv_layers'],
            p_dropout=0.0,
            readout_hidden_layer_channels_list=[config['unet_first_conv_channels']],
            readout_kernel_size=config['unet_readout_kernel_size'],
            activation=activation_from_str(config['unet_activation']),
            device=device,
            dtype=dtype)
    
    def forward(
            self,
            batch_data
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        padded_sliced_diff_movie_ntxy = batch_data['padded_sliced_diff_movie_ntxy']
        if self.use_global_features:
            padded_global_features_nfxy = batch_data['padded_global_features_nfxy']
        
        t_total = padded_sliced_diff_movie_ntxy.shape[1]
        t_tandem = t_total - self.t_order

        # calculate processed features
        unet_endpoint_ntxy = torch.cat([
                (self.unet(
                    padded_sliced_diff_movie_ntxy[:, i_t:i_t+self.t_order, :, :],
                    padded_global_features_nfxy)
                if self.use_global_features else
                self.unet(padded_sliced_diff_movie_ntxy[:, i_t:i_t+self.t_order, :, :]))['readout']
            for i_t in range(t_tandem + 1)], dim=1)
        
        return unet_endpoint_ntxy
    
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
        
        n_frames = ws_denoising.n_frames
        t_mid = (self.t_order - 1) // 2
        mid_frame_begin = max(t_begin, t_mid)
        mid_frame_end = min(t_end, n_frames - t_mid)
        
        denoised_movie_txy_list = []
        
        if self.use_global_features:
            padded_global_features_1fxy = ws_denoising.get_feature_slice(
                x0=x0,
                y0=y0,
                x_window=x_window,
                y_window=y_window)
        
        with torch.no_grad():
            for i_t in range(mid_frame_begin - t_mid, mid_frame_end - t_mid):
                padded_sliced_movie_1txy = ws_denoising.get_movie_slice(
                    t_begin_index=i_t,
                    t_end_index=i_t + self.t_order,
                    x0=x0,
                    y0=y0,
                    x_window=x_window,
                    y_window=y_window)['diff']
                
                denoised_movie_txy_list.append(
                    (self.unet(padded_sliced_movie_1txy, padded_global_features_nfxy)
                        if self.use_global_features else
                        self.unet(padded_sliced_movie_1txy))['readout'][0, ...])
        
        # fill in edge frames with the ends of the middle frame interval
        denoised_movie_txy_full_list = \
            [denoised_movie_txy_list[0] for i in range(mid_frame_begin - t_begin)] + \
            denoised_movie_txy_list + \
            [denoised_movie_txy_list[-1] for i in range(t_end - mid_frame_end)]
        
        denoised_movie_txy = torch.cat(denoised_movie_txy_full_list, dim=0)
        return denoised_movie_txy

    @staticmethod
    def get_best_window_padding(
            config: dict,
            output_min_size_lo: int,
            output_min_size_hi: int) -> Tuple[int, int, int, int]:
        input_size, output_size = get_best_gunet_input_output_size(
            unet_kernel_size=config['spatial_unet_kernel_size'],
            n_conv_layers=config['spatial_unet_n_conv_layers'],
            depth=config['spatial_unet_depth'],
            ds_rate=2,
            output_min_size_lo=output_min_size_lo,
            output_min_size_hi=output_min_size_lo)
        padding = (input_size - output_size) // 2
        return output_size, output_size, padding, padding
