import numpy as np
import torch
from torch import nn
from typing import List, Tuple, Dict

import sys
sys.path.append('../cellmincer')

from .denoising_model import DenoisingModel

from opto_models import \
    ConditionalUNet, \
    TemporalDenoiser, \
    activation_from_str
from opto_utils import crop_center
from opto_ws import OptopatchDenoisingWorkspace


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
        
        # TODO: allow for disabled global features
        
        self.spatial_unet = ConditionalUNet(
            in_channels=1,
            out_channels=1,
            data_dim=2,
            feature_channels=config['n_global_features'],
            depth=config['spatial_unet_depth'],
            wf=config['spatial_unet_wf'],
            out_channels_before_readout=config['spatial_unet_out_channels_before_readout'],
            pad=config['spatial_unet_padding'],
            batch_norm=config['spatial_unet_batch_norm'],
            unet_kernel_size=config['spatial_unet_kernel_size'],
            n_conv_layers=config['spatial_unet_n_conv_layers'],
            readout_kernel_size=config['spatial_unet_readout_kernel_size'],
            readout_hidden_layer_channels_list=config['spatial_unet_readout_hidden_layer_channels_list'],
            activation=activation_from_str(config['spatial_unet_activation']),
            final_trans=torch.nn.Identity(),
            device=device,
            dtype=dtype)
        
        self.temporal_denoiser = TemporalDenoiser(
            in_channels=config['spatial_unet_out_channels_before_readout'],
            feature_channels=config['n_global_features'],
            t_order=self.t_order,
            kernel_size=config['temporal_denoiser_kernel_size'],
            hidden_conv_channels=config['temporal_denoiser_conv_channels'],
            hidden_dense_layer_dims=config['temporal_denoiser_hidden_dense_layer_dims'],
            activation=activation_from_str(config['temporal_denoiser_activation']),
            final_trans=torch.nn.Identity(),
            device=device,
            dtype=dtype)
    
    def forward(
            self,
            batch_data
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        padded_sliced_diff_movie_ntxy = batch_data['padded_sliced_diff_movie_ntxy']
        padded_global_features_nfxy = batch_data['padded_global_features_nfxy']
        
        n_batch, t_total = padded_sliced_diff_movie_ntxy.shape[:2]
        n_global_features = padded_global_features_nfxy.shape[-3]
        t_tandem = t_total - self.t_order
        x_window = batch_data['x_window']
        y_window = batch_data['y_window']

        cropped_unet_endpoint_nxy_list = []
        cropped_temporal_endpoint_nxy_list = []

        # calculate processed features
        unet_output_list = [
            self.spatial_unet(
                padded_sliced_diff_movie_ntxy[:, i_t:i_t+1, :, :],
                padded_global_features_nfxy)
            for i_t in range(t_total)]
        unet_features_nctxy = torch.stack([output['features_ncxy'] for output in unet_output_list], dim=-3)
        unet_features_width, unet_features_height = unet_features_nctxy.shape[-2:]
        unet_cropped_global_features_nfxy = crop_center(
            padded_global_features_nfxy,
            target_width=unet_features_width,
            target_height=unet_features_height)

        # compute temporal-denoised convolutions for all t_order-length windows
        cropped_temporal_endpoint_ntxy = torch.stack([
            crop_center(
                self.temporal_denoiser(
                    unet_features_nctxy[:, :, i_t:(i_t + self.t_order), :, :],
                    unet_cropped_global_features_nfxy),
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
        
        # TODO: decide if a true deque structure offers significant improvement
        unet_features_ncxy_list = []
        
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

                unet_output = self.spatial_unet(padded_sliced_movie_1txy, padded_global_features_1fxy)
                unet_features_ncxy_list.append(unet_output['features_ncxy'])

            unet_features_width, unet_features_height = unet_features_ncxy_list[0].shape[-2:]
            t_mid = (self.t_order - 1) // 2
            for i_t in range(t_begin + t_mid, t_end - t_mid):
                padded_sliced_movie_1txy = ws_denoising.get_movie_slice(
                    t_begin_index=i_t + t_mid,
                    t_end_index=i_t + t_mid + 1,
                    x0=x0,
                    y0=y0,
                    x_window=x_window,
                    y_window=y_window)['diff']

                unet_output = self.spatial_unet(padded_sliced_movie_1txy, padded_global_features_1fxy)
                unet_features_ncxy_list.append(unet_output['features_ncxy'])

                denoised_movie_txy[i_t - t_begin] = crop_center(
                    self.temporal_denoiser(
                        # features from t_order tandem frames
                        torch.stack(unet_features_ncxy_list, dim=-3),
                        # cropped global features
                        crop_center(
                            padded_global_features_1fxy,
                            target_width=unet_features_width,
                            target_height=unet_features_height)),
                    target_width=x_window,
                    target_height=y_window)

                unet_features_ncxy_list.pop(0)
        
        denoised_movie_txy[:t_mid] = denoised_movie_txy[t_mid]
        denoised_movie_txy[-t_mid:] = denoised_movie_txy[-t_mid - 1]
        return denoised_movie_txy
    
    @staticmethod
    def get_minimum_padding(
            config: dict,
            training_x_window: int,
            training_y_window: int) -> Tuple[int, int]:
        def _get_unet_input_size(
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
        
        if not config['spatial_unet_padding']:
            padded_x_window = _get_unet_input_size(
                output_min_size=(
                    training_x_window
                    + (config['spatial_unet_readout_kernel_size'] - 1)
                        * len(config['spatial_unet_readout_hidden_layer_channels_list']) + 1),
                kernel_size=config['spatial_unet_kernel_size'],
                n_conv_layers=config['spatial_unet_n_conv_layers'],
                depth=config['spatial_unet_depth'])
            padded_y_window = _get_unet_input_size(
                output_min_size=(
                    training_y_window
                    + (config['spatial_unet_readout_kernel_size'] - 1)
                        * len(config['spatial_unet_readout_hidden_layer_channels_list']) + 1),
                kernel_size=config['spatial_unet_kernel_size'],
                n_conv_layers=config['spatial_unet_n_conv_layers'],
                depth=config['spatial_unet_depth'])

        else:
            padded_x_window = _get_unet_input_size(
                output_min_size=training_x_window,
                kernel_size=1,
                n_conv_layers=config['spatial_unet_n_conv_layers'],
                depth=config['spatial_unet_depth'])
            padded_y_window = _get_unet_input_size(
                output_min_size=training_y_window,
                kernel_size=1,
                n_conv_layers=config['spatial_unet_n_conv_layers'],
                depth=config['spatial_unet_depth'])

        x_padding = (padded_x_window - training_x_window) // 2
        y_padding = (padded_y_window - training_y_window) // 2

        return x_padding, y_padding