import numpy as np
import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union

from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.loggers.base import DummyExperiment

from torchinfo import summary

from .components import \
    activation_from_str, \
    GUNet, \
    get_unet_input_size, \
    TemporalDenoiser

from .denoising_model import DenoisingModel

from cellmincer.util import \
    OptopatchDenoisingWorkspace, \
    const, \
    crop_center, \
    generate_lr_scheduler, \
    generate_optimizer


class SpatialUnet2dTemporalDenoiser(DenoisingModel):
    def __init__(
            self,
            config: dict):
        
        super(SpatialUnet2dTemporalDenoiser, self).__init__(
            name=config['type'],
            t_order=SpatialUnet2dTemporalDenoiser.get_temporal_order_from_config(config))
        
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
            final_trans=torch.nn.Identity(),
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
            activation=activation_from_str(config['spatial_unet_activation']))
        
        self.temporal_denoiser = TemporalDenoiser(
            in_channels=self.spatial_unet.out_channels_before_readout,
            t_order=self.t_order,
            kernel_size=config['temporal_denoiser_kernel_size'],
            hidden_conv_channels=config['temporal_denoiser_conv_channels'],
            hidden_dense_layer_dims=config['temporal_denoiser_hidden_dense_layer_dims'],
            activation=activation_from_str(config['temporal_denoiser_activation']),
            final_trans=torch.nn.Identity())
    
    def forward(
            self,
            x: torch.Tensor,
            features: torch.Tensor) -> torch.Tensor:
        assert not(self.feature_mode != 'none' and (features is None))
        
        t_total = x.shape[1]
        t_tandem = t_total - self.t_order + 1

        # calculate processed features
        unet_output_list = [(
                self.spatial_unet(x[:, i_t:i_t+1, :, :], features) if self.feature_mode != 'none' else
                self.spatial_unet(x[:, i_t:i_t+1, :, :]))
            for i_t in range(t_total)]
        unet_features_nctxy = torch.stack([output['features'] for output in unet_output_list], dim=-3)
        
        # compute temporal-denoised convolutions for all t_order-length windows
        temporal_endpoint_ntxy = torch.stack([
            self.temporal_denoiser(unet_features_nctxy[:, :, i_t:(i_t + self.t_order), :, :])
            for i_t in range(t_tandem)], dim=1)
            
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
                    t_begin=i_t,
                    t_length=1,
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
                    t_begin=i_t + t_mid,
                    t_length=1,
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
            t_begin=0,
            t_length=self.t_order,
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
    def get_temporal_order_from_config(config: dict) -> int:
        return (1 +
            (config['temporal_denoiser_kernel_size'] - 1) *
            (config['temporal_denoiser_n_conv_layers']))

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
            model_config: dict,
            train_config: dict):
        super(PlSpatialUnet2dTemporalDenoiser, self).__init__()
        
        self.save_hyperparameters('model_config', 'train_config')
        
        self.neptune_run_id = None
        self.denoising_model = nn.SyncBatchNorm.convert_sync_batchnorm(SpatialUnet2dTemporalDenoiser(model_config))

    def forward(self,
                x: torch.Tensor,
                features: Optional[torch.Tensor] = None):
        return crop_center(
            self.denoising_model(x=x, features=features),
            target_width=self.hparams.train_config['x_window'],
            target_height=self.hparams.train_config['y_window'])

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        # batch is what the dataloader provides -> movie slice

        occluded_batch = self._occlude_data(batch)
        denoised_output_ntxy = self(occluded_batch['padded_diff'], batch['features'])
        loss_dict = self._compute_noise2self_loss(
            denoised_output_ntxy,
            expected_output_ntxy=occluded_batch['middle_frames'],
            occlusion_masks=occluded_batch['occlusion_masks'])

        loss = loss_dict["rec_loss"]
        self.log('train/loss', loss.item())
        if isinstance(self.logger.experiment, DummyExperiment):
            self.logger.experiment['train/loss'].log(loss.item())
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx: int = -1) -> Any:
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optim = generate_optimizer(
            denoising_model=self.denoising_model,
            optim_params=self.hparams.train_config['optim_params'],
            lr=self.hparams.train_config['lr_params']['max'])
        sched = generate_lr_scheduler(
            optim=optim,
            lr_params=self.hparams.train_config['lr_params'],
            n_iters=self.hparams.train_config['n_iters'])

        return [optim], [sched]

    def _compute_noise2self_loss(self,
            denoised_output_ntxy,
            expected_output_ntxy,
            occlusion_masks):
        """Calculates the loss of a Noise2Self predictor on a given minibatch."""

        # iterate over the middle frames and accumulate loss
        def _compute_lp_loss(_err, _norm_p, _scale=1.):
            return (_scale * (_err.abs() + const.EPS).pow(_norm_p)).sum()

        x_window, y_window = self.hparams.train_config['x_window'], self.hparams.train_config['y_window']
        total_pixels = x_window * y_window
        t_tandem = denoised_output_ntxy.shape[1]

        # reconstruction losses
        total_masked_pixels_t = occlusion_masks.sum(dim=(0, 2, 3))
        loss_scale_t = 1. / (t_tandem * (const.EPS + total_masked_pixels_t))
        loss_scale_ntxy = loss_scale_t[None, :, None, None]

        err_ntxy = occlusion_masks * (denoised_output_ntxy - expected_output_ntxy)
        rec_loss = _compute_lp_loss(_err=err_ntxy, _norm_p=self.hparams.train_config['norm_p'], _scale=loss_scale_ntxy)

        return {'rec_loss': rec_loss}

    def _occlude_data(self, batch):
        def _generate_bernoulli_mask(
                p: float,
                n_batch: int,
                t_tandem: int,
                width: int,
                height: int) -> torch.Tensor:
            return torch.distributions.Bernoulli(
                probs=torch.tensor(p, device=self.device)).sample(
                [n_batch, t_tandem, width, height]).byte()
        
        padded_diff_movie_ntxy = batch['padded_diff']
        features_nfxy = batch['features']
        trend_mean_feature_index = batch['trend_mean_feature_index']
        detrended_std_feature_index = batch['detrended_std_feature_index']

        x_window, y_window = self.hparams.train_config['x_window'], self.hparams.train_config['y_window']
        x_padding, y_padding = self.hparams.train_config['x_padding'], self.hparams.train_config['y_padding']
        occlusion_prob = self.hparams.train_config['occlusion_prob']
        padded_x_window = x_window + 2 * x_padding
        padded_y_window = y_window + 2 * y_padding
        
        tandem_start = self.denoising_model.t_order // 2
        tandem_end = padded_diff_movie_ntxy.shape[1] - tandem_start
        
        middle_frames_ntxy = padded_diff_movie_ntxy[:, tandem_start:tandem_end, x_padding:-x_padding, y_padding:-y_padding].clone()

        n_batch, t_tandem = middle_frames_ntxy.shape[:2]
        
        # generate a uniform bernoulli mask
        occlusion_masks_ntxy = _generate_bernoulli_mask(
            p=occlusion_prob,
            n_batch=n_batch,
            t_tandem=t_tandem,
            width=x_window,
            height=y_window)
        
        padded_occlusion_masks_ntxy = nn.functional.pad(occlusion_masks_ntxy, (y_padding, y_padding, x_padding, x_padding), 'constant', 0)

        padded_diff_movie_ntxy[:, tandem_start:tandem_end, :, :] = torch.where(
            padded_occlusion_masks_ntxy,
            torch.distributions.Normal(
                loc=features_nfxy[:, trend_mean_feature_index, :, :][:, None, ...].expand(
                    n_batch, t_tandem, padded_x_window, padded_y_window),
                scale=features_nfxy[:, detrended_std_feature_index, :, :][:, None, ...].expand(
                    n_batch, t_tandem, padded_x_window, padded_y_window) + const.EPS).sample(),
            padded_diff_movie_ntxy[:, tandem_start:tandem_end, :, :])

        return {
            'padded_diff': padded_diff_movie_ntxy,
            'middle_frames': middle_frames_ntxy,
            'occlusion_masks': occlusion_masks_ntxy}
    
    def on_save_checkpoint(self, checkpoint):
        if isinstance(self.logger, NeptuneLogger):
            checkpoint['neptune_run_id'] = self.logger.run._short_id

    def on_load_checkpoint(self, checkpoint):
        self.neptune_run_id = checkpoint.get('neptune_run_id', None)
