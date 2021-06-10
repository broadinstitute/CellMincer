import numpy as np
import torch
from torch import nn
from typing import List, Tuple, Optional, Union, Dict

from cellmincer.util.utils import crop_center
from cellmincer.util.ws import OptopatchBaseWorkspace, OptopatchDenoisingWorkspace


def center_crop_1d(layer: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _, _, layer_width = layer.size()
    _, _, target_width = target.size()
    assert layer_width >= target_width
    diff_x = (layer_width - target_width) // 2
    return layer[:, :,
        diff_x:(diff_x + target_width)]


def center_crop_2d(layer: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _, _, layer_height, layer_width = layer.size()
    _, _, target_height, target_width = target.size()
    assert layer_height >= target_height
    assert layer_width >= target_width
    diff_x = (layer_width - target_width) // 2
    diff_y = (layer_height - target_height) // 2
    return layer[:, :,
        diff_y:(diff_y + target_height),
        diff_x:(diff_x + target_width)]


def center_crop_3d(layer: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _, _, layer_depth, layer_height, layer_width = layer.size()
    _, _, target_depth, target_height, target_width = layer.size()
    assert layer_depth >= target_depth
    assert layer_height >= target_height
    assert layer_width >= target_width
    diff_x = (layer_width - target_width) // 2
    diff_y = (layer_height - target_height) // 2
    diff_z = (layer_depth - target_depth) // 2
    return layer[:, :,
        diff_z:(diff_z + target_depth),
        diff_y:(diff_y + target_height),
        diff_x:(diff_x + target_width)]


_CONV_DICT = {
    1: nn.Conv1d,
    2: nn.Conv2d,
    3: nn.Conv3d
}

_BATCH_NORM_DICT = {
    1: nn.BatchNorm1d,
    2: nn.BatchNorm2d,
    3: nn.BatchNorm3d
}

_CONV_TRANS_DICT = {
    1: nn.ConvTranspose1d,
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d
}

_AVG_POOL_DICT = {
    1: nn.functional.avg_pool1d,
    2: nn.functional.avg_pool2d,
    3: nn.functional.avg_pool3d
}

_MAX_POOL_DICT = {
    1: nn.functional.max_pool1d,
    2: nn.functional.max_pool2d,
    3: nn.functional.max_pool3d
}

_REFLECTION_PAD_DICT = {
    1: nn.ReflectionPad1d,
    2: nn.ReflectionPad2d
}

_CENTER_CROP_DICT = {
    1: center_crop_1d,
    2: center_crop_2d,
    3: center_crop_3d
}

_ACTIVATION_DICT = {
    'relu': nn.ReLU(),
    'elu': nn.ELU(),
    'selu': nn.SELU(),
    'sigmoid': nn.Sigmoid(),
    'leaky_relu': nn.LeakyReLU(),
    'softplus': nn.Softplus()
}


def activation_from_str(activation_str: str):
    return _ACTIVATION_DICT[activation_str]


class UNetConvBlock(nn.Module):
    def __init__(
            self,
            in_size: int,
            out_size: int,
            pad: bool,
            batch_norm: bool,
            data_dim: int,
            kernel_size: int,
            n_conv_layers: int,
            activation: nn.Module):
        super(UNetConvBlock, self).__init__()
        block = []
        
        assert n_conv_layers >= 1
        if pad:
            assert kernel_size % 2 == 1
        pad_flag = int(pad)
        
        block.append(_CONV_DICT[data_dim](
            in_size,
            out_size,
            kernel_size=kernel_size))
        block.append(activation)
        if batch_norm:
            block.append(_BATCH_NORM_DICT[data_dim](out_size))
        block.append(_REFLECTION_PAD_DICT[data_dim](pad_flag * (kernel_size - 1) // 2))
        for _ in range(n_conv_layers - 1):
            block.append(_CONV_DICT[data_dim](
                out_size,
                out_size,
                kernel_size=kernel_size))
            block.append(activation)
            if batch_norm:
                block.append(_BATCH_NORM_DICT[data_dim](out_size))
            block.append(_REFLECTION_PAD_DICT[data_dim](pad_flag * (kernel_size - 1) // 2))

        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(
            self,
            in_size: int,
            mid_size: int,
            bridge_size: int,
            out_size: int,
            up_mode: 'str',
            data_dim: int,
            pad: bool,
            batch_norm: bool,
            kernel_size: int,
            n_conv_layers: int,
            activation: nn.Module,
            scale_factor: int = 2):
        super(UNetUpBlock, self).__init__()
        
        # upsampling
        if up_mode == 'upconv':
            self.up = _CONV_TRANS_DICT[data_dim](in_size, mid_size, kernel_size=scale_factor, stride=scale_factor)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=scale_factor2),
                _CONV_DICT[data_dim](in_size, mid_size, kernel_size=1))
        
        self.data_dim = data_dim
        self.center_crop = _CENTER_CROP_DICT[data_dim]
        
        self.conv_block = UNetConvBlock(
            in_size=mid_size + bridge_size,
            out_size=out_size,
            pad=pad,
            batch_norm=batch_norm,
            data_dim=data_dim,
            kernel_size=kernel_size,
            n_conv_layers=n_conv_layers,
            activation=activation)
        
        self.use_bridge = bridge_size > 0
        
    def forward(self, x: torch.Tensor, bridge: torch.Tensor) -> torch.Tensor:
        up = self.up(x)
        
        if self.use_bridge:
            cropped_bridge = self.center_crop(bridge, up)
            out = torch.cat([up, cropped_bridge], - self.data_dim - 1)
            out = self.conv_block(out)
        else:
            out = self.conv_block(up)
            
        return out

    
class ConditionalUNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            data_dim: int,
            feature_channels: int,
            depth: int,
            wf: int,
            out_channels_before_readout: int,
            final_trans: nn.Module,
            pad: bool = False,
            batch_norm: bool = False,
            up_mode: str = 'upconv',
            unet_kernel_size: int = 3,
            n_conv_layers: int = 2,
            readout_hidden_layer_channels_list: List[int] = [],
            readout_kernel_size: int = 1,
            activation: nn.Module = nn.SELU(),
            device: torch.device = torch.device('cuda'),
            dtype: torch.dtype = torch.float32):
        super(ConditionalUNet, self).__init__()
        
        assert up_mode in ('upconv', 'upsample')
        
        if pad:
            assert readout_kernel_size % 2 == 1
        pad_flag = int(pad)
        
        self.data_dim = data_dim
        self.center_crop = _CENTER_CROP_DICT[data_dim]
        self.max_pool = _MAX_POOL_DICT[data_dim]
        self.avg_pool = _AVG_POOL_DICT[data_dim]
        
        # downward path
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(
                    in_size=prev_channels + feature_channels,
                    out_size=2 ** (wf + i),
                    pad=pad,
                    batch_norm=batch_norm,
                    data_dim=data_dim,
                    kernel_size=unet_kernel_size,
                    n_conv_layers=n_conv_layers,
                    activation=activation))
            prev_channels = 2 ** (wf + i)

        # upward path
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            up_in_channels = prev_channels
            up_bridge_channels = 2 ** (wf + i) + feature_channels
            up_mid_channels = 2 ** (wf + i)
            if i > 0:
                up_out_channels = 2 ** (wf + i)
            else:
                up_out_channels = out_channels_before_readout
            self.up_path.append(
                UNetUpBlock(
                    in_size=up_in_channels,
                    mid_size=up_mid_channels,
                    bridge_size=up_bridge_channels,
                    out_size=up_out_channels,
                    up_mode=up_mode,
                    pad=pad,
                    batch_norm=batch_norm,
                    data_dim=data_dim,
                    kernel_size=unet_kernel_size,
                    n_conv_layers=n_conv_layers,
                    activation=activation))
            prev_channels = up_out_channels

        # final readout
        readout = []
        for hidden_channels in readout_hidden_layer_channels_list:
            readout.append(
                _CONV_DICT[data_dim](
                    in_channels=prev_channels,
                    out_channels=hidden_channels,
                    kernel_size=readout_kernel_size))
            readout.append(activation)
            if batch_norm:
                readout.append(_BATCH_NORM_DICT[data_dim](hidden_channels))
            readout.append(_REFLECTION_PAD_DICT[data_dim](pad_flag * (readout_kernel_size - 1) // 2))                    
            prev_channels = hidden_channels
        readout.append(
            _CONV_DICT[data_dim](
                in_channels=prev_channels,
                out_channels=out_channels,
                kernel_size=readout_kernel_size))
        readout.append(_REFLECTION_PAD_DICT[data_dim](pad_flag * (readout_kernel_size - 1) // 2))
        self.readout = nn.Sequential(*readout)
        self.final_trans = final_trans
        
        # send to device
        self.to(device)
        
        if feature_channels == 0:
            self.forward = self._forward_wo_features
        else:
            self.forward = self._forward_w_features

    def _forward_w_features(self, x: torch.Tensor, features: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        features_list = []
        block_list = []
        for i, down_op in enumerate(self.down_path):            
            x = torch.cat([features, x], - self.data_dim - 1)
            x = down_op(x)
            if i != len(self.down_path) - 1:
                features = self.center_crop(features, x)
                features_list.append(features)
                block_list.append(x)
                features = self.avg_pool(features, 2)
                x = self.max_pool(x, 2)
            
        for i, up_op in enumerate(self.up_path):
            bridge = torch.cat([features_list[-i - 1], block_list[-i - 1]], - self.data_dim - 1)
            x = up_op(x, bridge)
            
        return {
            'features_ncxy': x,
            'readout_ncxy': self.final_trans(self.readout(x))
        }
    
    def _forward_wo_features(self, x: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        block_list = []
        for i, down_op in enumerate(self.down_path):
            x = down_op(x)
            if i != len(self.down_path) - 1:
                block_list.append(x)
                x = self.max_pool(x, 2)
            
        for i, up_op in enumerate(self.up_path):
            bridge = block_list[-i - 1]
            x = up_op(x, bridge)
            
        return {
            'features_ncxy': x,
            'readout_ncxy': self.final_trans(self.readout(x))
        }

    
class TemporalDenoiser(nn.Module):
    def __init__(
            self,
            in_channels: int,
            feature_channels: int,
            t_order: int,
            kernel_size: int,
            hidden_conv_channels: int,
            hidden_dense_layer_dims: List[int],
            activation: nn.Module,
            final_trans: nn.Module,
            device: torch.device,
            dtype: torch.dtype):
        super(TemporalDenoiser, self).__init__()
        
        assert t_order % 2 == 1
        assert (t_order - 1) % (kernel_size - 1) == 0
        
        n_conv_layers = (t_order - 1) // (kernel_size - 1)
        
        conv_blocks = []
        prev_channels = in_channels
        for _ in range(n_conv_layers):
            conv_blocks.append(
                nn.Conv3d(
                    in_channels=prev_channels,
                    out_channels=hidden_conv_channels,
                    kernel_size=(kernel_size, 1, 1)))
            conv_blocks.append(activation)
            prev_channels = hidden_conv_channels
            
        dense_blocks = []
        for hidden_dim in hidden_dense_layer_dims:
            dense_blocks.append(
                nn.Conv3d(
                    in_channels=prev_channels,
                    out_channels=hidden_dim,
                    kernel_size=(1, 1, 1)))
            dense_blocks.append(activation)
            prev_dim = hidden_dim
        dense_blocks.append(
            nn.Conv3d(
                in_channels=prev_dim,
                out_channels=1,
                kernel_size=(1, 1, 1)))
        
        self.conv_block = nn.Sequential(*conv_blocks)
        self.dense_block = nn.Sequential(*dense_blocks)
        self.final_trans = final_trans
        
        self.to(device)
        
    def forward(self, x, features):
        """
        args:
            x: (N, C, T, X, Y)
            features: (N, C, X, Y)
        """
        x = self.conv_block(x)
        x = self.dense_block(x)
        return self.final_trans(x[:, 0, 0, :, :])


class AttentionCNN(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            data_dim: int,
            feature_channels: int,
            depth: int,
            wf: int,
            out_channels_before_readout: int,
            final_trans: nn.Module,
            pad: bool = False,
            batch_norm: bool = False,
            up_mode: str = 'upconv',
            unet_kernel_size: int = 3,
            n_conv_layers: int = 2,
            readout_hidden_layer_channels_list: List[int] = [],
            readout_kernel_size: int = 1,
            activation: nn.Module = nn.SELU(),
            device: torch.device = torch.device('cuda'),
            dtype: torch.dtype = torch.float32):
        super(ConditionalUNet, self).__init__()
        
        assert up_mode in ('upconv', 'upsample')
        
        if pad:
            assert readout_kernel_size % 2 == 1
        pad_flag = int(pad)
        
        self.data_dim = data_dim
        self.center_crop = _CENTER_CROP_DICT[data_dim]
        self.max_pool = _MAX_POOL_DICT[data_dim]
        self.avg_pool = _AVG_POOL_DICT[data_dim]
        
        # downward path
        prev_channels = in_channels
        self.down_paths = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(
                    in_size=prev_channels + feature_channels,
                    out_size=2 ** (wf + i),
                    pad=pad,
                    batch_norm=batch_norm,
                    data_dim=data_dim,
                    kernel_size=unet_kernel_size,
                    n_conv_layers=n_conv_layers,
                    activation=activation))
            prev_channels = 2 ** (wf + i)

        # upward blocks to revert to full resolution
        self.up_blocks = nn.ModuleList()
        for i in range(depth - 1):
            up_in_channels = 2 ** (wf + i)
            up_mid_channels = 2 ** (wf + i)
            if i > 0:
                up_out_channels = 2 ** (wf + i)
            else:
                up_out_channels = out_channels_before_readout
            self.up_path.append(
                UNetUpBlock(
                    in_size=up_in_channels,
                    mid_size=up_mid_channels,
                    bridge_size=0,
                    out_size=up_out_channels,
                    up_mode=up_mode,
                    pad=pad,
                    batch_norm=batch_norm,
                    data_dim=data_dim,
                    kernel_size=unet_kernel_size,
                    n_conv_layers=n_conv_layers,
                    activation=activation))
            prev_channels = up_out_channels

        # final readout
        readout = []
        for hidden_channels in readout_hidden_layer_channels_list:
            readout.append(
                _CONV_DICT[data_dim](
                    in_channels=prev_channels,
                    out_channels=hidden_channels,
                    kernel_size=readout_kernel_size))
            readout.append(activation)
            if batch_norm:
                readout.append(_BATCH_NORM_DICT[data_dim](hidden_channels))
            readout.append(_REFLECTION_PAD_DICT[data_dim](pad_flag * (readout_kernel_size - 1) // 2))                    
            prev_channels = hidden_channels
        readout.append(
            _CONV_DICT[data_dim](
                in_channels=prev_channels,
                out_channels=out_channels,
                kernel_size=readout_kernel_size))
        readout.append(_REFLECTION_PAD_DICT[data_dim](pad_flag * (readout_kernel_size - 1) // 2))
        self.readout = nn.Sequential(*readout)
        self.final_trans = final_trans
        
        # send to device
        self.to(device)
        
        if feature_channels == 0:
            self.forward = self._forward_wo_features
        else:
            self.forward = self._forward_w_features
