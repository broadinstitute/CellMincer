import math
import numpy as np
import torch
from torch import nn

from typing import List, Optional, Tuple

from .functions import *

from .blocks import \
    UNetConvBlock, \
    UNetUpBlock

class GUNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            data_dim: int,
            feature_channels: int,
            noise_channels: int,
            depth: int,
            first_conv_channels: int,
            ch_growth_rate: float,
            ds_rate: int,
            final_trans: nn.Module,
            out_channels_before_readout: Optional[int] = None,
            pad: bool = False,
            layer_norm: bool = True,
            norm_mode: str = 'batch',
            attention: bool = True,
            feature_mode: str = 'repeat',
            up_mode: str = 'upconv',
            pool_mode: str = 'max',
            kernel_size: int = 3,
            n_conv_layers: int = 2,
            p_dropout: float = 0.0,
            readout_hidden_layer_channels_list: List[int] = [],
            readout_kernel_size: int = 1,
            activation: nn.Module = nn.SELU()):
        super(GUNet, self).__init__()
        
        assert feature_mode in {'repeat', 'once', 'none'}
        assert up_mode in {'upconv', 'upsample'}
        assert pool_mode in {'max', 'avg'}
        
        # if using features only at entry, bundle feature channels into entry point
        if feature_mode == 'once':
            in_channels += feature_channels
            
        # if not using repeat mode, silence repeat features
        if feature_mode != 'repeat':
            feature_channels = 0
        
        if pad:
            assert readout_kernel_size % 2 == 1
        pad_flag = int(pad)
        
        if out_channels_before_readout is None:
            out_channels_before_readout = first_conv_channels
        
        self.data_dim = data_dim
        self.kernel_size = kernel_size
        self.n_conv_layers = n_conv_layers
        self.depth = depth
        self.ds_rate = ds_rate
        self.noise_channels = noise_channels
        self.out_channels_before_readout = out_channels_before_readout
        self.center_crop = _CENTER_CROP_DICT[data_dim]
        if pool_mode == 'max':
            self.pool = _MAX_POOL_DICT[data_dim](kernel_size=ds_rate)
        elif pool_mode == 'avg':
            self.pool = _AVG_POOL_DICT[data_dim](kernel_size=ds_rate)
        self.drop = nn.Dropout2d(p=p_dropout)
        
        # downward path
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(
                    in_size=prev_channels + feature_channels,
                    out_size=math.ceil(first_conv_channels * ch_growth_rate ** i),
                    pad=pad,
                    layer_norm=layer_norm,
                    norm_mode=norm_mode,
                    data_dim=data_dim,
                    kernel_size=kernel_size,
                    n_conv_layers=n_conv_layers,
                    activation=activation))
            prev_channels = math.ceil(first_conv_channels * ch_growth_rate ** i)

        # upward path
        self.up_path = nn.ModuleList()        
        for i in reversed(range(depth - 1)):
            up_in_channels = prev_channels + noise_channels
            up_bridge_channels = math.ceil(first_conv_channels * ch_growth_rate ** i) + feature_channels
            up_mid_channels = math.ceil(first_conv_channels * ch_growth_rate ** i)
            if i > 0:
                up_out_channels = math.ceil(first_conv_channels * ch_growth_rate ** i)
            else:
                up_out_channels = out_channels_before_readout
            
            self.up_path.append(
                UNetUpBlock(
                    in_size=up_in_channels,
                    mid_size=up_mid_channels,
                    bridge_size=up_bridge_channels,
                    out_size=up_out_channels,
                    up_mode=up_mode,
                    ds_rate=ds_rate,
                    attention=attention,
                    pad=pad,
                    layer_norm=layer_norm,
                    norm_mode=norm_mode,
                    data_dim=data_dim,
                    kernel_size=kernel_size,
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
            if layer_norm:
                readout.append(_NORM_DICT[norm_mode][data_dim](hidden_channels))
            if pad_flag:
                readout.append(_REFLECTION_PAD_DICT[data_dim]((readout_kernel_size - 1) // 2))                    
            prev_channels = hidden_channels
        readout.append(
            _CONV_DICT[data_dim](
                in_channels=prev_channels,
                out_channels=out_channels,
                kernel_size=readout_kernel_size))
        if pad_flag:
            readout.append(_REFLECTION_PAD_DICT[data_dim]((readout_kernel_size - 1) // 2))
        self.readout = nn.Sequential(*readout)
        self.final_trans = final_trans
        
        if feature_channels == 0:
            self.forward = self._forward
        else:
            self.forward = self._forward_w_repeat_features

    def _forward_w_repeat_features(self, x: torch.Tensor, features: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        
        # if no persistent feature is provided, use input as features
        if features is None:
            features = x

        features_list = []
        block_list = []
        for i, down_op in enumerate(self.down_path):
            x = torch.cat([features, x], - self.data_dim - 1)
            x = down_op(x)
            if i != len(self.down_path) - 1:
                features = self.center_crop(features, x)
                features_list.append(features)
                block_list.append(x)
                features = self.pool(features)
                x = self.drop(self.pool(x))
            
        for i, up_op in enumerate(self.up_path):
            bridge = self.drop(torch.cat([features_list[-i - 1], block_list[-i - 1]], - self.data_dim - 1))
            if self.noise_channels > 0:
                noise_shape = (x.shape[0], self.noise_channels,) + x.shape[-self.data_dim:]
                noise = torch.randn(noise_shape)
                x = torch.cat([noise, x], - self.data_dim - 1)
            x = up_op(x, bridge)
            
        return {
            'features': x,
            'readout': self.final_trans(self.readout(x))
        }
    
    def _forward(self, x: torch.Tensor, features: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        if features is not None:
            x = torch.cat([features, x], - self.data_dim - 1)
        
        block_list = []
        for i, down_op in enumerate(self.down_path):            
            x = down_op(x)
            if i != len(self.down_path) - 1:
                block_list.append(x)
                x = self.pool(x)
            
        for i, up_op in enumerate(self.up_path):
            bridge = block_list[-i - 1]
            if self.noise_channels > 0:
                noise_shape = (x.shape[0], self.noise_channels,) + x.shape[-self.data_dim:]
                noise = torch.randn(noise_shape)
                x = torch.cat([noise, x], - self.data_dim - 1)
            x = up_op(x, bridge)
            
        return {
            'features': x,
            'readout': self.final_trans(self.readout(x))
        }

    
def get_unet_input_size(
        output_min_size: int,
        kernel_size: int,
        n_conv_layers: int,
        depth: int,
        ds_rate: int):
    """Smallest input size for output size >= `output_min_size`.
    
    .. note:
        The calculated input sizes guarantee that all of the layers have even dimensions.
        This is important to prevent aliasing in downsampling (pooling) operations.
    
    """
    delta = n_conv_layers * (kernel_size - 1)
    size = np.array(output_min_size)
    
    for i in range(depth - 1):
        size = np.ceil((size + delta) / ds_rate)
    
    input_size = np.array(size, dtype=int) + delta
    
    for i in range(depth - 1):
        input_size = input_size * ds_rate + delta
    
    return input_size
