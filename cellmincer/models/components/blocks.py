import torch
from torch import nn
from .functions import *

class UNetConvBlock(torch.nn.Module):
    def __init__(
            self,
            in_size: int,
            out_size: int,
            pad: bool,
            layer_norm: bool,
            norm_mode: str,
            data_dim: int,
            kernel_size: int,
            n_conv_layers: int,
            activation: torch.nn.Module):
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
        if layer_norm:
            block.append(_LAYER_NORM_DICT[norm_mode][data_dim](out_size))
        if pad_flag:
            block.append(_REFLECTION_PAD_DICT[data_dim]((kernel_size - 1) // 2))
        for _ in range(n_conv_layers - 1):
            block.append(_CONV_DICT[data_dim](
                out_size,
                out_size,
                kernel_size=kernel_size))
            block.append(activation)
            if layer_norm:
                block.append(_LAYER_NORM_DICT[norm_mode][data_dim](out_size))
            if pad_flag:
                block.append(_REFLECTION_PAD_DICT[data_dim]((kernel_size - 1) // 2))

        self.block = torch.nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return out


class UNetAttentionBlock(torch.nn.Module):
    def __init__(
            self,
            operator_size: int,
            operand_size: int,
            interact_size: int,
            data_dim: int,
            layer_norm: bool,
            norm_mode: str,
            activation: torch.nn.Module):
        super(UNetAttentionBlock, self).__init__()
        
        self.data_dim = data_dim
        self.activation = activation
        
        w_operator_blocks = []
        w_operator_blocks.append(
            _CONV_DICT[data_dim](in_channels=operator_size, out_channels=interact_size, kernel_size=1))
        if layer_norm:
            w_operator_blocks.append(
                _LAYER_NORM_DICT[norm_mode][data_dim](interact_size))
        self.w_operator = torch.nn.Sequential(*w_operator_blocks)
        
        w_operand_blocks = []
        w_operand_blocks.append(
            _CONV_DICT[data_dim](in_channels=operand_size, out_channels=interact_size, kernel_size=1))
        if layer_norm:
            w_operand_blocks.append(
                _LAYER_NORM_DICT[norm_mode][data_dim](interact_size))
        self.w_operand = torch.nn.Sequential(*w_operand_blocks)

        att_map_blocks = []
        att_map_blocks.append(
             _CONV_DICT[data_dim](in_channels=interact_size, out_channels=1, kernel_size=1))
        if layer_norm:
            att_map_blocks.append(
                _LAYER_NORM_DICT[norm_mode][data_dim](1))
        att_map_blocks.append(torch.nn.Sigmoid())
        self.att_map = torch.nn.Sequential(*att_map_blocks)
        
    def forward(self, operator: torch.Tensor, operand: torch.Tensor) -> torch.Tensor:
        operator_features = self.w_operator(operator)
        operand_features = self.w_operand(operand)
        att_map = self.att_map(self.activation(operator_features + operand_features))
        return att_map * operand


class UNetUpBlock(torch.nn.Module):
    def __init__(
            self,
            in_size: int,
            mid_size: int,
            bridge_size: int,
            out_size: int,
            ds_rate: int,
            attention: bool,
            up_mode: 'str',
            data_dim: int,
            pad: bool,
            layer_norm: bool,
            norm_mode: str,
            kernel_size: int,
            n_conv_layers: int,
            activation: torch.nn.Module):
        super(UNetUpBlock, self).__init__()
        
        # upsampling
        if up_mode == 'upconv':
            self.up = _CONV_TRANS_DICT[data_dim](in_size, mid_size, kernel_size=ds_rate, stride=ds_rate)
        elif up_mode == 'upsample':
            self.up = torch.nn.Sequential(
                torch.nn.Upsample(mode='bilinear', scale_factor=ds_rate, align_corners=True),
                _CONV_DICT[data_dim](in_size, mid_size, kernel_size=1))
        
        self.attention = attention
        self.data_dim = data_dim
        self.center_crop = _CENTER_CROP_DICT[data_dim]
        
        self.conv_block = UNetConvBlock(
            in_size=mid_size + bridge_size,
            out_size=out_size,
            pad=pad,
            layer_norm=layer_norm,
            norm_mode=norm_mode,
            data_dim=data_dim,
            kernel_size=kernel_size,
            n_conv_layers=n_conv_layers,
            activation=activation)
        
        if attention:
            self.attention_block = UNetAttentionBlock(
                operator_size=mid_size,
                operand_size=bridge_size,
                interact_size=2 * mid_size,
                data_dim=data_dim,
                layer_norm=layer_norm,
                norm_mode=norm_mode,
                activation=activation)
        
    def forward(self, x: torch.Tensor, bridge: torch.Tensor) -> torch.Tensor:
        up = self.up(x)
        bridge = self.center_crop(bridge, up)
        if self.attention:
            bridge = self.attention_block(up, bridge)
        out = torch.cat([up, bridge], - self.data_dim - 1)
        out = self.conv_block(out)
        return out
