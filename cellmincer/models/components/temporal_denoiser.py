import numpy as np
import torch
from torch import nn

from typing import List


class TemporalDenoiser(nn.Module):
    def __init__(
            self,
            in_channels: int,
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
        
    def forward(self, x):
        """
        args:
            x: (N, C, T, X, Y)
        """
        x = self.conv_block(x)
        x = self.dense_block(x)
        return self.final_trans(x[:, 0, 0, :, :])
