import torch
from torch import nn

__all__ = [
    '_CONV_DICT',
    '_CONV_TRANS_DICT',
    '_AVG_POOL_DICT',
    '_MAX_POOL_DICT',
    '_NORM_DICT',
    '_REFLECTION_PAD_DICT',
    '_CENTER_CROP_DICT',
    '_ACTIVATION_DICT',
    'activation_from_str'
]

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

_CONV_TRANS_DICT = {
    1: nn.ConvTranspose1d,
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d
}

_AVG_POOL_DICT = {
    1: nn.AvgPool1d,
    2: nn.AvgPool2d,
    3: nn.AvgPool3d
}

_MAX_POOL_DICT = {
    1: nn.MaxPool1d,
    2: nn.MaxPool2d,
    3: nn.MaxPool3d
}

_NORM_DICT = {
    'batch': {
        1: nn.BatchNorm1d,
        2: nn.BatchNorm2d,
        3: nn.BatchNorm3d
    }
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