import os

import numpy as np
import torch
from typing import Optional, Union, List, Tuple
from bisect import bisect_left, bisect_right

import logging


def get_cosine_similarity_with_sequence_np(
    movie_txy: np.ndarray,
    seq_t: np.ndarray,
    t_indices: Optional[np.ndarray] = None):
    """Calculates the pixel-wise cosine similarity between the zero-mean movie and
    a given temporal sequence.    
    """
    if t_indices is not None:
        seq_t = seq_t[t_indices]
        movie_txy = movie_txy[t_indices, ...]

    return np.einsum("txy,t->xy", movie_txy, seq_t) / (
        np.linalg.norm(movie_txy, axis=0) * np.linalg.norm(seq_t))


def get_cosine_similarity_with_sequence_torch(
    movie_txy: torch.tensor,
    seq_t: torch.Tensor,
    t_indices: Optional[torch.LongTensor] = None):
    """Calculates the pixel-wise cosine similarity between the zero-mean movie and
    a given temporal sequence.    
    """
    if t_indices is not None:
        seq_t = seq_t[t_indices]
        movie_txy = movie_txy[t_indices, ...]

    return torch.einsum("txy,t->xy", movie_txy, seq_t) / (
        torch.norm(movie_txy, dim=0) * torch.norm(seq_t))


def pad_images_np(images_ncxy: np.ndarray,
                  target_width: int,
                  target_height: int,
                  pad_value_nc: Optional[np.ndarray] = None) -> np.ndarray:
    assert images_ncxy.ndim == 4
    source_width, source_height = images_ncxy.shape[2], images_ncxy.shape[3]
    assert target_width >= source_width
    assert target_height >= source_height
    assert (target_width - source_width) % 2 == 0
    assert (target_height - source_height) % 2 == 0
    
    margin_width = (target_width - source_width) // 2
    margin_height = (target_height - source_height) // 2
    if pad_value_nc is None:
        pad_value_nc = np.mean(images_ncxy, axis=(-1, -2))
    output = np.tile(pad_value_nc[..., None, None], (1, 1, target_width, target_height))
    output[:, :,
           margin_width:(source_width + margin_width),
           margin_height:(source_height + margin_height)] = images_ncxy
    return output


def pad_images_torch(images_ncxy: torch.Tensor,
                     target_width: int,
                     target_height: int,
                     pad_value_nc: Optional[torch.Tensor] = None) -> torch.Tensor:
    assert images_ncxy.ndim == 4
    source_width, source_height = images_ncxy.shape[2], images_ncxy.shape[3]
    assert target_width >= source_width
    assert target_height >= source_height
    assert (target_width - source_width) % 2 == 0
    assert (target_height - source_height) % 2 == 0
    if (source_width == target_width) and (source_height == target_height):
        return images_ncxy
    
    margin_width = (target_width - source_width) // 2
    margin_height = (target_height - source_height) // 2
    if pad_value_nc is None:
        pad_value_nc = torch.mean(images_ncxy, dim=(-1, -2))
    
    output = pad_value_nc[..., None, None].repeat(1, 1, target_width, target_height)
    output[:, :,
           margin_width:(source_width + margin_width),
           margin_height:(source_height + margin_height)] = images_ncxy
    return output

'''
Crops the last two dimensions to the specified center window.
'''
def crop_center(input_ncxy: Union[torch.Tensor, np.ndarray],
                target_width: int,
                target_height: int) -> Union[torch.Tensor, np.ndarray]:
    input_width = input_ncxy.shape[-2]
    input_height = input_ncxy.shape[-1]
    assert input_width >= target_width, f'input {input_width} smaller than target {target_width}'
    assert input_height >= target_height, f'input {input_height} smaller than target {target_height}'
    margin_width = (input_width - target_width) // 2
    margin_height = (input_height - target_height) // 2
    if (margin_width == 0) and (margin_height == 0):
        return input_ncxy
    else:
        return input_ncxy[...,
               margin_width:(target_width + margin_width),
               margin_height:(target_height + margin_height)]


def get_nn_spatio_temporal_mean(movie_ntxy: torch.Tensor, i_t: int) -> torch.Tensor:
    ONE_OVER_TWENTY_SIX = 0.03846153846
    return ONE_OVER_TWENTY_SIX * (
        movie_ntxy[..., i_t, 0:-2, 0:-2]
            + movie_ntxy[..., i_t, 0:-2, 1:-1]
            + movie_ntxy[..., i_t, 0:-2, 2:]
            + movie_ntxy[..., i_t, 1:-1, 0:-2]
            + movie_ntxy[..., i_t, 1:-1, 2:]
            + movie_ntxy[..., i_t, 2:, 0:-2]
            + movie_ntxy[..., i_t, 2:, 1:-1]
            + movie_ntxy[..., i_t, 2:, 2:]
            + movie_ntxy[..., i_t - 1, 0:-2, 0:-2]
            + movie_ntxy[..., i_t - 1, 0:-2, 1:-1]
            + movie_ntxy[..., i_t - 1, 0:-2, 2:]
            + movie_ntxy[..., i_t - 1, 1:-1, 0:-2]
            + movie_ntxy[..., i_t - 1, 1:-1, 1:-1]
            + movie_ntxy[..., i_t - 1, 1:-1, 2:]
            + movie_ntxy[..., i_t - 1, 2:, 0:-2]
            + movie_ntxy[..., i_t - 1, 2:, 1:-1]
            + movie_ntxy[..., i_t - 1, 2:, 2:]
            + movie_ntxy[..., i_t + 1, 0:-2, 0:-2]
            + movie_ntxy[..., i_t + 1, 0:-2, 1:-1]
            + movie_ntxy[..., i_t + 1, 0:-2, 2:]
            + movie_ntxy[..., i_t + 1, 1:-1, 0:-2]
            + movie_ntxy[..., i_t + 1, 1:-1, 1:-1]
            + movie_ntxy[..., i_t + 1, 1:-1, 2:]
            + movie_ntxy[..., i_t + 1, 2:, 0:-2]
            + movie_ntxy[..., i_t + 1, 2:, 1:-1]
            + movie_ntxy[..., i_t + 1, 2:, 2:])


def get_nn_spatial_mean(movie_ntxy: torch.Tensor, i_t: int) -> torch.Tensor:
    return 0.125 * (
        movie_ntxy[..., i_t, 0:-2, 0:-2]
            + movie_ntxy[..., i_t, 0:-2, 1:-1]
            + movie_ntxy[..., i_t, 0:-2, 2:]
            + movie_ntxy[..., i_t, 1:-1, 0:-2]
            + movie_ntxy[..., i_t, 1:-1, 2:]
            + movie_ntxy[..., i_t, 2:, 0:-2]
            + movie_ntxy[..., i_t, 2:, 1:-1]
            + movie_ntxy[..., i_t, 2:, 2:])


def _get_overlapping_index_range(x_list: List[float], pos: float, radius: float) -> Tuple[int, int]:
    i_left = bisect_left(x_list, pos - radius)
    i_right = bisect_right(x_list, pos + radius)
    return i_left, i_right


def _first_leq_np(arr: np.ndarray, axis: int, value: float, invalid_val=-1):
    mask = arr <= value
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def _first_leq_torch(data: torch.Tensor, dim: int, value: float):
    mask = data <= value
    return ((mask.cumsum(dim) == 1) & mask).max(dim).indices


def rolling_circle_filter_np(
        x: np.ndarray,
        y: np.ndarray,
        radius_x: float,
        radius_y: float,
        eps: float = 1e-6):
    """Rolling circle filter.
    
    Args:
        x: (N,) ndarray
        y: (..., N) ndarray
    """
    assert x.shape[-1] == y.shape[-1]
    assert y.ndim >= 1
    assert x.ndim == 1

    n_points = x.size
    batch_shape = y.shape[:-1]
    y = y.reshape(-1, n_points)
    x_list = x.tolist()
    inv_radius_x2 = 1. / (radius_x ** 2)
    inv_radius_y2 = 1. / (radius_y ** 2)
    batch_ravel = list(range(y.shape[0]))
    y_bg = np.zeros_like(y)
    for i_x in range(len(x_list)):
        pos = x_list[i_x]
        i_left, i_right = _get_overlapping_index_range(x_list, pos, radius_x)
        x_slice = x[i_left:i_right]
        y_slice = y[:, i_left:i_right]
        y_center = y_slice - radius_y * np.sqrt(
            np.maximum(0, 1. - inv_radius_x2 * (x_slice - pos) ** 2))
        dist_mat = (
            inv_radius_y2 * (y_center[:, :, None] - y_slice[:, None, :]) ** 2 +
            inv_radius_x2 * (pos - x_slice[None, None, :]) ** 2)
        inside_count = np.sum(dist_mat < 1. - eps, axis=-1)
        under_count = np.sum((y_center[:, :, None] - radius_y) > y_slice[:, None, :], axis=-1) 
        indices = _first_leq_np(arr=inside_count + under_count, axis=-1, value=0)
        y_bg[:, i_x] = y_center[batch_ravel, indices] + radius_y
    
    return y_bg.reshape(batch_shape + (n_points,))


def rolling_circle_filter_torch(
        x: torch.Tensor,
        y: torch.Tensor,
        radius_x: float,
        radius_y: float,
        eps: float = 1e-6,
        log_progress: bool = False,
        log_every: int = 100):
    """Rolling circle filter.
    
    Args:
        x: (N,) ndarray
        y: (..., N) ndarray
    """
    assert x.shape[-1] == y.shape[-1]
    assert y.ndim >= 1
    assert x.ndim == 1

    n_points = x.numel()
    batch_shape = y.shape[:-1]
    y = y.view(-1, n_points)
    x_list = x.tolist()
    inv_radius_x2 = 1. / (radius_x ** 2)
    inv_radius_y2 = 1. / (radius_y ** 2)
    batch_ravel = list(range(y.shape[0]))
    y_bg = torch.zeros_like(y)
    for i_x in range(n_points):
        if log_progress & (i_x % log_every == 0):
            logging.debug(f"processing {i_x} / {n_points} ...")
        pos = x_list[i_x]
        i_left, i_right = _get_overlapping_index_range(x_list, pos, radius_x)
        x_slice = x[i_left:i_right]
        y_slice = y[:, i_left:i_right]
        y_center = y_slice - radius_y * torch.sqrt(
            torch.clamp(1. - inv_radius_x2 * (x_slice - pos).pow(2), min=0.))
        dist_mat = (
            inv_radius_y2 * (y_center[:, :, None] - y_slice[:, None, :]).pow(2) +
            inv_radius_x2 * (pos - x_slice[None, None, :]).pow(2))
        inside_count = torch.sum(dist_mat < 1. - eps, dim=-1)
        under_count = torch.sum((y_center[:, :, None] - radius_y) > y_slice[:, None, :], dim=-1) 
        indices = _first_leq_torch(data=inside_count + under_count, dim=-1, value=0)
        y_bg[:, i_x] = y_center[batch_ravel, indices] + radius_y
    
    return y_bg.reshape(batch_shape + (n_points,))
