import os
import logging
import pprint
import time

import json
import pickle

import numpy as np
import torch
from typing import List, Optional, Tuple, Union

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from cellmincer.models import \
    DenoisingModel, \
    init_model, \
    get_temporal_order_from_config, \
    get_window_padding_from_config

from cellmincer.util import \
    OptopatchBaseWorkspace, \
    OptopatchDenoisingWorkspace, \
    const

import warnings

# suppress DataLoader pthreadpool warning
warnings.filterwarnings('ignore')

def movie_collate_fn(data):
    return {
        'padded_diff': torch.cat([item['padded_diff'] for item in data], dim=0),
        'features': torch.cat([item['features'] for item in data], dim=0),
        'trend_mean_feature_index': data[0]['trend_mean_feature_index'],
        'detrended_std_feature_index': data[0]['detrended_std_feature_index']}

class MovieDataset(Dataset):
    def __init__(
            self,
            ws_denoising_list: List[OptopatchDenoisingWorkspace],
            x_window: int,
            y_window: int,
            x_padding: int,
            y_padding: int,
            t_total: int,
            length: int):
        self.ws_denoising_list = ws_denoising_list
        self.x_window = x_window
        self.y_window = y_window
        self.x_padding = x_padding
        self.y_padding = y_padding
        self.t_total = t_total
        self.length = length

    def __len__(self):
        return self.length
        
    def __getitem__(self, item):
        movie_idx = item % len(self.ws_denoising_list)
        return self._generate_random_slice(movie_idx)

    def _generate_random_slice(self, movie_idx):
        n_frames, width, height = self.ws_denoising_list[movie_idx].shape
        t0 = np.random.randint(n_frames - self.t_total + 1)
        x0 = np.random.randint(width - self.x_window + 1)
        y0 = np.random.randint(height - self.y_window + 1)
        diff_movie_slice_1txy = self.ws_denoising_list[movie_idx].get_movie_slice(
            include_bg=False,
            t_begin=t0,
            t_length=self.t_total,
            x0=x0,
            y0=y0,
            x_window=self.x_window,
            y_window=self.y_window,
            x_padding=self.x_padding,
            y_padding=self.y_padding)['diff']
        feature_slice_list_1fxy = self.ws_denoising_list[movie_idx].get_feature_slice(
            x0=x0,
            y0=y0,
            x_window=self.x_window,
            y_window=self.y_window,
            x_padding=self.x_padding,
            y_padding=self.y_padding)
        trend_mean_feature_index = self.ws_denoising_list[movie_idx].cached_features.get_feature_index('trend_mean_0')
        detrended_std_feature_index = self.ws_denoising_list[movie_idx].cached_features.get_feature_index('detrended_std_0')
        
        return {
            'padded_diff': diff_movie_slice_1txy,
            'features': feature_slice_list_1fxy,
            'trend_mean_feature_index': trend_mean_feature_index,
            'detrended_std_feature_index': detrended_std_feature_index}

class MovieDataModule(LightningDataModule):
    def __init__(
            self,
            ws_denoising_list: List[OptopatchDenoisingWorkspace],
            x_window: int,
            y_window: int,
            x_padding: int,
            y_padding: int,
            t_total: int,
            n_batch: int,
            length: int):
        # self.ws_denoising_list = ws_denoising_list
        # self.x_window = x_window
        # self.y_window = y_window
        # self.x_padding = x_padding
        # self.y_padding = y_padding
        # self.t_total = t_total
        self.n_batch = n_batch
        # self.length = length
        self.dataset = MovieDataset(
            ws_denoising_list=ws_denoising_list,
            x_window=x_window,
            y_window=y_window,
            x_padding=x_padding,
            y_padding=y_padding,
            t_total=t_total,
            length=length)
        self.n_global_features = ws_denoising_list[0].n_global_features

    def prepare_data(self):
        pass
    
    def setup(self, stage: Optional[str] = None):
        pass
        # self.dataset = MovieDataset(
        #     ws_denoising_list=self.ws_denoising_list,
        #     x_window=self.x_window,
        #     y_window=self.y_window,
        #     x_padding=self.x_padding,
        #     y_padding=self.y_padding,
        #     t_total=self.t_total,
        #     length=self.length)

    def train_dataloader(self) -> "torch.dataloader":
        return DataLoader(
            dataset=self.dataset,
            collate_fn=movie_collate_fn,
            batch_size=self.n_batch,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            shuffle=True)

    def val_dataloader(self) -> "torch.dataloader":
        pass

def build_ws_denoising(
        dataset: str,
        model_config: dict,
        use_memmap: bool,
        device: Optional[torch.device] = None):
    movie_diff = np.load(os.path.join(dataset, 'trend_subtracted.npy'))
    movie_bg_path = os.path.join(dataset, 'trend.npy')

    opto_noise_params_path = os.path.join(dataset, 'noise_params.json')
    with open(opto_noise_params_path, 'r') as f:
        noise_params = json.load(f)

    opto_feature_path = os.path.join(dataset, 'features.pkl')
    with open(opto_feature_path, 'rb') as f:
        feature_container = pickle.Unpickler(f).load()

    padding = max(get_window_padding_from_config(
        model_config=model_config,
        output_min_size=np.arange(1, max(movie_diff.shape[-2:]) + 1)))
    
    return OptopatchDenoisingWorkspace(
        movie_diff=movie_diff,
        movie_bg_path=movie_bg_path,
        noise_params=noise_params,
        features=feature_container,
        x_padding=padding,
        y_padding=padding,
        use_memmap=use_memmap,
        clip=model_config.get('clip', 0),
        padding_mode=model_config['padding_mode'],
        occlude_padding=model_config['occlude_padding'],
        device=device)

def build_datamodule(
        datasets: List[str],
        model_config: dict,
        train_config: dict,
        gpus: int,
        use_memmap: bool) -> MovieDataModule:
    logging.info('Loading datasets...')
    assert all([os.path.exists(dataset) for dataset in datasets])

    ws_denoising_list = []
    for i_dataset, dataset in enumerate(datasets):
        logging.info(f'({i_dataset + 1}/{len(datasets)}) {dataset}')
        ws_denoising_list.append(build_ws_denoising(dataset, model_config, use_memmap))

    return MovieDataModule(
        ws_denoising_list,
        x_window=train_config['x_window'],
        y_window=train_config['y_window'],
        x_padding=train_config['x_padding'],
        y_padding=train_config['y_padding'],
        t_total=get_temporal_order_from_config(model_config) + train_config['t_tandem'] - 1,
        n_batch=train_config['n_batch'],
        length=gpus * train_config['n_batch'])
