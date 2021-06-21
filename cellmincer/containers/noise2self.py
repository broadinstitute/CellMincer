import os
import logging
import pprint
import time

import json
import pickle

import numpy as np
import torch
from typing import List, Tuple, Optional

from cellmincer.models import DenoisingModel, init_model, get_best_window_padding
from cellmincer.util import \
    OptopatchBaseWorkspace, \
    OptopatchDenoisingWorkspace, \
    consts


class Noise2Self:
    def __init__(
            self,
            params: dict,
            x_padding: Optional[int] = None,
            y_padding: Optional[int] = None):
        self.params = params
        self.x_padding = x_padding
        self.y_padding = y_padding
        
    def load_datasets(self) -> List[OptopatchDenoisingWorkspace]:
        logging.info('Loading datasets...')
        datasets = self.params['datasets']
        dataset_dirs = [os.path.join(self.params['root_data_dir'], dataset) for dataset in datasets]

        assert all([os.path.exists(dataset_dir) for dataset_dir in dataset_dirs])

        ws_denoising_list = []
        for i_dataset in range(len(datasets)):
            
            base_diff_path = os.path.join(dataset_dirs[i_dataset], 'trend_subtracted.npy')
            ws_base_diff = OptopatchBaseWorkspace.from_npy(base_diff_path)

            base_bg_path = os.path.join(dataset_dirs[i_dataset], 'trend.npy')
            ws_base_bg = OptopatchBaseWorkspace.from_npy(base_bg_path)

            opto_noise_params_path = os.path.join(dataset_dirs[i_dataset], 'noise_params.json')
            with open(opto_noise_params_path, 'r') as f:
                noise_params = json.load(f)

            opto_feature_path = os.path.join(dataset_dirs[i_dataset], 'features.pkl')
            with open(opto_feature_path, 'rb') as f:
                feature_container = pickle.Unpickler(f).load()

            if self.x_padding is None:
                _, x_padding = get_best_window_padding(
                    model_config=self.params['model'],
                    output_min_size_lo=ws_base_diff.width,
                    output_min_size_hi=ws_base_diff.width)
            else:
                x_padding = self.x_padding

            if self.y_padding is None:
                _, y_padding = get_best_window_padding(
                    model_config=self.params['model'],
                    output_min_size_lo=ws_base_diff.height,
                    output_min_size_hi=ws_base_diff.height)
            else:
                y_padding = self.y_padding

            ws_denoising_list.append(
                OptopatchDenoisingWorkspace(
                    ws_base_diff=ws_base_diff,
                    ws_base_bg=ws_base_bg,
                    noise_params=noise_params,
                    features=feature_container,
                    x_padding=x_padding,
                    y_padding=y_padding,
                    device=self.params['device'],
                    dtype=consts.DEFAULT_DTYPE
                )
            )

        return ws_denoising_list
    
    def instance_model(
        self,
        n_global_features: int) -> DenoisingModel:
    
        self.params['model']['n_global_features'] = n_global_features
        model_dir = os.path.join(self.params['root_model_dir'], self.params['model']['type'])

        if 'state_index' in self.params:
            model_state_path = os.path.join(model_dir, f'{self.params["state_index"]:06d}', 'model_state.pt')
            denoising_model = init_model(
                self.params['model'],
                model_state_path=model_state_path,
                device=self.params['device'],
                dtype=consts.DEFAULT_DTYPE)
        else:
            denoising_model = init_model(
                self.params['model'],
                device=self.params['device'],
                dtype=consts.DEFAULT_DTYPE)

        return denoising_model

    def get_resources(self) -> Tuple[List[OptopatchDenoisingWorkspace], DenoisingModel]:
        ws_denoising_list = self.load_datasets()

        denoising_model = self.instance_model(
            n_global_features=ws_denoising_list[0].n_global_features)
        
        return ws_denoising_list, denoising_model
