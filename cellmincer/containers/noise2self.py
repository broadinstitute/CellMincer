import os
import logging
import pprint
import time

import json
import pickle

import numpy as np
import torch
from typing import List, Tuple

from cellmincer import consts
from cellmincer.models import DenoisingModel, init_model
from cellmincer.util import \
    OptopatchBaseWorkspace, \
    OptopatchDenoisingWorkspace, \
    get_tagged_dir


class Noise2Self:
    def __init__(
            self,
            params: dict):
        self.params = params
        
    def load_datasets(self) -> List[OptopatchDenoisingWorkspace]:
        logging.info('Loading datasets...')
        datasets = self.params['datasets']
        dataset_dirs = [get_tagged_dir(
                name=dataset,
                config_tag=self.params['data_tag'],
                root_dir=self.params['root_data_dir'])
            for dataset in datasets]

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

            ws_denoising_list.append(
                OptopatchDenoisingWorkspace(
                    ws_base_diff=ws_base_diff,
                    ws_base_bg=ws_base_bg,
                    noise_params=noise_params,
                    features=feature_container,
                    x_padding=0,
                    y_padding=0,
                    device=self.params['device'],
                    dtype=consts.DEFAULT_DTYPE
                )
            )

        return ws_denoising_list
    
    def instance_model(
        self,
        n_global_features: int) -> DenoisingModel:
    
        self.params['model']['n_global_features'] = n_global_features
        model_dir = get_tagged_dir(
            name=self.params['model']['type'],
            config_tag=self.params['model_tag'],
            root_dir=self.params['root_model_dir'])

        if self.params['state_index'] is None:
            denoising_model = init_model(
                self.params['model'],
                device=self.params['device'],
                dtype=consts.DEFAULT_DTYPE)
        else:
            model_state_path = os.path.join(model_dir, f'{self.params["state_index"]:06d}', 'model_state.pt')
            denoising_model = init_model(
                self.params['model'],
                model_state_path=model_state_path,
                device=self.params['device'],
                dtype=consts.DEFAULT_DTYPE)
    
        return denoising_model

    def get_resources(self) -> Tuple[List[OptopatchDenoisingWorkspace], DenoisingModel]:
        ws_denoising_list = self.load_datasets()

        denoising_model = self.instance_model(
            n_global_features=ws_denoising_list[0].n_global_features)
        
        return ws_denoising_list, denoising_model