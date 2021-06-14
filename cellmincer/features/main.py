import os
import yaml
import logging
import pprint
import time
import math

import numpy as np
import torch

from cellmincer import consts
from cellmincer.util import OptopatchBaseWorkspace, OptopatchGlobalFeatureExtractor, get_tagged_dir

class Features:
    def __init__(
            self,
            params: dict):
        self.params = params

    def run(self):
        device = torch.device(self.params['device'])
    
        # load datasets
        datasets = self.params['datasets']
        dataset_dirs = [get_tagged_dir(
                name=dataset,
                config_tag=self.params['tag'],
                root_dir=self.params['root_data_dir'])
            for dataset in datasets]

        assert all([os.path.exists(dataset_dir) for dataset_dir in dataset_dirs])

        for i_dataset in range(len(datasets)):
            logging.info(f'({i_dataset + 1}/{len(datasets)}) {datasets[i_dataset]}')
            
            ws_base = OptopatchBaseWorkspace.from_npy(os.path.join(dataset_dirs[i_dataset], 'trend_subtracted.npy'))

            feature_extractor = OptopatchGlobalFeatureExtractor(
                ws_base=ws_base,
                max_depth=1,
                device=self.params['device'])

            with open(os.path.join(dataset_dirs[i_dataset], "features.pkl"), 'wb') as f:
                pickle.dump(feature_extractor.features, f)
