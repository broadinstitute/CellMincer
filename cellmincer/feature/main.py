import os
import logging
import pprint
import time

import json
import pickle

import matplotlib.pylab as plt
import numpy as np
import torch

from cellmincer.util import OptopatchBaseWorkspace, OptopatchGlobalFeatureExtractor, consts

class Feature:
    def __init__(
            self,
            params: dict):
        
        self.root_data_dir = params['root_data_dir']
        self.device = torch.device(params['device'])
        self.datasets = params['datasets']

    def run(self):
        assert all([os.path.exists(os.path.join(self.root_data_dir, dataset)) for dataset in self.datasets])

        for i_dataset, dataset in enumerate(self.datasets):
            logging.info(f'({i_dataset + 1}/{len(self.datasets)}) {dataset}')
            
            ws_base = OptopatchBaseWorkspace.from_npy(os.path.join(self.root_data_dir, dataset, 'trend_subtracted.npy'))

            feature_extractor = OptopatchGlobalFeatureExtractor(
                ws_base=ws_base,
                max_depth=1,
                device=self.device)

            with open(os.path.join(self.root_data_dir, dataset, "features.pkl"), 'wb') as f:
                pickle.dump(feature_extractor.features, f)
