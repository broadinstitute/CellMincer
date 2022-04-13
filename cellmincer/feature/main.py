import os

import logging
import pprint
import time

import pickle

import numpy as np
import torch
from typing import Optional

from cellmincer.util import OptopatchBaseWorkspace, OptopatchGlobalFeatureExtractor

class Feature:
    def __init__(
            self,
            input_dir: str,
            use_active_range: bool):
        
        self.input_dir = input_dir
        input_file = os.path.join(input_dir, 'trend_subtracted.npy')
        self.ws_base = OptopatchBaseWorkspace.from_npy(input_file)
        
        mask_file = os.path.join(input_dir, 'active_mask.npy')
        self.active_mask = np.load(mask_file) if os.path.exists(mask_file) else None
        
        self.use_active_range = use_active_range

    def run(self):
        logging.info('Extracting features...')
        feature_extractor = OptopatchGlobalFeatureExtractor(
            ws_base=self.ws_base,
            active_mask=self.active_mask,
            select_active_t_range=self.use_active_range,
            max_depth=1)

        logging.info('Writing features to output directory...')
        with open(os.path.join(self.input_dir, 'features.pkl'), 'wb') as f:
            pickle.dump(feature_extractor.features, f)
