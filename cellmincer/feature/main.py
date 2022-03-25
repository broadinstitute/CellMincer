import os

import logging
import pprint
import time

import pickle

import numpy as np
import torch

from cellmincer.util import OptopatchBaseWorkspace, OptopatchGlobalFeatureExtractor

class Feature:
    def __init__(
            self,
            input_file: str,
            output_dir: str,
            use_active_range: bool):
        
        self.ws_base = OptopatchBaseWorkspace.from_npy(input_file)
        self.output_dir = output_dir
        
        self.use_active_range = use_active_range

    def run(self):
        logging.info('Extracting features...')
        feature_extractor = OptopatchGlobalFeatureExtractor(
            ws_base=self.ws_base,
            select_active_t_range=self.use_active_range,
            max_depth=1)

        logging.info('Writing features to output directory...')
        with open(os.path.join(self.output_dir, 'features.pkl'), 'wb') as f:
            pickle.dump(feature_extractor.features, f)
