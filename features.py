import os

import matplotlib.pylab as plt
import numpy as np
from time import time
import torch
import logging
import json
import pprint
import pickle

import yaml
import argparse

from cellmincer.opto_ws import OptopatchBaseWorkspace
from cellmincer.opto_features import OptopatchGlobalFeatureExtractor

from cellmincer.opto_utils import get_tagged_dir


def features(config_file):
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    device = torch.device(config['device'])
    
    # load datasets
    datasets = config['datasets']
    dataset_dirs = [get_tagged_dir(
            name=dataset,
            config_tag=config['tag'],
            root_dir=config['root_data_dir'])
        for dataset in datasets]
    
    assert all([os.path.exists(dataset_dir) for dataset_dir in dataset_dirs])
    
    for i_dataset in range(len(datasets)):
        ws_base = OptopatchBaseWorkspace.from_npy(os.path.join(dataset_dirs[i_dataset], 'trend_subtracted.npy'))
    
        feature_extractor = OptopatchGlobalFeatureExtractor(
            ws_base=ws_base,
            max_depth=1,
            device=config['device'])
        
        with open(os.path.join(dataset_dirs[i_dataset], "features.pkl"), 'wb') as f:
            pickle.dump(feature_extractor.features, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute global features.')
    parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()
    features(args.configfile)
