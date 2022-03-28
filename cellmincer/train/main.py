import os
import logging
import pprint
import time
from datetime import timedelta

import json
import pickle
import tarfile

import matplotlib.pylab as plt
import numpy as np
import skimage
import torch
import pandas as pd
from typing import List, Optional, Tuple

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

from cellmincer.datasets import build_datamodule

from cellmincer.models import \
    init_model, \
    get_window_padding_from_config, \
    load_model_from_checkpoint

from cellmincer.util import \
    const, \
    crop_center

import neptune.new as neptune
    
class Train:
    def __init__(
            self,
            inputs: List[str],
            output_dir: str,
            config: dict,
            gpus: int,
            pretrain: Optional[str] = None,
            checkpoint: Optional[str] = None):
        
        self.model = None
        if pretrain:
            try:
                self.model = load_model_from_checkpoint(
                    model_type=config['model']['type'],
                    ckpt_path=pretrain)
                logging.info('Loaded pretrained model state.')
            except EOFError:
                logging.warning('Bad checkpoint; ignoring pretrained state.')
        
        # if continuing from checkpoint
        self.ckpt_path = None
        if checkpoint is not None and os.path.exists(checkpoint):
            try:
                resume_model = load_model_from_checkpoint(
                    model_type=config['model']['type'],
                    ckpt_path=checkpoint)
                self.model = resume_model
                self.ckpt_path = checkpoint
                logging.info('Successfully loaded checkpoint.')
            except EOFError:
                logging.warning('Bad checkpoint; aborting checkpointed state. Expected behavior when Terra first initializes training.')
                pass
        
        if self.model:
            train_config = self.model.hparams.train_config
        else:
            train_config = config['train']
            
            # compute training padding with maximal output/input receptive field ratio
            output_min_size = np.arange(config['train']['output_min_size_lo'], config['train']['output_min_size_hi'] + 1)
            output_min_size = output_min_size[(output_min_size % 2) == 0]
            train_padding = get_window_padding_from_config(
                model_config=config['model'],
                output_min_size=output_min_size)
            best_train = np.argmax(output_min_size / (output_min_size + 2 * train_padding))
            train_config['x_window'] = train_config['y_window'] = output_min_size[best_train]
            train_config['x_padding'] = train_config['y_padding'] = train_padding[best_train]
            
        self.movie_dm = build_datamodule(
            datasets=inputs,
            model_config=config['model'],
            train_config=train_config,
            gpus=gpus)
        
        if self.model is None:
            logging.info('Initializing new model.')
            model_config = config['model']
            model_config['n_global_features'] = self.movie_dm.n_global_features
            self.model = init_model(
                model_config=model_config,
                train_config=train_config)
        
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        
        self.neptune_enabled = config['neptune']['enabled']
        pl_logger = True
        
        if self.neptune_enabled:
            if self.model.neptune_run_id is not None:
                logging.info('Reinitializing existing Neptune run...')
                neptune_run = neptune.init(
                    api_token=config['neptune']['api_token'],
                    project=config['neptune']['project'],
                    run=self.model.neptune_run_id,
                    tags=config['neptune']['tags'])
                pl_logger = NeptuneLogger(run=neptune_run)
            else:
                logging.info('Initializing new Neptune run...')
                pl_logger = NeptuneLogger(
                    api_key=config['neptune']['api_token'],
                    project=config['neptune']['project'],
                    tags=config['neptune']['tags'],
                    log_model_checkpoints=False)
                pl_logger.experiment['datasets'] = inputs

        self.trainer = Trainer(
            strategy='ddp',
            gpus=gpus,
            max_epochs=train_config['n_iters'],
            default_root_dir=self.output_dir,
            # TODO experiment with these settings because docs are ambiguous
            callbacks=[ModelCheckpoint(dirpath=self.output_dir, save_last=True)],
            logger=pl_logger)

    def run(self):
        logging.info('Training model...')
        
        self.trainer.fit(
            model=self.model,
            train_dataloaders=self.movie_dm,
            ckpt_path=self.ckpt_path)

        # save trained model
        logging.info('Training complete; saving model...')
        if self.neptune_enabled:
            self.model.logger.experiment['final'].upload(os.path.join(self.output_dir, 'last.ckpt'))
