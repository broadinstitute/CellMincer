'''Command-line tool functionality for `cellmincer train`.'''

import yaml
import logging
import os
import sys
from datetime import datetime

from cellmincer.cli.base_cli import AbstractCLI
from cellmincer.train.main import Train


class CLI(AbstractCLI):
    '''CLI implements AbstractCLI from the cellmincer.cli package.'''

    def __init__(self):
        self.name = 'train'
        self.args = None

    def get_name(self) -> str:
        return self.name

    def validate_args(self, args):
        '''Validate parsed arguments.'''

        # Ensure that if there's a tilde for $HOME in the file path, it works.
        try:
            args.inputs = [os.path.expanduser(x) for x in args.inputs]
            args.output_dir = os.path.expanduser(args.output_dir)
            args.config = os.path.expanduser(args.config)
            if args.pretrain:
                args.pretrain = os.path.expanduser(args.pretrain)
            if args.checkpoint:
                args.checkpoint = os.path.expanduser(args.checkpoint)
            if args.checkpoint_start:
                args.checkpoint_start = os.path.expanduser(args.checkpoint_start)
        except TypeError:
            raise ValueError('Problem with provided input paths.')

        self.args = args

        return args

    def run(self, args):
        '''Run the main tool functionality on parsed arguments.'''

        try:
            with open(args.config, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        except IOError:
            raise RuntimeError(f'Error loading the input YAML file {args.config}!')
        
        # Send logging messages to stdout as well as a log file.
        if 'log_dir' in config:
            log_file = os.path.join(config['log_dir'], 'cellmincer_train.log')
            logging.basicConfig(
                level=logging.INFO,
                format='cellmincer:train:%(asctime)s: %(message)s',
                filename=log_file,
                filemode='w')
        else:
            logging.basicConfig(level=logging.INFO)
        
        console = logging.StreamHandler()
        formatter = logging.Formatter('cellmincer:train:%(asctime)s: %(message)s', '%H:%M:%S')
        console.setFormatter(formatter)  # Use the same format for stdout.
        logging.getLogger('').addHandler(console)  # Log to stdout

        # Log the command as typed by user.
        logging.info('Command:\n' + ' '.join(['cellmincer', 'train'] + sys.argv[2:]))
                                      
        # train model
        Train(
            inputs=args.inputs,
            output_dir=args.output_dir,
            config=config,
            gpus=args.gpus,
            use_memmap=args.use_memmap,
            pretrain=args.pretrain,
            checkpoint=args.checkpoint,
            checkpoint_start=args.checkpoint_start).run()
