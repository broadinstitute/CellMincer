'''Command-line tool functionality for `cellmincer feature`.'''

import yaml
import logging
import os
import sys
from datetime import datetime

from cellmincer.cli.base_cli import AbstractCLI
from cellmincer.feature.main import Feature


class CLI(AbstractCLI):
    '''CLI implements AbstractCLI from the cellmincer.cli package.'''

    def __init__(self):
        self.name = 'feature'
        self.args = None

    def get_name(self) -> str:
        return self.name

    def validate_args(self, args):
        '''Validate parsed arguments.'''

        # Ensure that if there's a tilde for $HOME in the file path, it works.
        try:
            args.input_yaml_file = os.path.expanduser(args.input_yaml_file)
        except TypeError:
            raise ValueError('Problem with provided input paths.')

        self.args = args

        return args

    def run(self, args):
        '''Run the main tool functionality on parsed arguments.'''

        try:
            with open(args.input_yaml_file, 'r') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)
        except IOError:
            raise RuntimeError(f'Error loading the input YAML file {args.input_yaml_file}!')
        
        # Send logging messages to stdout as well as a log file.
        log_file = os.path.join(params['log_dir'], 'cellmincer_feature.log')
        logging.basicConfig(
            level=logging.INFO,
            format='cellmincer:feature:%(asctime)s: %(message)s',
            filename=log_file,
            filemode='w')
        console = logging.StreamHandler()
        formatter = logging.Formatter('cellmincer:feature:%(asctime)s: %(message)s', '%H:%M:%S')
        console.setFormatter(formatter)  # Use the same format for stdout.
        logging.getLogger('').addHandler(console)  # Log to stdout and a file.

        # Log the command as typed by user.
        logging.info('Command:\n' + ' '.join(['cellmincer', 'feature'] + sys.argv[2:]))
                                      
        # compute global features
        Feature(params).run()
