'''Command-line tool functionality for `cellmincer denoise`.'''

import yaml
import logging
import os
import sys
from datetime import datetime

from cellmincer.cli.base_cli import AbstractCLI
from cellmincer.denoise.main import Denoise


class CLI(AbstractCLI):
    '''CLI implements AbstractCLI from the cellmincer.cli package.'''

    def __init__(self):
        self.name = 'denoise'
        self.args = None

    def get_name(self) -> str:
        return self.name

    def validate_args(self, args):
        '''Validate parsed arguments.'''

        # Ensure that if there's a tilde for $HOME in the file path, it works.
        try:
            args.input_dir = os.path.expanduser(args.input_dir)
            if args.output_dir:
                args.output_dir = os.path.expanduser(args.output_dir)
            else:
                args.output_dir = args.input_dir
            args.model_ckpt = os.path.expanduser(args.model_ckpt)
        except TypeError:
            raise ValueError('Problem with provided input paths.')

        self.args = args

        return args

    def run(self, args):
        '''Run the main tool functionality on parsed arguments.'''
        
        # Send logging messages to stdout as well as a log file.
        logging.basicConfig(level=logging.INFO)
        
        console = logging.StreamHandler()
        formatter = logging.Formatter('cellmincer:denoise:%(asctime)s: %(message)s', '%H:%M:%S')
        console.setFormatter(formatter)  # Use the same format for stdout.
        logging.getLogger('').addHandler(console)  # Log to stdout

        # Log the command as typed by user.
        logging.info('Command:\n' + ' '.join(['cellmincer', 'denoise'] + sys.argv[2:]))
        
        # denoise data
        Denoise(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model_ckpt=args.model_ckpt,
            model_type=args.model_type,
            avi_enabled=args.avi_enabled,
            avi_frames=args.avi_frames,
            avi_sigma=args.avi_sigma).run()
