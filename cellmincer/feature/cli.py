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
            args.input_file = os.path.expanduser(args.input_file)
            args.output_dir = os.path.expanduser(args.output_dir)
            if args.active_range_file is not None:
                args.active_range_file = os.path.expanduser(args.active_range_file)
        except TypeError:
            raise ValueError('Problem with provided input paths.')

        self.args = args

        return args

    def run(self, args):
        '''Run the main tool functionality on parsed arguments.'''

        # Log the command as typed by user.
        logging.basicConfig(level=logging.INFO)
        console = logging.StreamHandler()
        formatter = logging.Formatter('cellmincer:feature:%(asctime)s: %(message)s', '%H:%M:%S')
        console.setFormatter(formatter)  # Use the same format for stdout.
        logging.getLogger('').addHandler(console)  # Log to stdout
        
        logging.info('Command:\n' + ' '.join(['cellmincer', 'feature'] + sys.argv[2:]))
                                      
        # compute global features
        Feature(
            input_file=args.input_file,
            output_dir=args.output_dir,
            use_active_range=args.active_range,
            active_range_file=args.active_range_file).run()
