import argparse


def add_subparser_args(subparsers: argparse) -> argparse:
    '''Add tool-specific arguments.
    Args:
        subparsers: Parser object before addition of arguments specific to
            `train`.
    Returns:
        parser: Parser object with additional parameters.
    '''

    subparser = subparsers.add_parser(
        'train',
        description='Trains denoising model.',
        help='Trains denoising model.')
    
    subparser.add_argument(
        '-i',
        '--input',
        nargs='+',
        type=str,
        dest='inputs',
        default=None,
        required=True,
        help='Model training input directory.')
    
    subparser.add_argument(
        '-o',
        '--output',
        nargs=None,
        type=str,
        dest='output_dir',
        default=None,
        required=True,
        help='Directory to save model/training state.')
    
    subparser.add_argument(
        '--config',
        nargs=None,
        type=str,
        dest='config',
        default=None,
        required=True,
        help='Model training configuration.')
    
    subparser.add_argument(
        '--pretrain',
        nargs=None,
        type=str,
        dest='pretrain',
        default=None,
        required=False,
        help='Pre-trained model weights.')
    
    subparser.add_argument(
        '--checkpoint',
        nargs=None,
        type=str,
        dest='checkpoint',
        default=None,
        required=False,
        help='Checkpoint file for resuming training on preemptible machines.')
    
    subparser.add_argument(
        '--checkpoint_start',
        nargs=None,
        type=str,
        dest='checkpoint_start',
        default=None,
        required=False,
        help='Checkpoint file for restarting training from an aborted run.')

    subparser.add_argument(
        '--gpus',
        nargs=None,
        type=int,
        dest='gpus',
        default=1,
        required=False,
        help='Number of GPUs to use for training.')
    
    subparser.add_argument(
        '--use_memmap',
        dest='use_memmap',
        action='store_true',
        help='Use memmapped training data for large operations.')

    return subparsers
