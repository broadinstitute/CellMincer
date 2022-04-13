import argparse


def add_subparser_args(subparsers: argparse) -> argparse:
    '''Add tool-specific arguments.
    Args:
        subparsers: Parser object before addition of arguments specific to
            `feature`.
    Returns:
        parser: Parser object with additional parameters.
    '''

    subparser = subparsers.add_parser(
        'feature',
        description='Computes global feature map.',
        help='Computes global feature map.')

    subparser.add_argument(
        '-i',
        '--input-dir',
        nargs=None,
        type=str,
        dest='input_dir',
        default=None,
        required=True,
        help='Input directory.')
    
    subparser.add_argument(
        '--no-active-range',
        dest='active_range',
        action='store_false',
        help='Disable active range.')

    return subparsers
