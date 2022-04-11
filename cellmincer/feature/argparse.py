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
        '--input-file',
        nargs=None,
        type=str,
        dest='input_file',
        default=None,
        required=True,
        help='Input movie object.')
    
    subparser.add_argument(
        '-o',
        '--output-dir',
        nargs=None,
        type=str,
        dest='output_dir',
        default=None,
        required=True,
        help='Output directory.')

    subparser.add_argument(
        '--active-range-file',
        nargs=None,
        type=str,
        dest='active_range_file',
        default=None,
        required=False,
        help='Active range mask.')
    
    subparser.add_argument(
        '--no-active-range',
        dest='active_range',
        action='store_false',
        help='Disable active range.')

    return subparsers
