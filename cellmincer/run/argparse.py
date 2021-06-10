import argparse


def add_subparser_args(subparsers: argparse) -> argparse:
    '''Add tool-specific arguments.
    Args:
        subparsers: Parser object before addition of arguments specific to
            `run`.
    Returns:
        parser: Parser object with additional parameters.
    '''

    subparser = subparsers.add_parser(
        'run',
        description='Runs model training and uses models to denoise.',
        help='Runs model training and uses models to denoise.')

    subparser.add_argument('--train', help='invokes training routine', action='store_true')
    subparser.add_argument('--denoise', help='invokes denoising routine', action='store_true')
    subparser.add_argument(
        '-i',
        '--input-yaml-file',
        nargs=None,
        type=str,
        dest='input_yaml_file',
        default=None,
        required=True,
        help='Input YAML file.')

    return subparsers
