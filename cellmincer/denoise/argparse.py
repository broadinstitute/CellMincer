import argparse


def add_subparser_args(subparsers: argparse) -> argparse:
    '''Add tool-specific arguments.
    Args:
        subparsers: Parser object before addition of arguments specific to
            `denoise`.
    Returns:
        parser: Parser object with additional parameters.
    '''

    subparser = subparsers.add_parser(
        'denoise',
        description='Denoises data with trained model.',
        help='Denoises data with trained model.')

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
