import argparse


def add_subparser_args(subparsers: argparse) -> argparse:
    '''Add tool-specific arguments.
    Args:
        subparsers: Parser object before addition of arguments specific to
            `insight`.
    Returns:
        parser: Parser object with additional parameters.
    '''

    subparser = subparsers.add_parser(
        'insight',
        description='Produces performance statistics from a denoised movie and a clean reference.',
        help='Produces performance statistics from a denoised movie and a clean reference.')

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
