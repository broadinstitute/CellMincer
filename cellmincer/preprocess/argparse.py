import argparse


def add_subparser_args(subparsers: argparse) -> argparse:
    '''Add tool-specific arguments.
    Args:
        subparsers: Parser object before addition of arguments specific to
            `preprocess`.
    Returns:
        parser: Parser object with additional parameters.
    '''

    subparser = subparsers.add_parser(
        'preprocess',
        description='Dejitters and detrends raw datasets.',
        help='Dejitters and detrends raw datasets.')

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
