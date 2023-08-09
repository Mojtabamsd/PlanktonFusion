import argparse
import os
import sys
from tools import Console
from data_preparation import sampling


def add_arguments(obj):
    obj.add_argument(
        "-c",
        "--configuration",
        type=str,
        default="georef_semantics.yaml",
        help="Path to the configuration file 'georef_semantics.yaml'",
    )
    obj.add_argument(
        "-o",
        "--output-folder",
        default="lga_output",
        type=str,
        help="Output path to write the results.",
    )


def main(args=None):

    os.system("")

    Console.banner()
    Console.info("Running georef_semantics version " + str(Console.get_version()))

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_sampling = subparsers.add_parser(
        "sampling", help="preparing dataset images for training"
    )
    add_arguments(parser_sampling)
    parser_sampling.set_defaults(func=call_sampling)

    if len(sys.argv) == 1 and args is None:
        # Show help if no args provided
        parser.print_help(sys.stderr)
        sys.exit(2)
    args = parser.parse_args(args)
    args.func(args)


def call_sampling(args):
    sampling(args.configuration)