from argparse import ArgumentParser
from pathlib import Path

from . import commands
from utils.logging import logger, setup_logger


def setup_parser():
    work_dir = Path.cwd()

    parser = ArgumentParser()

    parser.add_argument(
        "-l",
        "--log-file",
        dest="log_file_str",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--debug",
        dest="do_debug",
        action="store_true",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        dest="do_verbose",
        action="store_false",
    )

    subparsers = parser.add_subparsers()

    for command_cls in commands.available_commands:
        command = command_cls(work_dir)
        command.setup_parser(subparsers)

    return parser


def main():
    parser = setup_parser()
    args = vars(parser.parse_args())

    func = args.pop("func")

    log_file_str = args.get("log_file_str")
    do_debug = bool(args.get("do_debug"))
    do_verbose = bool(args.get("do_verbose"))

    setup_logger(log_file_str, do_debug, do_verbose)

    func(**args)
