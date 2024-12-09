from pathlib import Path
from argparse import ArgumentParser

from PIL import Image

from .basecommand import BaseCommand
from utils.logging import logger


class VisualizeCommand(BaseCommand):
    def setup_parser(self, subparser) -> ArgumentParser:
        if subparser is None:
            parser = ArgumentParser()
        else:
            parser = subparser.add_parser("visualize", help="Visualize data")

        parser.add_argument(
            "-f",
            "--image-file",
            dest="image_file_str",
            type=str,
            help="Path to file",
        )

        parser.set_defaults(func=self)

        return parser

    def __call__(self, image_file_str: str, *args, **kw_args):
        image_file = Path(image_file_str)
        logger.debug(f"Read and show {image_file.resolve()}")

        image = Image.open(image_file)
        image.show()
