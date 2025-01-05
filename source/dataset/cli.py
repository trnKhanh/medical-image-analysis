"""CLI for the dataset module."""

import click
from construct import BaseConstructorHandler
from labeling import BaseLabelingHandler


@click.group(
    name="dataset",
    help="Dataset management commands."
)
def main() -> None:
    """Command line interface for the dataset."""
    pass

@main.command(
    name="construct",
)
def construct() -> None:
    """Construct data from the dataset."""
    handler = BaseConstructorHandler()
    handler()

@main.command(
    name="labeling",
    help="Labeling data from the dataset."
)
def labeling() -> None:
    """Label data from the dataset."""
    handler = BaseLabelingHandler()
    handler()
