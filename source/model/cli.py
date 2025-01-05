"""CLI for the model module."""

import click
from nn_unet import *


@click.group(
    name="model",
    help="Model management commands."
)
def main() -> None:
    """Command line interface for the model."""
    pass

@main.command(
    name="execute",
)
def construct() -> None:
    """Model sample"""
    pass