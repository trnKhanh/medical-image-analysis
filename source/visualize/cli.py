"""CLI for the visualize module."""

import click


@click.group(
    name="visualize",
    help="Visualize management commands."
)
def main() -> None:
    """Command line interface for the dataset."""
    pass

@main.command(
    name="execute",
    help="Execute the dataset."
)
def construct() -> None:
    """Visualize Processing Template."""
    pass
