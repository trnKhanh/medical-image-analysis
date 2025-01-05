"""CLI for the dataset module."""

import click

@click.group(
    name="dataset",
    help="Dataset management commands."
)
def main():
    pass

if __name__ == "__main__":
    main()
    