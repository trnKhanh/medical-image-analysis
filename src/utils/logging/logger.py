import sys
import logging
import time
from pathlib import Path

from rich.logging import RichHandler
from rich.console import Console

logger = logging.getLogger("medical-images-analysis")


def setup_logger(log_file_str: str | None, do_debug: bool, do_verbose: bool):
    if do_verbose:
        shell_handler = RichHandler(
            console=Console(stderr=True),
            rich_tracebacks=True,
            show_time=False,
            tracebacks_show_locals=do_debug,
        )
        shell_handler.__module__
        shell_formatter = logging.Formatter(fmt="%(message)s")
        shell_handler.setFormatter(shell_formatter)
        logger.addHandler(shell_handler)

    if log_file_str:
        log_file = Path(log_file_str)
        if log_file.exists():
            log_file_str = f"{log_file.stem}@{time.time()}.{log_file.suffix}"
        file_handler = logging.FileHandler(log_file_str)
        file_formatter = logging.Formatter(
            fmt="%(levelname)s <%(asctime)s> [%(filename)s:%(funcName)s:%(lineno)s]: %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.setLevel(logging.DEBUG if do_debug else logging.INFO)
