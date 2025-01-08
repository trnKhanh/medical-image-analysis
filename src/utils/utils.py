from datetime import datetime
from pathlib import Path


def get_path(path: Path | str) -> Path:
    if isinstance(path, str):
        path = Path(path)
    return path


class dummy_context(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def get_current_time_str():
    return datetime.now().strftime("%d%m%Y_%H%M%S")
