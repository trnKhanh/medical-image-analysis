from abc import ABC, abstractmethod
from pathlib import Path
from argparse import ArgumentParser, _SubParsersAction


class BaseCommand(ABC):
    def __init__(self, work_dir: Path):
        self.work_dir: Path = work_dir

    @abstractmethod
    def setup_parser(self, subparser) -> ArgumentParser:
        pass

    @abstractmethod
    def __call__(self):
        pass
