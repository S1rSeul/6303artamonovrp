import logging
import sys
from typing import Set


class ModuleFilter(logging.Filter):
    def __init__(self, allowed_filenames: Set[str]):
        super().__init__()
        self.allowed_filenames = allowed_filenames

    def filter(self, record: logging.LogRecord) -> bool:
        return record.filename in self.allowed_filenames


def configure_logging(log_file: str = 'logs/metetl.log') -> None:
    modules = {"cli.py", "data_to_download.py", "aggregations.py",
                   "models.py", "processing.py", "__main__.py", "logging_config.py"}

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()

    module_filter = ModuleFilter(modules)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - PID %(process)d - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(module_filter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(module_filter)
    root_logger.addHandler(console_handler)