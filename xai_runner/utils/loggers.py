import logging
import sys
from typing import Optional

_INITIALIZED = False

def _initialize_root_logger(level: int = logging.INFO) -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)5s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    _INITIALIZED = True

def get_logger(name: Optional[str] = None) -> logging.Logger:

    _initialize_root_logger()
    return logging.getLogger(name if name is not None else __name__)
