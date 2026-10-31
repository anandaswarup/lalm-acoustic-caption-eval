"""Logging to the console and to a timestamped file under `outputs/logs/`."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from src.utils.paths import LOG_DIR, ensure_dir

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(name: str, level: str = "INFO", log_dir: str | Path | None = LOG_DIR) -> logging.Logger:
    """Configure the root logger and return a named logger.

    Pass `log_dir=None` to log to the console only.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir is not None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        handlers.append(logging.FileHandler(ensure_dir(log_dir) / f"{name}_{stamp}.log", encoding="utf-8"))
    logging.basicConfig(level=level.upper(), format=_FORMAT, handlers=handlers, force=True)
    return logging.getLogger(name)
