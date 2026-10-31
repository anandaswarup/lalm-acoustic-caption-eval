"""Shared command-line setup for the scripts in `scripts/`."""

from __future__ import annotations

import argparse
import logging

from src.utils.config import load_env
from src.utils.log import setup_logging
from src.utils.paths import LOG_DIR
from src.utils.run_info import collect_run_info
from src.utils.seed import DEFAULT_SEED, set_seed


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed (default: %(default)s)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--no-log-file", action="store_true", help="log to the console only")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set evaluation.split=evaluation (repeatable)",
    )
    return parser


def init_run(name: str, args: argparse.Namespace) -> logging.Logger:
    """Load `.env`, configure logging, seed RNGs and log run metadata."""
    load_env()
    logger = setup_logging(name, level=args.log_level, log_dir=None if args.no_log_file else LOG_DIR)
    set_seed(args.seed)
    logger.info("Run info: %s", collect_run_info(name, seed=args.seed))
    return logger


def not_implemented(logger: logging.Logger, experiment: str) -> int:
    logger.error("Not implemented yet; scheduled for %s in docs/experiment_plan.md.", experiment)
    return 2
