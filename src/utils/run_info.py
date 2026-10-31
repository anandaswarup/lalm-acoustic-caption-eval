"""Run metadata (code version, environment) recorded alongside every output for traceability."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any

from src.utils.paths import REPO_ROOT


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def collect_run_info(experiment: str, seed: int | None = None, **extra: Any) -> dict[str, Any]:
    status = _git("status", "--porcelain")
    return {
        "experiment": experiment,
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "seed": seed,
        **extra,
    }
