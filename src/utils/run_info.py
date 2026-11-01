"""Run metadata (code version, environment) recorded alongside every output for traceability."""

from __future__ import annotations

import os
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
    # Code snapshots without .git (e.g. Hugging Face Jobs) pass the commit in LALM_GIT_COMMIT; they are always
    # taken from a clean committed tree.
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    if commit is None and os.environ.get("LALM_GIT_COMMIT"):
        commit, status = os.environ["LALM_GIT_COMMIT"], ""
    return {
        "experiment": experiment,
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "seed": seed,
        **extra,
    }
