"""Scripts must run from the repository root without installing the project as a package."""

import subprocess
import sys

import pytest

from src.utils.paths import REPO_ROOT

SCRIPTS = sorted(path.name for path in (REPO_ROOT / "scripts").glob("*.py"))


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=REPO_ROOT, capture_output=True, text=True)


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_help(script):
    result = run(f"scripts/{script}", "--help")
    assert result.returncode == 0, result.stderr


def test_check_environment_runs():
    result = run("scripts/check_environment.py", "--no-log-file")
    assert result.returncode == 0, result.stderr
    assert "Environment check passed" in result.stderr
