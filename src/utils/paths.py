"""Repository-relative path management."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
MANIFEST_DIR = DATA_DIR / "manifests"
OUTPUT_DIR = REPO_ROOT / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
MANUSCRIPT_DIR = REPO_ROOT / "manuscript"


def resolve_path(path: str | Path) -> Path:
    """Resolve a path; relative paths are interpreted from the repository root."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if needed and return its resolved path."""
    path = resolve_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
