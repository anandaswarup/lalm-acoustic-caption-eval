"""External tools for the caption metrics: the JDK, aac-metrics jars and model checkpoints under `tools/` (E11).

`configs/metrics.yaml` (`tools:`) pins every download by URL, commit or SHA-256; `scripts/setup_metric_tools.py`
fetches them and writes `MANIFEST.json`, the SHA-256 of every file the metrics use.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.data.clotho import sha256sum
from src.utils.paths import resolve_path

MANIFEST_NAME = "MANIFEST.json"
# aac-metrics accepts Java 8-13; SPICE writes its output with Nashorn, which was removed in Java 15.
JAVA_MAJOR_MIN, JAVA_MAJOR_MAX = 11, 13
# Not hashed into the manifest: the JDK (its archive is pinned instead), scratch space and HF cache internals
# (blobs are reached through the snapshot links).
_UNHASHED = {"jdk-11", "tmp", "downloads", "blobs", ".locks", "xet", ".no_exist"}


def platform_key() -> str:
    return f"{platform.system()}-{platform.machine()}".lower()


def tool_paths(tools: dict[str, Any]) -> dict[str, Path]:
    root = resolve_path(tools["dir"])
    return {
        "root": root,
        "jdk": root / "jdk-11",
        "aac_metrics": root / "aac-metrics",
        "spice": root / "aac-metrics" / "spice",
        "hf_home": root / "hf",
        "torch_home": root / "torch",
        "bertscore_baseline": root / tools["bertscore_baseline"]["path"],
        "manifest": root / MANIFEST_NAME,
    }


def activate(tools: dict[str, Any], offline: bool = False) -> dict[str, Path]:
    """Point the Hugging Face and torch hub caches at `tools/`. Call before importing aac_metrics or msclap.

    With `offline`, Hugging Face models load only from the cache (the pinned revisions); nothing is downloaded.
    huggingface_hub reads these variables once, at import: if it was imported earlier with another cache, the settings
    would be silently ignored (and models fetched into the home cache), so that raises instead.
    """
    paths = tool_paths(tools)
    if "huggingface_hub" in sys.modules:
        constants = sys.modules["huggingface_hub"].constants
        other_cache = Path(constants.HF_HOME).resolve() != paths["hf_home"].resolve()
        if other_cache or (offline and not constants.HF_HUB_OFFLINE):
            raise RuntimeError("huggingface_hub was imported before activate(); activate the metric tools first")
    os.environ["HF_HOME"] = str(paths["hf_home"])
    os.environ["TORCH_HOME"] = str(paths["torch_home"])
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return paths


def clap_weights_path(tools: dict[str, Any]) -> Path:
    clap = tools["clap"]
    revision = tools["hf_models"][clap["repo"]]
    cache = tool_paths(tools)["hf_home"] / "hub" / f"models--{clap['repo'].replace('/', '--')}"
    return cache / "snapshots" / revision / clap["filename"]


def jdk_build(tools: dict[str, Any]) -> dict[str, str]:
    builds = tools["jdk"]["builds"]
    if platform_key() not in builds:
        raise RuntimeError(f"No pinned JDK for platform {platform_key()!r} (have {sorted(builds)}); set METRICS_JAVA")
    return builds[platform_key()]


def java_path(tools: dict[str, Any]) -> Path:
    if tools.get("java_path"):
        return resolve_path(tools["java_path"])
    return tool_paths(tools)["jdk"] / jdk_build(tools)["java_bin"]


def java_major(version: str) -> int:
    """Major version from `java -version`: "1.8.0_402" -> 8, "11.0.32.1" -> 11."""
    parts = version.split(".")
    return int(parts[1]) if parts[0] == "1" else int(re.match(r"\d+", parts[0]).group())


def check_java(java: str | Path) -> str:
    """Return the version of `java`; raise if it is missing or outside the supported major versions."""
    try:
        output = subprocess.run([str(java), "-version"], capture_output=True, text=True, check=True).stderr
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Cannot run {java} ({error}); run `make metric-tools`") from error
    match = re.search(r'version "([^"]+)"', output)
    if not match:
        raise RuntimeError(f"Cannot parse the version of {java}: {output.strip()!r}")
    version = match.group(1)
    if not JAVA_MAJOR_MIN <= java_major(version) <= JAVA_MAJOR_MAX:
        raise RuntimeError(f"{java} is Java {version}; the metrics need Java {JAVA_MAJOR_MIN}-{JAVA_MAJOR_MAX}")
    return version


def hf_revision_problem(hf_home: Path, repo: str, revision: str) -> str | None:
    """Problem with the cached Hugging Face model `repo` at the pinned commit, or None.

    The snapshot for `revision` must be cached. Models loaded by name resolve the `main` ref, so if one is cached it
    must point to `revision` too (a download by explicit commit, as for the CLAP weights, writes no `main` ref).
    """
    cache = hf_home / "hub" / f"models--{repo.replace('/', '--')}"
    if not (cache / "snapshots" / revision).is_dir():
        return f"{repo}: no cached snapshot for the pinned revision {revision}"
    ref = cache / "refs" / "main"
    if ref.exists() and (main := ref.read_text(encoding="utf-8").strip()) != revision:
        return f"{repo}: cached main is {main}, pinned {revision}"
    return None


def hash_tree(root: Path) -> dict[str, str]:
    """SHA-256 of every file below `root` (following file links), keyed by its relative POSIX path."""
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if _UNHASHED.intersection(relative.parts) or relative.as_posix() == MANIFEST_NAME or not path.is_file():
            continue
        files[relative.as_posix()] = sha256sum(path)
    return files


def write_manifest(root: Path, info: dict[str, Any]) -> Path:
    manifest = root / MANIFEST_NAME
    manifest.write_text(json.dumps({**info, "files": hash_tree(root)}, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_manifest(root: Path) -> list[str]:
    """Return problems with the files recorded in `MANIFEST.json` (missing, changed, or no manifest)."""
    manifest = root / MANIFEST_NAME
    if not manifest.exists():
        return [f"{manifest} not found; run `make metric-tools`"]
    expected = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    errors = []
    for name, digest in expected.items():
        path = root / name
        if not path.is_file():
            errors.append(f"{name}: missing")
        elif sha256sum(path) != digest:
            errors.append(f"{name}: checksum changed")
    return errors
