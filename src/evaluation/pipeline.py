"""Shared setup for the scoring scripts (E12-E14): output paths, input verification, evaluation inputs, the CLAP audio
cache and the timing block of a report.

Nothing here imports a model library, but the scripts still call `src.metrics.tools.activate` first (see its docstring).
"""

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace
from importlib import metadata
from pathlib import Path
from typing import Any

from src.data.clotho import read_jsonl
from src.data.manifest import verify_checksums
from src.evaluation.inputs import audio_paths, load_references
from src.metrics.clap import load_audio_cache, save_audio_cache
from src.metrics.registry import Timer
from src.metrics.tools import verify_manifest
from src.utils.paths import MANIFEST_DIR, OUTPUT_DIR, REPO_ROOT, resolve_path

DEFAULT_CACHE = OUTPUT_DIR / "cache" / "clap_audio.npz"


def relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path)


def add_run_arguments(parser: ArgumentParser, smoke_dir: Path) -> None:
    """--scores-dir, --report-dir, --cache, --limit and --force, shared by the scoring scripts."""
    smoke = f"{relative(smoke_dir)} with --limit"
    parser.add_argument("--scores-dir", help=f"default: {relative(OUTPUT_DIR / 'scores')} ({smoke})")
    parser.add_argument("--report-dir", help=f"default: {relative(OUTPUT_DIR / 'reports')} ({smoke})")
    cache = relative(DEFAULT_CACHE)
    parser.add_argument("--cache", default=cache, help="CLAP audio cache, 'none' to disable (default: %(default)s)")
    parser.add_argument("--limit", type=int, help="score the first N recordings only (smoke test)")
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")


def output_dirs(args: Namespace, smoke_dir: Path) -> tuple[Path, Path]:
    """Scores and report directories: the arguments, else the smoke directory with --limit, else outputs/."""
    smoke = smoke_dir if args.limit else None
    return (
        resolve_path(args.scores_dir or smoke or OUTPUT_DIR / "scores"),
        resolve_path(args.report_dir or smoke or OUTPUT_DIR / "reports"),
    )


def existing_outputs(outputs: dict[str, Path]) -> list[str]:
    return [relative(path) for path in outputs.values() if path.exists()]


def verify_inputs(tools_root: Path) -> list[str]:
    """Problems with the metric tools, the frozen manifests or the frozen captions (checked against their SHA-256s)."""
    problems = verify_manifest(tools_root)
    problems += [f"manifests: {e}" for e in verify_checksums(MANIFEST_DIR / "SHA256SUMS")]
    problems += [f"captions: {e}" for e in verify_checksums(OUTPUT_DIR / "captions" / "SHA256SUMS")]
    return problems


def evaluation_inputs(
    dataset: dict[str, Any], limit: int | None = None
) -> tuple[list[dict[str, Any]], Path, dict[str, list[str]], dict[str, Path]]:
    """Evaluation manifest (first `limit` records), references path, references and verified audio paths."""
    evaluation = dataset["evaluation"]
    manifest = read_jsonl(resolve_path(evaluation["manifest"]))
    if limit:
        manifest = manifest[:limit]
    references_path = resolve_path(evaluation["references"])
    all_references = load_references(references_path)
    references = {record["sample_id"]: all_references[record["sample_id"]] for record in manifest}
    audio = audio_paths(manifest, resolve_path(dataset["root"]))
    return manifest, references_path, references, audio


def clap_cache_key(settings: dict[str, Any], tools: dict[str, Any]) -> str:
    """What the cached audio embeddings depend on: CLAP settings, weights and msclap version."""
    key = {"clap": settings["clap"], "weights_sha256": tools["clap"]["sha256"], "msclap": metadata.version("msclap")}
    return json.dumps(key, sort_keys=True)


class AudioCache:
    """CLAP audio embeddings on disk, keyed by audio SHA-256; `path` None disables the cache."""

    def __init__(self, path: Path | None, key: str) -> None:
        self.path, self.key = path, key
        self.cached = load_audio_cache(path, key) if path else {}
        self.hits = 0

    def preload(self, clap, manifest: list[dict[str, Any]], audio: dict[str, Path]) -> int:
        hits = {
            str(audio[record["sample_id"]]): self.cached[record["audio_sha256"]]
            for record in manifest
            if record["audio_sha256"] in self.cached
        }
        clap.preload_audio(hits)
        self.hits = len(hits)
        return self.hits

    def save(self, clap, manifest: list[dict[str, Any]], audio: dict[str, Path]) -> int:
        """Add this run's recordings to the cache file; returns how many recordings it now holds."""
        if self.path is None:
            return 0
        by_path = clap.audio_embeddings()
        computed = {r["audio_sha256"]: by_path[str(audio[r["sample_id"]])] for r in manifest}
        self.cached |= computed
        save_audio_cache(self.path, self.key, self.cached)
        return len(self.cached)


def timing_block(timer: Timer, cache_hits: int) -> dict[str, Any]:
    return {
        "seconds": {name: round(value, 2) for name, value in timer.seconds.items() if name != "load_models"},
        "calls": dict(timer.calls),
        "load_models_seconds": round(timer.seconds["load_models"], 1),
        "scoring_seconds": round(sum(v for name, v in timer.seconds.items() if name != "load_models"), 1),
        "clap_cache_hits": cache_hits,
    }
