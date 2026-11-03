"""SPICE without its parse cache (E11).

aac-metrics always passes `-cache`, and SPICE's LMDB cache loads a native library that ships for x86 only, so it
fails on arm64. The cache only stores parses; scores are the same without it. Input and output follow aac-metrics:
one item per candidate, and the per-item score is the `All` F-score.
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

SPICE_JAR = "aac-metrics/spice/spice-1.0.jar"  # relative to tools.dir


def spice_input(candidates: Sequence[str], mult_references: Sequence[Sequence[str]]) -> list[dict[str, Any]]:
    if len(candidates) != len(mult_references):
        raise ValueError(f"{len(candidates)} candidates but {len(mult_references)} reference lists")
    return [
        {"image_id": index, "test": candidate, "refs": list(references)}
        for index, (candidate, references) in enumerate(zip(candidates, mult_references, strict=True))
    ]


def parse_spice_output(results: list[dict[str, Any]], num_items: int) -> np.ndarray:
    """Per-item `All` F-scores in input order; raise on missing, duplicate or non-finite items."""
    scores: dict[int, float] = {}
    for item in results:
        index = int(item["image_id"])
        if index in scores:
            raise RuntimeError(f"SPICE returned item {index} twice")
        try:
            value = float(item["scores"]["All"]["f"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"SPICE item {index} has no numeric All F-score") from error
        if not math.isfinite(value):
            raise RuntimeError(f"SPICE item {index} scored {value}")
        scores[index] = value
    if missing := sorted(set(range(num_items)) - set(scores)):
        raise RuntimeError(f"SPICE returned no score for {len(missing)} items, e.g. {missing[:5]}")
    if extra := sorted(set(scores) - set(range(num_items))):
        raise RuntimeError(f"SPICE returned unknown items {extra[:5]}")
    return np.array([scores[index] for index in range(num_items)], dtype=np.float64)


def spice_scores(
    candidates: Sequence[str],
    mult_references: Sequence[Sequence[str]],
    *,
    java: str | Path,
    tools_dir: str | Path,
    java_max_memory: str = "8G",
    threads: int | None = None,
) -> np.ndarray:
    """Per-item SPICE F-scores from one run of the SPICE jar."""
    with tempfile.TemporaryDirectory(prefix="spice_") as tmp:
        in_path, out_path = Path(tmp) / "input.json", Path(tmp) / "output.json"
        in_path.write_text(json.dumps(spice_input(candidates, mult_references)), encoding="utf-8")
        command = [str(java), "-jar", f"-Xmx{java_max_memory}", str(Path(tools_dir) / SPICE_JAR), str(in_path)]
        command += ["-out", str(out_path), "-subset"] + (["-threads", str(threads)] if threads else [])
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 or not out_path.exists():
            raise RuntimeError(f"SPICE failed (exit {result.returncode}): {result.stderr.strip()[-2000:]}")
        results = json.loads(out_path.read_text(encoding="utf-8"))
    return parse_spice_output(results, len(candidates))
