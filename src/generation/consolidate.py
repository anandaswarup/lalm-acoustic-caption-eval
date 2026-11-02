"""Merge the four models' captions into one frozen, aligned caption dataset (E10).

Normalization (decided in E05 and after E06): leading and trailing whitespace is stripped; a caption that is exactly
{"<text>"} is unwrapped to <text>; any other JSON-like caption is kept unchanged and flagged as malformed. Malformed
captions are scored as returned; nothing is excluded.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from src.data.clotho import read_jsonl
from src.generation.report import is_wrapped_sentence
from src.generation.runner import is_plain_text, prompt_sha256

NORMALIZATIONS = ("none", "unwrapped", "malformed")


def malformed_type(caption: str) -> str:
    """`refusal` ({"error": ...}), `result` ({"result": ...}), `fields` (an object with several keys) or `other`."""
    try:
        value = json.loads(caption)
    except ValueError:
        return "other"
    if not isinstance(value, dict):
        return "other"
    if list(value) == ["error"]:
        return "refusal"
    if list(value) == ["result"]:
        return "result"
    return "fields" if len(value) > 1 else "other"


def normalize_caption(raw: str) -> dict[str, Any]:
    """The scored caption and how it was derived from the raw model output."""
    text = raw.strip()
    if is_plain_text(text):
        return {"caption": text, "normalization": "none", "malformed_type": None}
    if is_wrapped_sentence(text):
        return {"caption": text[2:-2].strip(), "normalization": "unwrapped", "malformed_type": None}
    return {"caption": text, "normalization": "malformed", "malformed_type": malformed_type(text)}


def _last_records(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    """Last record per sample_id, and the number of lines in the file."""
    records = read_jsonl(path)
    return {record["sample_id"]: record for record in records}, len(records)


def consolidate(
    sources: dict[str, Path], manifest: Path, references: Path, prompt_text: str, captions_per_recording: int = 5
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    """Return (records, report, errors) for `sources` ({model_key: captions JSONL}, in output order).

    Records are sorted by manifest order, then by model in `sources` order. Any error means the dataset must not be
    frozen.
    """
    sample_ids = [record["sample_id"] for record in read_jsonl(manifest)]
    refs = json.loads(references.read_text(encoding="utf-8"))
    errors: list[str] = []
    if len(set(sample_ids)) != len(sample_ids):
        errors.append("manifest: duplicate sample_ids")
    if set(refs) != set(sample_ids):
        errors.append(f"references: {len(set(refs) ^ set(sample_ids))} sample_ids differ from the manifest")
    if bad := [key for key, value in refs.items() if len(value) != captions_per_recording]:
        errors.append(f"references: {len(bad)} recordings without {captions_per_recording} captions: {bad[:5]}")

    expected_sha = prompt_sha256(prompt_text)
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    models_report: dict[str, Any] = {}
    for model_key, path in sources.items():
        latest, lines = _last_records(path)
        by_model[model_key] = latest
        label = f"{model_key}"
        if missing := [sample_id for sample_id in sample_ids if sample_id not in latest]:
            errors.append(f"{label}: {len(missing)} samples missing: {missing[:5]}")
        if extra := sorted(set(latest) - set(sample_ids)):
            errors.append(f"{label}: {len(extra)} sample_ids not in the manifest: {extra[:5]}")
        for sample_id in sample_ids:
            record = latest.get(sample_id)
            if record is None:
                continue
            if record["status"] != "ok":
                errors.append(f"{label} {sample_id}: status {record['status']}")
            elif not (record.get("caption") or "").strip():
                errors.append(f"{label} {sample_id}: empty caption")
            if record.get("prompt_sha256") != expected_sha:
                errors.append(f"{label} {sample_id}: produced with another prompt")
            if record.get("model_key") not in (None, model_key):
                errors.append(f"{label} {sample_id}: record is for model {record['model_key']}")
        models_report[model_key] = {"source": path.name, "record_lines": lines, "samples": len(latest)}
    if errors:
        return [], {"errors": errors}, errors

    records = []
    for sample_id in sample_ids:
        for model_key in sources:
            source = by_model[model_key][sample_id]
            records.append(
                {
                    "sample_id": sample_id,
                    "model_key": model_key,
                    **normalize_caption(source["caption"]),
                    "caption_raw": source["caption"],
                    "model_id": source.get("model_id"),
                    "model_version": source.get("model_version"),
                }
            )

    valid_for_all = [
        sample_id
        for index, sample_id in enumerate(sample_ids)
        if all(
            record["normalization"] != "malformed"
            for record in records[index * len(sources) : (index + 1) * len(sources)]
        )
    ]
    for model_key in sources:
        own = [record for record in records if record["model_key"] == model_key]
        words = [len(record["caption"].split()) for record in own]
        types = Counter(record["malformed_type"] for record in own if record["malformed_type"])
        models_report[model_key].update(
            {
                "normalization": {name: sum(r["normalization"] == name for r in own) for name in NORMALIZATIONS},
                "malformed_types": dict(sorted(types.items())),
                "unwrapped_ids": [r["sample_id"] for r in own if r["normalization"] == "unwrapped"],
                "malformed_ids": {
                    r["sample_id"]: r["malformed_type"] for r in own if r["normalization"] == "malformed"
                },
                "model_versions": dict(Counter(str(r["model_version"]) for r in own)),
                "caption_words": {
                    "min": min(words),
                    "mean": round(statistics.fmean(words), 2),
                    "median": statistics.median(words),
                    "max": max(words),
                },
                "distinct_captions": len({r["caption"] for r in own}),
            }
        )
    report = {
        "recordings": len(sample_ids),
        "models": list(sources),
        "captions": len(records),
        "references": sum(len(value) for value in refs.values()),
        "prompt_sha256": expected_sha,
        "valid_for_all_models": {"count": len(valid_for_all), "sample_ids": valid_for_all},
        "per_model": models_report,
        "errors": [],
    }
    return records, report, []


def report_markdown(report: dict[str, Any], checksums: dict[str, str]) -> str:
    """Caption validation report: counts and sample_ids only, never caption text."""
    lines = ["# Caption validation report (E10)", ""]
    if report["errors"]:
        lines += ["## Errors (dataset not frozen)", ""] + [f"- {error}" for error in report["errors"]] + [""]
        return "\n".join(lines)
    lines += [
        "## Dataset",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Recordings | {report['recordings']} |",
        f"| Models | {', '.join(report['models'])} |",
        f"| Model captions | {report['captions']} |",
        f"| Human references | {report['references']} |",
        f"| Recordings valid for all models (sensitivity check) | {report['valid_for_all_models']['count']} |",
        "| Exclusions | none (malformed captions are scored as returned) |",
        f"| Prompt SHA-256 | `{report['prompt_sha256']}` |",
        "",
        "## Normalization per model",
        "",
        "| Model | none | unwrapped | malformed | Malformed types | Words min / mean / max | Distinct |",
        "|---|---|---|---|---|---|---|",
    ]
    for model_key, item in report["per_model"].items():
        norm, words = item["normalization"], item["caption_words"]
        types = ", ".join(f"{key}: {value}" for key, value in item["malformed_types"].items()) or "none"
        lines.append(
            f"| {model_key} | {norm['none']} | {norm['unwrapped']} | {norm['malformed']} | {types} | "
            f"{words['min']} / {words['mean']} / {words['max']} | {item['distinct_captions']} |"
        )
    lines.append("")
    for model_key, item in report["per_model"].items():
        if item["malformed_ids"]:
            lines += [f"## Malformed captions: {model_key}", "", "| sample_id | Type |", "|---|---|"]
            lines += [f"| `{sample_id}` | {kind} |" for sample_id, kind in item["malformed_ids"].items()]
            lines.append("")
    lines += ["## SHA-256", "", "| File | SHA-256 |", "|---|---|"]
    lines += [f"| `{name}` | `{sha}` |" for name, sha in checksums.items()]
    lines.append("")
    return "\n".join(lines)
