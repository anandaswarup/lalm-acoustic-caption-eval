"""Completion report for one model's caption run (E05-E08): counts and identifiers only, never caption text."""

from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from src.data.clotho import read_jsonl, sha256sum
from src.generation.runner import is_plain_text, latest_records, prompt_sha256
from src.utils.paths import REPO_ROOT

TOKEN_FIELDS = ("input", "input_audio", "input_text", "output", "thinking")


def _display(path: str | Path) -> str:
    """Repository-relative where possible, so committed reports carry no local paths."""
    path = Path(path).resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def is_wrapped_sentence(caption: str | None) -> bool:
    """True for a caption that is exactly {"<text>"}: the form the E09 normalization unwraps."""
    text = (caption or "").strip()
    return len(text) > 4 and text.startswith('{"') and text.endswith('"}') and '"' not in text[2:-2]


def normalize_usage(usage: dict[str, Any] | None) -> dict[str, int | None]:
    """Token counts under common names; None where the provider does not report a field.

    OpenAI: prompt_tokens / completion_tokens with prompt_tokens_details. Gemini: prompt_token_count /
    candidates_token_count / thoughts_token_count with per-modality prompt_tokens_details. HF: input_tokens /
    output_tokens.
    """
    usage = usage or {}
    counts: dict[str, int | None] = dict.fromkeys(TOKEN_FIELDS)
    if "prompt_tokens" in usage:
        details = usage.get("prompt_tokens_details") or {}
        output_details = usage.get("completion_tokens_details") or {}
        counts.update(
            input=usage.get("prompt_tokens"),
            input_audio=details.get("audio_tokens"),
            input_text=details.get("text_tokens"),
            output=usage.get("completion_tokens"),
            thinking=output_details.get("reasoning_tokens"),
        )
    elif "prompt_token_count" in usage:
        details = usage.get("prompt_tokens_details") or []
        modalities = {item.get("modality"): item.get("token_count") for item in details}
        counts.update(
            input=usage.get("prompt_token_count"),
            input_audio=modalities.get("AUDIO"),
            input_text=modalities.get("TEXT"),
            output=usage.get("candidates_token_count"),
            thinking=usage.get("thoughts_token_count"),
        )
    elif "input_tokens" in usage:
        counts.update(input=usage.get("input_tokens"), output=usage.get("output_tokens"))
    return counts


def _total(values: list[int | None]) -> int | None:
    known = [value for value in values if value is not None]
    return sum(known) if known else None


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "median": None, "max": None}
    return {
        "min": min(values),
        "mean": round(statistics.fmean(values), 2),
        "median": statistics.median(values),
        "max": max(values),
    }


def completion_report(
    output: str | Path, manifest: str | Path, prompt_text: str, meta: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Check the latest record per manifest sample against the frozen prompt and summarize the run.

    `complete` is True when every manifest sample has an `ok` record produced with `prompt_text`. JSON-like captions
    do not make a run incomplete: E09 unwraps exactly {"<text>"} and flags any other JSON-like output.
    """
    manifest_records = read_jsonl(manifest)
    sample_ids = [record["sample_id"] for record in manifest_records]
    all_latest = latest_records(output)
    latest = {sample_id: all_latest[sample_id] for sample_id in sample_ids if sample_id in all_latest}
    records = list(latest.values())
    ok = [record for record in records if record["status"] == "ok"]
    expected_sha = prompt_sha256(prompt_text)

    retried = [record for record in records if record.get("format_rejected")]
    json_like = [record for record in ok if not is_plain_text(record["caption"])]
    unwrappable = [record["sample_id"] for record in json_like if is_wrapped_sentence(record["caption"])]
    words = [len(record["caption"].split()) for record in ok]
    usages = [normalize_usage(record.get("usage")) for record in ok]
    peaks = [record["usage"]["peak_gpu_memory_gb"] for record in ok if "peak_gpu_memory_gb" in (record["usage"] or {})]
    latencies = [record["latency_s"] for record in records if record.get("latency_s") is not None]
    fingerprints = Counter((record.get("usage") or {}).get("system_fingerprint") for record in ok)
    fingerprints.pop(None, None)

    problems = {
        "missing": [sample_id for sample_id in sample_ids if sample_id not in latest],
        "not_ok": [
            {"sample_id": record["sample_id"], "status": record["status"], "error": (record.get("error") or "")[:200]}
            for record in records
            if record["status"] != "ok"
        ],
        "other_prompt": [record["sample_id"] for record in ok if record.get("prompt_sha256") != expected_sha],
    }
    report = {
        "output": {"path": _display(output), "sha256": sha256sum(output) if Path(output).exists() else None},
        "manifest": {"path": _display(manifest), "sha256": sha256sum(manifest), "samples": len(sample_ids)},
        "prompt_sha256": expected_sha,
        "complete": not any(problems.values()),
        "records": {
            "lines": len(read_jsonl(output)) if Path(output).exists() else 0,
            "samples_with_record": len(records),
            "not_in_manifest": len(set(all_latest) - set(sample_ids)),
        },
        "statuses": dict(sorted(Counter(record["status"] for record in records).items())),
        "problems": problems,
        "format": {
            "retried": len(retried),
            "fixed_by_retry": sum(is_plain_text(record["caption"]) for record in retried if record["status"] == "ok"),
            "rejected_responses": sum(len(record["format_rejected"]) for record in retried),
            "json_like": [record["sample_id"] for record in json_like],
            "json_like_unwrappable": unwrappable,
            "multiline": [record["sample_id"] for record in ok if "\n" in record["caption"].strip()],
        },
        "transient_retries": {
            "samples": sum((record.get("attempts") or 1) > 1 for record in records),
            "extra_attempts": sum((record.get("attempts") or 1) - 1 for record in records),
        },
        "finish_reasons": dict(sorted(Counter(str(record.get("finish_reason")) for record in ok).items())),
        "caption_words": _stats(words),
        "tokens": {name: _total([usage[name] for usage in usages]) for name in TOKEN_FIELDS},
        "peak_gpu_memory_gb": max(peaks) if peaks else None,
        "latency_s": {**_stats(latencies), "total": round(sum(latencies), 3)},
        "model_versions": dict(Counter(str(record.get("model_version")) for record in ok)),
        "system_fingerprints": dict(fingerprints),
        "deviations": dict(Counter(deviation for record in ok for deviation in record.get("deviations") or [])),
    }
    if meta is not None:
        # The meta file describes the last invocation only; a resumed run overwrites it.
        report["last_invocation"] = {
            "status": meta.get("status"),
            "started_utc": meta.get("started_utc"),
            "finished_utc": meta.get("finished_utc"),
            "git_commit": (meta.get("run_info") or {}).get("git_commit"),
            "git_dirty": (meta.get("run_info") or {}).get("git_dirty"),
            "generation_frozen": meta.get("generation_frozen"),
            "manifest_sha256_matches": (meta.get("manifest") or {}).get("sha256") == report["manifest"]["sha256"],
            "prompt_version": (meta.get("prompt") or {}).get("version"),
            "prompt_matches": (meta.get("prompt") or {}).get("text") == prompt_text,
            "decoding": meta.get("decoding"),
            "requests": meta.get("requests"),
            "adapter": meta.get("adapter"),
        }
    return report


def _ids(values: list[str], limit: int = 20) -> str:
    if not values:
        return "none"
    shown = ", ".join(f"`{value}`" for value in values[:limit])
    return f"{len(values)}: {shown}" + (f" … (+{len(values) - limit})" if len(values) > limit else "")


def _fmt(value: Any) -> str:
    if isinstance(value, dict):
        return "; ".join(f"{key}: {_fmt(item)}" for key, item in value.items()) or "none"
    return "n/a" if value is None else str(value)


def report_markdown(report: dict[str, Any], title: str) -> str:
    problems, fmt = report["problems"], report["format"]
    sections = [
        (
            "Completeness",
            [
                ("Complete", "**yes**" if report["complete"] else "**no**"),
                ("Manifest samples", report["manifest"]["samples"]),
                ("Samples with a record", report["records"]["samples_with_record"]),
                ("Record lines (all attempts)", report["records"]["lines"]),
                ("Statuses", _fmt(report["statuses"])),
                ("Missing", _ids(problems["missing"])),
                ("Not ok", _ids([f"{item['sample_id']} ({item['status']})" for item in problems["not_ok"]])),
                ("Ok with another prompt", _ids(problems["other_prompt"])),
                ("Prompt SHA-256", f"`{report['prompt_sha256']}`"),
            ],
        ),
        (
            "Format",
            [
                ("Format retries (samples)", fmt["retried"]),
                ("Fixed by a retry", fmt["fixed_by_retry"]),
                ("Rejected responses", fmt["rejected_responses"]),
                ("Still JSON-like", _ids(fmt["json_like"])),
                ('Of which exactly {"…"} (E09 unwraps)', _ids(fmt["json_like_unwrappable"])),
                ("Multi-line captions", _ids(fmt["multiline"])),
                ("Finish reasons", _fmt(report["finish_reasons"])),
                ("Caption words", _fmt(report["caption_words"])),
            ],
        ),
        (
            "Usage",
            [
                ("Tokens (ok responses)", _fmt(report["tokens"])),
                ("Peak GPU memory (GB)", _fmt(report["peak_gpu_memory_gb"])),
                ("Latency (s)", _fmt(report["latency_s"])),
                ("Samples with transient retries", report["transient_retries"]["samples"]),
                ("Extra attempts", report["transient_retries"]["extra_attempts"]),
            ],
        ),
        (
            "Model",
            [
                ("Model versions", _fmt(report["model_versions"])),
                ("System fingerprints", _fmt(report["system_fingerprints"])),
                ("Deviations", _fmt(report["deviations"])),
            ],
        ),
    ]
    if last := report.get("last_invocation"):
        sections.append(
            (
                "Last invocation (meta file)",
                [(key.replace("_", " ").capitalize(), _fmt(value)) for key, value in last.items()],
            )
        )
    lines = [f"# {title}", "", f"Output: `{report['output']['path']}` (SHA-256 `{report['output']['sha256']}`)", ""]
    for heading, rows in sections:
        lines += [f"## {heading}", "", "| Item | Value |", "|---|---|"]
        lines += [f"| {key} | {str(value).replace('|', '/')} |" for key, value in rows]
        lines.append("")
    return "\n".join(lines)
