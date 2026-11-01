"""Resumable caption generation over a frozen manifest (E04-E09)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.data.clotho import read_jsonl, sha256sum
from src.models.base import CaptionAdapter, CaptionRequest, CaptionResult, TransientError

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def load_samples(
    manifest: str | Path, clotho_root: str | Path, limit: int | None = None, sample_ids: Iterable[str] | None = None
) -> list[dict[str, Any]]:
    """Manifest records in manifest order, with `audio_path` resolved against `clotho_root`."""
    records = read_jsonl(manifest)
    if sample_ids is not None:
        wanted = set(sample_ids)
        if unknown := wanted - {record["sample_id"] for record in records}:
            raise ValueError(f"sample_ids not in manifest: {sorted(unknown)[:5]}")
        records = [record for record in records if record["sample_id"] in wanted]
    if limit is not None:
        records = records[:limit]
    root = Path(clotho_root)
    return [{**record, "resolved_audio_path": root / record["audio_path"]} for record in records]


def latest_records(output: str | Path) -> dict[str, dict[str, Any]]:
    """Last record per sample_id in an output JSONL (earlier attempts are kept in the file)."""
    path = Path(output)
    return {record["sample_id"]: record for record in read_jsonl(path)} if path.exists() else {}


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def completed_ids(output: str | Path, prompt: str) -> set[str]:
    """Samples whose latest record is `ok` and was produced with this exact prompt text."""
    sha = prompt_sha256(prompt)
    return {
        sample_id
        for sample_id, record in latest_records(output).items()
        if record["status"] == "ok" and record.get("prompt_sha256") == sha
    }


def append_record(output: Path, record: dict[str, Any]) -> None:
    with output.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def caption_with_retries(
    adapter: CaptionAdapter,
    request: CaptionRequest,
    max_retries: int,
    backoff_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> CaptionResult:
    """Retry TransientError with exponential backoff; any other exception becomes an error result."""
    attempts = 0
    start = time.perf_counter()
    while True:
        attempts += 1
        try:
            result = adapter.caption(request)
            break
        except TransientError as error:
            if attempts > max_retries:
                result = adapter.error_result(request, f"TransientError after {attempts} attempts: {error}")
                break
            delay = backoff_seconds * 2 ** (attempts - 1)
            logger.warning(
                "%s: transient failure (attempt %d), retrying in %.1fs: %s", request.sample_id, attempts, delay, error
            )
            sleep(delay)
        except Exception as error:  # recorded, not raised: one bad sample must not stop the run
            result = adapter.error_result(request, error)
            break
    result.latency_s = round(time.perf_counter() - start, 3)
    result.attempts = attempts
    result.timestamp_utc = _now()
    result.model_key = adapter.model_key
    return result


def is_plain_text(caption: str | None) -> bool:
    """False for JSON-like output (an object or array instead of a sentence)."""
    return not (caption or "").lstrip().startswith(("{", "["))


def caption_sample(
    adapter: CaptionAdapter,
    sample: dict[str, Any],
    prompt: str,
    decoding: dict[str, Any],
    requests_config: dict[str, Any],
    sleep: Callable[[float], None] = time.sleep,
) -> CaptionResult:
    """Caption one manifest sample; audio that is missing or does not match the manifest is never sent.

    An `ok` caption that is not plain text is requested again, up to `format_attempts` requests in total (the same
    rule for every model). The last response is kept as the result; the rejected ones go to `format_rejected`.
    """
    request = CaptionRequest(sample["sample_id"], sample["resolved_audio_path"], prompt, dict(decoding))
    if not request.audio_path.is_file():
        result = adapter.error_result(request, f"audio file not found: {request.audio_path}")
    elif sha256sum(request.audio_path) != sample["audio_sha256"]:
        result = adapter.error_result(request, "audio SHA-256 does not match the manifest")
    else:
        format_attempts = max(1, int(requests_config.get("format_attempts", 1)))
        rejected = []
        for attempt in range(1, format_attempts + 1):
            result = caption_with_retries(
                adapter,
                request,
                requests_config.get("max_retries", 5),
                requests_config.get("backoff_seconds", 2.0),
                sleep,
            )
            if result.status != "ok" or is_plain_text(result.caption) or attempt == format_attempts:
                break
            rejected.append(
                {
                    key: getattr(result, key)
                    for key in ("caption", "raw_response", "usage", "finish_reason", "latency_s", "timestamp_utc")
                }
            )
        result.format_rejected = rejected
    result.model_key = adapter.model_key
    result.prompt_sha256 = prompt_sha256(prompt)
    result.timestamp_utc = result.timestamp_utc or _now()
    return result


def caption_all(
    adapter: CaptionAdapter,
    samples: list[dict[str, Any]],
    concurrency: int,
    *args: Any,
) -> Iterator[tuple[dict[str, Any], CaptionResult]]:
    """Yield (sample, result) as requests finish: in manifest order when sequential, else in completion order."""
    if concurrency <= 1:
        for sample in samples:
            yield sample, caption_sample(adapter, sample, *args)
        return
    pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="caption")
    try:
        futures = {pool.submit(caption_sample, adapter, sample, *args): sample for sample in samples}
        for future in as_completed(futures):
            yield futures[future], future.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)  # on interrupt, queued requests are dropped


def summarize(output: str | Path, sample_ids: list[str]) -> dict[str, Any]:
    """Status counts and latency over the latest record of each sample in `sample_ids`."""
    latest = latest_records(output)
    records = [latest[sample_id] for sample_id in sample_ids if sample_id in latest]
    statuses = Counter(record["status"] for record in records)
    latencies = sorted(record["latency_s"] for record in records if record.get("latency_s") is not None)
    return {
        "samples": len(sample_ids),
        "with_record": len(records),
        "statuses": dict(sorted(statuses.items())),
        "missing": [sample_id for sample_id in sample_ids if sample_id not in latest],
        "failed": [record["sample_id"] for record in records if record["status"] != "ok"],
        "latency_s": {
            "median": latencies[len(latencies) // 2] if latencies else None,
            "max": latencies[-1] if latencies else None,
            "total": round(sum(latencies), 3),
        },
    }


def write_meta(path: Path, meta: dict[str, Any]) -> None:
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def run_generation(
    adapter: CaptionAdapter,
    samples: list[dict[str, Any]],
    output: str | Path,
    prompt: str,
    decoding: dict[str, Any],
    requests_config: dict[str, Any],
    meta: dict[str, Any],
    resume: bool = True,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Caption every sample not already `ok` in `output`; return the summary over all `samples`.

    With `requests_config["concurrency"]` > 1, that many requests run at once in threads; records are still
    appended one at a time from this thread, in completion order (records are keyed by sample_id).
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output.with_name(output.name.removesuffix(".jsonl") + ".meta.json")
    done = completed_ids(output, prompt) if resume else set()
    todo = [sample for sample in samples if sample["sample_id"] not in done]
    logger.info(
        "%d samples, %d already ok, %d to caption (concurrency %d) -> %s",
        len(samples),
        len(samples) - len(todo),
        len(todo),
        requests_config.get("concurrency", 1),
        output,
    )

    adapter.setup()
    started = _now()
    meta = {**meta, "adapter": adapter.describe(), "output": str(output), "started_utc": started, "status": "running"}
    write_meta(meta_path, meta)

    concurrency = int(requests_config.get("concurrency", 1))
    results = caption_all(adapter, todo, concurrency, prompt, decoding, requests_config, sleep)
    for index, (sample, result) in enumerate(results, 1):
        append_record(output, result.to_dict())
        logger.info(
            "[%d/%d] %s %s (%.1fs) %s",
            index,
            len(todo),
            sample["sample_id"],
            result.status,
            result.latency_s or 0.0,
            (result.caption or result.error or "")[:100],
        )
        if result.error_traceback:
            logger.warning("%s traceback:\n%s", sample["sample_id"], result.error_traceback.rstrip())

    summary = summarize(output, [sample["sample_id"] for sample in samples])
    write_meta(meta_path, {**meta, "finished_utc": _now(), "status": "finished", "summary": summary})
    return summary
