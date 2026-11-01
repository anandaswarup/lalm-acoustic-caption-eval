"""Run open-weight caption generation on Hugging Face Jobs (E04, E08, E09).

All data lives in one private bucket (configs/models.yaml `hf_jobs.bucket`), mounted read-write in the job:

    clotho/audio/{evaluation,validation}/*.wav   the manifest WAVs only (upload-audio)
    code/<commit>/                               snapshot of the committed code (submit)
    outputs/<run>/<model>_captions.jsonl         caption outputs, resumable (fetch pulls them back)
    <prefix>/<run>/<model>_captions.jsonl        test runs kept apart with --output-prefix (e.g. smoke)

Subcommands:
    compute MODEL     print the model's compute placement (used by the Makefile)
    upload-audio      upload the WAVs referenced by the manifests
    submit            snapshot the committed code and submit a detached job
    fetch             pull outputs/ from the bucket into outputs/captions/
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clotho import read_jsonl  # noqa: E402
from src.utils.cli import base_parser, init_run  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import OUTPUT_DIR, REPO_ROOT, resolve_path  # noqa: E402

# Paths shipped to the job; anything else (e.g. .env, docs/, outputs/) never leaves the laptop.
SNAPSHOT_PATHS = ["src", "scripts", "configs", "data/manifests", "requirements-hf.txt"]
# Bucket prefixes an --output-prefix must not write into.
RESERVED_PREFIXES = {"clotho", "code"}


def bucket_url(settings: dict, prefix: str = "") -> str:
    return f"hf://buckets/{settings['bucket']}" + (f"/{prefix}" if prefix else "")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def manifest_audio_paths(generation: dict) -> list[str]:
    paths = set()
    for run in generation["runs"].values():
        paths.update(record["audio_path"] for record in read_jsonl(resolve_path(run["manifest"])))
    return sorted(paths)


def stage_audio(clotho_root: Path, audio_paths: list[str], staging: Path) -> None:
    """Mirror only the needed WAVs under `staging` (hard links where possible; copies keep mtimes)."""
    for relative in audio_paths:
        source, target = clotho_root / relative, staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def output_prefix(value: str) -> str:
    """argparse type: a single path segment outside the audio and code prefixes."""
    if not value or "/" in value or value.startswith(".") or value in RESERVED_PREFIXES:
        raise argparse.ArgumentTypeError(f"{value!r} must be one path segment, not one of {sorted(RESERVED_PREFIXES)}")
    return value


def job_script(commit: str, model: str, run: str, mount: str, extra_args: list[str], prefix: str = "outputs") -> str:
    code = f"{mount}/code/{commit}"
    generate = [
        "python",
        "scripts/generate_captions.py",
        "--model",
        model,
        "--run",
        run,
        "--output-dir",
        f"{mount}/{prefix}/{run}",
        "--no-log-file",
        *extra_args,
    ]
    return " && ".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(code)}",
            "python --version",
            "(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true)",
            "uv pip install --system --index-strategy unsafe-best-match -r requirements-hf.txt",
            shlex.join(generate),
        ]
    )


def submit_command(
    settings: dict, model_key: str, model: dict, run: str, commit: str, extra_args: list[str], prefix: str = "outputs"
) -> list[str]:
    mount = settings["mount"]
    return [
        "hf", "jobs", "run", "--detach",
        "--flavor", model["hf_jobs"]["flavor"],
        "--timeout", str(settings["timeout"]),
        "--secrets", "HF_TOKEN",
        "--env", f"CLOTHO_ROOT={mount}/clotho",
        "--env", f"LALM_GIT_COMMIT={commit}",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--env", "PYTHONUNBUFFERED=1",
        "--label", f"model={model_key}", "--label", f"run={run}", "--label", f"commit={commit[:12]}",
        "--label", f"prefix={prefix}",
        "--volume", f"{bucket_url(settings)}:{mount}",
        settings["image"],
        "bash", "-c", job_script(commit, model_key, run, mount, extra_args, prefix),
    ]  # fmt: skip


def run_or_print(logger, command: list[str], dry_run: bool) -> int:
    logger.info("%s %s", "Would run:" if dry_run else "Running:", shlex.join(command))
    return 0 if dry_run else subprocess.run(command, cwd=REPO_ROOT).returncode


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    dry_run = argparse.ArgumentParser(add_help=False)
    dry_run.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    prefix = argparse.ArgumentParser(add_help=False)
    prefix.add_argument(
        "--output-prefix", type=output_prefix, default="outputs", help="bucket prefix for outputs (default: outputs)"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("compute", help="print a model's compute placement").add_argument("model")
    commands.add_parser("upload-audio", parents=[dry_run], help="upload the manifest WAVs to the bucket")
    submit = commands.add_parser("submit", parents=[dry_run, prefix], help="submit a caption-generation job")
    submit.add_argument("--model", required=True)
    submit.add_argument("--run", required=True)
    submit.add_argument("--limit", type=int)
    submit.add_argument("--sample-ids", nargs="+")
    fetch = commands.add_parser("fetch", parents=[dry_run, prefix], help="pull caption outputs from the bucket")
    fetch.add_argument("--dest", help="local directory (default: outputs/captions, or outputs/<prefix> otherwise)")
    args = parser.parse_args()

    models_config = load_config("models", args.overrides)
    models, settings = models_config["models"], models_config["hf_jobs"]
    if args.command == "compute":
        print(models[args.model]["compute"])
        return 0

    logger = init_run(Path(__file__).stem, args)
    generation = load_config("generation", args.overrides)

    if args.command == "upload-audio":
        clotho_root = resolve_path(load_config("dataset")["root"])
        audio_paths = manifest_audio_paths(generation)
        logger.info("Uploading %d WAVs from %s to %s", len(audio_paths), clotho_root, bucket_url(settings, "clotho"))
        with tempfile.TemporaryDirectory(prefix="lalm_audio_") as staging:
            stage_audio(clotho_root, audio_paths, Path(staging))
            command = ["hf", "buckets", "sync", staging, bucket_url(settings, "clotho")]
            return run_or_print(logger, command + (["--dry-run"] if args.dry_run else []), dry_run=False)

    if args.command == "fetch":
        default_dest = OUTPUT_DIR / ("captions" if args.output_prefix == "outputs" else args.output_prefix)
        dest = resolve_path(args.dest) if args.dest else default_dest
        command = ["hf", "buckets", "sync", bucket_url(settings, args.output_prefix), str(dest)]
        return run_or_print(logger, command, args.dry_run)

    # submit
    if args.model not in models or models[args.model]["compute"] != "hf_jobs":
        logger.error("%r is not an hf_jobs model; run it locally with scripts/generate_captions.py", args.model)
        return 1
    if args.run not in generation["runs"]:
        logger.error("Unknown run %r; expected one of %s", args.run, sorted(generation["runs"]))
        return 1
    # Untracked files (e.g. caption outputs) never reach the job: the snapshot is `git archive HEAD`.
    if git("status", "--porcelain", "--untracked-files=no"):
        logger.error("Uncommitted changes: jobs run committed code only, so commit first")
        return 1
    commit = git("rev-parse", "HEAD")
    extra = (["--limit", str(args.limit)] if args.limit else []) + (
        ["--sample-ids", *args.sample_ids] if args.sample_ids else []
    )
    with tempfile.TemporaryDirectory(prefix="lalm_code_") as snapshot:
        archive = subprocess.run(
            ["git", "archive", "--format=tar", "HEAD", *SNAPSHOT_PATHS], cwd=REPO_ROOT, capture_output=True, check=True
        ).stdout
        subprocess.run(["tar", "-x", "-C", snapshot], input=archive, check=True)
        sync = ["hf", "buckets", "sync", snapshot, bucket_url(settings, f"code/{commit}")]
        if code := run_or_print(logger, sync, args.dry_run):
            return code
    command = submit_command(settings, args.model, models[args.model], args.run, commit, extra, args.output_prefix)
    if code := run_or_print(logger, command, args.dry_run):
        return code
    if not args.dry_run:
        fetch = "make fetch-hf"
        if args.output_prefix != "outputs":
            fetch = f"scripts/hf_jobs.py fetch --output-prefix {args.output_prefix}"
        logger.info("Follow with: hf jobs logs <job_id> --follow; then %s", fetch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
