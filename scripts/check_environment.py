"""Validate the execution environment: Python, dependencies, configs, tools, dataset and credentials.

Credentials are reported only as present/missing; their values are never logged.
"""

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.cli import base_parser, init_run  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import CONFIG_DIR, resolve_path  # noqa: E402

MIN_PYTHON = (3, 12)
REQUIRED_MODULES = ["numpy", "pandas", "yaml", "dotenv", "soundfile", "tqdm"]
OPTIONAL_TOOLS = {"java": "METEOR/SPICE (E11)", "latexmk": "manuscript build", "hf": "Hugging Face Jobs"}
CREDENTIALS = ["OPENAI_API_KEY", "GEMINI_API_KEY", "HF_TOKEN"]


def check_dataset(config: dict) -> tuple[bool, str]:
    root = resolve_path(config["root"])
    split = config["evaluation"]["split"]
    audio_dir = root / "audio" / split
    captions = root / "captions" / f"clotho_captions_{split}.csv"
    if not audio_dir.is_dir() or not captions.is_file():
        return False, f"not found under {root} (set CLOTHO_ROOT in .env)"
    num_wavs = sum(1 for _ in audio_dir.glob("*.wav"))
    expected = config["evaluation"]["expected_recordings"]
    return num_wavs == expected, f"{root}: {num_wavs} {split} WAV files (expected {expected})"


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--strict", action="store_true", help="fail on missing dataset, tools or credentials")
    args = parser.parse_args()
    logger = init_run(Path(__file__).stem, args)

    errors, warnings = [], []

    version = sys.version_info[:2]
    if version < MIN_PYTHON:
        errors.append(f"Python {version[0]}.{version[1]} < required {MIN_PYTHON[0]}.{MIN_PYTHON[1]}")
    logger.info("Python %s (%s)", sys.version.split()[0], sys.executable)

    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
        except ImportError:
            errors.append(f"missing Python module '{module}' (run: make install)")

    configs = {}
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        try:
            configs[path.stem] = load_config(path.stem)
        except Exception as exc:  # noqa: BLE001 - report any config failure
            errors.append(f"config {path.name}: {exc}")
    logger.info("Loaded configs: %s", ", ".join(configs))

    for tool, purpose in OPTIONAL_TOOLS.items():
        if shutil.which(tool) is None:
            warnings.append(f"'{tool}' not found on PATH (needed for {purpose})")
    if shutil.which("java"):
        result = subprocess.run(["java", "-version"], capture_output=True, text=True)
        logger.info("Java: %s", (result.stderr or result.stdout).splitlines()[0])

    if "dataset" in configs:
        ok, message = check_dataset(configs["dataset"])
        if ok:
            logger.info("Dataset: %s", message)
        else:
            warnings.append(f"dataset: {message}")

    for name in CREDENTIALS:
        if not os.environ.get(name):
            warnings.append(f"credential {name} is not set")
    logger.info("Credentials set: %s", [name for name in CREDENTIALS if os.environ.get(name)])

    for message in warnings:
        logger.warning(message)
    for message in errors:
        logger.error(message)

    failed = bool(errors) or (args.strict and bool(warnings))
    logger.info(
        "Environment check %s (%d errors, %d warnings)", "FAILED" if failed else "passed", len(errors), len(warnings)
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
