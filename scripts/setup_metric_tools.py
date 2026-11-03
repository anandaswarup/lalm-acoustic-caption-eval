"""Download and verify the external tools the caption metrics need (E11).

Installs, under `tools.dir` of configs/metrics.yaml (git-ignored): a JDK 11 for METEOR, SPICE and the PTB tokenizer
(aac-metrics rejects Java > 13 and SPICE needs Nashorn); CoreNLP 3.6.0 from Stanford's mirror; the jars and models
fetched by aac-metrics; the MS-CLAP 2023 weights and its GPT-2 text encoder; and the BERTScore rescaling baseline.
Every download is checked against the SHA-256 or commit pinned in the config, then MANIFEST.json records the SHA-256 of
every file. Files already in place are kept, so the script can be rerun; --force downloads everything again.
Run it with the metrics environment: .venv-metrics/bin/python (make metric-tools).
"""

import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clotho import sha256sum  # noqa: E402
from src.metrics.tools import (  # noqa: E402
    activate,
    check_java,
    hf_revision_problem,
    java_path,
    jdk_build,
    platform_key,
    verify_manifest,
    write_manifest,
)
from src.utils.cli import base_parser, init_run  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.run_info import collect_run_info  # noqa: E402

CORENLP_ZIP = "SPICE-1.0/stanford-corenlp-full-2015-12-09.zip"  # where aac-metrics looks for it, below spice/


def download(url: str, dest: Path, sha256: str, logger) -> None:
    """Download `url` to `dest` unless it is already there with the expected SHA-256; fail on a mismatch."""
    if dest.is_file() and sha256sum(dest) == sha256:
        logger.info("Present: %s", dest.name)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    logger.info("Downloading %s", url)
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as stream:
        shutil.copyfileobj(response, stream, 1 << 20)
    if (actual := sha256sum(partial)) != sha256:
        partial.unlink()
        raise RuntimeError(f"{url}: SHA-256 {actual} does not match the pinned {sha256}")
    partial.replace(dest)


def install_jdk(tools: dict, paths: dict, logger) -> str:
    java = java_path(tools)
    if tools.get("java_path") or java.exists():
        return check_java(java)
    build = jdk_build(tools)
    archive = paths["root"] / "downloads" / unquote(Path(urlparse(build["url"]).path).name)
    download(build["url"], archive, build["sha256"], logger)
    staging = paths["root"] / "tmp" / "jdk"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        tar.extractall(staging, filter="data")
    (top,) = staging.iterdir()
    top.rename(paths["jdk"])
    shutil.rmtree(staging)
    archive.unlink()
    return check_java(java)


def install_aac_metrics(tools: dict, paths: dict, logger) -> None:
    """Jars and models fetched by aac-metrics, with CoreNLP placed first so aac-metrics skips its unreachable URL."""
    if not (paths["spice"] / "lib" / "stanford-corenlp-3.6.0.jar").exists():
        download(tools["corenlp"]["url"], paths["spice"] / CORENLP_ZIP, tools["corenlp"]["sha256"], logger)
    from aac_metrics.download import download_metrics

    download_metrics(cache_path=paths["root"], tmp_path=paths["root"] / "tmp", clap=False, verbose=1)


def install_clap(tools: dict, logger) -> Path:
    """MS-CLAP 2023 weights at the pinned commit; loading the model once also caches its GPT-2 text encoder."""
    from huggingface_hub import hf_hub_download
    from msclap import CLAP

    clap = tools["clap"]
    weights = Path(hf_hub_download(clap["repo"], clap["filename"], revision=tools["hf_models"][clap["repo"]]))
    if (actual := sha256sum(weights)) != clap["sha256"]:
        raise RuntimeError(f"{weights}: SHA-256 {actual} does not match the pinned {clap['sha256']}")
    logger.info("Loading MS-CLAP 2023 once to cache its text encoder")
    CLAP(model_fp=weights, version="2023", use_cuda=False)
    return weights


def check_pins(tools: dict, paths: dict) -> list[str]:
    errors = []
    for name, digest in tools["files"].items():
        path = paths["root"] / name
        if not path.is_file():
            errors.append(f"{name}: missing")
        elif sha256sum(path) != digest:
            errors.append(f"{name}: SHA-256 does not match the pinned value")
    for repo, revision in tools["hf_models"].items():
        if problem := hf_revision_problem(paths["hf_home"], repo, revision):
            errors.append(problem)
    return errors


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="delete tools.dir and download everything again")
    parser.add_argument("--check", action="store_true", help="only verify tools.dir against MANIFEST.json")
    args = parser.parse_args()
    logger = init_run(Path(__file__).stem, args)
    tools = load_config("metrics", args.overrides)["tools"]
    paths = activate(tools)

    if args.check:
        errors = verify_manifest(paths["root"])
        if not errors:
            try:
                logger.info("Java %s at %s", check_java(java_path(tools)), java_path(tools))
            except RuntimeError as error:
                errors.append(str(error))
        for error in errors:
            logger.error("check: %s", error)
        if not errors:
            logger.info("All files in %s match", paths["manifest"])
        return 1 if errors else 0

    if args.force and paths["root"].exists():
        logger.warning("--force: deleting %s", paths["root"])
        shutil.rmtree(paths["root"])
    paths["root"].mkdir(parents=True, exist_ok=True)

    java_version = install_jdk(tools, paths, logger)
    logger.info("Java %s at %s", java_version, java_path(tools))
    install_aac_metrics(tools, paths, logger)
    clap_weights = install_clap(tools, logger)
    baseline = tools["bertscore_baseline"]
    download(baseline["url"], paths["bertscore_baseline"], baseline["sha256"], logger)
    shutil.rmtree(paths["root"] / "tmp", ignore_errors=True)
    shutil.rmtree(paths["root"] / "downloads", ignore_errors=True)

    if errors := check_pins(tools, paths):
        for error in errors:
            logger.error("%s", error)
        logger.error("%d pinned files or revisions do not match; MANIFEST.json was not written", len(errors))
        return 1
    java = java_path(tools)
    manifest = write_manifest(
        paths["root"],
        {
            "run_info": collect_run_info(Path(__file__).stem, seed=args.seed),
            "platform": platform_key(),
            "java": {
                "path": str(java.relative_to(paths["root"]) if java.is_relative_to(paths["root"]) else java),
                "version": java_version,
            },
            "jdk_release": tools["jdk"]["release"],
            "clap_weights": str(clap_weights.relative_to(paths["root"])),
        },
    )
    logger.info("Wrote %s", manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
