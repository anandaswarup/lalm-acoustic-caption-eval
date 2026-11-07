"""Blinded audio-grounded review material for the qualitative sample.

E18 Part 1: run once the E17 labels are final. Verifies the blinding key against sample_manifest.json and the E17
template against review_items.csv, copies each sampled recording (checked against its manifest SHA-256) to
`output_dir`/audio/<code>.wav, and writes the local listening page audio_review.html (git-ignored, like the audio), the
text view audio_review_sheet.md, a template with one row per caption (audio rating and notes), a template with one row
per E17 disputed phrase (verdict and note) and audio_review_manifest.json. The material keeps the E16 codes and
labels and holds no references, E17 categories, model names, file names or scores. Templates that hold labels are
never overwritten.
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.annotations import check_annotations, read_table  # noqa: E402
from src.analysis.audio_review import (  # noqa: E402
    AUDIO_COLUMNS,
    AUDIO_FIELDS,
    PHRASE_COLUMNS,
    PHRASE_FIELDS,
    WAV_CHUNKS,
    audio_template,
    has_labels,
    phrase_template,
    review_page_html,
    review_sheet_markdown,
    text_leaks,
    wav_chunks,
)
from src.analysis.qualitative import REFERENCE_COLUMNS  # noqa: E402
from src.data.clotho import read_jsonl, sha256sum  # noqa: E402
from src.data.manifest import write_json  # noqa: E402
from src.evaluation.inputs import audio_paths  # noqa: E402
from src.evaluation.pipeline import existing_outputs, relative  # noqa: E402
from src.evaluation.scoring import write_csv  # noqa: E402
from src.utils.cli import base_parser, init_run  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import resolve_path  # noqa: E402


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs (never labelled templates)")
    args = parser.parse_args()
    logger = init_run(Path(__file__).stem, args)
    config = load_config("qualitative", args.overrides)
    if config["reviewers"] != 1:
        logger.error("Only one reviewer is supported (configs/qualitative.yaml has %s)", config["reviewers"])
        return 1
    codes = {category["code"] for category in config["categories"]}

    out_dir = resolve_path(config["output_dir"])
    inputs = {
        "review": out_dir / "review_items.csv",
        "annotations": out_dir / "annotations_reviewer_1.csv",
        "sample_manifest": out_dir / "sample_manifest.json",
        "key": out_dir / "blinding_key.csv",
    }
    templates = {
        "audio_template": out_dir / "audio_annotations_reviewer_1.csv",
        "phrase_template": out_dir / "disputed_verification_reviewer_1.csv",
    }
    outputs = {
        "audio_dir": out_dir / "audio",
        "page": out_dir / "audio_review.html",
        "sheet": out_dir / "audio_review_sheet.md",
        **templates,
        "manifest": out_dir / "audio_review_manifest.json",
    }
    fields = {"audio_template": AUDIO_FIELDS, "phrase_template": PHRASE_FIELDS}
    labelled = [relative(p) for name, p in templates.items() if p.is_file() and has_labels(read_table(p), fields[name])]
    if labelled:
        logger.error("Templates already hold labels and are never overwritten: %s", labelled)
        return 1
    if not args.force and (existing := existing_outputs(outputs)):
        logger.error("Outputs exist (use --force to overwrite): %s", existing)
        return 1

    sample_manifest = json.loads(inputs["sample_manifest"].read_text(encoding="utf-8"))
    if not inputs["key"].is_file() or sha256sum(inputs["key"]) != sample_manifest["key"]["sha256"]:
        logger.error("%s is missing or does not match sample_manifest.json", relative(inputs["key"]))
        return 1
    review, annotations = read_table(inputs["review"]), read_table(inputs["annotations"])
    if errors := check_annotations(annotations, review, codes):
        for error in errors[:20]:
            logger.error("%s", error)
        logger.error("The E17 template is not complete; finish E17 first")
        return 1

    dataset, models = load_config("dataset", args.overrides), load_config("models", args.overrides)["models"]
    key = read_table(inputs["key"])
    code_to_sample = {row["recording_code"]: row["sample_id"] for row in key}
    by_sample = {r["sample_id"]: r for r in read_jsonl(resolve_path(dataset["evaluation"]["manifest"]))}
    records = [by_sample[sample_id] for sample_id in code_to_sample.values()]
    try:
        sources = audio_paths(records, resolve_path(dataset["root"]))  # verified against the manifest SHA-256
    except ValueError as error:
        logger.error("%s", error)
        return 1
    if extra := sorted({c for p in sources.values() for c in wav_chunks(p)} - set(WAV_CHUNKS)):
        logger.error("Audio files carry chunks that may hold metadata: %s", extra)
        return 1

    audio_rows, phrase_rows = audio_template(review), phrase_template(annotations)
    page = review_page_html(review, phrase_rows, "audio")
    sheet = review_sheet_markdown(review, phrase_rows)
    terms = [*models, *(model["display_name"] for model in models.values())]
    terms += [record["file_name"] for record in records]
    terms += sorted({row[column] for row in review for column in REFERENCE_COLUMNS})
    texts = {"page": page, "sheet": sheet}
    texts |= {f"{row['item_id']}": " ".join(row.values()) for row in audio_rows}
    texts |= {f"{row['phrase_id']}": row["phrase"] for row in phrase_rows}
    if leaks := text_leaks(texts, terms):
        for leak in leaks[:50]:
            logger.error("Blinding: %s", leak)
        return 1

    outputs["audio_dir"].mkdir(parents=True, exist_ok=True)
    for code, sample_id in code_to_sample.items():
        target = outputs["audio_dir"] / f"{code}.wav"
        shutil.copyfile(sources[sample_id], target)
        if sha256sum(target) != by_sample[sample_id]["audio_sha256"]:
            logger.error("%s: copy does not match the manifest SHA-256", relative(target))
            return 1
    outputs["page"].write_text(page, encoding="utf-8")
    outputs["sheet"].write_text(sheet, encoding="utf-8")
    write_csv(audio_rows, outputs["audio_template"], AUDIO_COLUMNS)
    write_csv(phrase_rows, outputs["phrase_template"], PHRASE_COLUMNS)
    manifest = {
        "task": "E18",
        "num_recordings": len(code_to_sample),
        "num_items": len(audio_rows),
        "num_phrases": len(phrase_rows),
        "reviewers": config["reviewers"],
        "templates": [relative(path) for path in templates.values()],
        "inputs": {name: {"path": relative(path), "sha256": sha256sum(path)} for name, path in inputs.items()},
        "local_only": [relative(outputs["audio_dir"]), relative(outputs["page"])],
    }
    write_json(manifest, outputs["manifest"])
    counts = len(code_to_sample), len(audio_rows), len(phrase_rows)
    logger.info("%d recordings, %d captions, %d disputed phrases", *counts)
    logger.info("Wrote %s", sorted(relative(path) for path in outputs.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
