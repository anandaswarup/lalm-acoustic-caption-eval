"""Representative caption comparisons, tables and qualitative figures.

E19: run after E18 (make audio-review). Joins the review items (references and captions) with the unblinded E17/E18
labels, selects one recording per pattern by the rules in src/analysis/examples.py, and writes the example tables and
a supplementary table of every sampled recording (Markdown and LaTeX) to outputs/tables/, the figures (E17 categories
and E18 ratings per model, E17 x E18 heatmap) to outputs/figures/, and the selection manifest
outputs/qualitative/example_manifest.json with every eligible recording and the input checksums.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.annotations import read_table  # noqa: E402
from src.analysis.examples import (  # noqa: E402
    PATTERNS,
    by_recording,
    caption_records,
    examples_latex,
    examples_markdown,
    recording_example,
    select_examples,
)
from src.analysis.qualitative_figures import CATEGORY_COLORS, RATING_COLORS, heatmap, save, stacked_bars  # noqa: E402
from src.data.clotho import read_jsonl, sha256sum  # noqa: E402
from src.data.manifest import write_json  # noqa: E402
from src.evaluation.pipeline import existing_outputs, relative  # noqa: E402
from src.utils.cli import base_parser, init_run  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import resolve_path  # noqa: E402
from src.utils.run_info import collect_run_info  # noqa: E402

FIGURES = ("e17_categories_by_model", "e18_ratings_by_model", "e17_e18_heatmap")


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = parser.parse_args()
    logger = init_run(Path(__file__).stem, args)
    run_info = collect_run_info(Path(__file__).stem, seed=args.seed)  # before this run writes any output
    config = load_config("qualitative", args.overrides)
    settings = config["examples"]
    out_dir = resolve_path(config["output_dir"])
    tables_dir, figures_dir = resolve_path(settings["tables_dir"]), resolve_path(settings["figures_dir"])
    inputs = {
        "review": out_dir / "review_items.csv",
        "captions": out_dir / "audio_review_annotations.csv",
        "phrases": out_dir / "disputed_verification.csv",
        "audio_review_manifest": out_dir / "audio_review_manifest.json",
    }
    outputs = {
        "examples_md": tables_dir / "representative_examples.md",
        "examples_tex": tables_dir / "representative_examples.tex",
        "all_md": tables_dir / "qualitative_sample_all.md",
        "all_tex": tables_dir / "qualitative_sample_all.tex",
        **{f"{name}_{fmt}": figures_dir / f"{name}.{fmt}" for name in FIGURES for fmt in settings["figure_formats"]},
        "manifest": out_dir / "example_manifest.json",
    }
    if not args.force and (existing := existing_outputs(outputs)):
        logger.error("Outputs exist (use --force to overwrite): %s", existing)
        return 1
    if missing := [relative(path) for path in inputs.values() if not path.is_file()]:
        logger.error("Missing inputs (run make audio-review first): %s", missing)
        return 1
    audio_manifest = json.loads(inputs["audio_review_manifest"].read_text(encoding="utf-8"))
    if sha256sum(inputs["review"]) != audio_manifest["inputs"]["review"]["sha256"]:
        logger.error("review_items.csv differs from the version recorded in audio_review_manifest.json")
        return 1

    review = read_table(inputs["review"])
    try:
        records = caption_records(review, read_table(inputs["captions"]), read_table(inputs["phrases"]))
    except ValueError as error:
        logger.error("%s", error)
        return 1
    models_config = load_config("models", args.overrides)["models"]
    models, names = list(models_config), {key: model["display_name"] for key, model in models_config.items()}
    recordings = by_recording(records)
    if any(sorted(r["model_key"] for r in items) != sorted(models) for items in recordings.values()):
        logger.error("Every recording must have exactly one caption per model in models.yaml")
        return 1
    dataset = load_config("dataset", args.overrides)
    file_names = {r["sample_id"]: r["file_name"] for r in read_jsonl(resolve_path(dataset["evaluation"]["manifest"]))}

    selection = select_examples(records, PATTERNS, settings["seed"])
    if empty := [entry["pattern"] for entry in selection if entry["recording_code"] is None]:
        logger.warning("No eligible recording for %s", empty)
    for entry in selection:
        logger.info(
            "%s: %s (%d qualifying captions; %d eligible recordings)",
            entry["pattern"], entry["recording_code"], len(entry["qualifying_items"]),
            entry["num_eligible_recordings"],
        )  # fmt: skip
    selected = [entry for entry in selection if entry["recording_code"]]
    examples = [recording_example(recordings[e["recording_code"]], models, file_names) for e in selected]
    every = [recording_example(items, models, file_names) for _, items in sorted(recordings.items())]
    headings = {}
    for entry in selected:  # a recording chosen by two patterns (only when a pool is exhausted) gets both names
        code = entry["recording_code"]
        headings[code] = f"{headings[code]} / {entry['name']}" if code in headings else entry["name"]
    examples = list({e["recording_code"]: e for e in examples}.values())

    legend = (
        "Ref.: reference-agreement category (1 strong agreement ... 6 weak agreement or mismatch); Audio: audio rating "
        "(1 accurate, 2 minor errors, 3 major error or wrong sound source); disputed phrases are the caption phrases "
        "no reference supports, verified against the audio as S supported, U unsupported, A ambiguous."
    )
    rules = ["| Pattern | Rule | Eligible recordings | Qualifying captions | Selected |", "|---|---|---|---|---|"]
    rules += [
        f"| {e['name']} | {e['rule']} | {e['num_eligible_recordings']} | {e['num_qualifying_captions']} | "
        f"{e['recording_code'] or 'none'} |"
        for e in selection
    ]
    intro = [
        f"Generated by `scripts/prepare_examples.py` at commit `{run_info['git_commit']}` "
        f"(dirty: {run_info['git_dirty']}). One recording per pattern, chosen by rule (seed {settings['seed']}): the "
        "recording with the most qualifying captions, preferring recordings not chosen for another pattern; all "
        "eligible recordings are listed in `outputs/qualitative/example_manifest.json`.",
        "",
        *rules,
        "",
        legend,
    ]
    tables_dir.mkdir(parents=True, exist_ok=True)
    outputs["examples_md"].write_text(
        examples_markdown("Representative caption comparisons", intro, examples, names, headings),
        encoding="utf-8",
    )
    all_intro = [f"Every recording of the qualitative sample (supplementary). {legend}"]
    outputs["all_md"].write_text(
        examples_markdown("Qualitative sample: all recordings", all_intro, every, names, headings),
        encoding="utf-8",
    )
    table_captions = {e["recording_code"]: f"{headings[e['recording_code']]} (recording {e['recording_code']}, "
                      f"Clotho file {e['file_name']}). {legend}" for e in examples}  # fmt: skip
    outputs["examples_tex"].write_text(examples_latex(examples, names, table_captions, "tab:example-"), "utf-8")
    all_captions = {e["recording_code"]: f"Recording {e['recording_code']} (Clotho file {e['file_name']})."
                    for e in every}  # fmt: skip
    all_tex = examples_latex(every, names, all_captions, "tab:sample-", clearpage_every=2)
    outputs["all_tex"].write_text(all_tex, "utf-8")

    categories = [c["code"] for c in config["categories"]]
    ratings = [r["code"] for r in config["audio_review"]["ratings"]]
    per_model = {m: [r for r in records if r["model_key"] == m] for m in models}
    short_c, short_r = settings["short_categories"], settings["short_ratings"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    formats, dpi = settings["figure_formats"], settings["dpi"]
    n = len(recordings)
    save(
        stacked_bars(
            {names[m]: [sum(r["category"] == c for r in rows) for c in categories] for m, rows in per_model.items()},
            [f"{c} {label}" for c, label in zip(categories, short_c, strict=True)],
            CATEGORY_COLORS,
            "Reference agreement",
            f"Captions (of {n} per model)",
        ),
        figures_dir / FIGURES[0], formats, dpi,
    )  # fmt: skip
    save(
        stacked_bars(
            {names[m]: [sum(r["audio_rating"] == c for r in rows) for c in ratings] for m, rows in per_model.items()},
            [f"{c} {label}" for c, label in zip(ratings, short_r, strict=True)],
            RATING_COLORS,
            "Audio-grounded rating",
            f"Captions (of {n} per model)",
        ),
        figures_dir / FIGURES[1], formats, dpi,
    )  # fmt: skip
    save(
        heatmap(
            [[sum(r["category"] == c and r["audio_rating"] == g for r in records) for g in ratings]
             for c in categories],  # fmt: skip
            [f"{c} {label}" for c, label in zip(categories, short_c, strict=True)],
            [f"{g} {label}" for g, label in zip(ratings, short_r, strict=True)],
            None,  # the paper caption carries the title
            "Audio-grounded rating",
        ),
        figures_dir / FIGURES[2], formats, dpi,
    )  # fmt: skip

    manifest = {
        "run_info": run_info,
        "task": "E19",
        "seed": settings["seed"],
        "selection_method": (
            "Per pattern, the recording with the most qualifying captions, preferring recordings not chosen for "
            "another pattern; ties broken by a shuffle seeded with '<seed>:<pattern>'; patterns with fewer "
            "eligible recordings choose first."
        ),
        "num_recordings": n,
        "num_captions": len(records),
        "selection": [
            {
                **entry,
                "sample_id": recordings[entry["recording_code"]][0]["sample_id"] if entry["recording_code"] else None,
            }
            for entry in selection
        ],  # fmt: skip
        "recordings": {
            code: {
                "sample_id": items[0]["sample_id"],
                "file_name": file_names[items[0]["sample_id"]],
                "items": {r["item_id"]: r["model_key"] for r in items},
            }
            for code, items in sorted(recordings.items())
        },  # fmt: skip
        "inputs": {name: {"path": relative(path), "sha256": sha256sum(path)} for name, path in inputs.items()},
        "outputs": {name: relative(path) for name, path in outputs.items()},
    }
    write_json(manifest, outputs["manifest"])
    logger.info("Wrote %s", sorted(manifest["outputs"].values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
