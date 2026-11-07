"""Regenerate manuscript figures from frozen result artifacts.

E20: E11 scores by model (item means and 95% bootstrap CIs, five references) and E13 model - human differences
(matched four references, 95% CIs, Holm significance from E15), one panel per metric of `comparison.metrics` in
configs/metrics.yaml, written to `figures.dir`. The qualitative figures (E17, E18) come from make examples (E19).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.annotations import read_table  # noqa: E402
from src.analysis.figures import dot_panels, e11_panels, e13_panels  # noqa: E402
from src.analysis.qualitative_figures import save  # noqa: E402
from src.evaluation.pipeline import existing_outputs, relative  # noqa: E402
from src.metrics.registry import BY_NAME  # noqa: E402
from src.utils.cli import base_parser, init_run  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import OUTPUT_DIR, resolve_path  # noqa: E402

INPUTS = {
    "summary": OUTPUT_DIR / "scores" / "five_reference_summary.csv",
    "tests": OUTPUT_DIR / "scores" / "statistics_tests.csv",
}
FIGURES = ("e11_scores_by_model", "e13_model_minus_human")
SHORT_LABELS = {"bertscore_f1_rescaled": "BERTScore (resc.)"}  # panel titles are about 20 characters wide


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = parser.parse_args()
    logger = init_run(Path(__file__).stem, args)
    config = load_config("metrics", args.overrides)
    settings, metrics = config["figures"], config["comparison"]["metrics"]
    figures_dir = resolve_path(settings["dir"])
    outputs = {f"{name}_{fmt}": figures_dir / f"{name}.{fmt}" for name in FIGURES for fmt in settings["formats"]}
    if not args.force and (existing := existing_outputs(outputs)):
        logger.error("Outputs exist (use --force to overwrite): %s", existing)
        return 1
    if missing := [relative(path) for path in INPUTS.values() if not path.is_file()]:
        logger.error("Missing inputs: %s", missing)
        return 1
    models = load_config("models", args.overrides)["models"]
    systems, names = list(models), {key: model["display_name"] for key, model in models.items()}
    labels = {metric: SHORT_LABELS.get(metric, BY_NAME[metric].label) for metric in metrics}

    figures_dir.mkdir(parents=True, exist_ok=True)
    e11 = e11_panels(read_table(INPUTS["summary"]), metrics, labels)
    e13 = e13_panels(read_table(INPUTS["tests"]), metrics, labels)
    if any(set(panel["rows"]) != set(systems) for panel in e11 + e13):
        logger.error("Every panel needs one row per model in models.yaml")
        return 1
    title = "Scores by model, five references (E11): item mean and 95% CI over 1,045 recordings"
    save(dot_panels(e11, systems, names, title), figures_dir / FIGURES[0], settings["formats"], settings["dpi"])
    title = "Model minus human, matched four references (E13): 95% CI; hollow = n.s. after Holm"
    save(dot_panels(e13, systems, names, title, zero_line=True), figures_dir / FIGURES[1], settings["formats"],
         settings["dpi"])  # fmt: skip
    logger.info("Wrote %s", sorted(relative(path) for path in outputs.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
