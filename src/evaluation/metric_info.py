"""Metric interpretation notes and package versions recorded in the evaluation reports (E12-E14)."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import metadata

PACKAGES = (
    "aac-metrics",
    "msclap",
    "torch",
    "torchaudio",
    "transformers",
    "sentence-transformers",
    "torchmetrics",
    "huggingface-hub",
    "numpy",
    "pandas",
    "librosa",
    "soundfile",
)

LIMITATIONS = [
    "CIDEr-D, SPICE, SPIDEr and SPIDEr-FL depend on the reference set (CIDEr-D's document frequencies come from the "
    "references of each call), so five-reference (E12) and four-reference (E13/E14) scores are not directly "
    "comparable.",
    "BLEU and METEOR corpus values are computed from corpus statistics, not as the mean of per-item scores; "
    "bootstrap intervals are on the per-item mean and are reported next to the corpus value.",
    "BERTScore follows aac-metrics: roberta-large's last layer (24), no IDF, the maximum over references. Raw F1 is "
    "compressed near 1; the rescaled F1 (layer-24 baseline) is reported as a secondary column.",
    "FENSE, SPIDEr-FL and MACE multiply a caption's score by a fixed factor when the fluency-error detector flags it "
    "(probability above 0.9, or 0.97 for MACE). The detector also flags some short, plain captions, and its input is "
    "cut at 64 subwords.",
    "Malformed (JSON-like) captions are scored as returned after PTB tokenization: braces and quotes are removed but "
    "key names remain, which lowers lexical and semantic scores.",
    "MS-CLAP 2023 sees 7 s of audio at a time; clips are embedded whole as a duration-weighted mean of 7 s chunks, "
    "following the MACE paper. aac-metrics' own MACE instead uses one random 7 s crop, is text-only by default and "
    "returns the corpus mean as every per-item score.",
    "SPICE runs without its parse cache (the cache's native library is x86-only); this changes speed, not scores.",
    "Every metric runs on CPU in float32/float64; scores are reproducible on the same platform and versions but may "
    "differ in the last digits on other hardware.",
]


def package_versions(names: Sequence[str] = PACKAGES) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions
