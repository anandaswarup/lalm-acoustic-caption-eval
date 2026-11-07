"""Representative caption comparisons from the qualitative sample (E19).

Examples are whole recordings: the five references and all four model captions, each with its E17 agreement
category, E18 audio rating and disputed-phrase verdicts. Selection is rule-based and deterministic: every pattern is a
predicate on one caption's labels; for each pattern the recording with the most qualifying captions is chosen,
preferring recordings no other pattern has chosen, with ties broken by a seeded shuffle. Patterns with the fewest
eligible recordings choose first, so a rare pattern is not left without an unused recording. Every eligible recording
is recorded in the manifest.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.analysis.qualitative import REFERENCE_COLUMNS


@dataclass(frozen=True)
class Pattern:
    key: str
    name: str
    rule: str
    qualifies: Callable[[dict[str, Any]], bool]


PATTERNS = (
    Pattern(
        "strong_agreement",
        "Strong agreement",
        "E17 category 1 or 2 and E18 rating 1",
        lambda c: c["category"] in (1, 2) and c["audio_rating"] == 1,
    ),
    Pattern(
        "supported_extra_detail",
        "Reference-absent detail supported by the audio",
        "E17 category 3 or 5 with at least one disputed phrase verified as supported (S)",
        lambda c: c["category"] in (3, 5) and "S" in c["verdicts"],
    ),
    Pattern(
        "unsupported_detail",
        "Detail unsupported by the audio",
        "At least one disputed phrase verified as unsupported (U)",
        lambda c: "U" in c["verdicts"],
    ),
    Pattern(
        "omission",
        "Omission of referenced content",
        "E17 category 4",
        lambda c: c["category"] == 4,
    ),
    Pattern(
        "real_mismatch",
        "Sound-source or event mismatch",
        "E17 category 6 and E18 rating 3",
        lambda c: c["category"] == 6 and c["audio_rating"] == 3,
    ),
    Pattern(
        "incomplete_references",
        "Correct caption, incomplete references",
        "E17 category 6 and E18 rating 1",
        lambda c: c["category"] == 6 and c["audio_rating"] == 1,
    ),
)


def caption_records(
    review: list[dict[str, str]], captions: list[dict[str, str]], phrases: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """One record per caption, in review order: identifiers, references, caption text, labels and its disputed
    phrases with verdicts. `captions` and `phrases` are the unblinded E18 tables."""
    by_item = {row["item_id"]: row for row in captions}
    if set(by_item) != {row["item_id"] for row in review}:
        raise ValueError("review_items.csv and the unblinded E18 captions cover different items")
    found: dict[str, list[tuple[str, str]]] = {}
    for row in phrases:
        found.setdefault(row["item_id"], []).append((row["phrase"], row["verdict"]))
    records = []
    for row in review:
        labels = by_item[row["item_id"]]
        item_phrases = found.get(row["item_id"], [])
        records.append(
            {
                "item_id": row["item_id"],
                "recording_code": row["recording_code"],
                "sample_id": labels["sample_id"],
                "model_key": labels["model_key"],
                "references": [row[column] for column in REFERENCE_COLUMNS],
                "caption": row["caption"],
                "category": int(labels["category"]),
                "audio_rating": int(labels["audio_rating"]),
                "phrases": item_phrases,
                "verdicts": [verdict for _, verdict in item_phrases],
            }
        )
    return records


def by_recording(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        out.setdefault(record["recording_code"], []).append(record)
    return out


def eligible(records: list[dict[str, Any]], pattern: Pattern) -> dict[str, list[str]]:
    """Recording code -> item IDs of its captions that fit the pattern (only recordings with at least one)."""
    out = {}
    for code, items in by_recording(records).items():
        if hits := [record["item_id"] for record in items if pattern.qualifies(record)]:
            out[code] = hits
    return out


def select_examples(records: list[dict[str, Any]], patterns: tuple[Pattern, ...], seed: int) -> list[dict[str, Any]]:
    """One recording per pattern (see the module docstring), returned in pattern order."""
    pools = {pattern.key: eligible(records, pattern) for pattern in patterns}
    order = sorted(patterns, key=lambda p: (len(pools[p.key]), [q.key for q in patterns].index(p.key)))
    chosen: dict[str, str | None] = {}
    for pattern in order:
        pool = pools[pattern.key]
        codes = sorted(pool)
        random.Random(f"{seed}:{pattern.key}").shuffle(codes)
        ranked = sorted(codes, key=lambda code: (code in chosen.values(), -len(pool[code])))  # stable: keeps shuffle
        chosen[pattern.key] = ranked[0] if ranked else None
    out = []
    for pattern in patterns:
        pool, code = pools[pattern.key], chosen[pattern.key]
        out.append(
            {
                "pattern": pattern.key,
                "name": pattern.name,
                "rule": pattern.rule,
                "recording_code": code,
                "qualifying_items": pool.get(code, []) if code else [],
                "num_eligible_recordings": len(pool),
                "num_qualifying_captions": sum(len(items) for items in pool.values()),
                "eligible": {c: pool[c] for c in sorted(pool)},
            }
        )
    return out


def recording_example(items: list[dict[str, Any]], models: list[str], file_names: dict[str, str]) -> dict[str, Any]:
    """A recording's references and its captions in model order, with identifiers for traceability."""
    first = items[0]
    by_model = {record["model_key"]: record for record in items}
    return {
        "recording_code": first["recording_code"],
        "sample_id": first["sample_id"],
        "file_name": file_names[first["sample_id"]],
        "references": first["references"],
        "captions": [by_model[model] for model in models],
    }


def _phrase_text(record: dict[str, Any]) -> str:
    return "; ".join(f"{phrase} ({verdict})" for phrase, verdict in record["phrases"])


def examples_markdown(
    title: str, intro: list[str], examples: list[dict[str, Any]], names: dict[str, str], headings: dict[str, str]
) -> str:
    """Markdown tables, one per recording. `headings` maps a recording code to a heading prefix (the pattern)."""
    lines = [f"# {title}", "", *intro, ""]
    for example in examples:
        prefix = headings.get(example["recording_code"])
        heading = f"{prefix}: {example['recording_code']}" if prefix else example["recording_code"]
        lines += [
            f"## {heading}",
            "",
            f"Sample `{example['sample_id']}`, Clotho file `{example['file_name']}`.",
            "",
            "| Source | Text | E17 | E18 | Disputed phrases (verdict) |",
            "|---|---|---|---|---|",
            *[f"| Reference {i} | {_md(text)} | | | |" for i, text in enumerate(example["references"], start=1)],
        ]
        for record in example["captions"]:
            lines.append(
                f"| {names[record['model_key']]} (`{record['item_id']}`) | {_md(record['caption'])} | "
                f"{record['category']} | {record['audio_rating']} | {_md(_phrase_text(record))} |"
            )
        lines.append("")
    return "\n".join(lines)


def _md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


LATEX_SPECIAL = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_\allowbreak{}",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}  # fmt: skip


def latex_escape(text: str) -> str:
    return "".join(LATEX_SPECIAL.get(char, char) for char in text)


def examples_latex(
    examples: list[dict[str, Any]],
    names: dict[str, str],
    captions: dict[str, str],
    label_prefix: str,
    clearpage_every: int | None = None,
) -> str:
    """One booktabs/tabularx table per recording (needs \\usepackage{booktabs,tabularx}). `captions` maps a recording
    code to its table caption. `clearpage_every` flushes the floats after every that many tables, so a long run of
    tables with no text between them does not overflow LaTeX's float queue. Underscores (file names) may break."""
    lines = [
        "% Generated by scripts/prepare_examples.py (E19); do not edit by hand.",
        "% Requires \\usepackage{booktabs} and \\usepackage{tabularx}.",
        "",
    ]
    for number, example in enumerate(examples, start=1):
        code = example["recording_code"]
        lines += [
            "\\begin{table}[htbp]",
            "\\centering\\small",
            f"\\caption{{{latex_escape(captions[code])}}}",
            f"\\label{{{label_prefix}{code.lower()}}}",
            "\\begin{tabularx}{\\linewidth}{@{}lXcc@{}}",
            "\\toprule",
            "Source & Text & E17 & E18 \\\\",
            "\\midrule",
            *[
                f"Reference {i} & {latex_escape(text)} & & \\\\"
                for i, text in enumerate(example["references"], start=1)
            ],
            "\\midrule",
        ]
        for record in example["captions"]:
            text = latex_escape(record["caption"])
            if record["phrases"]:
                text += f" \\newline \\emph{{Disputed: {latex_escape(_phrase_text(record))}}}"
            name = latex_escape(names[record["model_key"]])
            lines.append(f"{name} & {text} & {record['category']} & {record['audio_rating']} \\\\")
        lines += ["\\bottomrule", "\\end{tabularx}", "\\end{table}", ""]
        if clearpage_every and number % clearpage_every == 0 and number < len(examples):
            lines += ["\\clearpage", ""]
    return "\n".join(lines)
