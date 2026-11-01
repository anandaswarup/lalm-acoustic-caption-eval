"""Synthetic Clotho-like fixture shared by the tests (CI has no copy of Clotho)."""

import csv
import hashlib
import json

import numpy as np
import soundfile as sf

SPLITS = {"evaluation": ["c.wav", "a sound.wav", "b.wav"], "validation": ["v1.wav", "v2.wav", "v3.wav", "v4.wav"]}


def write_wav(path, seconds=0.1, sample_rate=44100, channels=1, subtype="PCM_16"):
    frames = int(seconds * sample_rate)
    data = np.zeros((frames, channels), dtype=np.int16)
    sf.write(path, data, sample_rate, subtype=subtype)
    return frames


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def write_zenodo_record(root):
    files = [
        {"key": path.name, "size": path.stat().st_size, "checksum": f"md5:{md5(path)}"}
        for path in sorted((root / "captions").glob("*.csv")) + sorted((root / "metadata").glob("*.csv"))
    ]
    record = {"modified": "2021-05-21", "metadata": {"version": "2.1"}, "files": files}
    (root / "provenance" / "zenodo-record-4783391.json").write_text(json.dumps(record), encoding="utf-8")


def make_clotho(root):
    """A tiny Clotho-like dataset: 3 evaluation and 4 validation recordings."""
    for name in ("captions", "metadata", "manifests", "provenance"):
        (root / name).mkdir(parents=True)
    for split, names in SPLITS.items():
        (root / "audio" / split).mkdir(parents=True)
        captions = {name: [f"{name} caption {i}." for i in range(1, 6)] for name in names}
        frames = {
            name: write_wav(root / "audio" / split / name, seconds=0.1 + 0.05 * i) for i, name in enumerate(names)
        }
        write_csv(
            root / "captions" / f"clotho_captions_{split}.csv",
            ["file_name"] + [f"caption_{i}" for i in range(1, 6)],
            [[name, *captions[name]] for name in names],
        )
        write_csv(
            root / "metadata" / f"clotho_metadata_{split}.csv",
            ["file_name", "keywords", "sound_id", "sound_link", "start_end_samples", "manufacturer", "license"],
            [[name, "kw", f"{split}-{i}", f"https://fs.org/{i}", "", "x", "cc0"] for i, name in enumerate(names)],
        )
        with (root / "manifests" / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for name in names:
                stream.write(json.dumps({"file_name": name, "captions": captions[name], "num_frames": frames[name]}))
                stream.write("\n")
    (root / "LICENSE").write_text("Copyright notice\n", encoding="utf-8")
    write_zenodo_record(root)
    return root
