"""Metric tool helpers (E11): Java checks, pinned revisions and the tools manifest. No JDK or downloads needed."""

import json
import os
import sys

import pytest

from src.metrics import tools as metric_tools
from src.utils.config import load_config


def fake_java(tmp_path, version):
    java = tmp_path / "java"
    java.write_text(f"#!/bin/sh\necho 'openjdk version \"{version}\" 2026-08-18' >&2\n", encoding="utf-8")
    java.chmod(0o755)
    return java


@pytest.mark.parametrize(
    ("version", "major"), [("1.8.0_402", 8), ("11.0.32.1", 11), ("13", 13), ("27.0.1.0.3", 27), ("17-ea", 17)]
)
def test_java_major(version, major):
    assert metric_tools.java_major(version) == major


def test_check_java_accepts_11_to_13(tmp_path):
    assert metric_tools.check_java(fake_java(tmp_path, "11.0.32.1")) == "11.0.32.1"


@pytest.mark.parametrize("version", ["1.8.0_402", "27.0.1.0.3"])
def test_check_java_rejects_other_versions(tmp_path, version):
    with pytest.raises(RuntimeError, match="need Java 11-13"):
        metric_tools.check_java(fake_java(tmp_path, version))


def test_check_java_missing(tmp_path):
    with pytest.raises(RuntimeError, match="make metric-tools"):
        metric_tools.check_java(tmp_path / "nope")


def test_java_path_override_and_default(tmp_path, monkeypatch):
    tools = load_config("metrics")["tools"]
    monkeypatch.setattr(metric_tools, "platform_key", lambda: "linux-x86_64")
    assert metric_tools.java_path({**tools, "java_path": ""}).as_posix().endswith("tools/jdk-11/bin/java")
    assert metric_tools.java_path({**tools, "java_path": str(tmp_path / "java")}) == tmp_path / "java"
    monkeypatch.setattr(metric_tools, "platform_key", lambda: "plan9-mips")
    with pytest.raises(RuntimeError, match="No pinned JDK"):
        metric_tools.java_path({**tools, "java_path": ""})


def test_config_pins_are_complete():
    tools = load_config("metrics")["tools"]
    for build in tools["jdk"]["builds"].values():
        assert len(build["sha256"]) == 64 and build["url"].startswith("https://")
    assert all(len(digest) == 64 for digest in tools["files"].values())
    assert all(len(revision) == 40 for revision in tools["hf_models"].values())
    assert tools["clap"]["repo"] in tools["hf_models"]


def test_hf_revision_problem(tmp_path):
    cache = tmp_path / "hub" / "models--org--name"
    assert "no cached snapshot" in metric_tools.hf_revision_problem(tmp_path, "org/name", "abc123")
    (cache / "snapshots" / "abc123").mkdir(parents=True)
    assert metric_tools.hf_revision_problem(tmp_path, "org/name", "abc123") is None  # explicit-commit download
    (cache / "refs").mkdir()
    (cache / "refs" / "main").write_text("abc123\n", encoding="utf-8")
    assert metric_tools.hf_revision_problem(tmp_path, "org/name", "abc123") is None
    (cache / "refs" / "main").write_text("def456\n", encoding="utf-8")
    assert "cached main is def456" in metric_tools.hf_revision_problem(tmp_path, "org/name", "abc123")


def test_manifest_round_trip(tmp_path):
    (tmp_path / "aac-metrics").mkdir()
    (tmp_path / "aac-metrics" / "a.jar").write_bytes(b"jar")
    for skipped in ("jdk-11/bin/java", "tmp/x", "hf/hub/models--m/blobs/abc"):
        (tmp_path / skipped).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / skipped).write_bytes(b"x")
    snapshot = tmp_path / "hf/hub/models--m/snapshots/rev/config.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(tmp_path / "hf/hub/models--m/blobs/abc")

    metric_tools.write_manifest(tmp_path, {"platform": "test"})
    files = json.loads((tmp_path / metric_tools.MANIFEST_NAME).read_text(encoding="utf-8"))["files"]
    assert sorted(files) == ["aac-metrics/a.jar", "hf/hub/models--m/snapshots/rev/config.json"]
    assert metric_tools.verify_manifest(tmp_path) == []

    (tmp_path / "aac-metrics" / "a.jar").write_bytes(b"changed")
    snapshot.unlink()
    assert sorted(metric_tools.verify_manifest(tmp_path)) == [
        "aac-metrics/a.jar: checksum changed",
        "hf/hub/models--m/snapshots/rev/config.json: missing",
    ]


def test_verify_manifest_without_manifest(tmp_path):
    assert "make metric-tools" in metric_tools.verify_manifest(tmp_path)[0]


def test_activate_refuses_an_earlier_huggingface_hub_import(tmp_path, monkeypatch):
    import types

    tools = {**load_config("metrics")["tools"], "dir": str(tmp_path)}
    for name in ("HF_HOME", "TORCH_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(name, raising=False)

    def fake_hub(hf_home, offline):
        constants = types.SimpleNamespace(HF_HOME=str(hf_home), HF_HUB_OFFLINE=offline)
        monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(constants=constants))

    fake_hub(tmp_path / "elsewhere", True)
    with pytest.raises(RuntimeError, match="imported before activate"):
        metric_tools.activate(tools, offline=True)
    fake_hub(tmp_path / "hf", False)
    with pytest.raises(RuntimeError, match="imported before activate"):
        metric_tools.activate(tools, offline=True)
    fake_hub(tmp_path / "hf", True)
    paths = metric_tools.activate(tools, offline=True)
    assert os.environ["HF_HOME"] == str(paths["hf_home"]) and os.environ["HF_HUB_OFFLINE"] == "1"
