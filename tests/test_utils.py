import random

import numpy as np
import pytest

from src.utils.config import apply_overrides, load_config, load_yaml
from src.utils.paths import CONFIG_DIR, REPO_ROOT, resolve_path
from src.utils.seed import set_seed

CONFIG_NAMES = ["dataset", "models", "generation", "metrics", "qualitative"]


def test_repo_root_contains_project_files():
    assert (REPO_ROOT / "requirements.txt").is_file()
    assert (REPO_ROOT / "Makefile").is_file()


def test_resolve_path_relative_to_repo_root(tmp_path):
    assert resolve_path("configs") == REPO_ROOT / "configs"
    assert resolve_path(tmp_path) == tmp_path


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_configs_load(name):
    assert isinstance(load_config(name), dict)


def test_dataset_config_uses_full_evaluation_split():
    config = load_config("dataset")
    assert config["version"] == "2.1"
    assert config["evaluation"]["expected_recordings"] == 1045


def test_models_config_lists_four_models_with_unique_outputs():
    models = load_config("models")["models"]
    assert len(models) == 4
    outputs = [model["output"] for model in models.values()]
    assert len(set(outputs)) == 4
    assert all(model["compute"] in {"local", "hf_jobs"} for model in models.values())


def test_env_expansion(monkeypatch, tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("a: ${LALM_TEST_VAR}\nb: ${LALM_UNSET_VAR:-fallback}\n")
    monkeypatch.setenv("LALM_TEST_VAR", "value")
    monkeypatch.delenv("LALM_UNSET_VAR", raising=False)
    assert load_yaml(path) == {"a": "value", "b": "fallback"}


def test_env_expansion_missing_variable_raises(monkeypatch, tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("a: ${LALM_UNSET_VAR}\n")
    monkeypatch.delenv("LALM_UNSET_VAR", raising=False)
    with pytest.raises(KeyError):
        load_yaml(path)


def test_overrides_parse_values_and_do_not_mutate_input():
    config = {"decoding": {"temperature": 0.0}}
    updated = apply_overrides(config, ["decoding.temperature=0.7", "new.key=true"])
    assert updated == {"decoding": {"temperature": 0.7}, "new": {"key": True}}
    assert config == {"decoding": {"temperature": 0.0}}


def test_overrides_reject_malformed():
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals_sign"])


def test_set_seed_is_reproducible():
    set_seed(42)
    first = (random.random(), np.random.rand())
    set_seed(42)
    assert (random.random(), np.random.rand()) == first


def test_config_dir_has_only_expected_files():
    assert sorted(p.stem for p in CONFIG_DIR.glob("*.yaml")) == sorted(CONFIG_NAMES)
