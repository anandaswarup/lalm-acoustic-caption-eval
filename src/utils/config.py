"""YAML configuration loading with environment-variable expansion and overrides."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.utils.paths import CONFIG_DIR, REPO_ROOT

# Matches ${VAR} and ${VAR:-default}.
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def load_env(env_file: str | Path | None = None) -> None:
    """Load variables from `.env` at the repository root without overriding the shell."""
    load_dotenv(env_file or REPO_ROOT / ".env", override=False)


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):

        def substitute(match: re.Match) -> str:
            name, default = match.group(1), match.group(2)
            resolved = os.environ.get(name, default)
            if resolved is None:
                raise KeyError(f"Environment variable '{name}' is not set and has no default")
            return resolved

        return _ENV_PATTERN.sub(substitute, value)
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value


def load_yaml(path: str | Path, expand_env: bool = True) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    return _expand_env(data) if expand_env else data


def load_config(name: str, overrides: list[str] | None = None, expand_env: bool = True) -> dict[str, Any]:
    """Load `configs/<name>.yaml` and apply `key.subkey=value` overrides."""
    config = load_yaml(CONFIG_DIR / f"{name}.yaml", expand_env=expand_env)
    return apply_overrides(config, overrides or [])


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Return a copy of `config` with dotted-key overrides; values are parsed as YAML."""
    config = copy.deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must have the form key=value: {override!r}")
        dotted_key, raw_value = override.split("=", 1)
        *parents, leaf = dotted_key.strip().split(".")
        node = config
        for key in parents:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot override below non-mapping key '{key}' in {dotted_key!r}")
        node[leaf] = yaml.safe_load(raw_value)
    return config
