"""Load experiment defaults from TOML files."""

from __future__ import annotations

import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeVar


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
T = TypeVar("T")


def load_config_section(
    path: str | Path,
    section: str,
    *,
    expected_model: str | None = None,
) -> Mapping[str, object]:
    """Return one TOML configuration table."""
    config_path = Path(path)
    with config_path.open("rb") as file:
        payload = tomllib.load(file)

    if expected_model is not None and payload.get("model") != expected_model:
        raise ValueError(
            f"config {config_path} is for model {payload.get('model')!r}, "
            f"expected {expected_model!r}"
        )
    return payload[section]


def config_value(
    values: Mapping[str, object],
    key: str,
    converter: Callable[[str], T],
) -> T:
    """Read a value using the same converter as the command-line parser."""
    return converter(str(values[key]))


def config_path(values: Mapping[str, object], key: str) -> Path:
    return config_value(values, key, Path)


def config_device(values: Mapping[str, object]) -> str:
    return config_value(values, "device", str)
