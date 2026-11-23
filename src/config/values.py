from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any, Callable, TypeVar

import yaml


def positive_integer(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{key} must be a positive integer')
    return value


def strict_bool(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false")
    return value


def dataset_path_key(path: str | Path) -> str:
    """Match dataset folders consistently for relative paths and batch child processes."""
    return os.path.normcase(str(Path(path).resolve()))


def normalize_data_dirs(value: Any) -> str | list[str]:
    if value is None:
        return "data/UrbanEV"
    paths = value if isinstance(value, list) else [value]
    if not paths or any(not isinstance(path, str) or not path.strip() for path in paths):
        raise ValueError('run.data_dir must be a folder path or a non-empty list of folder paths')
    paths = [path.strip() for path in paths]
    resolved = [dataset_path_key(path) for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError('run.data_dir contains duplicate dataset folders')
    return paths if isinstance(value, list) else paths[0]


_DatasetValue = TypeVar("_DatasetValue")


def normalize_dataset_overrides(
    value: Any, name: str, validate: Callable[[Any, str], _DatasetValue],
) -> dict[str, _DatasetValue]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"run.{name} must be a mapping keyed by dataset folder paths")
    overrides = {}
    for folder, setting in value.items():
        if not isinstance(folder, str) or not folder.strip():
            raise ValueError(f"run.{name} keys must be non-empty dataset folder paths")
        key = dataset_path_key(folder.strip())
        if key in overrides:
            raise ValueError(f"run.{name} contains duplicate dataset folder: {folder}")
        overrides[key] = validate(setting, f"run.{name}[{folder}]")
    return overrides


def normalize_experiment_zone_counts(value: Any) -> dict[str, int]:
    return normalize_dataset_overrides(value, "experiment_zone_counts", positive_integer)


_ENV_REF_RE = re.compile(r"""
                         \$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\} 
                         |\$(?P<plain>[A-Za-z_][A-Za-z0-9_]*) 
                         |%(?P<windows>[A-Za-z_][A-Za-z0-9_]*)%
                         """, re.VERBOSE)


def _expand_env_vars(obj: Any) -> Any:
    if isinstance(obj, str):
        missing_env = []
        for match in _ENV_REF_RE.finditer(obj):
            name = match.group("braced") or match.group("plain") or match.group("windows")
            if name not in os.environ:
                missing_env.append(name)
        if missing_env:
            raise ValueError(f"Missing environment variable(s): {', '.join(missing_env)}")
        return os.path.expandvars(obj)
    elif isinstance(obj, list):
        return [_expand_env_vars(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    else:
        return obj


def read_config_mapping(path: Path, *, expand_env: bool = True) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return _expand_env_vars(value) if expand_env else value


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_forecast_model_name(value: str | None) -> str:
    normalized = (value or "timesfm").strip().lower().replace("-", "_")
    if normalized in {"xgb", "xg_boost"}:
        return "xgboost"
    return "AR" if normalized == "ar" else normalized


def optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def optional_bool(value: Any, default: bool) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean config value: {value!r}")


def normalize_zone_id_list(value: Any) -> list[str] | None:
    return normalize_string_list(value)


def normalize_string_list(value: Any) -> list[str] | None:
    if value in (None, ""):
        return None
    raw_values = value if isinstance(value, list) else [value]
    values: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        for part in str(raw).replace(";", ",").split(","):
            value_text = part.strip()
            if value_text and value_text not in seen:
                values.append(value_text)
                seen.add(value_text)
    return values or None


_ListItem = TypeVar("_ListItem")


def normalize_typed_list(value: Any, convert: Callable[[str], _ListItem]) -> list[_ListItem] | None:
    """Parse once, then deduplicate converted values while keeping input order."""
    return list(dict.fromkeys(convert(item) for item in normalize_string_list(value) or [])) or None


def normalize_int_list(value: Any) -> list[int] | None:
    return normalize_typed_list(value, int)


def normalize_float_list(value: Any) -> list[float] | None:
    """Normalize an alpha candidate list; null inherits, an empty list is invalid."""
    if value is None:
        return None
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, bool) or (isinstance(value, list) and any(isinstance(item, bool) for item in value)):
        raise ValueError("Diurnal blend alpha must be a finite number between 0 and 1")
    def convert(item: str) -> float:
        number = float(item)
        if not math.isfinite(number) or number < 0.0 or number > 1.0:
            raise ValueError(f"Diurnal blend alpha must be a finite number between 0 and 1: {item}")
        return round(number, 6)

    values = normalize_typed_list(value, convert)
    if not values:
        raise ValueError("Diurnal blend alpha lists cannot be empty; use null to inherit")
    return values
