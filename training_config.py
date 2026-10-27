"""Offline per-zone training settings, independent of forecast experiments."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from typing import Any


TRAINING_BACKENDS = ("lstm", "xgboost")
TRAINING_EXOG_COLS = ["T", "U", "nRAIN", "e_price", "s_price", "is_weekend", "temp_price_idx"]
TRAINING_PARAMETER_DEFAULTS = {
    "lstm_hidden_size": 32, "lstm_num_layers": 1,
    "lstm_batch_size": 64, "lstm_learning_rate": .001, "lstm_epochs": 100,
    "lstm_early_stopping_patience": 10, "lstm_dropout": 0.0,
    "lstm_head_dropout": .1, "lstm_weight_decay": 1e-4, "lstm_huber_delta": 1.0,
    "lstm_grad_clip": 1.0, "lstm_min_delta": 1e-4,
    "lstm_seed": 42, "lstm_device": "cuda", "lstm_exog_cols": TRAINING_EXOG_COLS,
    "xgboost_exog_cols": TRAINING_EXOG_COLS,
    "xgboost_n_estimators": 800, "xgboost_max_depth": 3,
    "xgboost_learning_rate": .03, "xgboost_min_child_weight": 10.0,
    "xgboost_subsample": .8, "xgboost_colsample_bytree": .9,
    "xgboost_reg_lambda": 10.0, "xgboost_reg_alpha": .1,
    "xgboost_early_stopping_rounds": 50, "xgboost_tree_method": "hist",
    "xgboost_device": "cuda", "xgboost_n_jobs": 2, "xgboost_max_bin": 256,
    "xgboost_seed": 42,
}
TRAINING_PARAMETER_NAMES = tuple(TRAINING_PARAMETER_DEFAULTS)
INTEGER_PARAMETERS = {
    "lstm_hidden_size", "lstm_num_layers", "lstm_batch_size",
    "lstm_epochs", "lstm_early_stopping_patience", "lstm_seed",
    "xgboost_n_estimators", "xgboost_max_depth", "xgboost_early_stopping_rounds",
    "xgboost_n_jobs", "xgboost_max_bin", "xgboost_seed",
}
LIST_PARAMETERS = {"lstm_exog_cols", "xgboost_exog_cols"}
STRING_PARAMETERS = {"lstm_device", "xgboost_device", "xgboost_tree_method"}


def training_path_key(value: str | Path) -> str:
    return os.path.normcase(str(Path(value).resolve()))


def validate_training_parameters(raw: Any, label: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must contain a mapping of prefixed training parameters")
    result = deepcopy(raw)
    for key, value in result.items():
        name = f"{label}.{key}"
        if key not in TRAINING_PARAMETER_DEFAULTS:
            raise ValueError(f"Unknown training parameter {name}")
        if value is None:
            raise ValueError(f"{name} cannot be null; omit it to inherit the default")
        if key in INTEGER_PARAMETERS:
            minimum = 0 if key.endswith("_seed") else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
            if key == "xgboost_max_bin" and value < 2:
                raise ValueError(f"{name} must be at least 2")
        elif key in LIST_PARAMETERS:
            if not isinstance(value, list):
                raise ValueError(f"{name} must be a list")
            if any(not isinstance(item, str) or not item.strip() for item in value):
                raise ValueError(f"{name} must contain nonempty feature names")
            if len(set(value)) != len(value):
                raise ValueError(f"{name} cannot contain duplicate values")
            if key.endswith("_exog_cols") and set(value) - {*TRAINING_EXOG_COLS, "hour", "occupancy"}:
                raise ValueError(f"{name} may contain only known covariates, never actual load or evaluation fields")
        elif key in STRING_PARAMETERS:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
            if key == "xgboost_tree_method" and value != "hist":
                raise ValueError(f"{name} must be hist for offline training")
            if (key == "xgboost_device" and value not in {"cpu", "cuda"}
                    and not (value.startswith("cuda:") and value[5:].isascii() and value[5:].isdigit())):
                raise ValueError(f"{name} must be cpu, cuda, or cuda:<nonnegative index>")
            if (key == "lstm_device" and value not in {"auto", "cpu", "cuda", "mps"}
                    and not (value.startswith("cuda:") and value[5:].isascii() and value[5:].isdigit())):
                raise ValueError(f"{name} must be auto, cpu, cuda, cuda:<index>, or mps")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if key in {"lstm_dropout", "lstm_head_dropout"}:
                valid = 0 <= value < 1
            elif key in {"xgboost_subsample", "xgboost_colsample_bytree"}:
                valid = 0 < value <= 1
            elif key in {"lstm_weight_decay", "lstm_min_delta", "xgboost_min_child_weight", "xgboost_reg_lambda", "xgboost_reg_alpha"}:
                valid = value >= 0
            else:
                valid = value > 0
            if not valid:
                raise ValueError(f"{name} is outside its supported range")
    return result


def _path_overrides(raw: Any, label: str, *, zones: bool = False) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must map dataset paths to overrides")
    result = {}
    for folder, values in raw.items():
        if not isinstance(folder, str) or not folder.strip():
            raise ValueError(f"{label} requires nonempty dataset paths")
        path = training_path_key(folder)
        if path in result:
            raise ValueError(f"Duplicate dataset path in {label}: {folder}")
        if zones:
            if not isinstance(values, dict):
                raise ValueError(f"{label}[{folder}] must map zone IDs to parameter mappings")
            result[path] = {}
            for zone_id, parameters in values.items():
                if not isinstance(zone_id, str) or not zone_id.strip():
                    raise ValueError(f"{label}[{folder}] requires string zone IDs; quote numeric IDs in YAML")
                result[path][zone_id] = validate_training_parameters(parameters, f"{label}[{folder}][{zone_id}]")
        else:
            result[path] = validate_training_parameters(values, f"{label}[{folder}]")
    return result


@dataclass(frozen=True)
class TrainingConfig:
    train_ratio: float = 0.8
    validation_ratio: float = 0.2
    min_history_days: int = 45
    min_samples: int = 512
    sample_stride_hours: int = 1
    seed: int = 42
    lstm_workers: int = 1
    xgboost_workers: int = 2
    defaults: dict[str, Any] = field(default_factory=dict)
    dataset_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    zone_overrides: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    def __post_init__(self):
        for key in ("min_history_days", "min_samples", "sample_stride_hours", "seed", "lstm_workers", "xgboost_workers"):
            value = getattr(self, key)
            minimum = 0 if key == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"training.{key} must be an integer >= {minimum}")
        for key in ("train_ratio", "validation_ratio"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"training.{key} must be a finite number between 0 and 1")
        if not math.isclose(self.train_ratio + self.validation_ratio, 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError("training.train_ratio + training.validation_ratio must equal 1")
        object.__setattr__(self, "defaults", validate_training_parameters(self.defaults, "training.defaults"))
        object.__setattr__(self, "dataset_overrides", _path_overrides(self.dataset_overrides, "training.dataset_overrides"))
        object.__setattr__(self, "zone_overrides", _path_overrides(self.zone_overrides, "training.zone_overrides", zones=True))

    @classmethod
    def from_mapping(cls, raw: Any = None) -> "TrainingConfig":
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError("training must contain a mapping")
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError("Unknown training setting(s): " + ", ".join(sorted(unknown)))
        return cls(**raw)

    def resolve(self, data_dir: str | Path, zone_id: str, backend: str,
                cli_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        backend = backend.lower()
        if backend not in TRAINING_BACKENDS:
            raise ValueError("Offline training supports lstm and xgboost only")
        values = deepcopy(TRAINING_PARAMETER_DEFAULTS)
        values.update(lstm_seed=self.seed, xgboost_seed=self.seed)
        dataset_key = training_path_key(data_dir)
        for override in (self.defaults, self.dataset_overrides.get(dataset_key, {}),
                         self.zone_overrides.get(dataset_key, {}).get(str(zone_id), {}),
                         validate_training_parameters(cli_overrides, "CLI training override")):
            values.update(override)
        result = {key: value for key, value in values.items() if key.startswith(backend + "_")}
        return validate_training_parameters(result, f"resolved {backend} training")
