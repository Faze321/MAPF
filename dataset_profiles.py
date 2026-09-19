"""Dataset switches and output isolation without changing legacy UrbanEV runs."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
import os
from pathlib import Path

from config import DataConfig, RunConfig, read_config_mapping, _expand_env_vars

CITIES = ("AMS", "JHB", "LOA", "MEL", "SPO", "SZH")


def normalize_dataset(value: str) -> str:
    name = value.lower().replace("-", "_")
    if name == "mpevdata":
        name = "mp_evdata"
    if name not in {"urbanev", "charged", "mp_evdata"}:
        raise ValueError(f"Unknown dataset: {value}")
    return name


def apply_dataset_profile(run: RunConfig, data: DataConfig, name: str, city: str | None, config_path: Path):
    name = normalize_dataset(name)
    if city and name != "charged":
        raise ValueError("--city applies only to CHARGED")
    city = (city or "JHB").upper()
    if city not in CITIES:
        raise ValueError(f"Unsupported CHARGED city: {city}")
    paths = {"urbanev": "data/UrbanEV", "charged": f"data/CHARGED/{city}", "mp_evdata": "data/MP-EVData"}
    origins = {"urbanev": "2022-10-14 00:00:00", "charged": "2023-06-01 00:00:00", "mp_evdata": "2024-11-01 00:00:00"}
    # Remove every input dependent on the former dataset, including artifact overrides.
    data = DataConfig(adapter=name)
    run = replace(run, data_dir=paths[name], weather_file="weather.csv" if name == "charged" else "weather_airport.csv",
                  forecast_start=origins[name], forecast_starts=None, zone_ids=None,
                  forecaster_output_dir=None, agent_output_dir=None, precomputed_window_data=None)
    raw = read_config_mapping(config_path, expand_env=False) if config_path.is_file() else {}
    overrides = _expand_env_vars((raw.get("datasets") or {}).get(name) or {})
    if not isinstance(overrides, dict):
        raise ValueError(f"datasets.{name} must be a mapping")
    data_values = {**asdict(data), **(overrides.get("data") or {})}
    if data_values["adapter"] != name:
        raise ValueError("Dataset profile cannot change its adapter")
    run_overrides = overrides.get("run") or {}
    run_values = {**asdict(run), **run_overrides}
    if "zones" in run_overrides and "zone_ids" not in run_overrides:
        run_values["zone_ids"] = run_overrides["zones"]
    return RunConfig.from_mapping(run_values), DataConfig.from_mapping(data_values)


def scoped_output(root: Path, adapter: str, data_path: Path, *, explicit_dataset: bool) -> Path:
    name = adapter.lower().replace("-", "_")
    if name == "mpevdata":
        name = "mp_evdata"
    if name == "charged":
        return root / name / data_path.name
    if name == "mp_evdata" or explicit_dataset:
        return root / name
    return root


@contextmanager
def output_lock(directory: Path):
    """An OS-released process lock prevents simultaneous writers to one output root."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".mapf-run.lock").open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another run is writing to {directory}; use a different --output-folder") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
