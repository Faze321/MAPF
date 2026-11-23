from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from src.data.adapters.base import DatasetAdapter
from src.data.adapters.charged import ChargedDatasetAdapter
from src.data.adapters.long_format import LongFormatDatasetAdapter
from src.data.adapters.mp_evdata import MPEVDataDatasetAdapter
from src.data.adapters.urbanev import UrbanEVDatasetAdapter
from src.data.canonical import CanonicalDataset, DatasetSpec


def adapter_for_spec(spec: DatasetSpec) -> DatasetAdapter:
    normalized = str(spec.adapter or "urbanev").strip().lower().replace("-", "_")
    if normalized == "auto":
        normalized = detect_dataset_adapter(spec.path, timeseries_file=spec.timeseries_file)
    if normalized in {"urbanev", "urban_ev"}:
        return UrbanEVDatasetAdapter()
    if normalized in {"long", "long_format", "generic"}:
        return LongFormatDatasetAdapter()
    if normalized == "charged":
        return ChargedDatasetAdapter()
    if normalized in {"mp_evdata", "mpevdata"}:
        return MPEVDataDatasetAdapter()
    raise ValueError(f"Unsupported dataset adapter: {spec.adapter}")


def detect_dataset_adapter(path: Path, *, timeseries_file: str | None = None) -> str:
    """Recognize published file layouts, independent of the folder's name."""
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {path}")
    matches = []
    if (path / MPEVDataDatasetAdapter.load_filename).is_file() and (path / "price.xlsx").is_file():
        matches.append("mp_evdata")
    if all((path / name).is_file() for name in ("volume.csv", "e_price.csv")):
        if (path / "sites.csv").is_file():
            matches.append("charged")
        if (path / "inf.csv").is_file():
            matches.append("urbanev")
    if timeseries_file and (path / timeseries_file).is_file():
        matches.append("long_format")
    if len(matches) != 1:
        raise ValueError(
            f"Cannot unambiguously detect dataset format in {path} (matches: {matches}). "
            "Select a folder containing the raw files (for CHARGED, a city folder), "
            "or set data.adapter explicitly for a custom dataset."
        )
    return matches[0]


def resolve_dataset_spec(spec: DatasetSpec) -> DatasetSpec:
    return replace(spec, adapter=adapter_for_spec(spec).name)


def load_canonical_dataset(spec: DatasetSpec, *, force_cache: bool = False) -> CanonicalDataset:
    return adapter_for_spec(spec).load(spec, force_cache=force_cache)


def validate_forecaster_dataset_source(spec: DatasetSpec, source: dict[str, Any]) -> None:
    """Reject a handoff from a different adapter or native dataset/city directory."""
    recorded_adapter = source.get("adapter")
    if not recorded_adapter:
        raise ValueError("Forecaster manifest is missing its dataset adapter")
    actual = adapter_for_spec(spec).name
    recorded = adapter_for_spec(DatasetSpec(path=spec.path, adapter=str(recorded_adapter))).name
    if actual != recorded:
        raise ValueError(f"Forecaster belongs to adapter {recorded}, not {actual}; check run.data_dir")
    recorded_path = source.get("data_dir")
    if recorded_path and Path(recorded_path).resolve() != spec.path.resolve():
        raise ValueError("Forecaster belongs to a different dataset directory; check run.data_dir")
