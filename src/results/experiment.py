from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def append_experiment_frame(
    frames: list[pd.DataFrame],
    path: Path | None,
    metadata: dict[str, Any],
) -> None:
    if path is None or not path.exists():
        return
    frame = pd.read_csv(path, dtype={"zone_id": str})
    insert_at = 0
    for column, value in metadata.items():
        if column in frame.columns:
            frame[column] = value
        else:
            frame.insert(insert_at, column, value)
            insert_at += 1
    frames.append(frame)


def write_experiment_summary(frames: list[pd.DataFrame], path: Path) -> pd.DataFrame:
    if frames:
        frame = pd.concat(frames, ignore_index=True)
    else:
        frame = pd.DataFrame()
    frame.to_csv(path, index=False)
    return frame


def write_numeric_summary(
    frame: pd.DataFrame,
    path: Path,
    *,
    metric_columns: list[str],
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    group_columns = group_columns or ["backend", "agent_mode", "diurnal_blend_alpha", "validation_bias", "validation_days"]
    output_columns = [
        *group_columns,
        "metric",
        "n",
        "mean",
        "std",
        "sem",
        "min",
        "max",
    ]
    if frame.empty:
        empty = pd.DataFrame(columns=output_columns)
        empty.to_csv(path, index=False)
        return empty

    groups = [column for column in group_columns if column in frame.columns]
    rows = []
    grouped = frame.groupby(groups, dropna=False) if groups else [((), frame)]
    for key, group in grouped:
        key_values = key if isinstance(key, tuple) else (key,)
        group_meta = dict(zip(groups, key_values))
        for metric in metric_columns:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            rows.append(
                {
                    **group_meta,
                    "metric": metric,
                    "n": int(len(values)),
                    "mean": round(float(values.mean()), 6),
                    "std": round(std, 6),
                    "sem": round(std / (len(values) ** 0.5), 6) if len(values) > 1 else 0.0,
                    "min": round(float(values.min()), 6),
                    "max": round(float(values.max()), 6),
                }
            )
    summary = pd.DataFrame(rows)
    if summary.empty:
        summary = pd.DataFrame(columns=output_columns)
    summary.to_csv(path, index=False)
    return summary


def write_decision_quality_summary(
    prices: pd.DataFrame,
    rationales: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not prices.empty:
        frames.append(
            long_metric_frame(
                prices,
                metrics=[
                    "price_accuracy",
                    "avg_predicted_minus_baseline_energy_price",
                    "avg_predicted_vs_baseline_pct",
                ],
                metric_family="price_decision",
            )
        )
    if not rationales.empty:
        frames.append(
            long_metric_frame(
                rationales,
                metrics=["stress_accuracy", "miss_stress_rate", "final_price_shift_pct"],
                metric_family="stress_and_price_trace",
            )
        )
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    columns = [
        "backend",
        "agent_mode",
        "diurnal_blend_alpha",
        "validation_bias",
        "validation_days",
        "metric_family",
        "metric",
        "n",
        "mean",
        "std",
        "sem",
        "min",
        "max",
    ]
    if combined.empty:
        summary = pd.DataFrame(columns=columns)
        summary.to_csv(path, index=False)
        return summary

    groups = [
        column
        for column in ["backend", "agent_mode", "diurnal_blend_alpha", "validation_bias", "validation_days", "metric_family", "metric"]
        if column in combined.columns
    ]
    rows = []
    for key, group in combined.groupby(groups, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        group_meta = dict(zip(groups, key_values))
        values = pd.to_numeric(group["value"], errors="coerce").dropna()
        if values.empty:
            continue
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(
            {
                **group_meta,
                "n": int(len(values)),
                "mean": round(float(values.mean()), 6),
                "std": round(std, 6),
                "sem": round(std / (len(values) ** 0.5), 6) if len(values) > 1 else 0.0,
                "min": round(float(values.min()), 6),
                "max": round(float(values.max()), 6),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        summary = pd.DataFrame(columns=columns)
    summary.to_csv(path, index=False)
    return summary


def long_metric_frame(frame: pd.DataFrame, *, metrics: list[str], metric_family: str) -> pd.DataFrame:
    group_cols = [col for col in ["backend", "agent_mode", "diurnal_blend_alpha", "validation_bias", "validation_days"] if col in frame.columns]
    rows = []
    for metric in metrics:
        if metric not in frame.columns:
            continue
        for _, row in frame.iterrows():
            value = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            rows.append(
                {
                    **{col: row.get(col) for col in group_cols},
                    "metric_family": metric_family,
                    "metric": metric,
                    "value": float(value),
                }
            )
    return pd.DataFrame(rows)
