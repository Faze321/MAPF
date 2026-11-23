"""Replay price scenarios with a frozen forecaster; never call a pricing agent."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.control.policy import LOAD_POLICY_ID, MEDIUM_STRESS, medium_gap_kwh, medium_load_bounds
from src.control.pricing import energy_price_with_predicted_windows
from src.control.scoring import schedule_score
from src.control.windows import build_pricing_windows_3h
from src.data.adapters.registry import load_canonical_dataset
from src.data.canonical import DatasetSpec
from src.results.manifest import load_forecaster_bundle


def replay(artifact, context, history):
    zone = str(artifact.zone)
    future = pd.DataFrame(artifact.known_future)
    future["time"] = pd.to_datetime(future["time"])
    original = future[["time", "e_price"]].rename(columns={"e_price": zone})
    hist = history.loc[history.index < pd.Timestamp(artifact.forecast_origin)].sort_index()
    if hist.empty or hist.index.has_duplicates:
        raise ValueError(f"Missing or duplicate price history for zone {zone}")
    energy = hist["energy_price"]
    pairs = (set(zip(energy.round(8), hist.service_price.round(8))) if "service_price" in hist else None)
    limits = dict(historical_min_load_kwh=context["historical_min_load_3h_kwh"],
                  historical_max_load_kwh=context["historical_max_load_3h_kwh"])
    lower, upper = medium_load_bounds(limits["historical_min_load_kwh"], limits["historical_max_load_kwh"])
    probes = 0

    def evaluate(prices, shifts):
        nonlocal probes
        probes += 1
        hourly = artifact.predict(prices).hourly.rename(columns={"timestamp": "time", "predicted_load": "predicted_kwh"})
        windows = build_pricing_windows_3h(hourly, stress_thresholds=limits)
        for w, shift in zip(windows, shifts, strict=True):
            w.update(price_valid=True, price_conditioned_load_stress_level=w["load_stress_level"],
                     price_conditioned_baseline_load_kwh=w["sum_predicted_kwh"], suggested_price_shift_pct=shift)
        return windows

    zero = [0.0] * len(context["pricing_windows_3h"])
    baseline = evaluate(original, zero)
    saved_windows = context["pricing_windows_3h"]
    for saved, reproduced in zip(saved_windows, baseline, strict=True):
        keys = ("window_start", "window_end")
        if (any(saved[k] != reproduced[k] for k in keys) or
                not np.isclose(saved["sum_predicted_kwh"], reproduced["sum_predicted_kwh"], atol=1e-4, rtol=1e-6)):
            raise ValueError(f"Frozen replay did not reproduce saved R0 at zone {zone}, {reproduced['window_start']}")
    combined = zero.copy()
    candidates, diagnostics = [], []
    for i, target in enumerate(baseline):
        if target["load_stress_level"] == MEDIUM_STRESS:
            continue
        start, end = pd.Timestamp(target["window_start"]), pd.Timestamp(target["window_end"])
        mask = future.time.between(start, end)
        base_mean = original.loc[mask, zone].mean()
        if not np.isfinite(base_mean) or base_mean < 0:
            raise ValueError(f"Invalid original price for {zone} at {start}")
        width = int(mask.sum())
        # Use only complete same-clock historical windows, never bridge missing hours.
        means = energy.asfreq("h").rolling(width, min_periods=width).mean()
        means = means[means.index.hour == end.hour].dropna()
        grid = {float(v): [f"fixed:{v}"] for v in (0, -5, 5, -10, 10, -20, 20, -40, 40)}
        # A zero-priced window cannot reach a positive historical mean by scaling.
        if base_mean > 0 and not means.empty:
            for q in (.10, .25, .50, .75, .90):
                shift = round(100 * (means.quantile(q) / base_mean - 1), 10)
                grid.setdefault(shift, []).append(f"historical:P{q*100:.0f}")
        best = schedule_score(baseline)
        hits = []
        for shift, sources in grid.items():
            shifts = zero.copy()
            shifts[i] = shift
            schedule = energy_price_with_predicted_windows(original, zone, [dict(target, suggested_price_shift_pct=shift)])
            evaluated = evaluate(schedule, shifts) if shift else baseline
            w = evaluated[i]
            changed = schedule.loc[mask, zone]
            other = [(a["load_stress_level"] == MEDIUM_STRESS, b["load_stress_level"] == MEDIUM_STRESS)
                     for j, (a, b) in enumerate(zip(baseline, evaluated)) if j != i]
            hit = w["load_stress_level"] == MEDIUM_STRESS
            if hit:
                hits.append(shift)
            score = schedule_score(evaluated)
            if score < best:
                best, combined[i] = score, shift
            candidates.append(dict(zone_id=zone, window_start=start, window_end=end,
                original_mean_energy_price=base_mean, historical_quantiles_reachable=base_mean > 0,
                shift_pct=shift, candidate_sources="|".join(sources), mean_energy_price=changed.mean(),
                forecast_load_kwh=w["sum_predicted_kwh"], stress=w["load_stress_level"],
                gap_kwh=medium_gap_kwh(w["sum_predicted_kwh"], lower, upper),
                other_entered=sum(not a and b for a, b in other), other_exited=sum(a and not b for a, b in other),
                within_historical_range=bool(changed.between(energy.min(), energy.max()).all()),
                observed_price_pairs=(all((p,s) in pairs for p,s in zip(changed.round(8), future.loc[mask,"s_price"].round(8)))
                                      if pairs is not None and "s_price" in future else None)))
        diagnostics.append((i, bool(hits)))

    schedule = energy_price_with_predicted_windows(original, zone, [
        dict(w, suggested_price_shift_pct=s) for w, s in zip(baseline, combined)])
    final = evaluate(schedule, combined) if any(combined) else baseline
    final_wins = schedule_score(final) < schedule_score(baseline)
    hits = dict(diagnostics)
    summary = [dict(zone_id=zone, window_start=w["window_start"], window_end=w["window_end"],
                    original_load_kwh=b["sum_predicted_kwh"], original_stress=b["load_stress_level"],
                    any_tested_single_window_hit=hits.get(i), combined_shift_pct=combined[i],
                    combined_load_kwh=w["sum_predicted_kwh"], combined_stress=w["load_stress_level"],
                    selected_shift_pct=combined[i] if final_wins else 0,
                    combined_beats_original=final_wins, zone_forecast_calls=probes)
               for i, (b, w) in enumerate(zip(baseline, final))]
    return candidates, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forecaster_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, help="Local copy of the original dataset")
    parser.add_argument("--zone", action="append", help="Repeat to select zones (default: all)")
    args = parser.parse_args()
    bundle = load_forecaster_bundle(args.forecaster_dir)
    contexts = bundle.contexts
    if args.zone:
        contexts = [c for c in contexts if str(c["zone_id"]) in args.zone]
        if set(args.zone) != {str(c["zone_id"]) for c in contexts}:
            parser.error("Unknown zone in this forecaster artifact")
    source = bundle.manifest["data_source"]
    dataset = load_canonical_dataset(DatasetSpec(args.data_dir or Path(source["data_dir"]),
        adapter=source["adapter"], weather_file=source["weather_file"]))
    candidates, summaries = [], []
    for context in contexts:
        zone = str(context["zone_id"])
        artifact = bundle.scenario_artifacts[zone]
        hist = dataset.timeseries.loc[dataset.timeseries.zone_id.astype(str) == zone].set_index("timestamp")
        # A relocated copy can have a different cache fingerprint; verify the observed price series.
        for records in (artifact.history, artifact.validation, artifact.known_future):
            if records:
                ref = pd.DataFrame(records)
                joined = hist.reindex(pd.to_datetime(ref.time))
                for raw, canonical in (("e_price", "energy_price"), ("s_price", "service_price")):
                    if raw in ref and canonical in joined and not np.allclose(
                            ref[raw].astype(float), joined[canonical].astype(float), equal_nan=True):
                        raise ValueError(f"Historical source mismatch: {zone} {raw}")
        rows, results = replay(artifact, context, hist)
        candidates.extend(rows)
        summaries.extend(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("price_candidates", candidates), ("combined_windows", summaries)):
        pd.DataFrame(rows).assign(load_policy_id=LOAD_POLICY_ID).to_csv(args.output_dir / f"{name}.csv", index=False)
