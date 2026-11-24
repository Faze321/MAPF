from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.control.numbers import optional_float
from src.control.policy import MEDIUM_STRESS, STRESS_LEVELS
from src.control.pricing import shifted_energy_prices
from src.time_utils import parse_datetime_24h


PRICE_RANGE_FIELDS = (
    "historical_min_energy_price", "historical_max_energy_price", "historical_price_hours",
    "executed_min_energy_price", "executed_max_energy_price", "executed_price_hours",
    "price_range_status", "hours_below_historical_min", "hours_above_historical_max",
)
OUTSIDE_PRICE_RANGE = {"below", "above", "both"}
PRICE_RANGE_TABLES = ("final_price_out_of_range_windows", "final_price_out_of_range_non_medium_windows")


def write_price_range_tables(directory: Path, final_rows: list[dict[str, Any]], forecast_origin: str | None) -> None:
    columns = ["backend", "agent_mode", "zone_id", "window_start", "window_end", "selected_attempt",
               "mean_energy_price", "suggested_price_shift_pct", "proposed_energy_price", *PRICE_RANGE_FIELDS,
               "price_conditioned_baseline_load_kwh", "price_conditioned_load_stress_level"]
    rows = pd.DataFrame((w for w in final_rows if w.get("price_range_status") in OUTSIDE_PRICE_RANGE),
                        columns=columns)
    rows.insert(0, "forecast_origin", forecast_origin)
    non_medium = rows["price_conditioned_load_stress_level"].isin(set(STRESS_LEVELS) - {MEDIUM_STRESS})
    for name, frame in zip(PRICE_RANGE_TABLES, (rows, rows.loc[non_medium]), strict=True):
        frame.to_csv(directory / f"{name}.csv", index=False)


def annotate_final_price_ranges(reports: list[dict[str, Any]], energy_price: pd.DataFrame,
                                forecast_origin: pd.Timestamp) -> None:
    """Audit only selected schedules. Never feed these diagnostics back to the agents."""
    times = pd.DatetimeIndex(energy_price["time"])
    for report in reports:
        zone = str(report["zone_id"])
        prices = (pd.Series(pd.to_numeric(energy_price[zone], errors="coerce").to_numpy(), index=times)
                  if zone in energy_price else pd.Series(index=times, dtype=float))
        prices = prices.where(np.isfinite(prices))
        history = prices.loc[prices.index < forecast_origin].dropna()
        lo, hi = optional_float(history.min()), optional_float(history.max())
        for w in report.get("price_change_windows_3h") or []:
            w.update(historical_min_energy_price=lo, historical_max_energy_price=hi,
                     historical_price_hours=len(history), executed_min_energy_price=None,
                     executed_max_energy_price=None, executed_price_hours=0,
                     hours_below_historical_min=None, hours_above_historical_max=None,
                     price_range_status="unavailable")
            shift = optional_float(w.get("suggested_price_shift_pct"))
            if not w.get("price_valid") or shift is None or shift < -100:
                continue
            hours = pd.date_range(parse_datetime_24h(w["window_start"]), parse_datetime_24h(w["window_end"]), freq="h")
            original = prices.reindex(hours).dropna()
            executed = shifted_energy_prices(original, shift)
            executed = executed[np.isfinite(executed)]
            w.update(executed_min_energy_price=optional_float(executed.min()),
                     executed_max_energy_price=optional_float(executed.max()), executed_price_hours=len(executed))
            if shift == 0 or (len(executed) == len(hours) and executed.eq(original).all()):
                w["price_range_status"] = "unchanged"
                continue
            if lo is None or hi is None or executed.empty:
                continue
            # Decimal tariffs at an inclusive endpoint can differ by a floating-point ULP.
            below = int(((executed < lo) & ~np.isclose(executed, lo, rtol=1e-12, atol=1e-12)).sum())
            above = int(((executed > hi) & ~np.isclose(executed, hi, rtol=1e-12, atol=1e-12)).sum())
            status = ("both" if below and above else "below" if below else "above" if above
                      else "within" if len(executed) == len(hours) else "unavailable")
            w.update(hours_below_historical_min=below, hours_above_historical_max=above, price_range_status=status)
