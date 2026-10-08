"""Pure, offline window/round analysis for saved control result schemas 2 and 3.

Medium refers to the predicted load band. It is deliberately independent of the
controller's ``control_success`` flag, which can include price-policy checks.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import math
from typing import Any, Iterable


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number.is_integer() else None


def _time_identity(value: Any) -> str:
    """Unify ISO timestamps while retaining legacy opaque test/window labels."""
    if value is None:
        raise ValueError("Window identity requires window_start and window_end")
    text = str(value).strip()
    if not text:
        raise ValueError("Window identity cannot be empty")
    # Short numbers and labels are identities, never inferred epoch dates.
    if len(text) >= 10 and text[4:5] == "-" and text[7:8] == "-":
        try:
            timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if timestamp.tzinfo is not None:
                timestamp = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
            return timestamp.isoformat()
        except ValueError:
            pass
    return text


def _windows_by_identity(windows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for window in windows:
        if not isinstance(window, dict):
            continue
        key = (_time_identity(window.get("window_start")), _time_identity(window.get("window_end")))
        if key in result:
            raise ValueError(f"Duplicate window identity: {key}")
        result[key] = window
    return result


def _stress(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().lower().replace("_", " ")
    names = {"low": "Low", "medium": "Medium", "moderate": "Medium",
             "high": "High", "extremely high": "Extremely High",
             "extreme high": "Extremely High", "critical": "Extremely High"}
    return names.get(normalized)


def _medium(stress: str | None) -> bool | None:
    return stress == "Medium" if stress is not None else None


def _first_present(record: dict[str, Any], *names: str) -> Any:
    for name in names:
        if record.get(name) is not None:
            return record[name]
    return None


def _product(price: float | None, load: float | None) -> float | None:
    if price is None or load is None:
        return None
    product = price * load
    return product if math.isfinite(product) else None


def _sum_finite(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    try:
        total = math.fsum(values)
    except OverflowError:
        return None
    return total if math.isfinite(total) else None


def _percentage(change: float | None, baseline: float | None) -> float | None:
    if change is None or baseline is None or baseline == 0:
        return None
    value = change / baseline * 100
    return value if math.isfinite(value) else None


def _zone_identity(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = _integer(value)
        return str(number) if number is not None else None
    text = str(value).strip()
    return text or None


def _hourly_time(value: Any) -> datetime | None:
    try:
        text = _time_identity(value)
        if len(text) < 10 or text[4:5] != "-" or text[7:8] != "-":
            return None
        return datetime.fromisoformat(text)
    except (ValueError, TypeError, OverflowError):
        return None


def _index_baseline_hourly(source: Any) -> tuple[dict[str, dict[datetime, list[dict[str, Any]]]], set[str]]:
    """Index once per result; invalid times cannot be silently dropped from a zone."""
    if source is None:
        return {}, set()
    records = source.to_dict(orient="records") if hasattr(source, "to_dict") else source
    indexed: dict[str, dict[datetime, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    invalid_zones: set[str] = set()
    for row in records:
        if not isinstance(row, dict):
            continue
        zone_id = _zone_identity(row.get("zone_id"))
        if zone_id is None:
            continue
        hour = _hourly_time(row.get("time"))
        if hour is None:
            invalid_zones.add(zone_id)
        else:
            indexed[zone_id][hour].append(row)
    return indexed, invalid_zones


def _baseline_hourly_revenue(
    indexed: dict[str, dict[datetime, list[dict[str, Any]]]], invalid_zones: set[str],
    zone_id: Any, key: tuple[str, str],
) -> float | None:
    """Sum exact hourly products; the window end is its inclusive last hour."""
    zone_key = _zone_identity(zone_id)
    start, end = (_hourly_time(value) for value in key)
    if zone_key is None or zone_key in invalid_zones or start is None or end is None or end < start:
        return None
    duration, remainder = divmod(end - start, timedelta(hours=1))
    if remainder:
        return None
    hours = [(hour, matches) for hour, matches in indexed.get(zone_key, {}).items()
             if start <= hour <= end]
    if len(hours) != duration + 1:
        return None
    revenues = []
    for hour, matches in hours:
        if (hour - start) % timedelta(hours=1) or len(matches) != 1:
            return None
        row = matches[0]
        if row.get("e_price_valid") is False or row.get("price_valid") is False:
            return None
        revenue = _product(_number(row.get("e_price")), _number(row.get("predicted_kwh")))
        if revenue is None:
            return None
        revenues.append(revenue)
    return _sum_finite(revenues)


def _history_fields(baseline: bool | None, history: list[tuple[int, bool | None]]) -> dict[str, Any]:
    states = [baseline, *(state for _, state in history)]
    ever = True if True in states else None if None in states else False
    first_round: int | None = 0 if baseline is True else None
    first_known = baseline is not None
    if baseline is False:
        for round_number, state in history:
            if state is None:
                first_known = False
            elif state is True:
                first_round = round_number if first_known else None
                break
    return {
        "first_entry_round": first_round,
        "first_entry_known": first_known,
        "ever_medium": ever,
        "ever_newly_medium": ever if baseline is False else False if baseline is True else None,
    }


def extract_window_rounds(control_result: dict[str, Any], *, baseline_hourly: Any = None) -> list[dict[str, Any]]:
    """Extract baseline, every control round, and authoritative final observations.

    Missing rounds stay unknown. Only an explicitly successful global early stop
    can supply later rounds from final state. Recorded frozen windows retain the
    prediction recorded for that round rather than an older successful value.
    ``baseline_hourly`` accepts a DataFrame or iterable of dictionaries with
    zone_id/time/e_price/predicted_kwh. Only exact, unique, complete hourly
    coverage supplies the additional hourly-price baseline; no interpolation or
    replacement of the original window-mean baseline is performed.
    """
    if control_result.get("schema_version") not in (2, 3):
        raise ValueError("Expected control_results.json schema version 2 or 3")
    hourly_index, invalid_hourly_zones = _index_baseline_hourly(baseline_hourly)
    zones = control_result.get("zones") or []
    prepared = []
    seen_zones: set[str] = set()
    all_attempt_numbers: list[int] = []
    for zone in zones:
        zone_id = str(zone["zone_id"])
        if zone_id in seen_zones:
            raise ValueError(f"Duplicate Zone: {zone_id}")
        seen_zones.add(zone_id)
        final = _windows_by_identity(zone.get("final_windows") or [])
        attempts = {}
        for attempt in zone.get("attempt_trace") or []:
            number = _integer(attempt.get("attempt"))
            if number is None or number < 1:
                continue
            if number in attempts:
                raise ValueError(f"Duplicate control attempt {number} in Zone {zone_id}")
            attempts[number] = _windows_by_identity(attempt.get("windows") or [])
            all_attempt_numbers.append(number)
        identities = sorted(set(final).union(*(set(value) for value in attempts.values())))
        prepared.append((zone, zone_id, final, attempts, identities))
    run_used = _integer(control_result.get("attempts_used"))
    if run_used is None:
        run_used = max([0, *all_attempt_numbers, *[
            _integer(zone.get("attempts_used")) or 0 for zone in zones
        ]])
    limit = max([run_used, _integer(control_result.get("attempt_limit")) or 0, *all_attempt_numbers])
    early_stop = control_result.get("status") == "success" and run_used < limit
    run_window_count = sum(len(item[4]) for item in prepared)
    rows: list[dict[str, Any]] = []
    for zone, zone_id, final, attempts, identities in prepared:
        zone_used = _integer(zone.get("attempts_used"))
        if zone_used is None:
            zone_used = run_used
        for key in identities:
            final_window = final.get(key, {})
            baseline_window = dict(final_window)
            # Very old exports can omit a final window but retain its baseline
            # fields in attempt records. Only those explicit fields are reused.
            for number in sorted(attempts):
                for name in ("load_stress_level", "mean_energy_price", "sum_predicted_kwh"):
                    if baseline_window.get(name) is None and attempts[number].get(key, {}).get(name) is not None:
                        baseline_window[name] = attempts[number][key][name]
            baseline_stress = _stress(baseline_window.get("load_stress_level"))
            baseline_medium = _medium(baseline_stress)
            baseline_price = _number(baseline_window.get("mean_energy_price"))
            if baseline_window.get("baseline_price_valid") is False:
                baseline_price = None
            baseline_load = _number(baseline_window.get("sum_predicted_kwh"))
            baseline_revenue = _product(baseline_price, baseline_load)
            baseline_hourly_revenue = _baseline_hourly_revenue(
                hourly_index, invalid_hourly_zones, zone.get("zone_id"), key,
            )
            metadata = {
                "zone_id": zone_id, "window_start": key[0], "window_end": key[1],
                "baseline_medium": baseline_medium, "baseline_stress": baseline_stress,
                "baseline_price": baseline_price, "baseline_load_kwh": baseline_load,
                "baseline_revenue": baseline_revenue,
                "baseline_hourly_revenue": baseline_hourly_revenue,
                "zone_expected_window_count": len(identities),
                "run_expected_window_count": run_window_count,
                "run_expected_zone_count": len(prepared),
            }

            def make_row(phase: str, number: int, window: dict[str, Any], source: str,
                         previous: bool | None, history: list[tuple[int, bool | None]]) -> dict[str, Any]:
                if phase == "baseline":
                    stress, price, load = baseline_stress, baseline_price, baseline_load
                    success = None
                else:
                    stress = _stress(_first_present(window, "reforecast_load_stress_level", "price_conditioned_load_stress_level"))
                    price = _number(window.get("proposed_energy_price"))
                    if window.get("price_valid") is False:
                        price = None
                    load = _number(_first_present(window, "reforecast_load_kwh", "price_conditioned_baseline_load_kwh"))
                    success = window.get("control_success")
                    success = success if isinstance(success, bool) else None
                revenue = _product(price, load)
                change = revenue - baseline_revenue if revenue is not None and baseline_revenue is not None else None
                hourly_change = revenue - baseline_hourly_revenue if revenue is not None and baseline_hourly_revenue is not None else None
                row = {
                    **metadata, "phase": phase, "round": number, "medium": _medium(stress),
                    "stress": stress, "control_success": success, "previous_medium": previous,
                    "price": price, "load_kwh": load, "revenue": revenue,
                    "revenue_change": change,
                    "revenue_change_pct": _percentage(change, baseline_revenue),
                    "revenue_change_vs_hourly": hourly_change,
                    "revenue_change_vs_hourly_pct": _percentage(hourly_change, baseline_hourly_revenue),
                    "observation_source": source,
                    "proposal_status": window.get("proposal_status"),
                    "early_stop_carried": source == "early_stop_carry",
                }
                row.update(_history_fields(baseline_medium, history))
                return row

            baseline = make_row("baseline", 0, baseline_window, "baseline", None, [])
            rows.append(baseline)
            history: list[tuple[int, bool | None]] = []
            round_states: dict[int, bool | None] = {0: baseline_medium}
            for number in range(1, limit + 1):
                window = attempts.get(number, {}).get(key)
                source = "attempt_trace"
                if window is None:
                    if early_stop and number > run_used and final_window:
                        window, source = final_window, "early_stop_carry"
                    else:
                        window, source = {}, "missing"
                current = _medium(_stress(_first_present(window, "reforecast_load_stress_level", "price_conditioned_load_stress_level")))
                history.append((number, current))
                rows.append(make_row("round", number, window, source, round_states[number - 1], history))
                round_states[number] = current
            final_medium = _medium(_stress(_first_present(final_window, "reforecast_load_stress_level", "price_conditioned_load_stress_level")))
            final_history = [(number, state) for number, state in history if number < zone_used]
            if zone_used > 0:
                final_history.append((zone_used, final_medium))
            rows.append(make_row("final", zone_used, final_window,
                                 "final_windows" if final_window else "missing",
                                 round_states.get(zone_used - 1), final_history))
    return rows


def _summary_level(rows: list[dict[str, Any]], level: str) -> dict[str, int]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row.get("batch"), row.get("run_id"))
        if level == "zone":
            key += (row.get("zone_id"),)
        grouped[key].append(row)
    complete_count = all_medium_count = 0
    for group in grouped.values():
        expected = max((_integer(row.get(f"{level}_expected_window_count")) or 0 for row in group), default=0)
        identities = {(row.get("zone_id"), row.get("window_start"), row.get("window_end")) for row in group}
        complete = len(identities) >= expected and all(isinstance(row.get("medium"), bool) for row in group)
        complete_count += int(complete)
        all_medium_count += int(complete and all(row["medium"] for row in group))
    return {f"{level}_count": len(grouped), f"{level}_complete_count": complete_count,
            f"{level}_unknown_count": len(grouped) - complete_count,
            f"{level}_all_medium_count": all_medium_count}


def summarize_windows(rows: list[dict[str, Any]], *, include_revenue: bool = False) -> dict[str, Any]:
    """Pool rows for one phase/round without treating unknown values as failures.

    Revenue is deliberately opt-in. Callers must keep both dataset and currency
    separate; ``build_window_tables`` applies those grouping boundaries.
    """
    medium_known = [row for row in rows if isinstance(row.get("medium"), bool)]
    baseline_pairs = [row for row in rows if isinstance(row.get("baseline_medium"), bool) and isinstance(row.get("medium"), bool)]
    adjacent_pairs = [row for row in rows if isinstance(row.get("previous_medium"), bool) and isinstance(row.get("medium"), bool)]
    new = sum(row["baseline_medium"] is False and row["medium"] is True for row in baseline_pairs)
    lost = sum(row["baseline_medium"] is True and row["medium"] is False for row in baseline_pairs)
    gained = sum(row["previous_medium"] is False and row["medium"] is True for row in adjacent_pairs)
    exited = sum(row["previous_medium"] is True and row["medium"] is False for row in adjacent_pairs)
    medium_count = sum(row["medium"] for row in medium_known)
    baseline_known = [row for row in rows if isinstance(row.get("baseline_medium"), bool)]
    control_known = [row for row in rows if isinstance(row.get("control_success"), bool)]
    result: dict[str, Any] = {
        "window_count": len(rows), "medium_known_count": len(medium_known),
        "medium_unknown_count": len(rows) - len(medium_known), "medium_count": medium_count,
        "medium_coverage_pct": 100 * len(medium_known) / len(rows) if rows else None,
        "medium_rate_pct": 100 * medium_count / len(medium_known) if medium_known else None,
        "baseline_medium_count": sum(row["baseline_medium"] for row in baseline_known),
        "baseline_medium_known_count": len(baseline_known),
        "baseline_medium_unknown_count": len(rows) - len(baseline_known),
        "baseline_comparable_count": len(baseline_pairs),
        "baseline_unknown_transition_count": len(rows) - len(baseline_pairs),
        "new_medium_count": new, "lost_medium_count": lost, "net_medium_count": new - lost,
        "adjacent_comparable_count": len(adjacent_pairs),
        "adjacent_unknown_transition_count": len(rows) - len(adjacent_pairs),
        "gained_medium_count": gained, "exited_medium_count": exited,
        "adjacent_net_medium_count": gained - exited,
        "first_entry_count": sum(row.get("baseline_medium") is False and row.get("first_entry_known") is True
                                 and row.get("first_entry_round") == row.get("round") for row in rows),
        "first_entry_unknown_count": sum(row.get("first_entry_known") is not True for row in rows),
        "first_entry_round_distribution": dict(sorted(Counter(
            str(row["first_entry_round"]) for row in rows
            if row.get("baseline_medium") is False and row.get("first_entry_known") is True
            and row.get("first_entry_round") is not None
        ).items(), key=lambda item: int(item[0]))),
        "never_entered_medium_count": sum(row.get("baseline_medium") is False and row.get("first_entry_known") is True
                                          and row.get("first_entry_round") is None for row in rows),
        "ever_newly_medium_count": sum(row.get("ever_newly_medium") is True for row in rows),
        "ever_newly_medium_unknown_count": sum(row.get("ever_newly_medium") is None for row in rows),
        "ever_medium_final_exit_count": sum(row.get("phase") == "final" and row.get("ever_medium") is True and row.get("medium") is False for row in rows),
        "ever_newly_medium_final_exit_count": sum(row.get("phase") == "final" and row.get("ever_newly_medium") is True and row.get("medium") is False for row in rows),
        "control_success_count": sum(row["control_success"] for row in control_known),
        "control_success_known_count": len(control_known),
        "control_success_unknown_count": len(rows) - len(control_known),
        "stress_distribution": dict(sorted(Counter(row.get("stress") or "Unknown" for row in rows).items())),
        **_summary_level(rows, "zone"), **_summary_level(rows, "run"),
    }
    if not include_revenue:
        return result
    for name in ("dataset", "dataset_identity", "revenue_unit"):
        identities = {str(row[name]) for row in rows if row.get(name) is not None}
        if len(identities) > 1:
            raise ValueError(f"Revenue must not pool different {name} values")
    for field in ("baseline_revenue", "baseline_hourly_revenue", "revenue"):
        observed = [_number(row.get(field)) for row in rows]
        known = [value for value in observed if value is not None]
        total = _sum_finite(known)
        complete = bool(rows) and len(known) == len(rows) and total is not None and math.isfinite(total)
        result.update({field: total if complete else None, f"known_{field}": total,
                       f"{field}_complete": complete, f"{field}_known_count": len(known),
                       f"{field}_unknown_count": len(rows) - len(known),
                       f"{field}_coverage_pct": 100 * len(known) / len(rows) if rows else None})
    comparable = [(base, value) for row in rows
                  if (base := _number(row.get("baseline_revenue"))) is not None
                  and (value := _number(row.get("revenue"))) is not None]
    known_change = _sum_finite(value - base for base, value in comparable)
    result["revenue_comparable_count"] = len(comparable)
    result["known_revenue_change"] = known_change
    change = result["revenue"] - result["baseline_revenue"] if result["revenue_complete"] and result["baseline_revenue_complete"] else None
    result["revenue_change"] = change
    result["revenue_change_pct"] = _percentage(change, result["baseline_revenue"])
    hourly_comparable = [(base, value) for row in rows
                         if (base := _number(row.get("baseline_hourly_revenue"))) is not None
                         and (value := _number(row.get("revenue"))) is not None]
    result["revenue_hourly_comparable_count"] = len(hourly_comparable)
    result["known_revenue_change_vs_hourly"] = _sum_finite(value - base for base, value in hourly_comparable)
    hourly_change = result["revenue"] - result["baseline_hourly_revenue"] if result["revenue_complete"] and result["baseline_hourly_revenue_complete"] else None
    result["revenue_change_vs_hourly"] = hourly_change
    result["revenue_change_vs_hourly_pct"] = _percentage(hourly_change, result["baseline_hourly_revenue"])
    return result


def build_window_tables(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Build deterministic control and revenue tables from metadata-enriched rows."""
    control_groups = {
        "round_overall": (),
        "round_by_dataset": ("dataset", "dataset_identity"),
        "round_by_dataset_model": ("dataset", "dataset_identity", "forecast_model"),
        "round_by_mode": ("agent_mode",),
        "round_by_origin_model_mode": ("dataset", "dataset_identity", "forecast_origin", "forecast_model", "agent_mode"),
        "round_by_zone": ("dataset", "dataset_identity", "forecast_model", "agent_mode", "forecast_origin", "zone_id"),
    }
    revenue_groups = {
        "revenue_by_dataset": ("dataset", "dataset_identity", "revenue_unit"),
        "revenue_by_dataset_model_mode": ("dataset", "dataset_identity", "forecast_model", "agent_mode", "revenue_unit"),
        "revenue_by_origin_model_mode": ("dataset", "dataset_identity", "forecast_model", "agent_mode", "forecast_origin", "revenue_unit"),
        "revenue_by_zone": ("dataset", "dataset_identity", "forecast_model", "agent_mode", "forecast_origin", "zone_id", "revenue_unit"),
    }
    tables = {}
    for table_name, dimensions in {**control_groups, **revenue_groups}.items():
        keys = ("batch", *dimensions, "phase", "round")
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[tuple(row.get(key) for key in keys)].append(row)
        table = []
        def sort_key(item: tuple[tuple[Any, ...], Any]) -> tuple[Any, ...]:
            values = item[0]
            return (*[str(value) for value in values[:-2]],
                    {"baseline": 0, "round": 1, "final": 2}.get(values[-2], 3),
                    _integer(values[-1]) or 0)

        for values, group in sorted(grouped.items(), key=sort_key):
            table.append({**dict(zip(keys, values)), **summarize_windows(group, include_revenue=table_name in revenue_groups)})
        tables[table_name] = table
    return tables
