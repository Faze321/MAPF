"""Offline result analysis. Run: python evaluate.py --input output/my_experiment.

No forecasting models or provider clients are loaded. An optional callable can
perform future LLM analysis of each completed control result.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd


TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
DEFINITIONS = {
    "forecast_scope": "Pool finite hourly actual/predicted pairs across Zones; never average Zone metrics.",
    "MAE": "mean(abs(actual - predicted)), kWh",
    "RMSE": "sqrt(mean((actual - predicted)**2)), kWh",
    "RAE": "sum(abs(error)) / sum(abs(actual - global_mean_actual)), ratio",
    "MAPE_pct": "100 * mean(abs(error / actual)); exclude abs(actual) <= 1e-9",
    "WAPE_pct": "100 * sum(abs(error)) / sum(abs(actual))",
    "undefined": "Undefined metrics and missing observations are null, never zero.",
    "first_success": "Success after proposal 1, not the uncontrolled baseline or ever-success.",
    "final_success": "Authoritative status/control_success in control_results.json; predicted control outcome, not observed demand.",
    "transitions": "Match run, Zone and window identity; report retained, gained, lost, never and unknown.",
    "calls": "Recorded agent_call_count, including recorded repair/retry calls; proposals are counted separately.",
    "tokens": "Known provider tokens only; full totals/averages are null if any usage is incomplete.",
    "averages": "Calls/tokens per run and per Zone; tokens per call = pooled tokens / pooled calls. All runs, including failures, are included.",
    "forecast_deduplication": "Within each model/blend, identical samples with the same dataset, forecast parameters, Zone, time and predictions are counted once across copied Agent-mode outputs. Per-run metrics retain all rows.",
}


class AgentAnalyzer(Protocol):
    """Reserved LLM hook: accept a JSON-ready payload, return a JSON-ready dict.

    Payload contains control_result and quantitative_analysis. Evaluation output
    must not be fed back into the control loop or its Agent context.
    """

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]: ...


def evaluate_forecaster(hourly: pd.DataFrame) -> dict[str, Any]:
    """Compute micro/global metrics from raw observations, with explicit coverage."""
    values = hourly[["actual_kwh", "predicted_kwh"]].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
    actual, predicted = values.loc[valid].to_numpy(dtype=float).T
    result: dict[str, Any] = {
        "n_total": len(values), "n": len(actual), "n_excluded": int((~valid).sum()),
        "n_mape": 0, "MAE": None, "RMSE": None, "RAE": None,
        "MAPE_pct": None, "WAPE_pct": None,
    }
    if not len(actual):
        return result
    error = np.abs(actual - predicted)
    nonzero = np.abs(actual) > 1e-9
    rae_denominator = float(np.abs(actual - actual.mean()).sum())
    wape_denominator = float(np.abs(actual).sum())
    result.update(
        MAE=float(error.mean()), RMSE=float(np.sqrt(np.mean(error ** 2))),
        RAE=float(error.sum() / rae_denominator) if rae_denominator > 1e-9 else None,
        MAPE_pct=float(np.mean(error[nonzero] / np.abs(actual[nonzero])) * 100) if nonzero.any() else None,
        WAPE_pct=float(error.sum() / wape_denominator * 100) if wape_denominator > 1e-9 else None,
        n_mape=int(nonzero.sum()),
    )
    return result


def _status(value: Any) -> bool | None:
    return {"success": True, "fail": False}.get(value)


def _all_success(states: list[bool | None]) -> bool | None:
    if not states:
        return None
    if False in states:
        return False
    return None if None in states else True


def _transition_counts(pairs: list[tuple[bool | None, bool | None]]) -> dict[str, Any]:
    known_first = [a for a, _ in pairs if a is not None]
    known_final = [b for _, b in pairs if b is not None]
    result = {
        "count": len(pairs), "first_known_count": len(known_first),
        "final_known_count": len(known_final),
        "first_success_count": sum(known_first), "final_success_count": sum(known_final),
        "first_success_rate_pct": 100 * sum(known_first) / len(known_first) if known_first else None,
        "final_success_rate_pct": 100 * sum(known_final) / len(known_final) if known_final else None,
    }
    for name, pair in {"retained": (True, True), "gained": (False, True),
                       "lost": (True, False), "never": (False, False)}.items():
        result[f"{name}_count"] = sum(item == pair for item in pairs)
    result["unknown_transition_count"] = sum(a is None or b is None for a, b in pairs)
    return result


def _number(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if np.isfinite(number) and number >= 0 and number.is_integer() else None


def _usage(result: dict[str, Any]) -> dict[str, Any]:
    # Use one authoritative total, never sum global + Zone + attempt records.
    raw = result.get("agent_cumulative_usage")
    if not isinstance(raw, dict) or not raw:
        return {"agent_call_count": None, "token_usage_complete": False,
                **{field: None for field in TOKEN_FIELDS}}
    values = {field: _number(raw.get(field)) for field in TOKEN_FIELDS}
    return {
        "agent_call_count": _number(raw.get("agent_call_count")), **values,
        "token_usage_complete": raw.get("token_usage_complete") is True
        and all(value is not None for value in values.values()),
    }


def _window_states(windows: list[dict[str, Any]]) -> dict[tuple[str, str], bool | None]:
    states = {}
    for window in windows:
        key = (str(window["window_start"]), str(window["window_end"]))
        if key in states:
            raise ValueError(f"Duplicate window identity: {key}")
        value = window.get("control_success")
        states[key] = value if isinstance(value, bool) else None
    return states


def evaluate_agent(
    control_result: dict[str, Any], *, analyzer: AgentAnalyzer | None = None,
) -> dict[str, Any]:
    """Analyze one authoritative result (schema 2/3), optionally using an LLM hook."""
    if control_result.get("schema_version") not in (2, 3):
        raise ValueError("Expected control_results.json schema version 2 or 3")
    zones = []
    seen_zones: set[str] = set()
    for zone in control_result["zones"]:
        zone_id = str(zone["zone_id"])
        if zone_id in seen_zones:
            raise ValueError(f"Duplicate Zone: {zone_id}")
        seen_zones.add(zone_id)
        attempts = zone.get("attempt_trace") or []
        first = next((a for a in attempts if a.get("attempt") == 1), {})
        initial = _window_states(first.get("windows") or [])
        final = _window_states(zone.get("final_windows") or [])
        windows = [
            {"window_start": key[0], "window_end": key[1],
             "first_success": initial.get(key), "final_success": final.get(key)}
            for key in sorted(initial.keys() | final.keys())
        ]
        zones.append({
            "zone_id": zone_id, "first_success": _status(first.get("control_status")),
            "final_success": _status(zone.get("status")),
            "attempts_used": _number(zone.get("attempts_used")),
            "usage": _usage(zone), "windows": windows,
            "window_summary": _transition_counts([
                (w["first_success"], w["final_success"]) for w in windows
            ]),
        })
    analysis = {
        "forecast_model": control_result.get("forecast_model"),
        "agent_mode": control_result.get("agent_mode"),
        "forecast_origin": control_result.get("forecast_origin"),
        "first_success": _all_success([z["first_success"] for z in zones]),
        "final_success": _status(control_result.get("status")),
        "attempts_used": _number(control_result.get("attempts_used")),
        "usage": _usage(control_result), "zones": zones,
    }
    analysis["summary"] = summarize_agents([analysis])
    if analyzer is None:
        analysis["llm_analysis"] = {"status": "not_configured"}
    else:
        response = analyzer(copy.deepcopy({
            "control_result": control_result, "quantitative_analysis": analysis,
        }))
        if not isinstance(response, dict):
            raise TypeError("Agent analyzer must return a JSON-serializable dict")
        json.dumps(response, allow_nan=False)
        analysis["llm_analysis"] = {"status": "completed", "result": response}
    return analysis


def summarize_agents(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool outcomes and resource totals before computing means."""
    zones = [zone for run in runs for zone in run["zones"]]
    windows = [window for zone in zones for window in zone["windows"]]
    result: dict[str, Any] = {}
    for scope, records in (("run", runs), ("zone", zones), ("window", windows)):
        counts = _transition_counts([(r["first_success"], r["final_success"]) for r in records])
        result.update({f"{scope}_{key}": value for key, value in counts.items()})
    for scope, records in (("run", runs), ("zone", zones)):
        attempts = [r["attempts_used"] for r in records]
        result[f"mean_attempts_per_{scope}"] = (
            sum(attempts) / len(attempts) if attempts and None not in attempts else None
        )
    usages = [run["usage"] for run in runs]
    calls = [u["agent_call_count"] for u in usages]
    call_total = sum(calls) if calls and None not in calls else None
    result["agent_call_count"] = call_total
    for scope, count in (("run", len(runs)), ("zone", len(zones))):
        result[f"mean_calls_per_{scope}"] = call_total / count if count and call_total is not None else None
    complete = bool(usages) and all(u["token_usage_complete"] for u in usages)
    result["token_usage_complete"] = complete
    result["incomplete_usage_run_count"] = sum(not u["token_usage_complete"] for u in usages)
    for field in TOKEN_FIELDS:
        observed = [u[field] for u in usages if u[field] is not None]
        known = sum(observed) if observed else None
        result[f"known_{field}"] = known
        result[field] = known if complete else None
        for scope, count in (("run", len(runs)), ("zone", len(zones)), ("call", call_total)):
            result[f"mean_{field}_per_{scope}"] = known / count if complete and count else None
    return result


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def evaluate_directory(
    input_dir: Path | str, *, analyzer: AgentAnalyzer | None = None,
    section: str = "all",
) -> dict[str, Any]:
    """Discover native outputs recursively; works for one run or an experiment matrix."""
    root = Path(input_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"Input directory does not exist: {root}")
    if section not in ("all", "forecaster", "agent"):
        raise ValueError(f"Unknown section: {section}")
    forecast_runs = []
    forecast_groups: dict[tuple[str, str], list[pd.DataFrame]] = {}
    if section != "agent":
        detail_dirs = sorted({p.parent for p in root.rglob("zone_*_forecast_vs_actual.csv")})
        for detail_dir in detail_dirs:
            paths = sorted(detail_dir.glob("zone_*_forecast_vs_actual.csv"))
            frame = pd.concat([pd.read_csv(p, dtype={"zone_id": str}) for p in paths], ignore_index=True)
            metadata_path = detail_dir.parent / "forecast_metrics.csv"
            metadata = pd.read_csv(metadata_path) if metadata_path.exists() else pd.DataFrame()
            model = "unknown"
            blend = "unknown"
            for column in ("forecast_model", "diurnal_blend_alpha"):
                unique = metadata[column].dropna().unique() if column in metadata else []
                if len(unique) > 1:
                    raise ValueError(f"Mixed {column} in {metadata_path}")
                if len(unique):
                    if column == "forecast_model":
                        model = str(unique[0])
                    else:
                        blend = str(unique[0])
            run_id = str(detail_dir.relative_to(root))
            forecast_runs.append({"run_id": run_id, "forecast_model": model,
                                  "diurnal_blend_alpha": blend, **evaluate_forecaster(frame)})
            manifest_path = detail_dir.parent.parent / "forecaster_manifest.json"
            # Shared forecaster copies across Agent modes must not weight a model
            # more heavily. Without provenance, keep each source independent.
            identity = str(detail_dir)
            if manifest_path.exists():
                manifest = _read_json(manifest_path)
                provenance = {key: manifest.get(key) for key in ("data_source", "forecast_parameters", "zone_ids")}
                if provenance["data_source"] and provenance["forecast_parameters"]:
                    identity = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
            frame["_forecast_identity"] = identity
            forecast_groups.setdefault((model, blend), []).append(frame)
    forecast_summary = []
    for (model, blend), frames in sorted(forecast_groups.items()):
        pooled = pd.concat(frames, ignore_index=True)
        keys = ["_forecast_identity", "zone_id", "time", "actual_kwh", "predicted_kwh"]
        unique = pooled.drop_duplicates(keys) if all(key in pooled for key in keys) else pooled
        forecast_summary.append({
            "forecast_model": model, "diurnal_blend_alpha": blend, "run_count": len(frames),
            "duplicate_sample_count": len(pooled) - len(unique), **evaluate_forecaster(unique),
        })
    agent_runs = []
    if section != "forecaster":
        for path in sorted(root.rglob("control_results.json")):
            try:
                analysis = evaluate_agent(_read_json(path), analyzer=analyzer)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Cannot evaluate {path}: {exc}") from exc
            analysis["run_id"] = str(path.relative_to(root))
            agent_runs.append(analysis)
    if not forecast_runs and not agent_runs:
        raise ValueError(f"No {section} result artifacts found under {root}")
    agent_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in agent_runs:
        key = (str(run["forecast_model"]), str(run["agent_mode"]))
        agent_groups.setdefault(key, []).append(run)
    return {
        "schema_version": 1, "input_dir": str(root), "definitions": DEFINITIONS,
        "forecaster": {"runs": forecast_runs, "global_by_model": forecast_summary},
        "agent": {
            "runs": agent_runs, "overall": summarize_agents(agent_runs),
            "by_model_and_mode": [
                {"forecast_model": model, "agent_mode": mode, **summarize_agents(runs)}
                for (model, mode), runs in sorted(agent_groups.items())
            ],
        },
    }


def write_evaluation(report: dict[str, Any], output_dir: Path | str) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "evaluation.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tables = {
        "forecaster_global_metrics.csv": report["forecaster"]["global_by_model"],
        "forecaster_run_metrics.csv": report["forecaster"]["runs"],
        "agent_summary.csv": report["agent"]["by_model_and_mode"],
        "agent_run_metrics.csv": [
            {"run_id": r["run_id"], "forecast_model": r["forecast_model"],
             "agent_mode": r["agent_mode"], **r["summary"]}
            for r in report["agent"]["runs"]
        ],
    }
    for filename, rows in tables.items():
        if rows:
            pd.DataFrame(rows).to_csv(output / filename, index=False, encoding="utf-8-sig")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("output"), help="Run or experiment directory")
    parser.add_argument("--output", type=Path, help="Default: <input>/analysis")
    parser.add_argument("--section", choices=("all", "forecaster", "agent"), default="all")
    parser.add_argument("--agent-analyzer", help="Optional callable module:function for LLM analysis")
    args = parser.parse_args()
    analyzer = None
    if args.agent_analyzer:
        module_name, separator, attribute = args.agent_analyzer.partition(":")
        if not separator or not module_name or not attribute:
            parser.error("--agent-analyzer must use module:function")
        if args.section == "forecaster":
            parser.error("--agent-analyzer requires the agent section")
        analyzer = getattr(importlib.import_module(module_name), attribute)
        if not callable(analyzer):
            parser.error("--agent-analyzer must reference a callable")
    report = evaluate_directory(args.input, analyzer=analyzer, section=args.section)
    target = write_evaluation(report, args.output or args.input / "analysis")
    print(f"Evaluated {len(report['forecaster']['runs'])} forecaster runs and "
          f"{len(report['agent']['runs'])} agent runs. Report: {target.resolve()}")


if __name__ == "__main__":
    main()
