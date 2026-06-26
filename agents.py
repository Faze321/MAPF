from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

from config import AgentConfig
from prompts import SYSTEM_MESSAGE, behavior_prompt, economist_prompt, grid_prompt, repair_economist_prompt


GRID_STRESS_LEVELS = ("Low", "Medium", "High", "Extreme High")
MODEL_RESPONSE_FAILED = "MODEL_RESPONSE_FAILED"
ECONOMIST_AGENT_OUTPUT_KEY = "_economist_agent_output"
AGENT_COMPLETION_USAGE_KEY = "_agent_completion_usage"
NASH_MAX_ITERATIONS = 6
NASH_PRICE_STABILITY_EPSILON_PCT = 0.5
NASH_MAX_DISCOMFORT_SCORE = 1.0
NASH_MAX_ABS_PRICE_SHIFT_PCT = 25.0
NASH_MIN_ELASTICITY = 0.05
NASH_MAX_ELASTICITY = 0.7
GRID_STRESS_LEVEL_BY_KEY = {level.lower(): level for level in GRID_STRESS_LEVELS}
GRID_STRESS_LEVEL_BY_KEY.update(
    {
        "moderate": "Medium",
        "critical": "Extreme High",
        "critica": "Extreme High",
        "extrame high": "Extreme High",
        "extreme_high": "Extreme High",
    }
)


class AgentStageError(RuntimeError):
    def __init__(self, *, stage: str, zone_id: Any, agent: str, original: Exception) -> None:
        self.stage = stage
        self.zone_id = zone_id
        self.agent = agent
        self.original = original
        message = (
            f"{agent} failed for zone {zone_id} at {stage}: "
            f"{type(original).__name__}: {original}"
        )
        super().__init__(message)


class ChatClient(Protocol):
    async def complete_json(self, prompt: str, *, temperature: float) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class AgentCallResult:
    content: dict[str, Any]
    usage: dict[str, Any]


@dataclass
class AgentChatClient:
    config: AgentConfig

    def __post_init__(self) -> None:
        if not self.config.api_key:
            raise ValueError("agent.api_key is required when dry-run is disabled")
        from openai import AsyncOpenAI

        headers = {}
        # if self.config.http_referer:
        #     headers["HTTP-Referer"] = self.config.http_referer
        # if self.config.title:
        #     headers["X-Title"] = self.config.title
        #     headers["X-OpenRouter-Title"] = self.config.title
        self._client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            default_headers=headers or None,
            timeout=self.config.timeout_seconds,
        )

    async def complete_json(self, prompt: str, *, temperature: float) -> dict[str, Any]:
        response = await self._client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        content = response.choices[0].message.content or "{}"
        payload = extract_json_object(content)
        payload[AGENT_COMPLETION_USAGE_KEY] = response_token_usage(response)
        return payload


@dataclass
class DryRunChatClient:
    async def complete_json(self, prompt: str, *, temperature: float) -> dict[str, Any]:
        raise RuntimeError("DryRunChatClient should not receive raw prompts")


async def run_zone_chain(
    context: dict[str, Any],
    *,
    client: ChatClient | None,
    temperature: float = 0.2,
    heuristic_source: str = "dry-run",
) -> dict[str, Any]:
    if client is None:
        return heuristic_zone_chain(context, source=heuristic_source)

    grid_result = await complete_agent_json(
        client,
        grid_prompt(context),
        context=context,
        temperature=temperature,
        stage="agent.grid",
        agent="Grid Analyst",
    )
    grid = merge_grid_fallback(
        grid_result.content,
        context,
    )
    behavior_result = await complete_agent_json(
        client,
        behavior_prompt(context, grid),
        context=context,
        temperature=temperature,
        stage="agent.behavior",
        agent="Behavioural Agent",
    )
    behavior = merge_behavior_fallback(
        behavior_result.content,
        context,
    )
    economist_report, economist_call_usage = await complete_validated_economist_report(
        client,
        context,
        grid,
        behavior,
        temperature=temperature,
    )
    economist_debug = economist_report.get(ECONOMIST_AGENT_OUTPUT_KEY)
    economist = merge_economist_fallback(
        economist_report,
        context,
    )
    agent_call_usage = [grid_result.usage, behavior_result.usage, *economist_call_usage]
    return combine_reports(
        context,
        grid,
        behavior,
        economist,
        source="model",
        economist_debug=economist_debug,
        agent_call_usage=agent_call_usage,
    )


async def run_all_zone_chains(
    contexts: list[dict[str, Any]],
    *,
    client: ChatClient | None,
    temperature: float = 0.2,
    heuristic_source: str = "dry-run",
) -> list[dict[str, Any]]:
    tasks = [
        run_zone_chain(
            context,
            client=client,
            temperature=temperature,
            heuristic_source=heuristic_source,
        )
        for context in contexts
    ]
    return await asyncio.gather(*tasks)


async def complete_agent_json(
    client: ChatClient,
    prompt: str,
    *,
    context: dict[str, Any],
    temperature: float,
    stage: str,
    agent: str,
) -> AgentCallResult:
    started = time.perf_counter()
    try:
        response = await client.complete_json(prompt, temperature=temperature)
    except AgentStageError:
        raise
    except Exception as exc:
        raise AgentStageError(
            stage=stage,
            zone_id=context.get("zone_id"),
            agent=agent,
            original=exc,
        ) from exc
    elapsed = round(time.perf_counter() - started, 4)
    token_usage = response.pop(AGENT_COMPLETION_USAGE_KEY, {})
    return AgentCallResult(
        content=response,
        usage=agent_call_usage_record(
            stage=stage,
            agent=agent,
            zone_id=context.get("zone_id"),
            elapsed_seconds=elapsed,
            token_usage=token_usage,
        ),
    )


async def complete_validated_economist_report(
    client: ChatClient,
    context: dict[str, Any],
    grid: dict[str, Any],
    behavior: dict[str, Any],
    *,
    temperature: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected_windows = context.get("pricing_windows_3h", [])
    initial_result = await complete_agent_json(
        client,
        economist_prompt(context, grid, behavior),
        context=context,
        temperature=temperature,
        stage="agent.economist",
        agent="Market Economist",
    )
    report = initial_result.content
    call_usage = [initial_result.usage]
    errors = validate_economist_report(report, expected_windows)
    debug: dict[str, Any] = {
        "zone_id": context.get("zone_id"),
        "category": context.get("category"),
        "forecast_start": context.get("forecast_start"),
        "forecast_end": context.get("forecast_end"),
        "expected_price_window_count": len(expected_windows) if isinstance(expected_windows, list) else 0,
        "initial_response": report,
        "initial_validation_errors": errors,
        "repair_attempted": False,
        "repair_response": None,
        "repair_validation_errors": None,
        "selected_response_source": "initial",
        "selected_validation_errors": errors,
    }
    if not errors:
        debug["agent_call_usage"] = call_usage
        debug["agent_usage_summary"] = summarize_agent_call_usage(call_usage)
        return attach_economist_debug(report, debug), call_usage

    repair_result = await complete_agent_json(
        client,
        repair_economist_prompt(context, grid, behavior, report, errors),
        context=context,
        temperature=min(temperature, 0.1),
        stage="agent.economist_repair",
        agent="Market Economist Repair",
    )
    repaired = repair_result.content
    call_usage.append(repair_result.usage)
    repair_errors = validate_economist_report(repaired, expected_windows)
    debug.update(
        {
            "repair_attempted": True,
            "repair_response": repaired,
            "repair_validation_errors": repair_errors,
        }
    )
    if not repair_errors or len(repair_errors) < len(errors):
        debug["selected_response_source"] = "repair"
        debug["selected_validation_errors"] = repair_errors
        debug["agent_call_usage"] = call_usage
        debug["agent_usage_summary"] = summarize_agent_call_usage(call_usage)
        return attach_economist_debug(repaired, debug), call_usage
    debug["agent_call_usage"] = call_usage
    debug["agent_usage_summary"] = summarize_agent_call_usage(call_usage)
    return attach_economist_debug(report, debug), call_usage


def attach_economist_debug(report: dict[str, Any], debug: dict[str, Any]) -> dict[str, Any]:
    tagged = dict(report)
    tagged[ECONOMIST_AGENT_OUTPUT_KEY] = debug
    return tagged


def response_token_usage(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    prompt_tokens = usage_value(usage, "prompt_tokens", "input_tokens")
    completion_tokens = usage_value(usage, "completion_tokens", "output_tokens")
    total_tokens = usage_value(usage, "total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def usage_value(usage: Any, *keys: str) -> int | None:
    if usage is None:
        return None
    for key in keys:
        value = None
        if isinstance(usage, dict):
            value = usage.get(key)
        else:
            value = getattr(usage, key, None)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def agent_call_usage_record(
    *,
    stage: str,
    agent: str,
    zone_id: Any,
    elapsed_seconds: float,
    token_usage: dict[str, Any] | None,
) -> dict[str, Any]:
    usage = token_usage if isinstance(token_usage, dict) else {}
    return {
        "stage": stage,
        "agent": agent,
        "zone_id": zone_id,
        "elapsed_seconds": elapsed_seconds,
        "prompt_tokens": optional_int_usage(usage.get("prompt_tokens")),
        "completion_tokens": optional_int_usage(usage.get("completion_tokens")),
        "total_tokens": optional_int_usage(usage.get("total_tokens")),
    }


def summarize_agent_call_usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "elapsed_seconds": round(sum(optional_float_usage(record.get("elapsed_seconds")) for record in records), 4),
        "prompt_tokens": sum_token_usage(records, "prompt_tokens"),
        "completion_tokens": sum_token_usage(records, "completion_tokens"),
        "total_tokens": sum_token_usage(records, "total_tokens"),
    }


def sum_token_usage(records: list[dict[str, Any]], field: str) -> int:
    total = 0
    for record in records:
        value = optional_int_usage(record.get(field))
        if value is not None:
            total += value
    return total


def optional_int_usage(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def optional_float_usage(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def heuristic_zone_chain(context: dict[str, Any], *, source: str = "dry-run") -> dict[str, Any]:
    grid = heuristic_grid(context)
    behavior = heuristic_behavior(context)
    economist = heuristic_economist(context, grid)
    return combine_reports(context, grid, behavior, economist, source=source)


def heuristic_grid(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "forecast_total_kwh": context["forecast_total_kwh"],
        "forecast_peak_kwh": context["forecast_peak_kwh"],
        "predicted_change_pct": context["predicted_change_pct"],
        "grid_stress_level": normalize_grid_stress_level(context.get("grid_stress_level"), "Low"),
        "forecast_summary": (
            f"{context['category']} zone forecast is anchored to the prior-week hourly "
            f"shape with a {context['predicted_change_pct']:+.1f}% change versus the "
            "recent comparable window."
        ),
    }


def heuristic_behavior(context: dict[str, Any]) -> dict[str, Any]:
    category = context["category"]
    weather = context["weather"]
    shape = context["hourly_shape"]
    windows = context.get("pricing_windows_3h") or []
    horizon_days = context.get("forecast_horizon_days", "configured")
    forecast_start = str(context.get("forecast_start") or "the forecast start")
    forecast_end = str(context.get("forecast_end") or "the forecast end")
    change = as_float(context.get("predicted_change_pct"), 0)
    drivers = []
    high_windows = [
        window
        for window in windows
        if normalize_grid_stress_level(window.get("load_stress_level") or window.get("grid_stress_level"), "Low")
        in {"High", "Extreme High"}
    ]
    peak_window = max(windows, key=lambda window: as_float(window.get("sum_predicted_kwh"), 0), default=None)
    if high_windows:
        drivers.append(f"{len(high_windows)} high-stress 3-hour windows")
    if peak_window:
        drivers.append(
            "peak window "
            f"{peak_window.get('window_start')} to {peak_window.get('window_end')} "
            f"at {as_float(peak_window.get('sum_predicted_kwh'), 0):.1f} kWh"
        )
    if abs(change) >= 5:
        drivers.append(f"{change:+.1f}% load change versus comparable history")
    else:
        drivers.append("stable total load versus comparable history")
    if weather["rain_hours"] > 0:
        drivers.append(f"{weather['rain_hours']} rainy hours")
    if shape["night_20_6"] >= shape["morning_7_10"]:
        drivers.append("night plateau")
    if shape["evening_17_22"] >= shape["morning_7_10"]:
        drivers.append("evening lift")
    profile = context.get("profile") or {}
    poi_total = profile.get("poi_total", 0)
    return {
        "agent_reasoning": (
            f"For the {horizon_days}-day window from {forecast_start} to {forecast_end}, "
            f"{category} demand is best explained by {', '.join(drivers)} and the "
            f"local POI mix of {poi_total} assigned POIs."
        ),
        "demand_drivers": drivers,
        "elasticity_factor": estimate_elasticity_factor(context, {}, peak_window or {}),
        "tolerance_threshold_pct": estimate_price_tolerance_pct(context, peak_window or {}),
        "confidence": "medium",
    }


def heuristic_economist(context: dict[str, Any], grid: dict[str, Any]) -> dict[str, Any]:
    category = context["category"]
    stress = grid["grid_stress_level"]
    change = float(grid["predicted_change_pct"])
    shift, label = base_price_shift(stress, category, change)
    window_shifts = build_heuristic_price_windows(context, grid)
    return {
        "suggested_price_shift_pct": shift,
        "action_label": label,
        "price_rationale": (
            f"{stress} stress with {change:+.1f}% expected load change; "
            f"category elasticity proxy is {category}."
        ),
        "price_change_windows_3h": window_shifts,
    }


def combine_reports(
    context: dict[str, Any],
    grid: dict[str, Any],
    behavior: dict[str, Any],
    economist: dict[str, Any],
    *,
    source: str,
    economist_debug: dict[str, Any] | None = None,
    agent_call_usage: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    price_windows = normalize_price_windows(
        economist.get("price_change_windows_3h"),
        context.get("pricing_windows_3h", []),
    )
    price_windows = apply_nash_equilibrium_to_windows(context, behavior, price_windows)
    nash_summary = summarize_nash_equilibrium(price_windows)
    final_shift = average_price_shift(price_windows, as_float(economist.get("suggested_price_shift_pct"), 0))
    usage_summary = summarize_agent_call_usage(agent_call_usage or [])
    report = {
        "category": context["category"],
        "zone_id": context["zone_id"],
        "predicted_load_kwh": as_float(grid.get("forecast_total_kwh"), context["forecast_total_kwh"]),
        "predicted_peak_kwh": as_float(grid.get("forecast_peak_kwh"), context["forecast_peak_kwh"]),
        "predicted_change_pct": as_float(grid.get("predicted_change_pct"), context["predicted_change_pct"]),
        "actual_load_kwh": context.get("actual_total_kwh"),
        "mae_kwh": context.get("mae_kwh"),
        "rmse_kwh": context.get("rmse_kwh"),
        "mape_pct": context.get("mape_pct"),
        "rae": context.get("rae"),
        "wape_pct": context.get("wape_pct"),
        "grid_stress_level": normalize_grid_stress_level(
            grid.get("grid_stress_level"),
            context["grid_stress_level"],
        ),
        "actual_grid_stress_level": context.get("actual_grid_stress_level"),
        "stress_accuracy": context.get("stress_accuracy"),
        "miss_stress_rate": context.get("miss_stress_rate"),
        "stress_eval_windows": context.get("stress_eval_windows"),
        "stress_miss_count": context.get("stress_miss_count"),
        "agent_reasoning": str(behavior.get("agent_reasoning") or ""),
        "suggested_price_shift_pct": final_shift,
        "action_label": str(economist.get("action_label") or ""),
        "price_rationale": str(economist.get("price_rationale") or ""),
        "price_change_windows_3h": price_windows,
        "nash_equilibrium_reached": nash_summary["nash_equilibrium_reached"],
        "nash_equilibrium_windows": nash_summary["nash_equilibrium_windows"],
        "nash_equilibrium_reached_windows": nash_summary["nash_equilibrium_reached_windows"],
        "nash_equilibrium_rounds": nash_summary["nash_equilibrium_rounds"],
        "nash_equilibrium_summary": nash_summary["nash_equilibrium_summary"],
        "agent_time_cost_seconds": usage_summary["elapsed_seconds"],
        "agent_prompt_tokens": usage_summary["prompt_tokens"],
        "agent_completion_tokens": usage_summary["completion_tokens"],
        "agent_total_tokens": usage_summary["total_tokens"],
        "agent_call_usage": agent_call_usage or [],
        "source": source,
    }
    if economist_debug is not None:
        report[ECONOMIST_AGENT_OUTPUT_KEY] = economist_debug
    return report


def recompute_report_nash(context: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    price_windows = apply_nash_equilibrium_to_windows(
        context,
        {},
        report.get("price_change_windows_3h") or [],
    )
    nash_summary = summarize_nash_equilibrium(price_windows)
    updated = dict(report)
    updated["price_change_windows_3h"] = price_windows
    updated["suggested_price_shift_pct"] = average_price_shift(
        price_windows,
        as_float(report.get("suggested_price_shift_pct"), 0),
    )
    updated["nash_equilibrium_reached"] = nash_summary["nash_equilibrium_reached"]
    updated["nash_equilibrium_windows"] = nash_summary["nash_equilibrium_windows"]
    updated["nash_equilibrium_reached_windows"] = nash_summary["nash_equilibrium_reached_windows"]
    updated["nash_equilibrium_rounds"] = nash_summary["nash_equilibrium_rounds"]
    updated["nash_equilibrium_summary"] = nash_summary["nash_equilibrium_summary"]
    return updated


def merge_grid_fallback(report: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    fallback = heuristic_grid(context)
    merged = {**fallback, **{key: value for key, value in report.items() if value not in (None, "")}}
    merged["grid_stress_level"] = normalize_grid_stress_level(
        merged.get("grid_stress_level"),
        fallback["grid_stress_level"],
    )
    return merged


def merge_behavior_fallback(report: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    fallback = heuristic_behavior(context)
    return {**fallback, **{key: value for key, value in report.items() if value not in (None, "")}}


def merge_economist_fallback(report: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    fallback = heuristic_economist(context, heuristic_grid(context))
    merged = {**fallback, **{key: value for key, value in report.items() if value not in (None, "")}}
    if not isinstance(report.get("price_change_windows_3h"), list):
        merged["price_change_windows_3h"] = []
    if not has_text(report.get("action_label")):
        merged["action_label"] = model_response_failed("missing action_label")
    if not has_text(report.get("price_rationale")):
        merged["price_rationale"] = model_response_failed("missing price_rationale")
    return merged


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def validate_economist_report(report: dict[str, Any], expected_windows: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(report, dict):
        return ["response is not a JSON object"]

    if not is_number_like(report.get("suggested_price_shift_pct")):
        errors.append("missing or invalid suggested_price_shift_pct")
    if not has_text(report.get("action_label")):
        errors.append("missing action_label")
    if not has_text(report.get("price_rationale")):
        errors.append("missing price_rationale")

    expected = expected_windows if isinstance(expected_windows, list) else []
    actual = report.get("price_change_windows_3h")
    if not isinstance(actual, list):
        errors.append("missing price_change_windows_3h")
        return errors
    if len(actual) != len(expected):
        errors.append(f"price_change_windows_3h length {len(actual)} != expected {len(expected)}")

    for idx, expected_window in enumerate(expected):
        if idx >= len(actual) or not isinstance(actual[idx], dict):
            errors.append(f"price_change_windows_3h[{idx}] missing")
            continue
        item = actual[idx]
        for field in ("window_start", "window_end", "action_label", "price_rationale"):
            if not has_text(item.get(field)):
                errors.append(f"price_change_windows_3h[{idx}] missing {field}")
        if not is_number_like(item.get("suggested_price_shift_pct")):
            errors.append(f"price_change_windows_3h[{idx}] missing or invalid suggested_price_shift_pct")
        for field in ("window_start", "window_end"):
            expected_value = expected_window.get(field) if isinstance(expected_window, dict) else None
            if has_text(expected_value) and has_text(item.get(field)) and str(item.get(field)) != str(expected_value):
                errors.append(f"price_change_windows_3h[{idx}] {field} does not match context")
    return errors


def is_number_like(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def as_float(value: Any, fallback: float) -> float:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return round(float(fallback), 2)


def base_price_shift(stress: str, category: str, change: float) -> tuple[int, str]:
    if stress == "Extreme High" and category == "Residential":
        return 15, "Peak deflection fee"
    if stress == "Extreme High":
        return 10, "Congestion fee"
    if stress == "High":
        return 8, "High-load premium"
    if stress == "Medium" and change > 10:
        return 5, "Medium peak premium"
    if stress == "Low" and change < -10:
        return -5, "Utilization incentive"
    return 0, "Hold price"


def build_heuristic_price_windows(
    context: dict[str, Any],
    grid: dict[str, Any],
) -> list[dict[str, Any]]:
    windows = context.get("pricing_windows_3h") or []
    averages = context.get("hourly_averages") or {}
    mean_load = as_float(averages.get("mean_predicted_kwh"), 0)
    mean_service_price = as_float(averages.get("mean_service_price"), 0)
    mean_energy_price = as_float(averages.get("mean_energy_price"), 0)
    stress = normalize_grid_stress_level(grid.get("grid_stress_level"), context.get("grid_stress_level"))
    category = str(context.get("category") or "")
    change = as_float(grid.get("predicted_change_pct"), context.get("predicted_change_pct", 0))
    results = []
    for window in windows:
        window_stress = normalize_grid_stress_level(
            window.get("load_stress_level") or window.get("grid_stress_level"),
            stress,
        )
        window_base_shift, _ = base_price_shift(window_stress, category, change)
        window_load = as_float(window.get("mean_predicted_kwh"), mean_load)
        window_service_price = as_float(window.get("mean_service_price"), mean_service_price)
        window_energy_price = as_float(window.get("mean_energy_price"), mean_energy_price)
        adjustment = 0
        if mean_load > 0:
            load_ratio = window_load / mean_load
            if load_ratio >= 1.15:
                adjustment += 3
            elif load_ratio <= 0.85:
                adjustment -= 3
        if mean_energy_price > 0:
            energy_ratio = window_energy_price / mean_energy_price
            if energy_ratio >= 1.08:
                adjustment += 2
            elif energy_ratio <= 0.92:
                adjustment -= 1
        shift = int(max(-15, min(15, round(window_base_shift + adjustment))))
        if shift > window_base_shift:
            label = "Raise price"
        elif shift < window_base_shift:
            label = "Reduce price"
        else:
            label = "Hold window price"
        results.append(
            {
                "window_start": window.get("window_start"),
                "window_end": window.get("window_end"),
                "sum_predicted_kwh": window.get("sum_predicted_kwh"),
                "mean_predicted_kwh": window.get("mean_predicted_kwh"),
                "sum_actual_kwh": window.get("sum_actual_kwh"),
                "mean_service_price": window.get("mean_service_price"),
                "mean_energy_price": window.get("mean_energy_price"),
                "load_stress_level": window.get("load_stress_level") or window.get("grid_stress_level"),
                "stress_load_3h_kwh": window.get("stress_load_3h_kwh"),
                "actual_load_stress_level": window.get("actual_load_stress_level") or window.get("actual_grid_stress_level"),
                "actual_stress_load_3h_kwh": window.get("actual_stress_load_3h_kwh"),
                "stress_correct": window.get("stress_correct"),
                "stress_missed": window.get("stress_missed"),
                "load_3h_q50_kwh": window.get("load_3h_q50_kwh"),
                "load_3h_q80_kwh": window.get("load_3h_q80_kwh"),
                "load_3h_q95_kwh": window.get("load_3h_q95_kwh"),
                "suggested_price_shift_pct": shift,
                "action_label": label,
                "price_rationale": (
                    f"{window_stress} 3-hour stress; mean load {window_load:.2f} kWh "
                    f"vs horizon mean {mean_load:.2f} kWh; service price {window_service_price:.2f}."
                ),
            }
        )
    return results


def normalize_price_windows(value: Any, fallback_windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        value = []
    normalized = []
    for idx, fallback in enumerate(fallback_windows):
        item_missing = idx >= len(value) or not isinstance(value[idx], dict)
        item = value[idx] if not item_missing else {}
        shift = as_float(item.get("suggested_price_shift_pct"), 0)
        normalized.append(
            {
                "window_start": item.get("window_start") or fallback.get("window_start"),
                "window_end": item.get("window_end") or fallback.get("window_end"),
                "hours": fallback.get("hours"),
                "price_conditioned_baseline_load_kwh": first_present(
                    item.get("price_conditioned_baseline_load_kwh"),
                    fallback.get("price_conditioned_baseline_load_kwh"),
                ),
                "price_conditioned_baseline_source": first_present(
                    item.get("price_conditioned_baseline_source"),
                    fallback.get("price_conditioned_baseline_source"),
                ),
                "price_conditioned_mean_predicted_kwh": first_present(
                    item.get("price_conditioned_mean_predicted_kwh"),
                    fallback.get("price_conditioned_mean_predicted_kwh"),
                ),
                "price_conditioned_peak_predicted_kwh": first_present(
                    item.get("price_conditioned_peak_predicted_kwh"),
                    fallback.get("price_conditioned_peak_predicted_kwh"),
                ),
                "price_conditioned_load_stress_level": first_present(
                    item.get("price_conditioned_load_stress_level"),
                    fallback.get("price_conditioned_load_stress_level"),
                ),
                "price_conditioned_service_price": first_present(
                    item.get("price_conditioned_service_price"),
                    fallback.get("price_conditioned_service_price"),
                ),
                "sum_predicted_kwh": fallback.get("sum_predicted_kwh"),
                "mean_predicted_kwh": fallback.get("mean_predicted_kwh"),
                "peak_predicted_kwh": fallback.get("peak_predicted_kwh"),
                "sum_actual_kwh": fallback.get("sum_actual_kwh"),
                "mean_service_price": fallback.get("mean_service_price"),
                "mean_energy_price": fallback.get("mean_energy_price"),
                "mean_occupancy": fallback.get("mean_occupancy"),
                "mean_temp_c": fallback.get("mean_temp_c"),
                "mean_humidity": fallback.get("mean_humidity"),
                "total_rain": fallback.get("total_rain"),
                "load_stress_level": fallback.get("load_stress_level") or fallback.get("grid_stress_level"),
                "stress_load_3h_kwh": fallback.get("stress_load_3h_kwh"),
                "actual_load_stress_level": fallback.get("actual_load_stress_level") or fallback.get("actual_grid_stress_level"),
                "actual_stress_load_3h_kwh": fallback.get("actual_stress_load_3h_kwh"),
                "stress_correct": fallback.get("stress_correct"),
                "stress_missed": fallback.get("stress_missed"),
                "stress_source_file": fallback.get("stress_source_file"),
                "stress_window_hours": fallback.get("stress_window_hours"),
                "load_3h_q50_kwh": fallback.get("load_3h_q50_kwh"),
                "load_3h_q80_kwh": fallback.get("load_3h_q80_kwh"),
                "load_3h_q95_kwh": fallback.get("load_3h_q95_kwh"),
                "suggested_price_shift_pct": shift,
                "action_label": required_model_text(item, "action_label", item_missing),
                "price_rationale": required_model_text(item, "price_rationale", item_missing),
            }
        )
    return normalized


def apply_nash_equilibrium_to_windows(
    context: dict[str, Any],
    behavior: dict[str, Any],
    windows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [solve_nash_equilibrium_window(context, behavior, window) for window in windows]


def solve_nash_equilibrium_window(
    context: dict[str, Any],
    behavior: dict[str, Any],
    window: dict[str, Any],
) -> dict[str, Any]:
    baseline_load = window_predicted_load(window)
    capacity_limit, capacity_source = capacity_limit_kwh(context, window)
    elasticity = estimate_elasticity_factor(context, behavior, window)
    tolerance_pct = estimate_price_tolerance_pct(context, window)
    previous_shift = 0.0
    raw_shift = as_float(window.get("suggested_price_shift_pct"), 0)
    required_reduction_pct = target_peak_reduction_pct(baseline_load, capacity_limit)
    needed_shift = required_reduction_pct / elasticity if required_reduction_pct > 0 else 0.0
    target_shift = max(raw_shift, needed_shift)
    if required_reduction_pct <= 0 and raw_shift < 0:
        target_shift = max(raw_shift, -tolerance_pct)
    target_shift = max(-NASH_MAX_ABS_PRICE_SHIFT_PCT, min(NASH_MAX_ABS_PRICE_SHIFT_PCT, target_shift))
    target_shift = max(-tolerance_pct, min(tolerance_pct, target_shift))
    trace: list[dict[str, Any]] = []
    final_state: dict[str, Any] = {}

    for iteration in range(1, NASH_MAX_ITERATIONS + 1):
        price_shift = round(target_shift, 4)
        expected_load = expected_load_after_price(baseline_load, price_shift, elasticity)
        discomfort_score = user_discomfort_score(price_shift, tolerance_pct)
        grid_safe = expected_load <= capacity_limit + 1e-9
        user_tolerant = discomfort_score <= NASH_MAX_DISCOMFORT_SCORE + 1e-9
        price_stable = abs(price_shift - previous_shift) < NASH_PRICE_STABILITY_EPSILON_PCT
        final_state = {
            "iteration": iteration,
            "price_shift_pct": round(price_shift, 4),
            "expected_load_kwh": round(expected_load, 4),
            "grid_safe": grid_safe,
            "user_tolerant": user_tolerant,
            "price_stable": price_stable,
            "discomfort_score": round(discomfort_score, 4),
        }
        trace.append(final_state)
        if grid_safe and user_tolerant and price_stable:
            break
        previous_shift = price_shift

    reached = bool(
        final_state.get("grid_safe")
        and final_state.get("user_tolerant")
        and final_state.get("price_stable")
    )
    enriched = dict(window)
    enriched.update(
        {
            "pre_nash_suggested_price_shift_pct": raw_shift,
            "suggested_price_shift_pct": round(float(final_state.get("price_shift_pct", target_shift)), 2),
            "nash_equilibrium_reached": reached,
            "nash_status": "reached" if reached else "not_reached",
            "nash_iterations": int(final_state.get("iteration", 0) or 0),
            "nash_iteration_trace": trace,
            "baseline_load_kwh": round(baseline_load, 4),
            "baseline_load_source": window_predicted_load_source(window),
            "target_peak_reduction_pct": round(required_reduction_pct, 4),
            "elasticity_factor": round(elasticity, 4),
            "capacity_limit_kwh": round(capacity_limit, 4),
            "capacity_limit_source": capacity_source,
            "expected_load_kwh": round(float(final_state.get("expected_load_kwh", baseline_load)), 4),
            "grid_safe": bool(final_state.get("grid_safe", False)),
            "user_tolerant": bool(final_state.get("user_tolerant", False)),
            "price_stable": bool(final_state.get("price_stable", False)),
            "discomfort_score": round(float(final_state.get("discomfort_score", 0)), 4),
            "max_discomfort_score": NASH_MAX_DISCOMFORT_SCORE,
            "price_stability_epsilon_pct": NASH_PRICE_STABILITY_EPSILON_PCT,
        }
    )
    return enriched


def summarize_nash_equilibrium(windows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(windows)
    reached = sum(1 for window in windows if window.get("nash_equilibrium_reached") is True)
    rounds = max((int(window.get("nash_iterations") or 0) for window in windows), default=0)
    all_reached = total > 0 and reached == total
    return {
        "nash_equilibrium_reached": all_reached,
        "nash_equilibrium_windows": total,
        "nash_equilibrium_reached_windows": reached,
        "nash_equilibrium_rounds": rounds,
        "nash_equilibrium_summary": (
            f"Nash equilibrium reached for {reached}/{total} pricing windows"
            if total
            else "No pricing windows available for Nash equilibrium evaluation"
        ),
    }


def average_price_shift(windows: list[dict[str, Any]], fallback: float) -> float:
    if not windows:
        return fallback
    values = [as_float(window.get("suggested_price_shift_pct"), fallback) for window in windows]
    return round(sum(values) / len(values), 2)


def window_predicted_load(window: dict[str, Any]) -> float:
    for key in ("price_conditioned_baseline_load_kwh", "nash_baseline_load_kwh", "sum_predicted_kwh"):
        load = optional_float(window.get(key))
        if load is not None:
            return max(0.0, load)
    mean_load = optional_float(window.get("mean_predicted_kwh")) or 0.0
    hours = optional_float(window.get("hours")) or 1.0
    return max(0.0, mean_load * hours)


def window_predicted_load_source(window: dict[str, Any]) -> str:
    for key, source in (
        ("price_conditioned_baseline_load_kwh", "price_conditioned_forecast_sum_predicted_kwh"),
        ("nash_baseline_load_kwh", "nash_baseline_load_kwh"),
        ("sum_predicted_kwh", "forecast_sum_predicted_kwh"),
    ):
        if optional_float(window.get(key)) is not None:
            return source
    return "forecast_mean_predicted_kwh_times_hours"


def capacity_limit_kwh(context: dict[str, Any], window: dict[str, Any]) -> tuple[float, str]:
    q95 = first_positive_float(
        window.get("load_3h_q95_kwh"),
        context.get("grid_stress_q95_kwh"),
    )
    if q95 is not None and q95 > 0:
        return q95, "load_3h_q95_kwh"
    return max(window_predicted_load(window), 0.0), "baseline_predicted_load"


def target_peak_reduction_pct(load: float, capacity_limit: float) -> float:
    if load <= 0:
        return 0.0
    return max(0.0, ((load - capacity_limit) / load) * 100)


def estimate_elasticity_factor(
    context: dict[str, Any],
    behavior: dict[str, Any],
    window: dict[str, Any],
) -> float:
    window_value = optional_float(window.get("elasticity_factor"))
    if window_value is not None:
        return clamp(abs(window_value), NASH_MIN_ELASTICITY, NASH_MAX_ELASTICITY)

    reported = optional_float(behavior.get("elasticity_factor"))
    if reported is not None:
        return clamp(abs(reported), NASH_MIN_ELASTICITY, NASH_MAX_ELASTICITY)

    category = str(context.get("category") or "").lower()
    if "residential" in category:
        base = 0.45
    elif "commercial" in category or "mall" in category:
        base = 0.32
    elif "cbd" in category or "office" in category:
        base = 0.24
    elif "transport" in category or "hub" in category:
        base = 0.2
    elif "industrial" in category:
        base = 0.18
    else:
        base = 0.28

    occupancy = normalized_rate(window.get("mean_occupancy"))
    rain = optional_float(window.get("total_rain"))
    if rain is None:
        weather = context.get("weather") if isinstance(context.get("weather"), dict) else {}
        rain = optional_float(weather.get("rain_hours")) or 0.0
    occupancy_factor = 1.0 - 0.25 * occupancy
    rain_factor = 0.85 if rain > 0 else 1.0
    return clamp(base * occupancy_factor * rain_factor, NASH_MIN_ELASTICITY, NASH_MAX_ELASTICITY)


def estimate_price_tolerance_pct(context: dict[str, Any], window: dict[str, Any]) -> float:
    category = str(context.get("category") or "").lower()
    if "residential" in category:
        base = 15.0
    elif "commercial" in category or "mall" in category:
        base = 12.0
    elif "industrial" in category:
        base = 9.0
    else:
        base = 10.0
    occupancy = normalized_rate(window.get("mean_occupancy"))
    rain = optional_float(window.get("total_rain")) or 0.0
    tolerance = base * (1.0 - 0.25 * occupancy)
    if rain > 0:
        tolerance *= 0.85
    return clamp(tolerance, 5.0, NASH_MAX_ABS_PRICE_SHIFT_PCT)


def expected_load_after_price(baseline_load: float, price_shift_pct: float, elasticity: float) -> float:
    response = elasticity * (price_shift_pct / 100.0)
    return max(0.0, baseline_load * (1.0 - response))


def user_discomfort_score(price_shift_pct: float, tolerance_pct: float) -> float:
    if tolerance_pct <= 0:
        return float("inf")
    return abs(price_shift_pct) / tolerance_pct


def normalized_rate(value: Any) -> float:
    number = optional_float(value)
    if number is None:
        return 0.0
    if number > 1.0:
        number /= 100.0
    return clamp(number, 0.0, 1.0)


def first_positive_float(*values: Any) -> float | None:
    for value in values:
        number = optional_float(value)
        if number is not None and number > 0:
            return number
    return None


def optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def required_model_text(item: dict[str, Any], field: str, item_missing: bool) -> str:
    if item_missing:
        return model_response_failed("missing price_change_windows_3h item")
    value = item.get(field)
    return str(value).strip() if has_text(value) else model_response_failed(f"missing {field}")


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def has_text(value: Any) -> bool:
    return bool(str(value).strip()) if value is not None else False


def model_response_failed(reason: str) -> str:
    return f"{MODEL_RESPONSE_FAILED}: {reason}"


def normalize_grid_stress_level(value: Any, fallback: Any = "Low") -> str:
    key = str(value or "").strip().lower()
    if key in GRID_STRESS_LEVEL_BY_KEY:
        return GRID_STRESS_LEVEL_BY_KEY[key]
    fallback_key = str(fallback or "").strip().lower()
    return GRID_STRESS_LEVEL_BY_KEY.get(fallback_key, "Low")
