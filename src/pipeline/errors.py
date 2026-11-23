from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from src.agents.client import AgentStageError
from src.paths import forecast_start_slug, safe_filename


ERROR_CODE_BY_STAGE = {
    "zone_selection": "MAPF-E010",
    "load_data": "MAPF-E020",
    "forecast": "MAPF-E100",
    "build_contexts": "MAPF-E110",
    "load_forecaster_output": "MAPF-E115",
    "precomputed_window_data": "MAPF-E120",
    "agent_chain": "MAPF-E200",
    "agent.grid": "MAPF-E210",
    "agent.behavior": "MAPF-E220",
    "agent.economist": "MAPF-E230",
    "agent.economist_repair": "MAPF-E231",
    "price_conditioned_forecast": "MAPF-E240",
    "write_outputs": "MAPF-E300",
    "unexpected": "MAPF-E900",
}


class PipelineStageError(RuntimeError):
    def __init__(
        self,
        *,
        stage: str,
        original: Exception,
        zone_id: Any = None,
        agent: str | None = None,
    ) -> None:
        self.stage = stage
        self.zone_id = zone_id
        self.agent = agent
        self.original = original
        location = f" zone={zone_id}" if zone_id is not None else ""
        agent_text = f" agent={agent}" if agent else ""
        super().__init__(
            f"{stage}{location}{agent_text}: {type(original).__name__}: {original}"
        )


def experiment_error_record(
    exc: Exception,
    *,
    experiment_dir: Path,
    record: dict[str, Any],
    run_index: int,
) -> dict[str, Any]:
    details = classify_experiment_error(exc)
    error = (
        f"{details['error_code']} {details['error_stage']}: "
        f"{details['error_type']}: {details['error_message']}"
    )
    failure_record = {
        "error": error,
        "error_code": details["error_code"],
        "error_stage": details["error_stage"],
        "error_zone_id": details["error_zone_id"],
        "error_agent": details["error_agent"],
        "error_type": details["error_type"],
        "error_message": details["error_message"],
    }
    traceback_path = write_error_traceback(
        exc,
        experiment_dir=experiment_dir,
        record={**record, **failure_record},
        run_index=run_index,
        error_code=details["error_code"],
    )
    return {
        **failure_record,
        "traceback_file": str(traceback_path),
    }


def classify_experiment_error(exc: Exception) -> dict[str, str]:
    source = exc
    stage = "unexpected"
    zone_id = ""
    agent = ""

    if isinstance(exc, AgentStageError):
        source = exc.original
        stage = exc.stage
        zone_id = "" if exc.zone_id is None else str(exc.zone_id)
        agent = exc.agent
    elif isinstance(exc, PipelineStageError):
        source = exc.original
        stage = exc.stage
        zone_id = "" if exc.zone_id is None else str(exc.zone_id)
        agent = exc.agent or ""
        if isinstance(source, AgentStageError):
            stage = source.stage
            zone_id = "" if source.zone_id is None else str(source.zone_id)
            agent = source.agent
            source = source.original

    return {
        "error_code": error_code_for_stage(stage),
        "error_stage": stage,
        "error_zone_id": zone_id,
        "error_agent": agent,
        "error_type": type(source).__name__,
        "error_message": str(source),
    }


def error_code_for_stage(stage: str) -> str:
    if stage in ERROR_CODE_BY_STAGE:
        return ERROR_CODE_BY_STAGE[stage]
    if stage.startswith("agent.discussion."):
        if stage.endswith(".grid"):
            return ERROR_CODE_BY_STAGE["agent.grid"]
        if stage.endswith(".behavior"):
            return ERROR_CODE_BY_STAGE["agent.behavior"]
        if stage.endswith(".economist.schema_repair"):
            return ERROR_CODE_BY_STAGE["agent.economist_repair"]
        if stage.endswith(".economist"):
            return ERROR_CODE_BY_STAGE["agent.economist"]
    return ERROR_CODE_BY_STAGE["unexpected"]


def format_failure_message(exc: Exception) -> str:
    details = classify_experiment_error(exc)
    lines = [
        "Execution stopped because an error occurred.",
        f"error_code: {details['error_code']}",
        f"stage: {details['error_stage']}",
    ]
    if details["error_zone_id"]:
        lines.append(f"zone_id: {details['error_zone_id']}")
    if details["error_agent"]:
        lines.append(f"agent: {details['error_agent']}")
    lines.extend(
        [
            f"error_type: {details['error_type']}",
            f"reason: {details['error_message']}",
        ]
    )
    return "\n".join(lines)


def write_error_traceback(
    exc: Exception,
    *,
    experiment_dir: Path,
    record: dict[str, Any],
    run_index: int,
    error_code: str,
) -> Path:
    errors_dir = experiment_dir / "errors"
    errors_dir.mkdir(parents=True, exist_ok=True)
    blend_value = record.get("diurnal_blend_alpha")
    blend_text = "blend" if blend_value in (None, "") else str(blend_value)
    name_parts = [
        f"{run_index:04d}",
        error_code,
        forecast_start_slug(str(record.get("forecast_origin") or "unknown")),
        safe_filename(record.get("backend") or "model"),
        safe_filename(record.get("agent_mode") or "mode"),
        safe_filename(blend_text),
    ]
    path = errors_dir / ("_".join(name_parts) + ".txt")
    traceback_record = {**record, "traceback_file": str(path)}
    metadata = "\n".join(f"{key}: {value}" for key, value in traceback_record.items())
    trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    path.write_text(f"{metadata}\n\nTRACEBACK:\n{trace}", encoding="utf-8")
    return path
