"""Stable output contracts for native structured Agent responses.

These schemas describe newly generated responses, not a migration or validator
for historical results. Window counts, timestamps, and pricing rules stay in
the existing runtime validation so the provider can reuse each role's schema.
"""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Any


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _array(items: dict[str, Any], description: str) -> dict[str, Any]:
    return {"type": "array", "items": items, "description": description}


_TEXT = {"type": "string"}
_NUMBER = {"type": "number"}
_CONFIDENCE = {
    "anyOf": [{"type": "string"}, {"type": "number"}],
    "description": "Confidence as a short qualitative label or numeric score.",
}
_WINDOW_ASSESSMENT = _object({
    "window_start": {"type": "string", "description": "Copy the context window_start exactly."},
    "window_end": {"type": "string", "description": "Copy the context window_end exactly."},
    "predicted_load_kwh": _NUMBER,
    "load_range_position_pct": {
        "anyOf": [{"type": "number"}, {"type": "null"}],
        "description": "Position in the historical load range, or null if unavailable.",
    },
    "grid_stress_level": {
        "type": "string", "enum": ["Low", "Medium", "High", "Extremely High"],
    },
    "adjustment_needed": {"type": "boolean"},
    "reasoning_summary": _TEXT,
})
_PRICE_WINDOW = _object({
    "window_start": {"type": "string", "description": "Copy the context window_start exactly."},
    "window_end": {"type": "string", "description": "Copy the context window_end exactly."},
    "suggested_price_shift_pct": _NUMBER,
    "proposed_energy_price": {
        "type": "number", "description": "Non-negative proposed energy price; excludes service price.",
    },
    "action_label": _TEXT,
    "price_rationale": _TEXT,
})
_WINDOW_ELASTICITY = _object({
    "window_start": {"type": "string", "description": "Copy the context window_start exactly."},
    "window_end": {"type": "string", "description": "Copy the context window_end exactly."},
    "elasticity_factor": {"type": "number", "description": "Positive price-response elasticity estimate."},
})
_GRID_FIELDS = {
    "reasoning_summary": _TEXT,
    "adjustment_needed": {"type": "boolean"},
    "confidence": _CONFIDENCE,
    "window_assessments": _array(_WINDOW_ASSESSMENT, "Assess the context pricing windows in their original order."),
}
_BEHAVIOR_FIELDS = {
    "reasoning_summary": _TEXT,
    "demand_drivers": _array(_TEXT, "Concise demand drivers grounded in the supplied context."),
    "elasticity_factor": {"type": "number", "description": "Positive overall price-response elasticity estimate."},
    "window_elasticities": _array(_WINDOW_ELASTICITY, "Per-window elasticity estimates in context window order."),
    "confidence": _CONFIDENCE,
}
_PRICING_FIELDS = {
    "reasoning_summary": _TEXT,
    "suggested_price_shift_pct": _NUMBER,
    "action_label": _TEXT,
    "price_rationale": _TEXT,
    "price_change_windows_3h": _array(_PRICE_WINDOW, "One item per context pricing window in its original order."),
}
_SINGLE_FIELDS = {
    "reasoning_summary": _TEXT,
    "window_assessments": _GRID_FIELDS["window_assessments"],
    "demand_drivers": _BEHAVIOR_FIELDS["demand_drivers"],
    "elasticity_factor": _BEHAVIOR_FIELDS["elasticity_factor"],
    "window_elasticities": _BEHAVIOR_FIELDS["window_elasticities"],
    "confidence": _CONFIDENCE,
    **{key: value for key, value in _PRICING_FIELDS.items() if key != "reasoning_summary"},
}
_DISCUSSION_FIELDS = {
    "agreements": _array(_TEXT, "Concise points of agreement with the other agents; empty if none."),
    "disagreements": _array(_TEXT, "Concise unresolved disagreements; empty if none."),
    "revisions_from_prior_round": _array(_TEXT, "Brief changes since the prior discussion round; empty if none."),
    "message_to_other_agents": _TEXT,
}
_ROLE_FIELDS = {
    "grid": _GRID_FIELDS,
    "behavior": _BEHAVIOR_FIELDS,
    "economist": _PRICING_FIELDS,
    "single_agent": _SINGLE_FIELDS,
}
_STAGE_ROLES = {
    "agent.grid": "grid",
    "agent.behavior": "behavior",
    "agent.economist": "economist",
    "agent.economist_repair": "economist",
    "agent.economist_retry": "economist",
    "agent.single_agent": "single_agent",
    "agent.single_agent_price_retry": "economist",
}
_DISCUSSION_STAGE = re.compile(r"agent\.discussion\.round_[1-9][0-9]*\.(grid|behavior|economist)")


def agent_output_schema(stage: str) -> dict[str, Any]:
    """Return an independent closed schema for a known Agent execution stage.

    Repairs and price retries ask only for pricing fields, even when repairing
    a single-agent or discussion response. Unknown stages fail explicitly so
    a caller cannot accidentally constrain a new task to an unrelated shape.
    """
    if not isinstance(stage, str):
        raise ValueError(f"Unknown Agent output stage: {stage!r}")
    repair = stage.endswith(".schema_repair")
    base_stage = stage.removesuffix(".schema_repair") if repair else stage
    discussion = _DISCUSSION_STAGE.fullmatch(base_stage)
    role = discussion.group(1) if discussion else _STAGE_ROLES.get(base_stage)
    if role is None:
        raise ValueError(f"Unknown Agent output stage: {stage!r}")
    if repair:
        fields = _PRICING_FIELDS
    elif discussion:
        fields = {**_ROLE_FIELDS[role], **_DISCUSSION_FIELDS}
    else:
        fields = _ROLE_FIELDS[role]
    return deepcopy(_object(fields))
