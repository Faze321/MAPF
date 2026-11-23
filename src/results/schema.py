"""Format identifiers shared by the forecast/control result writers and readers."""

from __future__ import annotations

from src.control.policy import LOAD_POLICY_ID


FORECASTER_MANIFEST_SCHEMA_VERSION = 10
CONTROL_RESULT_SCHEMA_VERSION = 5


def require_current_result(payload: dict, filename: str, version: int) -> None:
    if not isinstance(payload, dict) or payload.get("schema_version") != version:
        raise ValueError(f"{filename}: expected schema_version {version}. Regenerate the result.")
    if payload.get("load_policy_id") != LOAD_POLICY_ID:
        raise ValueError(f"{filename}: expected load_policy_id {LOAD_POLICY_ID}. Regenerate the result.")


MODEL_PROVENANCE_FIELDS = (
    "model_id", "model_training_start", "model_training_end_exclusive",
    "training_overlap_hours", "evaluation_is_out_of_sample",
)
