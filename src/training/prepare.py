from __future__ import annotations

from src.config.training import TrainingConfig
from src.data.selection import normalize_zone_ids


def prepare_offline_models(dataset, backends, training_config=None, cli_overrides=None,
                           *, zone_ids, history_days=35, horizon_days=1, force_train=False, data_dir=None,
                           backend_parameters=None):
    """Prepare only the selected sites once, before experiment expansion."""
    training_config = training_config or TrainingConfig()
    selected = training_config.enabled_backends(backends, cli_overrides=cli_overrides)
    if not selected:
        return {}
    from src.training.runner import ensure_dataset_models
    return ensure_dataset_models(
        dataset, backends=selected, training_config=training_config,
        zone_ids=normalize_zone_ids(zone_ids), history_days=history_days, horizon_days=horizon_days,
        cli_overrides=cli_overrides, force_train=force_train, data_dir=data_dir,
        backend_parameters=backend_parameters,
    )


def eligible_training_zone_ids(dataset, training_config=None, *, zone_ids=None,
                               history_days=35, horizon_days=1) -> list[str]:
    """Check sample availability without fitting models or consulting cached weights."""
    from src.training.windows import assess_dataset_training
    requested = normalize_zone_ids(zone_ids)
    assessment = assess_dataset_training(
        dataset, training_config or TrainingConfig(), zone_ids=requested or None,
        history_days=history_days, horizon_days=horizon_days,
    )
    eligible = [str(zone) for zone in assessment["eligible_zone_ids"]]
    if requested:
        unavailable = [zone for zone in requested if zone not in set(eligible)]
        if unavailable:
            reasons = {zone: assessment.get("zones", {}).get(zone, {"reason": "unknown_zone"})
                       for zone in unavailable}
            raise ValueError(f"Selected zones are not eligible for offline training: {reasons}")
        return requested
    if not eligible:
        raise ValueError("No zones have enough complete sliding windows for offline training")
    return eligible
