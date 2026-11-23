from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


_TIMESFM_MODEL_CACHE: dict[tuple[str, int, int], Any] = {}
_CHRONOS_MODEL_CACHE: dict[tuple[str, str], Any] = {}
DEFAULT_TIMESFM_EXOG_COLS = ["T", "U", "nRAIN", "e_price", "s_price", "is_weekend", "temp_price_idx"]
KNOWN_FUTURE_COVARIATES = frozenset([*DEFAULT_TIMESFM_EXOG_COLS, "hour", "occupancy"])


def load_timesfm_model(repo: str, context_hours: int, max_horizon: int):
    key = (repo, int(context_hours), int(max_horizon))
    if key in _TIMESFM_MODEL_CACHE:
        return _TIMESFM_MODEL_CACHE[key]

    try:
        from timesfm import ForecastConfig, TimesFM_2p5_200M_torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "timesfm is required for backend: timesfm. "
            "Install the official google-research/timesfm package and make sure "
            "this Python environment can import it."
        ) from exc

    patch_timesfm_hub_kwargs(TimesFM_2p5_200M_torch)
    model = TimesFM_2p5_200M_torch.from_pretrained(repo)
    model.compile(
        ForecastConfig(
            max_context=int(context_hours),
            max_horizon=int(max_horizon),
            normalize_inputs=True,
            fix_quantile_crossing=True,
            return_backcast=True,
        )
    )
    _TIMESFM_MODEL_CACHE[key] = model
    return model


def patch_timesfm_hub_kwargs(model_cls) -> None:
    if getattr(model_cls, "_mapf_hub_kwargs_compat", False):
        return

    original = model_cls._from_pretrained

    @classmethod
    def _from_pretrained_compat(cls, *args, **kwargs):
        kwargs.pop("proxies", None)
        kwargs.pop("resume_download", None)
        return original(*args, **kwargs)

    model_cls._from_pretrained = _from_pretrained_compat
    model_cls._mapf_hub_kwargs_compat = True


def load_chronos_model(repo: str, device: str):
    resolved_device = resolve_torch_device(device)
    key = (repo, resolved_device)
    if key in _CHRONOS_MODEL_CACHE:
        return _CHRONOS_MODEL_CACHE[key]

    try:
        import torch
        from chronos import Chronos2Pipeline
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "chronos-forecasting with Chronos2Pipeline is required for backend: chronos. "
            "Install it in this Python environment."
        ) from exc

    repo_lower = repo.lower()
    if "chronos-2" not in repo_lower and "chronos2" not in repo_lower:
        raise ValueError('This project only supports Chronos 2. Set chronos_repo: "amazon/chronos-2".')

    dtype = torch.bfloat16 if resolved_device.startswith("cuda") else torch.float32
    pipeline = Chronos2Pipeline.from_pretrained(
        repo,
        device_map=resolved_device,
        dtype=dtype,
    )
    _CHRONOS_MODEL_CACHE[key] = pipeline
    return pipeline


def resolve_torch_device(device: str) -> str:
    normalized = (device or "auto").strip().lower()
    if normalized == "auto":
        try:
            import torch
        except ModuleNotFoundError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized in {"gpu", "cuda"}:
        return "cuda"
    return normalized


def run_timesfm_prediction(
    model,
    context_load: np.ndarray,
    exog_context: np.ndarray | None,
    exog_horizon: np.ndarray | None,
    horizon: int,
    context_hours: int,
    exog_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ctx = context_load[-context_hours:].astype(np.float64)
    if exog_context is not None and exog_horizon is not None and hasattr(model, "forecast_with_covariates"):
        exog_ctx = exog_context[-len(ctx) :].astype(np.float64)
        dyn_cov = {}
        for idx, col in enumerate(exog_cols):
            if idx < exog_ctx.shape[1] and idx < exog_horizon.shape[1]:
                combined = np.concatenate([exog_ctx[:, idx], exog_horizon[:, idx]])
                dyn_cov[col] = [combined.tolist()]
        try:
            result = model.forecast_with_covariates(
                inputs=[ctx.tolist()],
                dynamic_numerical_covariates=dyn_cov if dyn_cov else None,
                xreg_mode="xreg + timesfm",
                normalize_xreg_target_per_input=True,
            )
        except ImportError as exc:
            raise RuntimeError(
                "TimesFM covariate forecasting requires the xreg dependencies. "
                "Install torch, jax, jaxlib, scikit-learn, and the official "
                "google-research/timesfm package."
            ) from exc
    else:
        result = model.forecast(horizon=int(horizon), inputs=[ctx])
    return parse_timesfm_result(result, horizon)


def run_chronos_prediction(
    pipeline,
    context_load: np.ndarray,
    horizon: int,
    context_hours: int,
    *,
    exog_context: np.ndarray | None = None,
    exog_horizon: np.ndarray | None = None,
    exog_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for backend: chronos.") from exc

    ctx = context_load[-context_hours:].astype(np.float32)
    inputs: list[Any]
    covariate_input = build_chronos_covariate_input(ctx, exog_context, exog_horizon, horizon, exog_cols or [])
    if covariate_input is not None:
        inputs = [covariate_input]
    else:
        inputs = [torch.tensor(ctx, dtype=torch.float32)]
    predict_kwargs: dict[str, Any] = {"limit_prediction_length": False}
    quantiles, _ = pipeline.predict_quantiles(
        inputs,
        prediction_length=int(horizon),
        quantile_levels=[0.1, 0.5, 0.9],
        **predict_kwargs,
    )
    if isinstance(quantiles, list):
        if not quantiles:
            raise RuntimeError("Chronos returned no quantile forecasts")
        q = np.asarray(quantiles[0].detach().cpu(), dtype=np.float64)
        if q.ndim == 3:
            q = q[0]
        q = q[np.newaxis, :, :]
    else:
        q = np.asarray(quantiles.detach().cpu(), dtype=np.float64)
    if q.ndim != 3 or q.shape[0] == 0 or q.shape[-1] < 3:
        raise RuntimeError(f"Unexpected Chronos quantile shape: {q.shape}")
    q10 = q[0, :horizon, 0]
    q50 = q[0, :horizon, 1]
    q90 = q[0, :horizon, 2]
    return np.clip(q50, 0, None), np.clip(q10, 0, None), np.clip(q90, 0, None)


def build_chronos_covariate_input(
    context_load: np.ndarray,
    exog_context: np.ndarray | None,
    exog_horizon: np.ndarray | None,
    horizon: int,
    exog_cols: list[str],
) -> dict[str, Any] | None:
    if not exog_cols or exog_context is None or exog_horizon is None:
        return None
    ctx = np.asarray(context_load, dtype=np.float32)
    if len(ctx) == 0:
        return None
    context_arr = np.asarray(exog_context, dtype=np.float64)
    if len(context_arr) > len(ctx):
        context_matrix = context_arr[-len(ctx) :]
    else:
        context_matrix = align_exog(context_arr, len(ctx))
    horizon_matrix = align_exog(np.asarray(exog_horizon, dtype=np.float64), int(horizon))
    if context_matrix.shape[1] == 0 or horizon_matrix.shape[1] == 0:
        return None

    past_covariates: dict[str, np.ndarray] = {}
    future_covariates: dict[str, np.ndarray] = {}
    for idx, col in enumerate(exog_cols):
        if idx >= context_matrix.shape[1] or idx >= horizon_matrix.shape[1]:
            continue
        past_covariates[col] = context_matrix[:, idx].astype(np.float32)
        future_covariates[col] = horizon_matrix[:, idx].astype(np.float32)
    if not past_covariates or not future_covariates:
        return None
    return {
        "target": ctx,
        "past_covariates": past_covariates,
        "future_covariates": future_covariates,
    }


def parse_timesfm_result(result: Any, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(result, tuple):
        point_out, quantile_out = result
    else:
        arr = np.asarray(result)
        if arr.ndim == 3 and arr.shape[-1] > 5:
            point_out = arr[:, :, 5]
            quantile_out = arr
        else:
            point_out = arr
            quantile_out = None

    point = first_series(point_out, horizon)
    if quantile_out is None:
        q10 = np.full(horizon, np.nan, dtype=np.float64)
        q90 = np.full(horizon, np.nan, dtype=np.float64)
    else:
        quantiles = np.asarray(quantile_out, dtype=np.float64)
        if quantiles.ndim == 3:
            q10 = quantiles[0, :horizon, 0]
            q90_idx = min(8, quantiles.shape[-1] - 1)
            q90 = quantiles[0, :horizon, q90_idx]
        else:
            q10 = np.full(horizon, np.nan, dtype=np.float64)
            q90 = np.full(horizon, np.nan, dtype=np.float64)
    return np.clip(point, 0, None), np.clip(q10, 0, None), np.clip(q90, 0, None)


def first_series(values: Any, horizon: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:horizon]
    return arr[0, :horizon]


def build_exog_matrix(frame: pd.DataFrame, exog_cols: list[str]) -> np.ndarray:
    if not exog_cols:
        return np.empty((len(frame), 0), dtype=np.float64)
    exog = frame.reindex(columns=exog_cols).copy()
    exog = exog.apply(pd.to_numeric, errors="coerce")
    exog = exog.ffill().bfill()
    means = exog.mean(numeric_only=True)
    exog = exog.fillna(means).fillna(0.0)
    return exog.to_numpy(dtype=np.float64)


def align_exog(matrix: np.ndarray, target: int) -> np.ndarray:
    if len(matrix) == target:
        return matrix
    if len(matrix) > target:
        return matrix[:target]
    if len(matrix) == 0:
        return np.zeros((target, 0), dtype=np.float64)
    pad = np.tile(matrix[-1], (target - len(matrix), 1))
    return np.vstack([matrix, pad])
