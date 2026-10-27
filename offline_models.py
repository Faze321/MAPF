"""Per-zone supervised models trained offline and loaded without fitting.

Each model maps a complete history window to an entire future window directly.
Validation selects the best weights from one training run. Preprocessing is fitted
only on the training prefix; blending and calibration belong to the forecasting layer.
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


DEFAULT_EXOG_COLS = ["T", "U", "nRAIN", "e_price", "s_price", "is_weekend", "temp_price_idx"]
KNOWN_EXOG_COLS = frozenset([*DEFAULT_EXOG_COLS, "hour", "occupancy"])
CALENDAR_FEATURES = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]
PREPROCESS_SCHEMA_VERSION = 2


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def complete_target_indices(frame: pd.DataFrame, context_hours: int,
                            horizon_hours: int = 1, stride_hours: int = 1) -> np.ndarray:
    """First target positions with continuous [t-H,t+P), strided per valid segment."""
    context_hours = _positive_integer(context_hours, "context_hours")
    horizon_hours = _positive_integer(horizon_hours, "horizon_hours")
    stride_hours = _positive_integer(stride_hours, "stride_hours")
    if len(frame) < context_hours + horizon_hours:
        return np.empty(0, dtype=np.int64)
    times = pd.to_datetime(frame["time"], errors="raise")
    new_segment = (~times.diff().eq(pd.Timedelta(hours=1))).to_numpy()
    positions = np.arange(len(frame), dtype=np.int64)
    segment_starts = np.maximum.accumulate(np.where(new_segment, positions, 0))
    targets = np.arange(context_hours, len(frame) - horizon_hours + 1, dtype=np.int64)
    # Using the last target's segment start also excludes gaps inside the labels.
    starts = segment_starts[targets + horizon_hours - 1]
    complete = targets - starts >= context_hours
    aligned = (targets - starts - context_hours) % stride_hours == 0
    return targets[complete & aligned]


def _checked_frame(frame: pd.DataFrame, *, actuals: bool) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    result["time"] = pd.to_datetime(result["time"], errors="raise").dt.as_unit("ns")
    if result["time"].isna().any() or not result["time"].is_monotonic_increasing or result["time"].duplicated().any():
        raise ValueError("Model frames must contain unique sorted valid timestamps")
    if actuals:
        result["actual_kwh"] = pd.to_numeric(result["actual_kwh"], errors="coerce")
        if not np.isfinite(result["actual_kwh"].to_numpy(dtype=float)).all():
            raise ValueError("Training/context load must be finite; remove invalid observations before model use")
    return result


def _parameters(backend: str, supplied: dict[str, Any]) -> dict[str, Any]:
    retired = {"lstm_context_hours", "xgboost_lags"}.intersection(supplied)
    if retired:
        raise ValueError("Retired one-step parameter(s): " + ", ".join(sorted(retired)) + "; use input_hours/output_hours")
    if backend == "lstm":
        defaults = dict(hidden_size=32, num_layers=1, epochs=100,
                        learning_rate=0.001, batch_size=64, device="cuda", seed=42,
                        early_stopping_patience=10, dropout=0.0, head_dropout=0.1,
                        weight_decay=1e-4, huber_delta=1.0, grad_clip=1.0, min_delta=1e-4)
    elif backend == "xgboost":
        defaults = dict(n_estimators=800, max_depth=3,
                        learning_rate=0.03, min_child_weight=10, subsample=0.8,
                        colsample_bytree=0.9, reg_lambda=10.0, reg_alpha=0.1,
                        early_stopping_rounds=50, tree_method="hist", device="cuda",
                        n_jobs=2, max_bin=256, seed=42)
    else:
        raise ValueError(f"Offline model backend must be lstm or xgboost, got {backend!r}")
    result = {f"{backend}_{key}": copy.deepcopy(value) for key, value in defaults.items()}
    for name in ("input_hours", "output_hours"):
        result[name] = _positive_integer(supplied.get(name), name)
    result["sample_stride_hours"] = _positive_integer(supplied.get("sample_stride_hours", 1), "sample_stride_hours")
    result[f"{backend}_exog_cols"] = list(DEFAULT_EXOG_COLS)
    result[f"{backend}_seed"] = int(supplied.get("training_seed", 42))
    result.update({key: value for key, value in supplied.items()
                   if key in result and key.startswith(backend + "_") and value is not None})
    exog = list(result[f"{backend}_exog_cols"])
    if len(set(exog)) != len(exog) or set(exog) - KNOWN_EXOG_COLS:
        raise ValueError("Exogenous columns must be unique known future covariates, never load or zone/category IDs")
    result[f"{backend}_exog_cols"] = exog
    positive = (["hidden_size", "num_layers", "epochs", "batch_size", "early_stopping_patience"]
                if backend == "lstm" else ["n_estimators", "max_depth", "early_stopping_rounds", "max_bin"])
    for name in positive:
        result[f"{backend}_{name}"] = _positive_integer(result[f"{backend}_{name}"], f"{backend}_{name}")
    if backend == "lstm":
        for name in ["dropout", "head_dropout"]:
            if not 0 <= float(result[f"lstm_{name}"]) < 1:
                raise ValueError(f"lstm_{name} must be in [0, 1)")
        for name in ["learning_rate", "huber_delta", "grad_clip"]:
            if not np.isfinite(result[f"lstm_{name}"]) or result[f"lstm_{name}"] <= 0:
                raise ValueError(f"lstm_{name} must be finite and positive")
        for name in ["weight_decay", "min_delta"]:
            if not np.isfinite(result[f"lstm_{name}"]) or result[f"lstm_{name}"] < 0:
                raise ValueError(f"lstm_{name} must be finite and nonnegative")
    else:
        device = result["xgboost_device"]
        if (result["xgboost_tree_method"] != "hist" or not isinstance(device, str)
                or (device not in {"cpu", "cuda"} and not (device.startswith("cuda:") and device[5:].isdigit()))):
            raise ValueError("Offline XGBoost requires tree_method='hist' and device='cpu', 'cuda' or 'cuda:<index>'")
    return result


def _raw_covariates(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    arrays = [pd.to_numeric(frame.get(name, pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(dtype=float)
              for name in columns]
    result = np.column_stack(arrays) if arrays else np.empty((len(frame), 0), dtype=float)
    result[~np.isfinite(result)] = np.nan
    return result


def _calendar(frame: pd.DataFrame) -> np.ndarray:
    times = pd.to_datetime(frame["time"])
    hour = times.dt.hour.to_numpy(dtype=float)
    weekday = times.dt.dayofweek.to_numpy(dtype=float)
    return np.column_stack([np.sin(hour * 2 * np.pi / 24), np.cos(hour * 2 * np.pi / 24),
                            np.sin(weekday * 2 * np.pi / 7), np.cos(weekday * 2 * np.pi / 7)])


def _fit_preprocess(backend: str, frame: pd.DataFrame, p: dict[str, Any],
                    selected_exog: list[str] | None = None) -> dict[str, Any]:
    requested = p[f"{backend}_exog_cols"]
    if selected_exog is None:
        raw = _raw_covariates(frame, requested)
        selected_exog = [name for i, name in enumerate(requested) if np.isfinite(raw[:, i]).any()]
    raw = _raw_covariates(frame, selected_exog)
    load = frame["actual_kwh"].to_numpy(dtype=float)
    load_std = float(np.std(load))
    state: dict[str, Any] = dict(schema_version=PREPROCESS_SCHEMA_VERSION, backend=backend,
                                parameters=p, input_hours=p["input_hours"], output_hours=p["output_hours"],
                                sample_stride_hours=p["sample_stride_hours"], prediction_mode="direct_window",
                                exog_cols=selected_exog, dropped_exog_cols=[c for c in requested if c not in selected_exog],
                                load_mean=float(np.mean(load)), load_std=load_std if load_std > 1e-8 else 1.0,
                                fitted_rows=len(frame), fitted_start=frame.iloc[0]["time"].isoformat(),
                                fitted_end=frame.iloc[-1]["time"].isoformat())
    cov_names = [*selected_exog, *CALENDAR_FEATURES, *[f"{c}_missing" for c in selected_exog]]
    state["future_feature_names"] = cov_names
    if backend == "lstm":
        medians = np.nanmedian(raw, axis=0) if raw.shape[1] else np.empty(0)
        filled = np.where(np.isnan(raw), medians, raw)
        continuous = np.column_stack([filled, _calendar(frame)])
        std = np.std(continuous, axis=0)
        state.update(exog_medians=medians.tolist(), feature_mean=np.mean(continuous, axis=0).tolist(),
                     feature_std=np.where(std > 1e-8, std, 1.0).tolist(),
                     feature_names=["normalized_load", *cov_names],
                     history_feature_names=["normalized_load", *cov_names])
    else:
        state["summary_windows"] = list(dict.fromkeys([min(24, p["input_hours"]), min(168, p["input_hours"])]))
        state["history_feature_names"] = [f"normalized_load_lag_{lag}" for lag in range(1, p["input_hours"] + 1)]
        state["feature_names"] = [*state["history_feature_names"],
                                   *[f"normalized_load_{stat}_{window}" for window in state["summary_windows"]
                                     for stat in ("mean", "std")],
                                   *[f"future_{step + 1}_{name}" for step in range(p["output_hours"]) for name in cov_names]]
    return state


def _exog_features(frame: pd.DataFrame, state: dict[str, Any]) -> np.ndarray:
    raw = _raw_covariates(frame, state["exog_cols"])
    missing = np.isnan(raw).astype(float)
    if state["backend"] == "lstm":
        raw = np.where(np.isnan(raw), np.asarray(state["exog_medians"]), raw)
        continuous = np.column_stack([raw, _calendar(frame)])
        continuous = (continuous - np.asarray(state["feature_mean"])) / np.asarray(state["feature_std"])
    else:
        continuous = np.column_stack([raw, _calendar(frame)])
    return np.column_stack([continuous, missing]).astype(np.float32)


def _normalized_load(frame: pd.DataFrame, state: dict[str, Any]) -> np.ndarray:
    return (frame["actual_kwh"].to_numpy(dtype=float) - state["load_mean"]) / state["load_std"]


def _xgb_features(frame: pd.DataFrame, targets: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    load = _normalized_load(frame, state).astype(np.float32)
    count, horizon = state["input_hours"], state["output_hours"]
    result = np.empty((len(targets), len(state["feature_names"])), dtype=np.float32)
    result[:, :count] = load[targets[:, None] - np.arange(1, count + 1)]
    past = pd.Series(load).shift(1)
    column = count
    for window in state["summary_windows"]:
        result[:, column] = past.rolling(window).mean().to_numpy()[targets]
        result[:, column + 1] = past.rolling(window).std(ddof=0).to_numpy()[targets]
        column += 2
    exog = _exog_features(frame, state)
    result[:, column:] = exog[targets[:, None] + np.arange(horizon)].reshape(len(targets), horizon * exog.shape[1])
    return result


def _target_matrix(frame: pd.DataFrame, targets: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    load = _normalized_load(frame, state).astype(np.float32)
    return load[targets[:, None] + np.arange(state["output_hours"])]


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _device(requested: str) -> str:
    import torch
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("LSTM CUDA was requested but is unavailable")
    return str(requested)


def _xgboost_device(requested: str) -> str:
    """Fail explicitly if requested GPU execution would silently fall back to CPU."""
    if requested == "cpu":
        return requested
    import torch
    import xgboost
    if not xgboost.build_info().get("USE_CUDA"):
        raise RuntimeError("XGBoost CUDA was requested but this XGBoost build has no CUDA support")
    if not torch.cuda.is_available():
        raise RuntimeError("XGBoost CUDA was requested but no CUDA device is available")
    index = int(requested.split(":", 1)[1]) if ":" in requested else 0
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"XGBoost CUDA device index {index} is unavailable")
    return f"cuda:{index}"


def _check_xgboost_device(model, requested: str) -> str:
    actual = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
    if actual != requested:
        raise RuntimeError(f"XGBoost requested device {requested} but activated {actual}")
    return actual


def _lstm_model(state: dict[str, Any]):
    import torch
    p = state["parameters"]

    class Regressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(len(state["history_feature_names"]), p["lstm_hidden_size"],
                                      num_layers=p["lstm_num_layers"], batch_first=True,
                                      dropout=p["lstm_dropout"] if p["lstm_num_layers"] > 1 else 0.0)
            self.head_dropout = torch.nn.Dropout(p["lstm_head_dropout"])
            head_width = p["lstm_hidden_size"] + state["output_hours"] * len(state["future_feature_names"])
            self.head = torch.nn.Linear(head_width, state["output_hours"])

        def forward(self, history_features, future_features):
            encoded, _ = self.lstm(history_features)
            combined = torch.cat([encoded[:, -1, :], future_features.flatten(start_dim=1)], dim=1)
            return self.head(self.head_dropout(combined))

    return Regressor()


def _lstm_dataset(frame: pd.DataFrame, targets: np.ndarray, state: dict[str, Any]):
    import torch
    load = _normalized_load(frame, state).astype(np.float32)
    exog = _exog_features(frame, state)
    context, horizon = state["input_hours"], state["output_hours"]

    class Windows(torch.utils.data.Dataset):
        def __len__(self):
            return len(targets)

        def __getitem__(self, index):
            target = int(targets[index])
            # Every encoder row pairs observed load with covariates at that same hour.
            features = np.column_stack([load[target - context:target],
                                        exog[target - context:target]])
            return (torch.from_numpy(features), torch.from_numpy(exog[target:target + horizon]),
                    torch.from_numpy(load[target:target + horizon]))

    return Windows()


def _fit_lstm(frame: pd.DataFrame, training_targets: np.ndarray, validation_targets: np.ndarray,
              state: dict[str, Any], rounds: int, phase: str):
    import torch
    p = state["parameters"]
    _seed(int(p["lstm_seed"]))
    torch.manual_seed(int(p["lstm_seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(p["lstm_seed"]))
    device = _device(p["lstm_device"])
    cuda_batches = torch.device(device).type == "cuda"
    model = _lstm_model(state).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p["lstm_learning_rate"], weight_decay=p["lstm_weight_decay"])
    loss_fn = torch.nn.HuberLoss(delta=p["lstm_huber_delta"])
    generator = torch.Generator().manual_seed(int(p["lstm_seed"]))
    training = torch.utils.data.DataLoader(_lstm_dataset(frame, training_targets, state),
                                          batch_size=p["lstm_batch_size"], shuffle=True, generator=generator,
                                          pin_memory=cuda_batches)
    validation = torch.utils.data.DataLoader(_lstm_dataset(frame, validation_targets, state),
                                            batch_size=p["lstm_batch_size"], shuffle=False,
                                            pin_memory=cuda_batches)
    history = []
    best_metric, best_round, stale = float("inf"), 0, 0
    best_state = None
    for epoch in range(1, rounds + 1):
        model.train()
        absolute_error, loss_total = 0.0, 0.0
        for features, future_features, targets in training:
            # The dataset and sequences stay on CPU; only this batch uses GPU memory.
            features = features.to(device, non_blocking=cuda_batches)
            future_features = future_features.to(device, non_blocking=cuda_batches)
            targets = targets.to(device, non_blocking=cuda_batches)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features, future_features)
            loss = loss_fn(prediction, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), p["lstm_grad_clip"])
            optimizer.step()
            absolute_error += float(torch.abs(prediction.detach() - targets).sum().cpu())
            loss_total += float(loss.detach().cpu()) * len(targets)
        validation_mae = None
        if len(validation_targets):
            model.eval()
            error = 0.0
            with torch.inference_mode():
                for features, future_features, targets in validation:
                    features = features.to(device, non_blocking=cuda_batches)
                    future_features = future_features.to(device, non_blocking=cuda_batches)
                    targets = targets.to(device, non_blocking=cuda_batches)
                    error += float(torch.abs(model(features, future_features) - targets).sum().cpu())
            validation_mae = error / (len(validation_targets) * state["output_hours"])
        history.append(dict(phase=phase, step=None, round=epoch,
                            train_normalized_mae=absolute_error / (len(training_targets) * state["output_hours"]),
                            validation_normalized_mae=validation_mae, train_loss=loss_total / len(training_targets)))
        if not np.isfinite(history[-1]["train_loss"]) or (validation_mae is not None and not np.isfinite(validation_mae)):
            raise RuntimeError("LSTM training produced a nonfinite loss or validation metric")
        if validation_mae is not None:
            if validation_mae < best_metric - p["lstm_min_delta"]:
                best_metric, best_round, stale = validation_mae, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
            if stale >= p["lstm_early_stopping_patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_round or rounds, history


def _fit_xgboost(frame: pd.DataFrame, training_targets: np.ndarray, validation_targets: np.ndarray,
                 state: dict[str, Any], rounds: int, phase: str):
    from xgboost import XGBRegressor
    p = state["parameters"]
    args = {key.removeprefix("xgboost_"): value for key, value in p.items()
            if key.startswith("xgboost_") and key not in {"xgboost_exog_cols", "xgboost_seed", "xgboost_early_stopping_rounds"}}
    args.update(n_estimators=rounds, objective="reg:squarederror", eval_metric="mae", random_state=int(p["xgboost_seed"]))
    args["device"] = _xgboost_device(p["xgboost_device"])
    if len(validation_targets):
        args["early_stopping_rounds"] = p["xgboost_early_stopping_rounds"]
    x = _xgb_features(frame, training_targets, state)
    y = _target_matrix(frame, training_targets, state)
    x_valid, y_valid = None, None
    if len(validation_targets):
        x_valid = _xgb_features(frame, validation_targets, state)
        y_valid = _target_matrix(frame, validation_targets, state)
    models, best_rounds, history = [], [], []
    for step in range(state["output_hours"]):
        model = XGBRegressor(**args)
        eval_set = [(x, y[:, step])]
        if x_valid is not None:
            eval_set.append((x_valid, y_valid[:, step]))
        model.fit(x, y[:, step], eval_set=eval_set, verbose=False)
        _check_xgboost_device(model, args["device"])
        scores = model.evals_result()
        history.extend(dict(phase=phase, step=step + 1, round=i + 1, train_normalized_mae=float(value), train_loss=None,
                            validation_normalized_mae=float(scores["validation_1"]["mae"][i]) if len(validation_targets) else None)
                       for i, value in enumerate(scores["validation_0"]["mae"]))
        models.append(model)
        best_rounds.append(int(model.best_iteration + 1) if len(validation_targets) else rounds)
    if any(not np.isfinite(row["train_normalized_mae"]) or
           (row["validation_normalized_mae"] is not None and not np.isfinite(row["validation_normalized_mae"])) for row in history):
        raise RuntimeError("XGBoost training produced a nonfinite metric")
    return models, best_rounds, history


def _predictor(backend: str, model, state: dict[str, Any], *, rounds: int | list[int] | None = None) -> Callable:
    def predict(context_frame: pd.DataFrame, future_frame: pd.DataFrame) -> pd.DataFrame:
        future = _checked_frame(future_frame.drop(columns=["actual_kwh", "load"], errors="ignore"), actuals=False)
        count, horizon = state["input_hours"], state["output_hours"]
        if len(future) != horizon:
            raise ValueError(f"Offline {backend} requires exactly {horizon} future hours for its direct output window")
        context = _checked_frame(context_frame, actuals=True)
        context = context[context["time"] < future.iloc[0]["time"]].tail(count).reset_index(drop=True)
        if len(context) < count:
            raise ValueError(f"Offline {backend} requires {count} complete context hours")
        combined = pd.concat([context, future], ignore_index=True)
        if not combined["time"].diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
            raise ValueError("Offline prediction requires consecutive hourly context and future rows; padding is forbidden")
        if backend == "lstm":
            import torch
            device = next(model.parameters()).device
            model.eval()
            features = np.column_stack([_normalized_load(context, state), _exog_features(context, state)]).astype(np.float32)
            future_features = _exog_features(future, state)
            with torch.inference_mode():
                normalized = model(torch.from_numpy(features).unsqueeze(0).to(device),
                                   torch.from_numpy(future_features).unsqueeze(0).to(device)).cpu().numpy().reshape(-1)
        else:
            if len(model) != horizon:
                raise ValueError("XGBoost model count does not match its output window")
            row = _xgb_features(combined, np.asarray([count]), state)
            values = []
            matrix = None
            if state["parameters"]["xgboost_device"].startswith("cuda"):
                from xgboost import DMatrix
                matrix = DMatrix(row, nthread=state["parameters"]["xgboost_n_jobs"])
            for step, step_model in enumerate(model):
                kwargs = {"iteration_range": (0, int(rounds[step]))} if rounds is not None else {}
                if state["parameters"]["xgboost_device"].startswith("cuda"):
                    value = step_model.get_booster().predict(matrix, **kwargs)[0]
                else:
                    value = step_model.predict(row, **kwargs)[0]
                values.append(float(value))
            normalized = np.asarray(values, dtype=float)
        if normalized.shape != (horizon,) or not np.isfinite(normalized).all():
            raise RuntimeError("Offline model produced a nonfinite or incorrectly shaped direct forecast")
        predictions = np.maximum(0.0, normalized * state["load_std"] + state["load_mean"])
        if not np.isfinite(predictions).all():
            raise RuntimeError("Offline model produced a nonfinite forecast")
        return pd.DataFrame({"time": future["time"].to_numpy(), "raw_predicted_kwh": predictions})
    return predict


def _direct_diagnostic(predict: Callable, frame: pd.DataFrame, start: pd.Timestamp,
                       end: pd.Timestamp, load_std: float, horizon_hours: int) -> dict[str, Any]:
    errors, actuals = [], []
    days, skipped = 0, 0
    day = start
    while day + pd.Timedelta(hours=horizon_hours) <= end:
        future = frame[(frame["time"] >= day) & (frame["time"] < day + pd.Timedelta(hours=horizon_hours))]
        expected = pd.date_range(day, periods=horizon_hours, freq="h")
        if len(future) != horizon_hours or not np.array_equal(future["time"].to_numpy(), expected.to_numpy()):
            skipped += 1
        else:
            try:
                prediction = predict(frame[frame["time"] < day], future)
            except ValueError as exc:
                if "context" not in str(exc) and "consecutive" not in str(exc):
                    raise
                skipped += 1
            else:
                observed = future["actual_kwh"].to_numpy(dtype=float)
                errors.extend((prediction["raw_predicted_kwh"].to_numpy() - observed).tolist())
                actuals.extend(observed.tolist())
                days += 1
        day += pd.Timedelta(days=1)
    error = np.asarray(errors, dtype=float)
    denominator = float(np.abs(actuals).sum())
    return dict(windows=days, skipped_windows=skipped, forecast_values=len(errors),
                mae_kwh=float(np.abs(error).mean()) if len(error) else None,
                rmse_kwh=float(np.sqrt(np.mean(error ** 2))) if len(error) else None,
                normalized_mae=float(np.abs(error).mean() / load_std) if len(error) else None,
                wape=float(np.abs(error).sum() / denominator) if denominator else None,
                history_policy="known_actuals_before_each_origin", output_hours=horizon_hours,
                origin_stride_hours=24, prediction_mode="direct_window")


def _scaler_summary(state: dict[str, Any]) -> dict[str, Any]:
    return {name: state[name] for name in ("load_mean", "load_std", "fitted_rows", "fitted_start", "fitted_end")}


def train_zone_model(backend: str, frame: pd.DataFrame, validation_start: pd.Timestamp,
                     validation_end: pd.Timestamp, parameters: dict[str, Any], model_dir: Path) -> dict[str, Any]:
    """Train once with chronological validation and publish the best direct model."""
    backend = str(backend).lower()
    p = _parameters(backend, parameters)
    frame = _checked_frame(frame, actuals=True)
    start, end = pd.Timestamp(validation_start), pd.Timestamp(validation_end)
    if start >= end:
        raise ValueError("Development validation_end must be after validation_start (exclusive)")
    development = frame[frame["time"] < start].reset_index(drop=True)
    if development.empty:
        raise ValueError("No development-training rows before validation_start")
    targets = complete_target_indices(frame, p["input_hours"], p["output_hours"], p["sample_stride_hours"])
    target_times = frame.iloc[targets]["time"]
    last_target_times = frame.iloc[targets + p["output_hours"] - 1]["time"].reset_index(drop=True)
    training_targets = targets[(last_target_times < start).to_numpy()]
    validation_targets = targets[((target_times.reset_index(drop=True) >= start) & (last_target_times < end)).to_numpy()]
    if not len(training_targets) or not len(validation_targets):
        raise ValueError("Development training and validation both require complete supervised windows")
    state = _fit_preprocess(backend, development, p)
    fit = _fit_lstm if backend == "lstm" else _fit_xgboost
    maximum = int(p["lstm_epochs"] if backend == "lstm" else p["xgboost_n_estimators"])
    model, best_rounds, history = fit(frame, training_targets, validation_targets, state, maximum, "train")
    diagnostic = _direct_diagnostic(_predictor(backend, model, state, rounds=best_rounds),
                                    frame, start, end, state["load_std"], p["output_hours"])
    state["best_rounds"] = best_rounds
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if backend == "lstm":
        import torch
        torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, model_dir / "model.pt")
        rounds_run = len(history)
        best_validation = history[best_rounds - 1]["validation_normalized_mae"]
        per_output_validation = None
        execution_device = str(next(model.parameters()).device)
    else:
        bundle = dict(schema_version=PREPROCESS_SCHEMA_VERSION, backend=backend,
                      model_strategy="independent_regressor_per_output", input_hours=p["input_hours"], output_hours=p["output_hours"],
                      models=[json.loads(step_model.get_booster()[:count].save_raw(raw_format="json").decode("utf-8"))
                              for step_model, count in zip(model, best_rounds)])
        (model_dir / "model.json").write_text(json.dumps(bundle, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        rounds_run = [sum(row["step"] == step + 1 for row in history) for step in range(p["output_hours"])]
        per_output_validation = [next(row["validation_normalized_mae"] for row in history
                                      if row["step"] == step + 1 and row["round"] == count)
                                 for step, count in enumerate(best_rounds)]
        best_validation = float(np.mean(per_output_validation))
        device = _xgboost_device(p["xgboost_device"])
        execution_device = _check_xgboost_device(model[0], device)
    (model_dir / "preprocess.json").write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False)
    report = dict(backend=backend, training_phases=["train"], two_phase_training=False,
                  full_fit_reset=False, preprocessing_refitted=False, best_rounds=best_rounds, rounds_run=rounds_run,
                  validation_start=start.isoformat(), validation_end_exclusive=end.isoformat(),
                  validation_selection_metric="direct_window_normalized_mae" if backend == "lstm" else "per_output_normalized_mae",
                  best_validation_normalized_mae=best_validation, per_output_validation_normalized_mae=per_output_validation,
                  sample_counts=dict(train=int(len(training_targets)), validation=int(len(validation_targets)),
                                     complete_windows=int(len(targets)),
                                     purged_windows=int(len(targets) - len(training_targets) - len(validation_targets))),
                  features=state["feature_names"], exog_cols=state["exog_cols"], dropped_exog_cols=state["dropped_exog_cols"],
                  input_hours=p["input_hours"], output_hours=p["output_hours"], sample_stride_hours=p["sample_stride_hours"],
                  history_feature_names=state["history_feature_names"], future_feature_names=state["future_feature_names"],
                  actual_dimensions=dict(input_hours=p["input_hours"], output_hours=p["output_hours"],
                                         history_channels=len(state["history_feature_names"]) if backend == "lstm" else 1,
                                         future_channels=len(state["future_feature_names"]),
                                         xgboost_feature_count=len(state["feature_names"]) if backend == "xgboost" else None),
                  preprocessing=_scaler_summary(state), preprocessing_fit_start=state["fitted_start"],
                  preprocessing_fit_end_exclusive=(development.iloc[-1]["time"] + pd.Timedelta(hours=1)).isoformat(),
                  training_label_start=frame.iloc[training_targets[0]]["time"].isoformat(),
                  training_label_end_exclusive=(frame.iloc[training_targets[-1] + p["output_hours"] - 1]["time"]
                                                + pd.Timedelta(hours=1)).isoformat(),
                  direct_horizon_diagnostic=diagnostic, parameters=p, execution_device=execution_device)
    # Enforce portable JSON here so errors cannot create seemingly valid cache reports.
    return json.loads(json.dumps(report, ensure_ascii=False, allow_nan=False))


def load_zone_predictor(backend: str, model_dir: Path) -> Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    """Load fixed weights and fixed preprocessing; this function never trains."""
    backend = str(backend).lower()
    model_dir = Path(model_dir)
    state = json.loads((model_dir / "preprocess.json").read_text(encoding="utf-8"))
    if state.get("schema_version") != PREPROCESS_SCHEMA_VERSION or state.get("backend") != backend:
        raise ValueError("Offline preprocessing schema/backend does not match requested model")
    if backend == "lstm":
        import torch
        model = _lstm_model(state)
        model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True))
        model.to(_device(state["parameters"]["lstm_device"]))
        model.eval()
    elif backend == "xgboost":
        from xgboost import XGBRegressor
        device = _xgboost_device(state["parameters"]["xgboost_device"])
        bundle = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
        if (bundle.get("schema_version") != PREPROCESS_SCHEMA_VERSION or bundle.get("backend") != backend
                or bundle.get("input_hours") != state["input_hours"] or bundle.get("output_hours") != state["output_hours"]
                or len(bundle.get("models", [])) != state["output_hours"]):
            raise ValueError("XGBoost bundle does not match its direct-window preprocessing schema")
        model = []
        for step, payload in enumerate(bundle["models"]):
            step_model = XGBRegressor()
            step_model.load_model(bytearray(json.dumps(payload, allow_nan=False).encode("utf-8")))
            step_model.set_params(device=device, n_jobs=state["parameters"]["xgboost_n_jobs"])
            _check_xgboost_device(step_model, device)
            if step_model.get_booster().num_boosted_rounds() != state["best_rounds"][step]:
                raise ValueError("XGBoost bundle contains trees outside its selected best iteration")
            model.append(step_model)
    else:
        raise ValueError(f"Unsupported offline backend: {backend}")
    return _predictor(backend, model, state, rounds=state["best_rounds"])
