"""Per-zone LoRA adapters for the native TimesFM 2.5 and Chronos 2 models.

Base weights are frozen. Complete chronological windows and validation select one
adapter; forecasting reuses the existing covariate implementations. Loading an
adapter always creates a fresh base model and never changes the zero-shot cache.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.training import LORA_BACKENDS, base_backend
from src.training.windows import (
    DEFAULT_EXOG_COLS,
    KNOWN_EXOG_COLS,
    checked_training_frame,
    raw_covariates,
    complete_target_indices,
)


FOUNDATION_PREPROCESS_SCHEMA_VERSION = 2

_TARGETS = {
    "timesfm": ["qkv_proj", "out"],
    "chronos": ["self_attention.q", "self_attention.v", "self_attention.k", "self_attention.o",
                "output_patch_embedding.output_layer"],
}

_REPOS = {"timesfm": "google/timesfm-2.5-200m-pytorch", "chronos": "amazon/chronos-2"}


def _parameters(backend: str, supplied: dict[str, Any]) -> dict[str, Any]:
    from src.config.training import TRAINING_PARAMETER_DEFAULTS, validate_training_parameters
    family = _adapter_family(backend)
    p = {name: value for name, value in TRAINING_PARAMETER_DEFAULTS.items() if name.startswith(family + "_")}
    p.update({name: value for name, value in supplied.items() if name in p})
    validate_training_parameters(p, f"{backend} LoRA")
    for name in ("input_hours", "output_hours", "sample_stride_hours"):
        value = supplied.get(name, 1 if name == "sample_stride_hours" else None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        p[name] = value
    p[f"{family}_repo"] = str(supplied.get(f"{family}_repo", _REPOS[family]))
    exog = supplied.get(f"{family}_exog_cols")
    exog = list(DEFAULT_EXOG_COLS if exog is None else exog)
    if len(set(exog)) != len(exog) or set(exog) - KNOWN_EXOG_COLS:
        raise ValueError("LoRA exog_cols must contain unique known covariates, never actual load")
    p[f"{family}_exog_cols"] = exog
    # Native TimesFM emits 128 values per decoding patch. Longer horizons need
    # a separate autoregressive training objective, not truncated target labels.
    if family == "timesfm" and p["output_hours"] > 128:
        raise ValueError("TimesFM LoRA currently supports output_hours <= 128")
    return p


def _adapter_family(backend: str) -> str:
    if backend not in LORA_BACKENDS:
        raise ValueError(f"Unsupported LoRA backend: {backend}; use timesfm_lora or chronos_lora")
    return base_backend(backend)


def _split_targets(frame, input_hours, output_hours, stride, start, end):
    targets = complete_target_indices(frame, input_hours, output_hours, stride)
    first = frame.iloc[targets]["time"].to_numpy()
    last = frame.iloc[targets + output_hours - 1]["time"].to_numpy()
    train = targets[last < start.to_datetime64()]
    valid = targets[(first >= start.to_datetime64()) & (last < end.to_datetime64())]
    if not len(train) or not len(valid):
        raise ValueError("LoRA training and validation both require complete supervised windows")
    return train, valid, len(targets) - len(train) - len(valid)


def _dependencies():
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
    except ImportError as exc:
        raise RuntimeError("LoRA requires torch and peft; install project dependencies before training. "
                           "Full-weight training is never used as a fallback.") from exc
    return torch, LoraConfig, PeftModel, get_peft_model


def _local_base_checksums(path: Path) -> dict[str, str]:
    files = [p for p in path.rglob("*") if p.is_file() and
             (p.suffix in {".safetensors", ".bin"} or p.name in {"config.json", "model.safetensors.index.json"})]
    result = {}
    for file in sorted(files):
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result[file.relative_to(path).as_posix()] = digest.hexdigest()
    if not any(Path(name).suffix in {".safetensors", ".bin"} for name in result):
        raise ValueError(f"No local base-model weights found in {path}")
    return result


def _load_base(backend: str, repo: str, provenance: dict | None = None):
    """Return a fresh wrapper plus an immutable Hub revision or local checksums."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local = Path(repo).is_dir()
    saved = dict(provenance or {})
    if local:
        checksums = _local_base_checksums(Path(repo))
        if saved and checksums != saved.get("local_file_checksums"):
            raise ValueError("Local base-model files changed since this LoRA adapter was trained")
        saved = {"repo": str(Path(repo).resolve()), "revision": None, "local_file_checksums": checksums}
    revision = saved.get("revision")
    if backend == "timesfm":
        from timesfm import TimesFM_2p5_200M_torch
        from src.forecast.foundation import patch_timesfm_hub_kwargs
        patch_timesfm_hub_kwargs(TimesFM_2p5_200M_torch)
        source = repo
        if not local:
            from huggingface_hub import hf_hub_download
            weights = Path(hf_hub_download(repo, "model.safetensors", revision=revision))
            revision = weights.parent.name
            if not revision or revision in {"snapshots", "blobs"}:
                raise ValueError("Could not identify the immutable TimesFM base revision")
            source = str(weights.parent)
        wrapper = TimesFM_2p5_200M_torch.from_pretrained(source, torch_compile=False)
        wrapper.model.to(device)
        wrapper.model.device = torch.device(device)
    else:
        from chronos import Chronos2Pipeline
        wrapper = Chronos2Pipeline.from_pretrained(repo, device_map=device, dtype=torch.float32,
                                                   **({"revision": revision} if revision else {}))
        revision = revision or getattr(wrapper.model.config, "_commit_hash", None)
        if not local and not revision:
            raise ValueError("Could not identify the immutable Chronos base revision")
    if not local:
        saved = {"repo": repo, "revision": revision, "local_file_checksums": None}
    return wrapper, saved


def _peft_ready_model(wrapper, backend):
    model = wrapper.model
    original = model.config
    if backend == "timesfm":
        # The native TimesFM configuration is a dataclass, whereas PEFT expects
        # a dict or a Transformers PretrainedConfig. Restore it after merging.
        model.config = dataclasses.asdict(original) if dataclasses.is_dataclass(original) else dict(original)
        model.config["model_type"] = "timesfm"
    return model, original


def _timesfm_quantiles(model, context, horizon):
    """Differentiable native one-patch forecast, including flip invariance."""
    import torch
    from timesfm.torch.util import revin, update_running_stats
    patch = model.p
    padding = (-context.shape[-1]) % patch
    values = torch.nn.functional.pad(context, (padding, 0))
    masks = torch.zeros_like(values, dtype=torch.bool)
    masks[:, :padding] = True
    # Match ForecastConfig(normalize_inputs=True), including left padding.
    outer_mu, outer_sigma = values.mean(-1, keepdim=True), values.std(-1, keepdim=True)
    values = revin(values, outer_mu, outer_sigma)
    patched_masks = masks.reshape(len(values), -1, patch)

    def forward(inputs):
        patches = inputs.reshape(len(inputs), -1, patch)
        count = torch.zeros(len(inputs), device=inputs.device)
        mu, sigma = torch.zeros_like(count), torch.zeros_like(count)
        means, scales = [], []
        for i in range(patches.shape[1]):
            (count, mu, sigma), _ = update_running_stats(count, mu, sigma, patches[:, i], patched_masks[:, i])
            means.append(mu)
            scales.append(sigma)
        means, scales = torch.stack(means, 1), torch.stack(scales, 1)
        inputs = torch.where(patched_masks, 0.0, revin(patches, means, scales))
        outputs, _ = model(inputs, patched_masks)
        predictions = outputs[2][:, -1].reshape(len(inputs), model.o, model.q)[:, :horizon]
        return revin(predictions, means[:, -1], scales[:, -1], reverse=True)

    positive, negative = forward(values), forward(-values)
    flipped = torch.cat([negative[..., :1], negative[..., 1:].flip(-1)], dim=-1)
    return revin((positive - flipped) / 2, outer_mu, outer_sigma, reverse=True)


def _batch_loss(backend, model, contexts, targets, past_covariates, future_covariates):
    import torch
    if backend == "timesfm":
        predictions = _timesfm_quantiles(model, contexts, targets.shape[-1])
        scale = contexts.std(-1, keepdim=True)
        scale = torch.where(scale > 1e-6, scale, 1.0)
        error = (targets[..., None] - predictions) / scale[..., None]
        quantiles = torch.as_tensor(model.config["quantiles"], device=contexts.device)
        pinball = torch.maximum(quantiles * error[..., 1:], (quantiles - 1) * error[..., 1:]).mean()
        return error[..., 0].square().mean() + 2 * pinball
    # A group contains the load and its known-future covariates. Only load is
    # a training target; the future covariate rows must never enter the loss.
    n, _, width = past_covariates.shape
    group_size = width + 1
    context = torch.cat([contexts[:, None, :], past_covariates.transpose(1, 2)], dim=1).reshape(n * group_size, -1)
    future = torch.cat([torch.full_like(targets[:, None, :], torch.nan), future_covariates.transpose(1, 2)], dim=1)
    target = torch.cat([targets[:, None, :], torch.full_like(future_covariates.transpose(1, 2), torch.nan)], dim=1)
    return model(context=context, group_ids=torch.arange(n, device=contexts.device).repeat_interleave(group_size),
                 future_covariates=future.reshape(n * group_size, -1),
                 future_target=target.reshape(n * group_size, -1),
                 num_output_patches=math.ceil(targets.shape[-1] / model.chronos_config.output_patch_size)).loss


def _covariates(frame, state):
    values = raw_covariates(frame, state["exog_cols"])
    return np.where(np.isfinite(values), values, np.asarray(state["exog_medians"])[None, :]).astype(np.float32)


def train_foundation_model(backend: str, frame: pd.DataFrame, validation_start: pd.Timestamp,
                           validation_end: pd.Timestamp, parameters: dict[str, Any], model_dir: Path) -> dict[str, Any]:
    """Train frozen-base LoRA once and save the best chronological validation adapter."""
    backend = backend.lower()
    family = _adapter_family(backend)
    p = _parameters(backend, parameters)
    frame = checked_training_frame(frame, actuals=True)
    start, end = pd.Timestamp(validation_start), pd.Timestamp(validation_end)
    if start >= end:
        raise ValueError("validation_end must be after validation_start")
    train, valid, purged = _split_targets(frame, p["input_hours"], p["output_hours"], p["sample_stride_hours"], start, end)
    torch, LoraConfig, _, get_peft_model = _dependencies()
    seed = p[f"{family}_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    wrapper, provenance = _load_base(family, p[f"{family}_repo"])
    base, _ = _peft_ready_model(wrapper, family)
    if family == "timesfm" and math.ceil(p["input_hours"] / base.p) * base.p + base.o > 16384:
        raise ValueError("TimesFM LoRA context plus output exceeds its context limit")
    if family == "chronos":
        if p["input_hours"] > base.chronos_config.context_length:
            raise ValueError("Chronos LoRA input_hours exceeds this base model's context_length; history cannot be truncated")
        if p["output_hours"] > base.chronos_config.max_output_patches * base.chronos_config.output_patch_size:
            raise ValueError("Chronos LoRA output_hours exceeds this base model's direct prediction limit")
    model = get_peft_model(base, LoraConfig(r=p[f"{family}_lora_rank"], lora_alpha=p[f"{family}_lora_alpha"],
        lora_dropout=p[f"{family}_lora_dropout"], target_modules=p[f"{family}_lora_target_modules"], bias="none"))
    model.peft_config["default"].base_model_name_or_path = provenance["repo"]
    model.peft_config["default"].revision = provenance["revision"]
    trained, total = model.get_nb_trainable_parameters()
    if not 0 < trained < total:
        raise ValueError("LoRA must train a nonempty subset of the base model parameters")
    development = frame[frame["time"] < start]
    requested = p[f"{family}_exog_cols"]
    raw = raw_covariates(development, requested)
    chosen = [i for i in range(len(requested)) if np.isfinite(raw[:, i]).any()]
    state = dict(schema_version=FOUNDATION_PREPROCESS_SCHEMA_VERSION, backend=backend, parameters=p,
                 input_hours=p["input_hours"], output_hours=p["output_hours"], base_model=provenance,
                 exog_cols=[requested[i] for i in chosen],
                 exog_medians=[float(np.nanmedian(raw[:, i])) for i in chosen],
                 prediction_mode="direct_window", adapter_type="lora")
    load = frame["actual_kwh"].to_numpy(dtype=np.float32)
    covariates = _covariates(frame, state)
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad],
                                  lr=p[f"{family}_learning_rate"], weight_decay=p[f"{family}_weight_decay"])
    batch_size, accumulation = p[f"{family}_batch_size"], p[f"{family}_gradient_accumulation_steps"]
    best, best_epoch, stale, history = math.inf, 0, 0, []
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    def tensors(indices):
        h, out = p["input_hours"], p["output_hours"]
        arrays = ([load[i - h:i] for i in indices], [load[i:i + out] for i in indices],
                  [covariates[i - h:i] for i in indices], [covariates[i:i + out] for i in indices])
        return [torch.as_tensor(np.stack(values), device=device, dtype=torch.float32) for values in arrays]

    for epoch in range(1, p[f"{family}_epochs"] + 1):
        model.train()
        order = rng.permutation(train)
        train_sum = 0.0
        batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        # Normalize by the actual number of windows in each accumulation group,
        # including the final short group and the final partial minibatch.
        for group_start in range(0, len(batches), accumulation):
            group = batches[group_start:group_start + accumulation]
            group_count = sum(len(indices) for indices in group)
            optimizer.zero_grad(set_to_none=True)
            for indices in group:
                loss = _batch_loss(family, model, *tensors(indices))
                if not torch.isfinite(loss):
                    raise RuntimeError("LoRA training produced a nonfinite loss")
                (loss * (len(indices) / group_count)).backward()
                train_sum += float(loss.detach().cpu()) * len(indices)
            torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad], p[f"{family}_grad_clip"])
            optimizer.step()
        model.eval()
        valid_sum = 0.0
        with torch.no_grad():
            for offset in range(0, len(valid), batch_size):
                indices = valid[offset:offset + batch_size]
                loss = _batch_loss(family, model, *tensors(indices))
                if not torch.isfinite(loss):
                    raise RuntimeError("LoRA validation produced a nonfinite loss")
                valid_sum += float(loss.cpu()) * len(indices)
        score = valid_sum / len(valid)
        history.append(dict(phase="train", epoch=epoch, train_loss=train_sum / len(train), validation_loss=score))
        if score < best:
            best, best_epoch, stale = score, epoch, 0
            model.save_pretrained(model_dir, safe_serialization=True)
        else:
            stale += 1
        if stale >= p[f"{family}_early_stopping_patience"]:
            break
    state["best_epoch"] = best_epoch
    (model_dir / "preprocess.json").write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False)
    files = ["adapter_model.safetensors", "adapter_config.json", "preprocess.json", "training_history.csv"]
    if any(not (model_dir / name).is_file() for name in files):
        raise RuntimeError("LoRA training did not produce a complete adapter artifact")
    return dict(backend=backend, adapter_type="lora", training_phases=["train"], two_phase_training=False,
                preprocess_schema_version=FOUNDATION_PREPROCESS_SCHEMA_VERSION, artifact_files=files,
                sample_counts=dict(train=len(train), validation=len(valid), purged_windows=purged),
                input_hours=p["input_hours"], output_hours=p["output_hours"], epochs_run=len(history),
                best_epoch=best_epoch, best_validation_loss=best, trainable_parameters=trained, total_parameters=total,
                execution_device=str(device), base_model=provenance,
                training_label_start=frame.iloc[train[0]]["time"].isoformat(),
                training_label_end_exclusive=(frame.iloc[train[-1] + p["output_hours"] - 1]["time"] + pd.Timedelta(hours=1)).isoformat(),
                validation_start=start.isoformat(), validation_end_exclusive=end.isoformat(),
                covariate_policy="native_xreg_at_inference" if family == "timesfm" else "known_future_covariates",
                history=history)


def load_foundation_predictor(backend: str, model_dir: Path):
    """Load an immutable adapter; no fitting and no shared zero-shot mutation."""
    backend, model_dir = backend.lower(), Path(model_dir)
    family = _adapter_family(backend)
    state = json.loads((model_dir / "preprocess.json").read_text(encoding="utf-8"))
    if (state.get("schema_version") != FOUNDATION_PREPROCESS_SCHEMA_VERSION
            or state.get("backend") != backend or state.get("adapter_type") != "lora"):
        raise ValueError("Foundation adapter preprocessing schema/backend mismatch")
    from src.forecast.foundation import run_timesfm_prediction, run_chronos_prediction
    _, _, PeftModel, _ = _dependencies()
    wrapper, _ = _load_base(family, state["base_model"]["repo"], state["base_model"])
    base, original_config = _peft_ready_model(wrapper, family)
    wrapper.model = PeftModel.from_pretrained(base, str(model_dir), is_trainable=False).merge_and_unload()
    if family == "timesfm":
        from timesfm import ForecastConfig
        wrapper.model.config = original_config
        wrapper.compile(ForecastConfig(max_context=state["input_hours"], max_horizon=max(32, state["output_hours"]),
                                       normalize_inputs=True, fix_quantile_crossing=True, return_backcast=True))
    wrapper.model.eval()

    def predict(context: pd.DataFrame, future: pd.DataFrame):
        context, future = checked_training_frame(context, actuals=True), checked_training_frame(future, actuals=False)
        h, out = state["input_hours"], state["output_hours"]
        if len(context) < h or len(future) != out:
            raise ValueError("LoRA prediction requires its full trained history and output windows")
        context = context.iloc[-h:].reset_index(drop=True)
        times = pd.concat([context["time"], future["time"]], ignore_index=True)
        if not times.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
            raise ValueError("LoRA prediction windows must be continuous and adjacent")
        values = context["actual_kwh"].to_numpy(dtype=np.float32)
        past_cov, future_cov = _covariates(context, state), _covariates(future, state)
        cols = state["exog_cols"]
        if family == "timesfm":
            raw, q10, q90 = run_timesfm_prediction(wrapper, values, past_cov if cols else None,
                                                  future_cov if cols else None, out, h, cols)
        else:
            raw, q10, q90 = run_chronos_prediction(wrapper, values, out, h, exog_context=past_cov,
                                                  exog_horizon=future_cov, exog_cols=cols)
        if not all(len(array) == out and np.isfinite(array).all() for array in (raw, q10, q90)):
            raise RuntimeError("LoRA model produced incomplete or nonfinite forecasts")
        return pd.DataFrame(dict(time=future["time"].to_numpy(), raw_predicted_kwh=raw, q10_kwh=q10, q90_kwh=q90))

    return predict
