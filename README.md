# MAPF: Forecaster–Agent Closed-Loop Load Control

MAPF separates forecasting from price control and keeps forecast-period ground truth out of every Agent prompt.

## Architecture

1. **Dataset adapter and cache** converts UrbanEV or a mapped long-format dataset to a canonical schema. Reusable dataset features are stored under `<dataset_path>/cache/`.
2. **Global forecaster** fits once at a fixed origin, calibrates validation bias, and persists a reusable artifact. Price proposals reuse this artifact; no control attempt retrains the model.
3. **Control engine** runs either Grid → Behaviour → Economist or a single Agent, proposes one energy price per continuous 3-hour window, reforecasts, and accepts a run only when every Zone/window is `Medium`.
4. **Evaluation** is kept separate from the no-leakage handoff and authoritative control result.

The load-policy bands use the position within each Zone's pre-origin historical 3-hour load min/max range:

- `Low`: below 35%
- `Medium`: 35% to below 80%
- `High`: 80% to below 90%
- `Extremely High`: 90% or above

## Dataset adapters and cache

UrbanEV is the default adapter. A generic long-format dataset can be configured with explicit semantic mappings:

```yaml
data:
  adapter: long_format
  timeseries_file: series.csv
  cache_dir: null  # defaults to <run.data_dir>/cache
  column_mapping:
    timestamp: when
    zone_id: area
    load_kwh: demand
    energy_price: tariff
```

Required canonical fields are timestamp, Zone, load, and a strictly positive baseline energy-price schedule. Optional dynamic fields include weather, occupancy, service price, and calendar values; optional static fields include coordinates, Zone type, POIs, geography, and charging capacity. Explicit mappings win over alias discovery, and ambiguous aliases fail fast.

The cache is fingerprinted by adapter/schema version, source paths, sizes, mtimes, column/header signatures, and mappings:

```text
<dataset_path>/cache/
  cache_manifest.json
  datasets/<dataset_fingerprint>/
    canonical_schema.json
    feature_manifest.json
    canonical_timeseries.csv.gz
    static_zone_features.csv
    poi_zone_counts.csv
    price_change_reference.json
    splits/<split_cache_key>/
      training_zone_profiles.csv
      historical_3h_load_policy.csv
```

`--force-cache` rebuilds caches. Writes use a temporary file followed by atomic replacement. Model state, validation bias, forecasts, Agent output, closed-loop traces, and evaluation are never stored in the dataset cache.

## Running

Copy `config.example.yaml` to `config.yaml`, configure the provider key or use `--dry-run`, then run:

```powershell
python main.py --dry-run --forecast-model AR --agent-mode multi_agent_economist_retry
```

The output root and run name are configured separately:

```yaml
run:
  output_folder: "output"
  experiment_name: null # null uses the existing automatic experiment name
```

The same values can be overridden independently from the CLI:

```powershell
python main.py --output-folder output --experiment-name my_experiment
```

`experiment_name` identifies one complete experiment matrix; it does not rename
ordinary single-model output. An unnamed matrix keeps the existing descriptive
name, for example `output/3zonesx4timesx4modes_2blends`. A custom name is
written as `output/my_experiment`.

Stages can be separated without changing the fitted state:

```powershell
python main.py --pipeline-stage forecaster --forecast-model AR
python main.py --pipeline-stage agent --forecast-model AR --agent-mode multi_agent_economist_retry
```

Supported control modes are:

- `multi_agent_economist_retry`: Grid → Behaviour → Economist initially; only Economist revises failed windows.
- `multi_agent_full_retry`: reruns all three Agents with the latest reforecast context.
- `single_agent_price_retry`: one Agent performs all roles; failure retries revise prices only.
- `single_agent_full_retry`: one Agent fully re-evaluates the updated context on failure.

Each run permits at most three price proposals. Successful windows retain their price for the next proposal, but every global reforecast re-evaluates all windows; a formerly successful window can become failed again. Negative prices consume an attempt and fail validation. There is no historical-P95 price cap or historical-mean multiplier cap.

## Fixed-origin and no leakage

- Profiles, scalers, stress thresholds, preprocessing, and validation calibration use only timestamps before `forecast_start`.
- Multi-step forecasts roll their own predictions, never forecast-period actual load.
- Agent context is constructed from an allowlist. It excludes future actual load/stress/percentile, future error metrics, evaluation nodes, and all derived correctness labels.
- Grid Agent assesses the forecaster values but cannot overwrite them.
- Only structured reasoning summaries are persisted; hidden chain-of-thought is neither requested nor stored.

## Historical price diagnostic

For every Zone, the adapter computes mean energy price in consecutive non-overlapping 3-hour windows, then pools adjacent-window absolute percentage changes across the whole dataset. Every proposal reports its empirical percentile, dataset P95, and `exceeds_historical_p95`. This diagnostic is display-only: it does not clip prices, determine success, or enter Agent context.

## Outputs

The authoritative control artifact is `control_results.json` (schema version 2). It records the dataset fingerprint, split cache key, feature manifest, forecast origin/model, experiment mode, global and Zone status, attempts, structured Agent summaries, reforecast trace, final window prices/load positions/stress, rationale, and historical price diagnostics. Each proposal attempt also stores per-Zone Agent call timing/token usage, while top-level round records store concurrent Agent batch wall time and cumulative usage. Missing provider usage remains explicitly incomplete instead of being estimated.

Companion outputs are:

- `control_final_windows.csv`
- `control_attempt_trace.csv`, with every Zone/window/attempt price transition, including frozen windows and previous-to-current price/shift deltas
- `agent_attempt_usage.csv`, with separate `global` and `zone` rows so token totals are not repeated for each window
- `control_experiment_summary.csv`
- `forecaster/forecaster_artifact.json` for refit-free Agent-only handoff. It stores
  backend-specific inference state and known-future covariates; every proposed price
  schedule is sent back through the original TimesFM, Chronos, LSTM, or AR forecast
  function. The control loop does not use a separate price-to-load approximation.
- `forecaster/context_snippets.json` for the no-leakage Agent handoff
- forecast evaluation CSV/plots, separate from `control_results.json`

Experiment matrices additionally aggregate these records into `experiment_agent_attempt_usage.csv` and `experiment_control_attempt_trace.csv`.

The run is `success` only when every selected Zone and every 3-hour window is `Medium`; otherwise the third proposal produces `fail` with the final summaries and reforecast state.

## Tests

```powershell
python -B -m unittest tests.test_refactor -v
```

The refactor tests cover cache hit/invalidation/recovery, dataset-local cache layout, fixed-origin cross-Zone artifacts, artifact round-trip and price-scenario reuse, Agent no-leakage filtering, attempt-level timing/token accounting, frozen/revised price trajectories, and versioned authoritative output.
