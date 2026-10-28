# MAPF: Forecaster–Agent Closed-Loop Load Control

MAPF separates forecasting from price control and keeps forecast-period ground truth out of every Agent prompt.

## Architecture

1. **Dataset adapter and cache** converts UrbanEV, CHARGED, MP-EVData, or a mapped long-format dataset to a canonical schema. Reusable dataset features are stored under `<dataset_path>/cache/`.
2. **Offline models** train a separate LSTM and XGBoost for each available Zone from that Zone's full valid history. Forecasting reads these cached weights, calibrates each origin, and persists reusable inference state. Price proposals reuse the same model and bias.
3. **Control engine** runs either Grid → Behaviour → Economist or a single Agent, proposes one energy price per continuous 3-hour window, reforecasts, and accepts a run only when every Zone/window is `Medium`.
4. **Evaluation** is kept separate from the no-leakage handoff and authoritative control result.

The load-policy bands use the position within each Zone's pre-origin historical 3-hour load min/max range:

- `Low`: below 35%
- `Medium`: 35% to below 80%
- `High`: 80% to below 90%
- `Extremely High`: 90% or above

## Dataset adapters and cache

Set `run.data_dir` in `config.yaml`, then run `python main.py`. The default
`data.adapter: auto` recognizes UrbanEV, CHARGED, and MP-EVData from their files.
Use a single folder to switch datasets, or a list to train in separate processes:

```yaml
data:
  adapter: auto
run:
  data_dir: data/MP-EVData # or data/UrbanEV, data/CHARGED/JHB
  # For parallel training, replace the value above with:
  # data_dir: [data/UrbanEV, data/CHARGED/JHB, data/MP-EVData]
  max_parallel_datasets: 2
  forecast_starts: null
  zones: null
```

Null dates select the dataset's last complete forecast window; null zones select
sites from that dataset. Explicit dates and sites remain supported for controlled
experiments. Devices are selected automatically by the existing model backends.
Outputs are grouped under `<output_folder>/<experiment_name>/<dataset>/`, with
the date, seed, and mode directories required by the experiment matrix. Every
single-model forecast uses `blend_<alpha>/<backend>/`. Parallel batches also
write individual logs and `batch_manifest.json`.
See [dataset setup, filtering rules and parallel training](docs/datasets.md).

A generic long-format dataset can be configured with explicit semantic mappings:

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

`--force-cache` rebuilds dataset caches. Writes use a temporary file followed by atomic replacement. Offline LSTM/XGBoost weights live under `<dataset_path>/cache/weight/<backend>/`; forecast-specific calibration, forecasts, Agent output and evaluation remain in experiment outputs.

## Running

Copy `config.example.yaml` to `config.yaml`, configure the provider key or use `--dry-run`, then run.

Multi-Agent and single-Agent provider settings are independent, so they may use
different API keys, endpoints, models, timeouts, and concurrency limits:

```yaml
agent:
  multi_agent:
    api_key: "${MULTI_AGENT_API_KEY}"
    base_url: "https://openrouter.ai/api/v1"
    model: "qwen/qwen3-14b"
  single_agent:
    api_key: "${SINGLE_AGENT_API_KEY}"
    base_url: "https://api.example.com/v1"
    model: "example/single-agent-model"
```

Every `multi_agent_*` mode (including the three-round discussion mode) uses
`agent.multi_agent`; every `single_agent_*` mode uses `agent.single_agent`.
Only the active profile is loaded, so its counterpart's environment variables
do not need to be set. The legacy flat `agent.api_key`/`agent.base_url` format and
`single_agent_model` remain supported. `--model` overrides the model of the active
profile for that run without changing the other profile.

```powershell
python main.py --dry-run --forecast-models AR --agent-mode multi_agent_economist_retry
```

The output root and run name are configured separately:

```yaml
run:
  output_folder: "output"
  experiment_name: null # null uses the stable folder name "experiment"
```

The same values can be overridden independently from the CLI:

```powershell
python main.py --output-folder output --experiment-name my_experiment
```

`experiment_name` identifies the run or experiment matrix under the output root.
An unnamed run uses `<output_folder>/experiment/<dataset>/`; a custom name uses
`<output_folder>/my_experiment/<dataset>/`. Forecast directories include the actual
alpha and backend, including when there is only one configuration.

Stages can be separated without changing the fitted state:

```powershell
python main.py --pipeline-stage forecaster --forecast-models AR
python main.py --pipeline-stage agent --forecast-models AR --agent-mode multi_agent_economist_retry
```

Supported control modes are:

- `multi_agent_economist_retry`: Grid → Behaviour → Economist initially; only Economist revises failed windows.
- `multi_agent_full_retry`: reruns all three Agents with the latest reforecast context.
- `multi_agent_discussion_3rounds`: runs up to three internal Grid/Behaviour/Economist discussion rounds for every price proposal. Rounds 2 and 3 receive only the previous round's conclusion summaries, disagreements, and compact key decisions; full prior Agent outputs are not repeated. If a new round's substantive stress, elasticity, and pricing decisions match the preceding round, the discussion stops immediately. A failed outer control attempt starts a fresh discussion with the latest reforecast context.
- `single_agent_price_retry`: one Agent performs all roles; failure retries revise prices only.
- `single_agent_full_retry`: one Agent fully re-evaluates the updated context on failure.

Each run permits at most three price proposals. Successful windows retain their price for the next proposal, but every global reforecast re-evaluates all windows; a formerly successful window can become failed again. Negative prices consume an attempt and fail validation. There is no historical-P95 price cap or historical-mean multiplier cap.

## Prompt prefix caching

Agent prompts place fixed role, policy, output-schema, and discussion instructions
before the forecast context. Actual horizon values stay in the context; current
Agent reports follow it, with discussion round/handoff and schema-repair errors
at the end. Retry feedback is the final field of the context object and retains
its `context.retry_feedback` path. Updated forecasts and prices are always sent.

JSON objects use recursively sorted keys and compact separators, so identical
inputs produce identical prompt text across dataset workers and Python hash seeds.
Window arrays retain their original order. Existing context filters, output keys,
and pricing rules are preserved. This layout is used automatically without a new
configuration option. Actual cache hits depend on the inference service's cache
support, activation, minimum prefix length, and retention/routing behavior; see
the [provider caching documentation](https://openrouter.ai/docs/guides/best-practices/prompt-caching).

## vLLM and SGLang endpoints

The Agent client uses the existing Chat Completions API. Select the server in each
Agent profile; for example, after starting a local vLLM server:

```yaml
agent:
  max_concurrent_requests_total: 4
  multi_agent:
    backend: vllm # sglang uses the same configuration fields
    base_url: http://127.0.0.1:8000/v1
    model: Qwen/Qwen3-14B # must match the server's model name or served alias
    api_key: null # use the configured key if the server requires authentication
    reasoning_effort: "none" # none=disable thinking; low/medium/high=enable and forward effort
    chat_template_kwargs: {} # normally keep empty to follow reasoning_effort
    max_tokens: null # optional positive output-token limit; allow room for every window
    max_concurrent_requests: 4
```

The same options apply to `single_agent`. `backend: auto` identifies OpenRouter by
its URL hostname; other URLs use `openai_compatible`. It does not guess a local
engine from a port number or model name. Both `vllm` and `sglang` now use the
existing `reasoning_effort` setting. Their HTTP request receives top-level
`reasoning_effort`, together with `chat_template_kwargs.enable_thinking` derived
from the setting. The SDK merges these fields from `extra_body` into the JSON
request. OpenRouter retains its `reasoning: {effort: ...}` format; other generic
endpoints keep their previous behavior.

| Local `reasoning_effort` | Request behavior |
| --- | --- |
| `"none"` (default) | Send `reasoning_effort: "none"` and `enable_thinking: false` |
| `"low"`, `"medium"`, `"high"` | Forward the selected effort and set `enable_thinking: true` |
| `"minimal"`, `"xhigh"`, `"max"` | Forward unchanged and enable thinking; requires a server/model version supporting that level |
| `null` or an empty string | Omit automatic effort and thinking fields; use server/model defaults |

Each profile can override an inherited effort, including `null` to clear it for
a local backend. Local values are normalized to lowercase and validated before
any request. Explicit `chat_template_kwargs.enable_thinking` takes precedence
over the derived switch; other template options are preserved. For example,
`reasoning_effort: high` with `enable_thinking: false` sends both fields but keeps
the explicit switch false. Leave the switch unset when effort should control it.
See the [vLLM reasoning protocol](https://docs.vllm.ai/en/latest/features/reasoning_outputs/#automatic-enable_thinking-activation)
and [SGLang request protocol](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/openai/protocol.py).

**Qwen3.5 limitation:** its standard template implements a thinking switch, so
`low`/`medium`/`high` all enable thinking; forwarding these labels does not create
different token budgets. Graded behavior depends on the served model and template.
The client does not invent an effort-to-token mapping. Configure the appropriate
reasoning parser on the server when enabling thinking (Qwen uses
`--reasoning-parser qwen3`). The prompt text is unchanged, and only final response
content is parsed as the existing JSON result. See the
[official Qwen3.5 template](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/chat_template.jinja).

Only explicit `vllm`/`sglang` profiles may omit an API key, in which case the SDK
receives an `EMPTY` placeholder. Real server authentication still requires its key.

`max_tokens: null` preserves the server's output budget; a positive integer sets
the request's total output-token limit (thinking plus final answer, not a separate
thinking budget). Different models and discussion modes can need
different budgets. These options do not change forecast models or select a device.

`max_concurrent_requests` remains the per-client limit. Optional
`max_concurrent_requests_total` additionally bounds requests to the same normalized
server URL across this checkout's processes, including parallel datasets and Agent
profiles. Configure the same total for every profile sharing an endpoint. Set it
under `agent` as a shared default, or in a profile to override that default. The
example leaves the shared limit disabled (`null`). An explicit profile-level
`null` also disables it instead of inheriting the parent value. If omitted from
both levels, only the per-client limit applies. Clients with a null total do not join the shared
budget. URL aliases such as `localhost` and `127.0.0.1`, distinct API paths, separate
checkouts, and remote machines are not combined; use a common URL and server-side
admission limits when those also need one budget.

Request slots use OS locks under `.cache/llm_request_limits/` and are released on
completion, cancellation, errors, or process exit. Waiting does not block the async
event loop. Changing a total is allowed while that endpoint is idle; conflicting
totals while requests are active produce a configuration error. This directory
stores only endpoint hashes and limit metadata, not prompts or KV cache content.

Enable caching and reporting in the inference server as supported by its installed
version: vLLM provides `--enable-prefix-caching` and
`--enable-prompt-tokens-details`; SGLang uses Radix caching (keep
`--disable-radix-cache` unset) and `--enable-cache-report`. See the
[vLLM arguments](https://docs.vllm.ai/en/latest/cli/serve/) and
[SGLang arguments](https://docs.sglang.io/docs/advanced_features/server_arguments).
Restarting the Python API client does not clear the server's KV cache.

Agent usage JSON and the step/attempt usage CSVs now include `cached_tokens`,
`cache_hit_ratio`, and `cache_usage_complete`. The ratio is cached input tokens
divided by input tokens (0 to 1), pooled across calls; it is not a request hit rate.
Missing or invalid cache counts remain unknown. Partial results expose
`known_cached_tokens` and `cache_reported_call_count`, while the full count/ratio
remain null. Calls with malformed-provider-response retries are marked incomplete
because the earlier attempts may have unreported usage. Existing prompt, completion,
and total token counts retain their meanings. The dataset cache is independent of
this server-side prefix cache.

## Native Anthropic / Claude endpoint

Set `backend: anthropic` to use the Anthropic SDK and Messages API independently
of the OpenAI-compatible client. This works in either Agent profile, including
mixed providers for single/multi-Agent experiments. For example:

```yaml
agent:
  single_agent:
    backend: anthropic
    api_key: "${ANTHROPIC_API_KEY}"
    base_url: "https://api.anthropic.com"
    model: "claude-sonnet-4-6"
    reasoning_effort: medium
    max_tokens: 8192
    chat_template_kwargs: {}
    anthropic_thinking: null
    anthropic_cache_control: {type: ephemeral}
```

Use native Claude model IDs and credentials for the selected endpoint. Explicit
`anthropic` selection is required: `auto` keeps its existing routing. Native
defaults when omitted are `https://api.anthropic.com`, `claude-sonnet-4-6`, and
an application-selected output cap of 8192 when `max_tokens` is null. The cap is
required by the Messages API and covers thinking plus the final answer. A trailing
`/v1` is removed from the configured base URL because the SDK appends
`/v1/messages`; custom gateway path prefixes are retained. Existing active
profiles are not switched automatically.

For this backend, `reasoning_effort: none` requests `thinking.type: disabled`;
`low`, `medium`, `high`, `xhigh`, or `max` requests adaptive thinking and forwards
the level in `output_config.effort`. `null`/blank leaves both unspecified.
Supported modes/levels depend on the chosen Claude model; older models may not
support adaptive thinking, and some newer models cannot disable thinking. An
explicit `anthropic_thinking` overrides the derived mode. To use a model's manual
budget mode, set `reasoning_effort: null` and
`anthropic_thinking: {type: enabled, budget_tokens: 2048}`. The budget must be at
least 1024 and less than `max_tokens`. See
[Anthropic effort](https://platform.claude.com/docs/en/build-with-claude/effort).

The adapter uses the installed Anthropic SDK 2.x API: it sends the system prompt
as `system` and converts `response_format: {type: json_object}` into native
`output_config.format: {type: json_schema, schema: ...}`. Agent calls automatically
select the fixed schema for Grid, Behaviour, pricing, single Agent, or discussion;
schema repairs and price retries use the pricing contract. All objects declare
their fields and set `additionalProperties: false`. No extra config setting is
needed. Schemas do not embed Zone IDs, timestamps, window counts, or discussion
round numbers, so the same stage can reuse its schema across experiments.

Native `format` and `effort` share `output_config`; thinking and cache settings
remain effective. The adapter retains the existing JSON parser, schema-repair
flow, and business validation of window identities, counts, and prices. It reads
only final text blocks, never thinking blocks. `temperature`, OpenAI
`response_format`, and local `chat_template_kwargs` are not sent as native request
parameters. Nonempty `chat_template_kwargs` is rejected; use the native thinking
option instead. A truncated (`max_tokens`) response raises an explicit error. See
the [SDK migration](https://platform.claude.com/docs/en/api/migration-2026-10-21)
and [structured-output requirements](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).

Custom code calling `AgentChatClient.complete_json` directly must supply a concrete
schema for native JSON-object mode, because `json_object` alone does not describe
the expected fields:

```python
result = await client.complete_json(
    prompt,
    temperature=0.2,
    response_format={"type": "json_object"},
    output_schema={
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    },
)
```

An explicit OpenAI-style `response_format` with `type: json_schema` and
`json_schema.schema` is also converted. The SDK's `transform_schema` adapts
unsupported constraints (such as numeric bounds) into descriptions; callers
should validate those constraints themselves. Missing schemas in direct native
JSON-object calls fail before sending a request. The ordinary Agent pipeline
supplies schemas automatically, while other backends keep their existing JSON
mode.

`anthropic_cache_control: null` sends no cache option; `{type: ephemeral}` enables
native automatic caching (optional `ttl: 5m` or `1h`). Cache hits depend on the
service's prefix/block matching and retention rules. Recorded input tokens equal
`input_tokens + cache_creation_input_tokens + cache_read_input_tokens`; only
`cache_read_input_tokens` counts as `cached_tokens`. Cache writes are input, not
hits. Missing fields remain unknown instead of becoming zero; complete counts
flow into the existing token, cache and evaluation summaries. See
[Anthropic cache accounting](https://platform.claude.com/docs/en/build-with-claude/prompt-caching#tracking-cache-performance).

Native SDK-internal retries are disabled so every attempt uses the existing
concurrency limit. `provider_json_retries` also controls native connection,
timeout, 408/409/429 and server-error retries, using the configured backoff.
Successful calls after a retry retain known response usage but mark complete
token/cache totals as unavailable because failed attempts may have consumed
unreported tokens. No native SDK or model request is loaded by offline evaluation.

## Offline training for each Zone

Each selected Zone has its own LSTM and XGBoost. The pipeline checks training
eligibility before selecting Zones, then prepares weights only for that selection.
Training uses each Zone's available historical series independently of forecast
origins, alpha values, Agent modes, and experiment seeds. Explicit training runs use:

```powershell
python main.py --stage train --forecast-models lstm xgboost
```

Forecaster and Full stages prepare missing or stale models before generating
forecasts. Agent-only requires the saved forecast artifact and its referenced
weights. `--force-train` creates a new immutable version; existing references
continue to identify their original model. Model IDs depend on the Zone's
training content, resolved training parameters, and input/output window lengths. Updating another Zone or
changing the forecast matrix does not invalidate an unchanged model.
The preparation stage rebuilds missing or corrupt caches into a fresh version and
records the recovery reason. Agent-only replay remains bound to the saved version
and fails if that version is missing or damaged.

`run.history_days` determines the input window H and `run.horizon_days` the
direct output window P, both in hours. Defaults are 720 input hours and 24 output
hours. With stride 1, N consecutive hourly observations yield N-H-P+1 complete
samples. Every input and output hour must exist; windows never cross gaps or pad
missing loads. A Zone needs at least 45 days of valid history and 512 complete
samples, so default windows require at least 1,255 consecutive hours. Duplicate
timestamps, insufficient samples, and constant load or price are rejected.

Complete samples are ordered chronologically and split using `train_ratio: 0.8`
and `validation_ratio: 0.2`. Training samples whose output labels reach the first
validation target are purged. Both ratios must be positive and sum to 1;
for example, `--training-train-ratio 0.7 --training-validation-ratio 0.3`.
Reports retain requested ratios, actual counts,
and purged counts. Preprocessing uses only the training prefix. Each model fits
once, stops early if validation no longer improves, and saves its best weights
or trees. It is not refitted on validation data. This internal split is separate
from the forecast's five-day bias calibration.

Defaults are fixed: LSTM hidden size 32, one layer, batch size 64, learning rate
0.001, and at most 100 epochs; XGBoost depth 3, learning rate 0.03, and at most
800 trees per output hour. LSTM produces P values in one forward pass; XGBoost
saves P independent regressors. Batch size controls LSTM batches. XGBoost uses
`xgboost_n_estimators` and `xgboost_learning_rate`, without an epoch or batch setting.

Training settings belong in a separate top-level section; old `run.lstm_*` and
`run.xgboost_*` training fields are rejected. Plural per-model alpha lists remain
under `run`. Override precedence is CLI, Zone, dataset, global defaults, then
built-in defaults. Explicit training seeds are independent of experiment seeds.

```yaml
training:
  train_ratio: 0.8
  validation_ratio: 0.2
  min_history_days: 45
  min_samples: 512
  sample_stride_hours: 1
  seed: 42
  lstm_workers: 1
  xgboost_workers: 2
  defaults:
    lstm_device: cuda
    lstm_hidden_size: 32
    lstm_num_layers: 1
    lstm_batch_size: 64
    lstm_learning_rate: 0.001
    lstm_epochs: 100
    lstm_early_stopping_patience: 10
    xgboost_device: cuda
    xgboost_n_jobs: 2
    xgboost_n_estimators: 800
    xgboost_max_depth: 3
    xgboost_learning_rate: 0.03
    xgboost_early_stopping_rounds: 50
  dataset_overrides:
    data/CHARGED/JHB:
      lstm_hidden_size: 32
  zone_overrides:
    data/MP-EVData:
      A10:
        lstm_batch_size: 32
```

LSTM and XGBoost default to CUDA for offline training and inference. Use
`cuda:0` to select a specific GPU, or `--lstm-device cpu --xgboost-device cpu`
to explicitly use CPU. Device settings are part of the training fingerprint, so
changing the device creates or selects the corresponding cached model version.

`lstm_workers` and `xgboost_workers` control concurrent Zone model training per
dataset and backend. They accept any positive integer, with defaults of 1 and 2.
There is no global training lock across datasets: up to `max_parallel_datasets`
datasets can each use their configured workers. XGBoost uses two host threads
per model by default. CLI overrides remain `--training-lstm-workers` and
`--training-xgboost-workers`.

Offline fitting and early-stopping validation can include dates later than a historical forecast origin.
The forecast is still generated without reading future observations as inputs,
but it is not a strict out-of-sample backtest when its labels influenced fitting or early stopping.
Results retain the exact model ID, training range, and per-hour training overlap;
offline evaluation keeps distinct model versions separate and reports overlap.
Zero overlap only means the evaluated hours were absent from fitted data; it does
not establish a temporal holdout if the model used later observations.
`evaluation_is_out_of_sample` requires the forecast origin to be at or after the
recorded end of training. Agent, window, usage and paired summaries also separate
model versions; the overall Agent inventory explicitly spans all versions.

## Forecast windows and daily bias calibration

All five backends use the same default forecast split: 30 days of history,
5 days of validation, and 1 day of forecasting. The lengths remain configurable.
For forecast origin `T`, the default history window is `[T-35d, T-5d)`,
validation is `[T-5d, T)`, and the forecast is `[T, T+1d)`. LSTM/XGBoost input
length equals `history_days`; TimesFM/Chronos retain their own context settings.

LSTM/XGBoost load the Zone's offline model without fitting at the forecast origin.
AR estimates its price response from the history window. Validation evaluates five
separate 24-hour predictions. LSTM/XGBoost predict the full configured output
window each time and use its first 24 hours for calibration. At each validation origin, context consists
of the most recent `history_days` of already observed load; observations from
the current or later validation day cannot enter its forecast input. Saved weights,
scalers, the AR price response, and the history diurnal profile remain frozen.

For each validation day, model predictions are blended once with the history
diurnal profile. Their signed errors (`blended prediction - actual`) are averaged
across validation days at each of the 24 horizon offsets. The final prediction
is `max(0, blended prediction - bias[offset % 24])`. Thus the default produces
24 biases from all 120 validation hours, rather than one scalar bias or a bias
from only the first validation day. Saved calibration records include daily
predictions, hourly residuals, and the averaged bias vector.

Incomplete or invalid validation load is an error. `validation_days: 0` disables
calibration. A longer configured forecast repeats the 24-hour bias vector; it
never consumes forecast-period actual load. LSTM and XGBoost directly predict
the entire configured horizon. The old `roll_actuals` and per-model
`step_horizon` settings have been removed.

Configuration selects dates and models through plural keys, even for one run:

```yaml
run:
  history_days: 30
  validation_days: 5
  horizon_days: 1
  forecast_starts: null
  forecast_start_count: 2
  forecast_models: [timesfm, chronos, lstm, xgboost, AR]
  diurnal_blend_alphas: [0.0, 0.5]
  timesfm_diurnal_blend_alphas: null
  chronos_diurnal_blend_alphas: null
  lstm_diurnal_blend_alphas: [0.0, 0.25]
  xgboost_diurnal_blend_alphas: [0.0]
  ar_diurnal_blend_alphas: null
```

A non-null per-model alpha list replaces the global list. Missing/null overrides
inherit it; the global default is `[0.0]`. Each setting also accepts one number.
Values must be finite and between 0 and 1; empty lists are rejected. Values are
deduplicated to six decimal places while preserving order. Global and per-model
CLI options use the same plural names with hyphens. CLI overrides its matching
YAML setting before per-model precedence is applied:

```powershell
python main.py --forecast-models lstm xgboost --diurnal-blend-alphas 0 0.5 --lstm-diurnal-blend-alphas 0.25
```

The matrix runs each backend only for its own effective alpha list. Independent
models' alpha lists are not multiplied together. Every output path contains a
`blend_<alpha>` component with the same precision as the configuration, including
single-alpha runs. Single-run artifacts record the resolved scalar alpha and
only that backend's inference parameters; changing an unrelated model's grid
does not change their identity. Forecaster, Full, and Agent-only stages share
the same path convention.

The singular date/model configuration and CLI options are removed, as are the
singular alpha configuration options. Deprecated inputs fail with a migration
message instead of silently falling back. Python single-run APIs and new output
fields use `forecast_origin` and `backend`. Offline evaluation accepts legacy
field names at its reading boundary and writes the new names. Model artifacts
use the new protocol and do not replay legacy calibration states.

## XGBoost forecaster

Install the updated dependencies (`uv sync` or `pip install -r requirements.txt`).
Select `run.forecast_models: [xgboost]`, or include `xgboost` alongside other
backends for a model comparison. This CLI override selects only XGBoost:

```powershell
python main.py --config config.yaml --forecast-models xgboost --pipeline-stage forecaster
```

`xgboost` (alias `xgb`) trains one `XGBRegressor` for each output hour per selected
Zone in the offline phase, with early stopping on internal validation.
Defaults use learning rate 0.03, column sampling 0.9, `tree_method: hist`, and `device: cuda`.
Training overrides are listed in `config.example.yaml`; global and independent
diurnal-blend sweeps reuse the same saved model.

Inputs contain all H historical loads, recent load summaries, hourly/weekly calendar
cycles, and the P hours of configured known covariates. Load normalization is
fitted only on the training prefix, and outputs are converted back to kWh.
Missing weather and other covariates remain NaN with missingness indicators. Target columns
cannot be selected as covariates. Future prices/weather must be available at
the forecast origin, as with the other covariate forecasters.

The model follows the shared daily bias-calibration protocol above. All P output
hours are predicted directly; blending and correction happen once
on the forecast output. The q10/q90 columns follow the empirical residual-width
convention and are not learned quantiles. With no validation, bias is zero and
interval bounds are unavailable.

The weight cache stores the P regressors in a bundle using XGBoost's
[JSON model format](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html),
alongside preprocessing parameters. The forecast artifact stores calibration and
an immutable model reference. Full and Agent-only runs reuse them for each proposed
price schedule without retraining. Load changes depend on the relationships
learned from history; tree predictions can remain unchanged for small price changes.

## Fixed-origin and no leakage

- Forecast contexts, stress thresholds and bias calibration use only observations before `forecast_origin`. Offline LSTM/XGBoost weights and scalers follow their independently recorded training period; overlap is reported for historical evaluations.
- LSTM/XGBoost produce direct multi-step outputs without forecast-period actual load.
- Agent context is constructed from an allowlist. It excludes future actual load/stress/percentile, future error metrics, evaluation nodes, and all derived correctness labels.
- Grid Agent assesses the forecaster values but cannot overwrite them.
- Only structured reasoning summaries are persisted; hidden chain-of-thought is neither requested nor stored.

## Historical price diagnostic

For every Zone, the adapter computes mean energy price in consecutive non-overlapping 3-hour windows, then pools adjacent-window absolute percentage changes across the whole dataset. Every proposal reports its empirical percentile, dataset P95, and `exceeds_historical_p95`. This diagnostic is display-only: it does not clip prices, determine success, or enter Agent context.

## Outputs

The authoritative control artifact is `control_results.json` (schema version 4). It records the dataset fingerprint, split cache key, feature manifest, forecast origin/model, experiment mode, global and Zone status, attempts, structured Agent summaries, reforecast trace, final window prices/load positions/stress, rationale, and historical price diagnostics. Agent time is not measured or emitted. Every concrete Grid, Behaviour, Economist, schema-repair, and retry call records prompt, completion, and total tokens. Attempt-level summaries and the final `agent_token_totals` provide Zone and global totals. Missing provider usage remains explicitly incomplete instead of being estimated.

Companion outputs are:

- `control_final_windows.csv`
- `control_attempt_trace.csv`, with every Zone/window/attempt price transition, including frozen windows and previous-to-current price/shift deltas
- `agent_attempt_usage.csv`, with separate `global` and `zone` rows so token totals are not repeated for each window
- `agent_step_token_usage.csv`, with one row per concrete Agent/provider call and its prompt/completion/total token usage
- `control_experiment_summary.csv`
- `forecaster/forecaster_artifact.json` for refit-free Agent-only handoff. It stores
  backend-specific inference state and known-future covariates; every proposed price
  schedule is sent back through the original TimesFM, Chronos, LSTM, XGBoost, or AR forecast
  function. The control loop does not use a separate price-to-load approximation.
- `forecaster/context_snippets.json` for the no-leakage Agent handoff
- forecast evaluation CSV/plots, separate from `control_results.json`

Experiment matrices additionally aggregate these records into `experiment_agent_attempt_usage.csv`, `experiment_agent_step_token_usage.csv`, and `experiment_control_attempt_trace.csv`.

`run.max_dataset_errors` sets the cumulative exception limit for each dataset's
experiment matrix. It must be a positive integer and defaults to `3` when omitted:
the first two failed combinations are skipped, and the third stops that dataset.
Every failed combination is recorded in `experiment_runs.csv`
with its error details and cumulative `dataset_error_count`, plus a traceback in
`errors/`. The counter starts at zero for each dataset invocation, spans all dates,
models, seeds, modes, and blend values, and is not reset by successful combinations.
Other dataset processes continue even when one reaches the limit. Configuration
or zone-selection errors before the matrix starts still stop that dataset immediately.
An ordinary control outcome of `fail` is not an exception and does not increment
this counter.

The run is `success` only when every selected Zone and every 3-hour window is `Medium`; otherwise the third proposal produces `fail` with the final summaries and reforecast state.

## Offline result analysis

`evaluate.py` reads existing outputs independently of forecasting and control.
It supports a single run or recursively scans an experiment matrix:

```powershell
python evaluate.py --input output/my_experiment
python evaluate.py --input output/my_experiment --section forecaster
python evaluate.py --input output/my_experiment --section agent --output output/my_analysis
```

The default destination is `<input>/analysis/`. `evaluation.json` includes metric
definitions, coverage, per-run/Zone/window details, and the optional LLM analysis.
CSV tables are `forecaster_global_metrics.csv`, `forecaster_run_metrics.csv`,
`agent_summary.csv`, and `agent_run_metrics.csv` (only populated sections are written).
The evaluator also writes `evaluation_report_zh.md`, a deterministic Chinese report,
and the four PPT-style tables `ppt_forecast.csv`, `ppt_medium.csv`,
`ppt_first_entries.csv`, and `ppt_tokens.csv`. It does not generate plots; the
existing optional LLM analyzer remains opt-in.

To compare multiple experiment batches, use unique labels and a separate output:

```powershell
python evaluate.py --batch 27B=output/random5_start2_qwen3_5 --batch 9B=output/random5_start2_qwen3_5_9b_single --output output/evaluate_qwen_comparison
```

The equivalent Python entry point is
`evaluate_experiments({"27B": first_directory, "9B": second_directory})`.
Labels are supplied by the caller; they do not verify the actual model used by
old runs that did not record its ID. Input batches may not overlap. Mode tables
use complete common runs within each batch; pairwise comparisons use the same
Agent mode and check dataset identity, forecast parameters, regions, timestamps,
baseline prices/loads/thresholds, and complete hourly observations/predictions
(including original hourly energy prices when saved).
`comparison_audit.csv` explains missing, ambiguous, unknown and inconsistent
matches. Full-population tables remain available separately.

`window_round_details.csv` records baseline, each outer control round, and final
states. `new` means originally non-Medium and currently Medium; adjacent entries,
exits, first entries and final losses are separate metrics. Round count is dynamic.
An explicitly successful early stop carries its final outcome with zero further
calls; missing intermediate observations are unknown. Internal discussion rounds
are not additional observed control outcomes.

Revenue is an independent, predicted electricity-income metric with two baselines:

- `baseline_revenue`: original `mean_energy_price * sum_predicted_kwh`, the existing
  window-mean-price approximation.
- `baseline_hourly_revenue`: `sum(e_price * predicted_kwh)` over the original hourly
  forecast records in that window. Both window endpoints are included. This uses
  predicted load, not `actual_kwh`, and requires complete, unique, valid hourly
  records. Missing hourly prices or loads leave this baseline unknown.

Each control round uses the actual `proposed_energy_price` times its reforecast
total kWh. `revenue_change` / `revenue_change_pct` retain the window baseline;
`revenue_change_vs_hourly` / `revenue_change_vs_hourly_pct` use the hourly baseline.
Each percentage divides by its own baseline. The `baseline` phase keeps the
window-mean revenue, so its hourly-relative change shows the difference between
the two baseline calculations. Directory evaluation loads hourly CSVs automatically;
direct callers can pass `evaluate_agent(control_result, baseline_hourly=hourly)`
with a DataFrame or record iterable containing `zone_id`, `time`, `e_price` and
`predicted_kwh`.

Do not multiply by three again, infer price from the declared percentage, include
service fees/costs, or add alternative rounds together.
`revenue_by_dataset_model_mode.csv` and finer tables preserve source
currency and never pool different datasets/currencies. Missing/invalid values
keep each full baseline/revenue unknown while exposing known subtotals and coverage; a zero
baseline has undefined relative change. Existing Medium success rules are unchanged.

`call_details.csv`, `usage_per_run_round.csv`, and grouped `usage_by_*.csv` include
input/output/total tokens, per-call means, medians/P95 where individual calls are
complete, repair/role/discussion breakdowns, and cached input tokens. Cache fields
are `cached_tokens`, `known_cached_tokens`, `cache_reported_call_count`,
`cache_usage_complete`, and `cache_hit_ratio`, plus mean cached tokens per call.
The cache ratio is pooled cached input tokens divided by pooled input tokens;
cached tokens are already included in input/total counts. Missing cache reports
are unknown, not zero. Partial call details retain known distributions with
coverage; provider retries do not invent unseen HTTP-request token distributions.
`usage_audit.csv` compares call, round and cumulative records without adding the
same usage more than once. Old files without cache reporting cannot recover those
counts retrospectively.

Forecaster MAE, RMSE, RAE, MAPE and WAPE are calculated from pooled hourly samples
across all Zones, not averaged from Zone metrics. Global comparisons are grouped
by model and diurnal blend; identical forecast samples copied across Agent modes
are deduplicated using dataset/forecast provenance. MAPE and WAPE use percentages
(`MAPE_pct`, `WAPE_pct`); RAE is a ratio using the pooled actual mean. Non-finite
pairs are excluded with coverage counts, zero actuals are excluded from MAPE,
and undefined denominators produce null.

Agent tables report first-proposal and final success counts/rates at run, Zone,
and 3-hour-window levels, plus retained/gained/lost/never-success transitions.
Success uses the recorded control outcome after reforecast, not observed demand.
Missing first-round observations remain unknown. Calls and proposal rounds are
separate: averages include failed runs and frozen Zones. Prompt/completion/total
tokens are available per run, Zone and call; incomplete provider usage exposes
known totals but leaves full totals and averages null. Global resource totals
are counted once from `agent_cumulative_usage`.

For future LLM review, pass `analyzer=callback` to `evaluate_agent` or
`evaluate_directory`, or use `--agent-analyzer my_module:analyze`. The callable
receives an independent dictionary with `control_result` and
`quantitative_analysis`, and returns a JSON-serializable dictionary. By default,
the hook is `not_configured` and makes no model calls. Its output is stored only
in the evaluation report and must not enter control Agent context.

## Tests

```powershell
python -B -m unittest discover -s tests -v
```

The tests cover dataset formats and filtering, cache consistency and process locks,
config-only switching and parallel dispatch, token accounting and evaluation,
fixed-origin prediction without future-load leakage, and fitted-state reuse.
TimesFM/Chronos/LSTM control-flow tests use deterministic backend doubles so they
run without downloading model weights or calling provider APIs.
See [shared code and regression validation](docs/refactoring.md) for the refactor boundaries.
