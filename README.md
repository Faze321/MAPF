# Multi-Agent Prescriptive Forecasting (MAPF)

This project implements the conference-work requirement in `Conference work.docx`:

1. Select five UrbanEV zones that behave like CBD/office, residential, transport hub, commercial/mall, and industrial demand profiles.
2. Build compact context snippets from `volume.csv`, a configurable weather file, `poi.csv`, `inf.csv`, `occupancy.csv`, `e_price.csv`, and `s_price.csv`.
3. Run a sequential multi-agent chain per zone:
   - Grid Analyst: forecast 1-4 days of load and assign a grid stress level.
   - Behavioural Agent: explain the demand drivers from POI mix, weather, time markers, and estimate a price-response elasticity factor.
   - Market Economist: prescribe an energy-price shift from stress, elasticity, and price context.
   - Price-conditioned Forecaster: rerun the forecaster with predicted energy prices plus observed weather, occupancy, and service-price conditions to estimate the current load baseline.
   - Nash Equilibrium Check: test whether each price-conditioned baseline can be moved into the Medium load band while keeping energy price between 0.4 and 2.0 times the zone mean energy price.
4. Execute all five zone chains concurrently with `asyncio`.
5. Export an explainability table with predicted vs. actual load, rationale, price shift, and Nash equilibrium status.

The model call path uses the OpenAI Python SDK with an OpenAI-compatible `base_url`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For local validation without an API key:

```powershell
Copy-Item config.example.yaml config.yaml
# Edit config.yaml run.zones / run.horizon_days / run.dry_run as needed
python main.py
```

For model-backed agents:

```powershell
Copy-Item config.example.yaml config.yaml
# Edit config.yaml and set agent.api_key / agent.model / agent.base_url
# Optional: set agent.single_model_model to use a stronger model only for --agent-mode single_model
# Set run.dry_run: false
python main.py
```

Useful options:

```powershell
python main.py
python main.py --dry-run --horizon-days 4 --history-days 7
python main.py --dry-run --zones 102 --horizon-days 1
python main.py --dry-run --zones 102,104,108 --horizon-days 1
python main.py --dry-run --zones 102 --weather-file weather_central.csv --forecast-start "2022-09-09 00:00:00" --horizon-days 6
python main.py --dry-run --forecast-model chronos
python main.py --dry-run --forecast-model lstm
python main.py --dry-run --forecast-model AR
python main.py --agent-mode agents_no_nash --zones 102 --forecast-model timesfm
python main.py --agent-mode single_model --model openai/gpt-4.1 --zones 102 --forecast-model timesfm
python main.py --zones 102 105 --forecast-starts "2022-09-09 00:00:00" "2022-10-14 00:00:00" "2022-12-16 00:00:00" "2023-02-24 00:00:00" --forecast-models timesfm chronos lstm AR
python main.py --forecast-starts "2022-09-09 00:00:00" "2022-10-14 00:00:00" --forecast-models chronos lstm --experiment-zone-count 12 --experiment-seeds 7 42 99
python main.py --forecast-starts "2022-09-09 00:00:00" --forecast-models chronos lstm --agent-modes agents rules --diurnal-blend-alphas 0.0 0.3 0.6
python main.py --config config.yaml --model anthropic/claude-sonnet-4.5 --forecast-start "2023-02-25 00:00:00"
python main.py --force-cache
python main.py --precomputed-window-data "output/timesfm/window_load_price_cache.csv"
```

Runtime defaults can be stored under `run:` in `config.yaml`, so common settings do not need to be typed each time. Command-line options override YAML values only for that run. When `run.zones` / `--zones` is omitted, the pipeline keeps the original five-category automatic zone selection. When zones are provided, the pipeline skips category selection and validates only the specified zone ids.

For broader experimental validation, set `run.forecast_starts` and `run.forecast_models`, or pass `--forecast-starts` and `--forecast-models`. If `run.zones` / `--zones` is omitted in an experiment matrix, the runner selects a representative zone set from the cached zone profiles. `run.experiment_zone_count` / `--experiment-zone-count` controls the size of that set; the example config uses 12 zones instead of the earlier two-zone quick check.

Experiment matrices can also sweep `run.experiment_seeds`, `run.agent_modes`, and `run.diurnal_blend_alphas`. Seeds are recorded for every model and passed into LSTM training. `agent_modes` supports `agents` for the full three-stage LLM chain with Nash equilibrium, `agents_no_nash` for the same chain without the Nash equilibrium adjustment, `single_model` for one stronger model call that replaces the Grid/Behaviour/Economist calls, and `rules` for the deterministic rule-only pricing baseline. This lets `price_accuracy`, price bias, `stress_accuracy`, and `miss_stress_rate` be compared directly across ablations. `diurnal_blend_alphas` applies the same daily-shape blend weight to TimesFM, Chronos, LSTM, and AR for fair blend ablations.

Runs are fail-fast. The first exception stops the current single run or experiment matrix immediately, prints the error code, stage, zone, agent, exception type, and original reason, and exits with status 1. Experiment matrices still save the failed row in `experiment_runs.csv` and a traceback under the experiment `errors/` directory before stopping.

For `single_model`, configure a dedicated model under `agent.single_model_model`. The normal three-stage `agents` mode continues to use `agent.model`; `--model` still overrides the active model for one command-line run.

```yaml
agent:
  model: "meta-llama/llama-3.1-8b-instruct"
  single_model_model: "meta-llama/llama-3.3-70b-instruct"
  reasoning_effort: "none"
```

`agent.reasoning_effort: "none"` disables model thinking for JSON-mode requests. This is required by Qwen3 providers that reject `response_format: json_object` while thinking is enabled. The client omits this parameter automatically for `meta-llama/*` models, which do not use Qwen3-style switchable thinking.

## Load And Pricing Policy

For each 3-hour window, the load percentage is:

```text
load_range_position_pct =
    (predicted_3h_load_kwh - historical_min_load_3h_kwh)
    / (historical_max_load_3h_kwh - historical_min_load_3h_kwh)
    * 100
```

The historical load interval is built from the minimum and maximum of all 3-hour load sums in `volume.csv` for that zone. The stress labels use fixed positions within this interval: below 35% is `Low`, 35% to below 80% is `Medium`, 80% to below 90% is `High`, and 90% or above is `Extremely High`.

The pricing controller targets the interior of the Medium band. For Low load it reduces energy price and targets the 35.5% position. For Medium load it permits at most a small +/-3% energy-price change, and rejects that change if it would move expected load outside Medium. For High and Extremely High load it increases energy price and targets the 79.5% position. The zone mean energy price is calculated from all positive `e_price` values in that zone. Every final energy price is clamped to `[0.4 * zone_mean_energy_price, 2.0 * zone_mean_energy_price]`; `s_price` remains the observed actual service price.

Multi-value runs execute the full forecast-start by forecaster by seed by agent-mode by blend matrix while keeping each combination in a separate folder under `output/experiments/`, for example `output/experiments/20zonesx4timesx3modes_11blends/2022-09-09_000000/seed_42/agent_rules/blend_0_3/lstm/`. The experiment root writes raw concatenated tables plus aggregate summaries with `n`, `mean`, `std`, and `sem`: `experiment_forecast_summary.csv`, `experiment_price_summary.csv`, `experiment_rationale_summary.csv`, and `experiment_decision_quality_summary.csv`.

Set `run.forecast_model: "timesfm"` to use `google/timesfm-2.5-200m-pytorch` for load forecasting. Set `run.forecast_model: "lstm"` to train a small local PyTorch LSTM per zone. Set `run.forecast_model: "AR"` for a fast autoregressive baseline run without TimesFM; AR applies a calibrated energy-price response adjustment when `e_price` is available.
Set `run.forecast_model: "chronos"` to use Chronos. The default Chronos config uses `amazon/chronos-2`, passes known future covariates including `e_price` and actual `s_price`, rolls actual observations into the context during retrospective multi-day evaluation, and exports the same `predicted_kwh`, `q10_kwh`, `q50_kwh`, and `q90_kwh` columns as TimesFM.

The TimesFM path now follows the `zone102_timefm1.ipynb` workflow:

- `run.weather_file` chooses the weather source. Use `weather_central.csv` to match `zone102_timefm1.ipynb`; the default project path uses `weather_airport.csv`.
- `run.history_days: 7` builds the context window.
- `run.validation_days: 1` reserves the day before `forecast_start` for bias calibration.
- `run.timesfm_exog_cols` controls dynamic numerical covariates. The notebook-style default is `T`, `U`, `nRAIN`, `e_price`, `s_price`, `is_weekend`, and `temp_price_idx`. The post-price baseline forecast forces predicted `e_price` into the TimesFM, LSTM, and Chronos covariates while retaining actual `s_price`; AR uses the predicted `e_price` through its local energy-price response adjustment.
- `run.timesfm_diurnal_blend_alpha` blends the TimesFM point forecast with the recent hourly load profile. `1.0` matches the notebook setting; `0.0` disables the blend.
- `run.timesfm_roll_actuals: true` rolls known actual values into the context during multi-day validation/forecast steps.

The same daily-shape blend is available for the other forecasting methods through `run.chronos_diurnal_blend_alpha`, `run.lstm_diurnal_blend_alpha`, and `run.ar_diurnal_blend_alpha`. Their defaults are `0.0`, so existing Chronos, LSTM, and AR results do not change unless you opt in.

The first TimesFM run may download model weights from Hugging Face. The dependency list installs TimesFM from the official `google-research/timesfm` repository, plus `torch`, `jax`/`jaxlib`, and `scikit-learn` for the PyTorch model class and covariate regression path.

The LSTM path uses the existing `torch` installation and trains only on the selected zone's history window. `run.lstm_context_hours`, `run.lstm_epochs`, `run.lstm_hidden_size`, and `run.lstm_exog_cols` control the local model size and training setup.

## Outputs

Generated result files are written under a forecast-model subfolder, for example `output/timesfm/`, `output/chronos/`, `output/lstm/`, or `output/AR/`:

- `selected_zones.csv`: the selected zones and the proxy features used for selection.
- `context_snippets.json`: token-efficient context passed to each agent.
- `rationale_trace.csv`: machine-readable explainability table, including zone-level Nash equilibrium status plus `agent_time_cost_seconds`, `agent_prompt_tokens`, `agent_completion_tokens`, and `agent_total_tokens`.
- `rationale_trace.md`: markdown table for a report or paper appendix.
- `rationale_trace.json`: full structured agent outputs, including per-agent call usage details under `agent_call_usage`.
- `agent_debug_outputs.json`: full debug payloads for model-backed agent calls, including economist repair details or single-model responses.
- `price_schedule_3h.csv` / `price_schedule_3h.md`: per-3-hour window price actions, `predicted_energy_price`, `actual_energy_price`, unchanged `actual_service_price`, stress labels, and price rationales. Policy diagnostics include `load_range_position_pct`, the historical 3-hour load range and 35/80/90 boundaries, `zone_mean_energy_price`, `min_allowed_energy_price`, `max_allowed_energy_price`, `energy_price_bound_source`, `target_load_kwh`, `expected_load_range_position_pct`, `load_in_medium_band`, and `price_within_allowed_bounds`. Price-conditioned baseline fields include `forecaster_input_predicted_energy_price`, `price_conditioned_baseline_load_kwh`, `baseline_load_kwh`, and `baseline_load_source`.
- `price_comparison_summary.csv` / `price_comparison_summary.md`: per-zone energy-pricing decision accuracy. A price action passes when `predicted_energy_price` is within 8% of `actual_energy_price`.
- `window_load_price_cache.csv`: stable per-window replay input containing the original load forecast, the load forecast conditioned on predicted energy price, the final predicted energy price, actual energy and service prices, `zone_mean_energy_price`, the 0.4-2.0 allowed bounds, load percentages, and Nash diagnostics.
- `explainability_rubric.md`: five-criterion human evaluation rubric for rationale quality.
- `explainability_review_packet.csv`: review template for two independent raters plus an operator sanity-check column.
- `forecast_metrics.csv` / `forecast_metrics.md`: per-zone forecast metrics including MAE, RMSE, MAPE, RAE, and WAPE.
- `forecast_details/zone_<id>_forecast_vs_actual.csv`: hourly actual vs predicted values, residuals, TimesFM raw/bias-corrected values, and P10/P50/P90 columns when TimesFM returns quantiles.
- `forecast_details/zone_<id>_forecast_plot.png`: per-zone actual/predicted plot, P10-P90 band when available, residual bars, and metric summary.

The first full run builds cached POI-to-zone assignments in `output/cache/`. Later runs reuse that shared cache unless `--force-cache` is passed.

`zone_3h_load_quantiles.csv` keeps its filename for backward compatibility but now stores the historical 3-hour load range policy. Its columns include `load_policy_id`, `historical_min_load_3h_kwh`, `historical_max_load_3h_kwh`, `historical_load_range_3h_kwh`, the three percentage boundaries, and their corresponding kWh thresholds. A legacy Q50/Q80/Q95 file, a different policy id, or stale percentages is automatically invalidated and overwritten on the next run.

Set `run.precomputed_window_data` or pass `--precomputed-window-data` to reuse a previous `window_load_price_cache.csv`. Matching uses `forecast_model`, `agent_mode`, `zone_id`, `window_start`, and `window_end` when the metadata columns are present. A zone is reused only when every expected 3-hour window has both `predicted_energy_price` and `price_conditioned_load_kwh`; a partial or missing zone falls back to the normal agent and price-conditioned forecast path. Reused zones skip the LLM pricing calls and the second load forecast, while the initial hourly load forecast still runs to produce metrics and plots.

## Data Notes

The POI file in this release contains only three broad POI labels: `food and beverage services`, `business and residential`, and `lifestyle services`. The five zone categories are therefore selected as operational proxies:

- CBD / Office: high business/residential density plus morning/noon load shape.
- Residential: night/evening charging plateau with a penalty for food/lifestyle POI dominance.
- Transport Hub: high charging capacity and bursty peaks.
- Commercial / Mall: high food/lifestyle density plus evening/weekend lift.
- Industrial: stable high base load with large charging capacity.

Each feature is median-filled, clipped to its 5th-95th percentile range, and standardized before scoring so a small number of extreme zones cannot dominate a category. The final five zones are selected jointly to maximize the total category score while requiring a different zone for every category.

## Data Folder

**data**: 1-hour resolution zone-level data of the UrbanEV dataset, which has been cleaned through outlier detection, zero-value checks, etc., and includes data from **275 zones**, **1,362 charging stations**, and **17,532 charging piles**.

* `adj.csv`: Adjacency matrix.
* `duration.csv`: Hourly EV charging duration (Unit: hour).
* `e_price.csv`: Electricity price (Unit: Yuan/kWh).
* `inf.csv`: Filtered station-level data for the 275 zones, including coordinates, charging capacities, area (Unit: m^2), and perimeter (Unit: m).
* `inf_raw.csv`: All station-level data for the same 275 zones, including coordinates, charging capacities, area (Unit: m^2), and perimeter (Unit: m).
* `occupancy.csv`: Hourly EV charging occupancy rate (Unit: %).
* `s_price.csv`: Service price (Unit: Yuan/kWh).
* `volume.csv`: Hourly EV charging volume (Unit: kWh). The pipeline uses this file as the project-wide load source. The volume in *volume.csv* is derived from the rated power of charging piles.
* `volume-11kW.csv`: Alternative vehicle-side estimation of charging volume to mitigate potential overestimation in `volume.csv`. It is kept in the data folder for reference, but it is not used by the pipeline by default.
* `weather_airport.csv`: Weather data from the meteorological station at Bao'an Airport (Shenzhen). These are the raw data collected, and it is recommended to use the **Max-Min** method for normalization.
* `weather_central.csv`: Weather data from Futian Meteorological Station in the city center of Shenzhen.
* `weather_header.txt`: Descriptions of the table headers in `weather_airport.csv` and `weather_central.csv`.
* `distance.csv`: Distance matrix between the 275 zones.
* `poi.csv`: Points of Interest categorized into three types: `food and beverage services`, `business and residential`, and `lifestyle services`. The coordinates used are based on the `WGS84` coordinate system.
* Notes: Our occupancy data is gathered from an availability perspective, while the duration and volume data is collected from a utilization standpoint. Specifically, the occupancy data records all unavailable or busy charging piles. In contrast, the duration and volume data only account for the piles actively providing electricity. You can select the data according to your research purpose.
