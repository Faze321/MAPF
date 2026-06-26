import asyncio
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from agents import (
    ECONOMIST_AGENT_OUTPUT_KEY,
    AgentChatClient,
    AgentStageError,
    extract_json_object,
    heuristic_behavior,
    heuristic_economist,
    merge_economist_fallback,
    merge_grid_fallback,
    normalize_price_windows,
    run_zone_chain,
    solve_nash_equilibrium_window,
    validate_economist_report,
)
from config import AgentConfig, AppConfig, RunConfig
from data_loader import available_zone_ids, build_zone_3h_load_quantiles, load_pipeline_data
from forecasting import (
    ar_forecast,
    build_zone_model_frame,
    chronos_forecast,
    compute_forecast_metrics,
    lstm_forecast,
    patch_timesfm_hub_kwargs,
    rebuild_quantile_interval,
)
from orchestrator import (
    apply_load_quantile_stress,
    build_agent_hourly_data,
    build_hourly_averages,
    build_pricing_windows_3h,
    classify_load_stress,
    forecast_output_dir,
    normalize_zone_ids,
    run_experiment_matrix,
    select_representative_zone_ids,
    select_requested_zones,
)
from prompts import compact_economist_context, economist_prompt, grid_prompt
from reporting import (
    build_price_comparison_summary,
    price_comparison_fields,
    split_economist_agent_outputs,
    write_outputs,
)
from zone_selection import select_zone_categories


class AgentParsingTests(unittest.TestCase):
    def test_agent_client_requests_json_response_format(self):
        class FakeCompletions:
            def __init__(self):
                self.kwargs = None

            async def create(self, **kwargs):
                self.kwargs = kwargs

                class Message:
                    content = '{"ok": true}'

                class Choice:
                    message = Message()

                class Response:
                    choices = [Choice()]

                return Response()

        class FakeChat:
            def __init__(self):
                self.completions = FakeCompletions()

        class FakeClient:
            def __init__(self):
                self.chat = FakeChat()

        fake_client = FakeClient()
        client = object.__new__(AgentChatClient)
        client.config = AgentConfig(api_key="sk-test", base_url="https://example.test", model="fake-model")
        client._client = fake_client

        result = asyncio.run(client.complete_json("Return JSON.", temperature=0.2))

        self.assertEqual(result, {"ok": True})
        self.assertEqual(fake_client.chat.completions.kwargs["response_format"], {"type": "json_object"})

    def test_extracts_json_from_markdown_fence(self):
        payload = extract_json_object('```json\n{"a": 1, "b": "x"}\n```')
        self.assertEqual(payload, {"a": 1, "b": "x"})

    def test_grid_stress_level_is_limited_to_known_levels(self):
        context = {
            "category": "Commercial",
            "forecast_total_kwh": 100.0,
            "forecast_peak_kwh": 20.0,
            "predicted_change_pct": 5.0,
            "grid_stress_level": "High",
        }
        self.assertEqual(
            merge_grid_fallback({"grid_stress_level": "Medium"}, context)["grid_stress_level"],
            "Medium",
        )
        self.assertEqual(
            merge_grid_fallback({"grid_stress_level": "Severe"}, context)["grid_stress_level"],
            "High",
        )
        self.assertEqual(
            merge_grid_fallback({"grid_stress_level": "extrame high"}, context)["grid_stress_level"],
            "Extreme High",
        )

    def test_heuristic_economist_returns_three_hour_price_windows(self):
        context = {
            "category": "Commercial",
            "grid_stress_level": "High",
            "predicted_change_pct": 5.0,
            "hourly_averages": {"mean_predicted_kwh": 10.0, "mean_energy_price": 1.0},
            "pricing_windows_3h": [
                {
                    "window_start": "2022-09-09 00:00:00",
                    "window_end": "2022-09-09 02:00:00",
                    "mean_predicted_kwh": 12.0,
                    "mean_energy_price": 1.1,
                },
                {
                    "window_start": "2022-09-09 03:00:00",
                    "window_end": "2022-09-09 05:00:00",
                    "mean_predicted_kwh": 8.0,
                    "mean_energy_price": 0.9,
                },
            ],
        }
        result = heuristic_economist(context, {"grid_stress_level": "High", "predicted_change_pct": 5.0})
        self.assertEqual(len(result["price_change_windows_3h"]), 2)
        self.assertEqual(result["price_change_windows_3h"][0]["window_start"], "2022-09-09 00:00:00")

    def test_normalized_price_windows_mark_missing_model_text_fields(self):
        windows = normalize_price_windows(
            [{"suggested_price_shift_pct": -0.5}],
            [
                {
                    "window_start": "2022-09-09 00:00:00",
                    "window_end": "2022-09-09 02:00:00",
                    "sum_predicted_kwh": 279.1,
                    "load_stress_level": "High",
                }
            ],
        )
        self.assertEqual(windows[0]["action_label"], "MODEL_RESPONSE_FAILED: missing action_label")
        self.assertEqual(windows[0]["price_rationale"], "MODEL_RESPONSE_FAILED: missing price_rationale")

    def test_economist_merge_does_not_hide_missing_window_response(self):
        context = {
            "category": "Commercial",
            "forecast_total_kwh": 100.0,
            "forecast_peak_kwh": 20.0,
            "predicted_change_pct": 5.0,
            "grid_stress_level": "High",
            "hourly_averages": {"mean_predicted_kwh": 10.0, "mean_energy_price": 1.0},
            "pricing_windows_3h": [
                {
                    "window_start": "2022-09-09 00:00:00",
                    "window_end": "2022-09-09 02:00:00",
                    "sum_predicted_kwh": 100.0,
                    "load_stress_level": "High",
                }
            ],
        }
        merged = merge_economist_fallback({}, context)
        self.assertEqual(merged["price_change_windows_3h"], [])
        windows = normalize_price_windows(merged["price_change_windows_3h"], context["pricing_windows_3h"])
        self.assertEqual(windows[0]["action_label"], "MODEL_RESPONSE_FAILED: missing price_change_windows_3h item")

    def test_economist_validation_finds_missing_window_fields(self):
        errors = validate_economist_report(
            {
                "suggested_price_shift_pct": 5,
                "action_label": "Raise",
                "price_rationale": "High demand",
                "price_change_windows_3h": [{"window_start": "2022-09-09 00:00:00"}],
            },
            [
                {
                    "window_start": "2022-09-09 00:00:00",
                    "window_end": "2022-09-09 02:00:00",
                }
            ],
        )
        self.assertIn("price_change_windows_3h[0] missing window_end", errors)
        self.assertIn("price_change_windows_3h[0] missing action_label", errors)

    def test_nash_equilibrium_marks_unresolved_grid_constraint(self):
        window = solve_nash_equilibrium_window(
            {
                "category": "Residential",
                "capacity_kw_proxy": 200.0,
            },
            {},
            {
                "window_start": "2022-09-09 18:00:00",
                "window_end": "2022-09-09 20:00:00",
                "hours": 3,
                "sum_predicted_kwh": 120.0,
                "mean_service_price": 1.0,
                "mean_energy_price": 0.8,
                "mean_occupancy": 0.9,
                "total_rain": 1.0,
                "load_3h_q95_kwh": 90.0,
                "suggested_price_shift_pct": 5,
                "action_label": "Raise price",
                "price_rationale": "High predicted demand.",
            },
        )

        self.assertFalse(window["nash_equilibrium_reached"])
        self.assertEqual(window["capacity_limit_source"], "historical_load_quantile")
        self.assertAlmostEqual(window["target_peak_reduction_pct"], 25.0)
        self.assertLessEqual(window["suggested_price_shift_pct"], window["max_discomfort_score"] * 15.0)
        self.assertFalse(window["grid_safe"])
        self.assertTrue(window["user_tolerant"])
        self.assertTrue(window["price_stable"])

    def test_run_zone_chain_repairs_invalid_economist_response(self):
        class FakeClient:
            def __init__(self):
                self.prompts = []
                self.responses = [
                    {
                        "forecast_total_kwh": 100.0,
                        "forecast_peak_kwh": 20.0,
                        "predicted_change_pct": 5.0,
                        "grid_stress_level": "High",
                        "forecast_summary": "High demand.",
                    },
                    {
                        "agent_reasoning": "One high-stress window.",
                        "demand_drivers": ["high-stress window"],
                        "confidence": "medium",
                    },
                    {
                        "suggested_price_shift_pct": 5,
                        "price_change_windows_3h": [{"suggested_price_shift_pct": 5}],
                    },
                    {
                        "suggested_price_shift_pct": 5,
                        "action_label": "Raise price",
                        "price_rationale": "High stress requires a higher service fee.",
                        "price_change_windows_3h": [
                            {
                                "window_start": "2022-09-09 00:00:00",
                                "window_end": "2022-09-09 02:00:00",
                                "suggested_price_shift_pct": 5,
                                "action_label": "Raise price",
                                "price_rationale": "High stress window requires a higher service fee.",
                            }
                        ],
                    },
                ]

            async def complete_json(self, prompt, *, temperature):
                self.prompts.append(prompt)
                return self.responses.pop(0)

        client = FakeClient()
        context = {
            "category": "Commercial",
            "zone_id": "102",
            "forecast_total_kwh": 100.0,
            "forecast_peak_kwh": 20.0,
            "predicted_change_pct": 5.0,
            "actual_total_kwh": 90.0,
            "mae_kwh": None,
            "rmse_kwh": None,
            "mape_pct": None,
            "rae": None,
            "wape_pct": None,
            "grid_stress_level": "High",
            "weather": {"rain_hours": 0},
            "hourly_shape": {"night_20_6": 1.0, "morning_7_10": 0.8, "evening_17_22": 0.7},
            "profile": {"poi_total": 10},
            "hourly_averages": {"mean_predicted_kwh": 10.0, "mean_energy_price": 1.0},
            "pricing_windows_3h": [
                {
                    "window_start": "2022-09-09 00:00:00",
                    "window_end": "2022-09-09 02:00:00",
                    "sum_predicted_kwh": 60.0,
                    "mean_predicted_kwh": 20.0,
                    "load_stress_level": "High",
                }
            ],
        }
        report = asyncio.run(run_zone_chain(context, client=client, temperature=0.2))
        self.assertEqual(len(client.prompts), 4)
        self.assertEqual(report["action_label"], "Raise price")
        self.assertEqual(report["price_change_windows_3h"][0]["action_label"], "Raise price")
        self.assertNotIn("MODEL_RESPONSE_FAILED", report["price_rationale"])
        debug = report[ECONOMIST_AGENT_OUTPUT_KEY]
        self.assertEqual(debug["zone_id"], "102")
        self.assertTrue(debug["repair_attempted"])
        self.assertEqual(debug["selected_response_source"], "repair")
        self.assertIn("missing action_label", debug["initial_validation_errors"])

    def test_splits_economist_agent_output_from_trace_reports(self):
        trace_reports, economist_outputs = split_economist_agent_outputs(
            [
                {
                    "zone_id": "102",
                    "action_label": "Raise price",
                    ECONOMIST_AGENT_OUTPUT_KEY: {
                        "zone_id": "102",
                        "initial_response": {"suggested_price_shift_pct": 8},
                        "initial_validation_errors": [],
                    },
                }
            ]
        )

        self.assertEqual(trace_reports, [{"zone_id": "102", "action_label": "Raise price"}])
        self.assertEqual(economist_outputs[0]["zone_id"], "102")
        self.assertEqual(economist_outputs[0]["initial_response"]["suggested_price_shift_pct"], 8)

    def test_write_outputs_saves_economist_agent_output_separately(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            outputs = write_outputs(
                output_dir=output_dir,
                selected_zones=pd.DataFrame([{"zone_id": "102", "category": "User-selected"}]),
                contexts=[],
                reports=[
                    {
                        "zone_id": "102",
                        "category": "User-selected",
                        "action_label": "Raise price",
                        ECONOMIST_AGENT_OUTPUT_KEY: {
                            "zone_id": "102",
                            "initial_response": {"suggested_price_shift_pct": 8},
                            "initial_validation_errors": [],
                        },
                    }
                ],
                forecast_results={},
            )

            economist_outputs = json.loads(outputs["economist_agent_outputs_json"].read_text(encoding="utf-8"))
            trace_reports = json.loads(outputs["rationale_trace_json"].read_text(encoding="utf-8"))
            self.assertEqual(economist_outputs[0]["zone_id"], "102")
            self.assertEqual(economist_outputs[0]["initial_response"]["suggested_price_shift_pct"], 8)
            self.assertNotIn(ECONOMIST_AGENT_OUTPUT_KEY, trace_reports[0])

    def test_prompt_includes_forecast_horizon_days(self):
        prompt = grid_prompt({"forecast_horizon_days": 3})
        self.assertIn("next 3 days", prompt)

    def test_economist_context_excludes_future_actuals_and_evaluation_fields(self):
        context = {
            "category": "User-selected",
            "zone_id": "102",
            "forecast_start": "2022-09-09T00:00:00",
            "forecast_end": "2022-09-10T23:00:00",
            "forecast_horizon_days": 2,
            "forecast_horizon_hours": 48,
            "forecast_total_kwh": 100.0,
            "forecast_peak_kwh": 20.0,
            "predicted_change_pct": 5.0,
            "grid_stress_level": "High",
            "actual_total_kwh": 120.0,
            "actual_grid_stress_level": "Extreme High",
            "hourly_averages": {
                "mean_predicted_kwh": 10.0,
                "mean_actual_kwh": 12.0,
                "mean_abs_pct_error": 15.0,
                "mean_service_price": 0.75,
            },
            "pricing_windows_3h": [
                {
                    "window_start": "2022-09-09 00:00:00",
                    "window_end": "2022-09-09 02:00:00",
                    "sum_predicted_kwh": 60.0,
                    "sum_actual_kwh": 90.0,
                    "actual_load_stress_level": "Extreme High",
                    "actual_grid_stress_level": "Extreme High",
                    "actual_stress_load_3h_kwh": 90.0,
                    "stress_correct": False,
                    "stress_missed": True,
                    "mean_abs_pct_error": 20.0,
                    "mean_service_price": 0.75,
                    "load_stress_level": "High",
                }
            ],
        }

        compact = compact_economist_context(context)
        prompt = economist_prompt(
            context,
            {"grid_stress_level": "High", "predicted_change_pct": 5.0},
            {"agent_reasoning": "Predicted high demand.", "demand_drivers": ["predicted stress"]},
        )
        serialized = json.dumps(compact, ensure_ascii=False)

        self.assertNotIn("actual_total_kwh", compact)
        self.assertNotIn("actual_grid_stress_level", compact)
        self.assertNotIn("mean_actual_kwh", compact["hourly_averages"])
        self.assertNotIn("mean_abs_pct_error", compact["hourly_averages"])
        self.assertNotIn("sum_actual_kwh", compact["pricing_windows_3h"][0])
        self.assertNotIn("actual_load_stress_level", compact["pricing_windows_3h"][0])
        self.assertNotIn("stress_correct", compact["pricing_windows_3h"][0])
        self.assertNotIn("stress_missed", compact["pricing_windows_3h"][0])
        self.assertNotIn("actual_", serialized)
        self.assertNotIn("stress_correct", prompt)
        self.assertNotIn("stress_missed", prompt)
        self.assertIn("sum_predicted_kwh", prompt)

    def test_heuristic_behavior_changes_with_forecast_window(self):
        base_context = {
            "category": "User-selected",
            "forecast_horizon_days": 1,
            "predicted_change_pct": 5.0,
            "weather": {"rain_hours": 0},
            "hourly_shape": {"night_20_6": 1.0, "morning_7_10": 0.8, "evening_17_22": 0.7},
            "profile": {"poi_total": 10},
        }
        first = heuristic_behavior(
            {
                **base_context,
                "forecast_start": "2022-09-09T00:00:00",
                "forecast_end": "2022-09-09T23:00:00",
                "pricing_windows_3h": [
                    {
                        "window_start": "2022-09-09 09:00:00",
                        "window_end": "2022-09-09 11:00:00",
                        "sum_predicted_kwh": 100.0,
                        "load_stress_level": "High",
                    }
                ],
            }
        )
        second = heuristic_behavior(
            {
                **base_context,
                "forecast_start": "2022-09-10T00:00:00",
                "forecast_end": "2022-09-10T23:00:00",
                "pricing_windows_3h": [
                    {
                        "window_start": "2022-09-10 18:00:00",
                        "window_end": "2022-09-10 20:00:00",
                        "sum_predicted_kwh": 80.0,
                        "load_stress_level": "Medium",
                    }
                ],
            }
        )
        self.assertNotEqual(first["agent_reasoning"], second["agent_reasoning"])


class ConfigTests(unittest.TestCase):
    def test_reads_agent_yaml_config(self):
        config = AgentConfig.from_file(Path("config.example.yaml"))
        self.assertEqual(config.api_key, "sk-...")
        self.assertEqual(config.model, "meta-llama/llama-3.1-8b-instruct")
        self.assertEqual(config.timeout_seconds, 90)

    def test_reads_run_yaml_config(self):
        config = AppConfig.from_file(Path("config.example.yaml"))
        self.assertTrue(config.run.dry_run)
        self.assertEqual(config.run.weather_file, "weather_central.csv")
        self.assertEqual(config.run.forecast_start, "2022-09-09 00:00:00")
        self.assertEqual(config.run.forecast_starts[:2], ["2022-09-09 00:00:00", "2022-10-14 00:00:00"])
        self.assertEqual(config.run.horizon_days, 2)
        self.assertEqual(config.run.history_days, 7)
        self.assertEqual(config.run.validation_days, 1)
        self.assertIsNone(config.run.zone_ids)
        self.assertEqual(config.run.experiment_zone_count, 12)
        self.assertEqual(config.run.forecast_model, "timesfm")
        self.assertEqual(config.run.forecast_models, ["timesfm", "chronos", "lstm", "AR"])
        self.assertEqual(config.run.experiment_seeds, [7, 42, 99])
        self.assertEqual(config.run.agent_mode, "agents")
        self.assertEqual(config.run.agent_modes, ["agents", "rules"])
        self.assertEqual(config.run.diurnal_blend_alphas, [0.0, 0.3, 0.6])
        self.assertEqual(config.run.timesfm_repo, "google/timesfm-2.5-200m-pytorch")
        self.assertEqual(config.run.timesfm_exog_cols[:4], ["T", "U", "nRAIN", "e_price"])
        self.assertEqual(config.run.ar_diurnal_blend_alpha, 0.0)
        self.assertEqual(config.run.chronos_repo, "amazon/chronos-2")
        self.assertEqual(config.run.chronos_context_hours, 512)
        self.assertEqual(config.run.chronos_diurnal_blend_alpha, 0.0)
        self.assertEqual(config.run.lstm_context_hours, 24)
        self.assertEqual(config.run.lstm_epochs, 50)
        self.assertEqual(config.run.lstm_diurnal_blend_alpha, 0.0)

    def test_reads_timesfm_config_keys(self):
        config = RunConfig.from_mapping(
            {
                "forecast_model": "timesfm",
                "timesfm_repo": "google/timesfm-2.5-200m-pytorch",
                "timesfm_context_hours": 48,
                "timesfm_exog_cols": ["T", "U"],
            }
        )
        self.assertEqual(config.forecast_model, "timesfm")
        self.assertEqual(config.timesfm_context_hours, 48)
        self.assertEqual(config.timesfm_exog_cols, ["T", "U"])


class ForecastingTests(unittest.TestCase):
    def test_timesfm_loader_strips_huggingface_hub_kwargs(self):
        class FakeTimesFM:
            seen_kwargs = None

            @classmethod
            def _from_pretrained(cls, **kwargs):
                cls.seen_kwargs = kwargs
                return cls()

        patch_timesfm_hub_kwargs(FakeTimesFM)
        FakeTimesFM._from_pretrained(
            model_id="fake/repo",
            revision=None,
            cache_dir=None,
            force_download=False,
            proxies={"http": "http://proxy"},
            resume_download=None,
            local_files_only=False,
            token=None,
            config={"model": "fake"},
        )

        self.assertNotIn("proxies", FakeTimesFM.seen_kwargs)
        self.assertNotIn("resume_download", FakeTimesFM.seen_kwargs)
        self.assertEqual(FakeTimesFM.seen_kwargs["config"], {"model": "fake"})

    def test_ar_forecast_keeps_hourly_horizon(self):
        history = pd.DataFrame(
            {
                "time": pd.date_range("2023-01-01", periods=24 * 7, freq="h"),
                "actual_kwh": [float(hour % 24) for hour in range(24 * 7)],
            }
        )
        result = ar_forecast(history, pd.Timestamp("2023-01-08"), 48)
        self.assertEqual(len(result), 48)
        self.assertIn("predicted_kwh", result)
        self.assertGreater(result["predicted_kwh"].sum(), 0)

    def test_forecast_metrics_include_error_standards(self):
        hourly = pd.DataFrame(
            {
                "actual_kwh": [10.0, 20.0, 30.0],
                "predicted_kwh": [12.0, 18.0, 33.0],
            }
        )
        metrics = compute_forecast_metrics(hourly)
        self.assertEqual(metrics["n"], 3)
        self.assertIn("MAE", metrics)
        self.assertIn("RMSE", metrics)
        self.assertIn("MAPE_pct", metrics)
        self.assertIn("RAE", metrics)

    def test_build_zone_model_frame_adds_notebook_covariates(self):
        times = pd.date_range("2023-01-01", periods=2, freq="h")
        load = pd.DataFrame({"time": times, "102": [10.0, 11.0]})
        service_price = pd.DataFrame({"time": times, "102": [0.5, 0.6]})
        energy_price = pd.DataFrame({"time": times, "102": [1.0, 1.2]})
        occupancy = pd.DataFrame({"time": times, "102": [0.1, 0.2]})
        weather = pd.DataFrame({"time": times, "T": [20.0, 21.0], "U": [60.0, 61.0], "nRAIN": [0.0, 1.0]})
        frame = build_zone_model_frame(load, service_price, energy_price, occupancy, weather, "102")
        self.assertIn("e_price", frame)
        self.assertIn("is_weekend", frame)
        self.assertIn("temp_price_idx", frame)
        self.assertAlmostEqual(frame["temp_price_idx"].iloc[1], 25.2)

    def test_rebuild_quantile_interval_centers_final_prediction(self):
        point = np.array([100.0, 2.0, 50.0])
        raw_q10 = np.array([40.0, 0.0, np.nan])
        raw_q90 = np.array([80.0, 10.0, np.nan])
        q10, q90 = rebuild_quantile_interval(point, raw_q10, raw_q90)
        self.assertEqual(q10[0], 80.0)
        self.assertEqual(q90[0], 120.0)
        self.assertEqual(q10[1], 0.0)
        self.assertEqual(q90[1], 7.0)
        self.assertTrue(np.isnan(q10[2]))
        self.assertTrue(np.isnan(q90[2]))
        self.assertTrue(np.all(q10[:2] <= point[:2]))
        self.assertTrue(np.all(q90[:2] >= point[:2]))

    def test_chronos_forecast_accepts_chronos2_list_output(self):
        test_case = self

        class FakeChronos2:
            def predict(
                self,
                inputs,
                prediction_length=None,
                batch_size=256,
                context_length=None,
                cross_learning=False,
                limit_prediction_length=False,
                **kwargs,
            ):
                return None

            def predict_quantiles(self, inputs, prediction_length, quantile_levels, **kwargs):
                test_case.assertEqual(kwargs, {"limit_prediction_length": False})
                center = torch.arange(1, prediction_length + 1, dtype=torch.float32)
                quantiles = torch.stack([center - 0.5, center, center + 0.5], dim=-1).unsqueeze(0)
                return [quantiles], [center.unsqueeze(0)]

        history = pd.DataFrame(
            {
                "time": pd.date_range("2023-01-01", periods=24, freq="h"),
                "actual_kwh": [float(hour + 1) for hour in range(24)],
            }
        )
        full_frame = pd.DataFrame(
            {
                "time": pd.date_range("2023-01-02", periods=2, freq="h"),
                "actual_kwh": [1.0, 2.0],
            }
        )
        with patch("forecasting.load_chronos_model", return_value=FakeChronos2()):
            result = chronos_forecast(
                history,
                pd.DataFrame(),
                full_frame,
                pd.Timestamp("2023-01-02"),
                2,
                repo="fake",
                context_hours=24,
                step_horizon=2,
                diurnal_blend_alpha=0.0,
                device="cpu",
                roll_actuals=False,
            )
        self.assertEqual(result["predicted_kwh"].tolist(), [1.0, 2.0])
        self.assertEqual(result["q10_kwh"].tolist(), [0.5, 1.5])
        self.assertEqual(result["q90_kwh"].tolist(), [1.5, 2.5])

    def test_lstm_forecast_produces_hourly_predictions(self):
        times = pd.date_range("2023-01-01", periods=60, freq="h")
        values = 20.0 + 5.0 * np.sin(np.arange(60) / 24.0 * 2.0 * np.pi)
        frame = pd.DataFrame({"time": times, "actual_kwh": values})
        history = frame.iloc[:48].reset_index(drop=True)
        validation = frame.iloc[48:54].reset_index(drop=True)
        full_frame = frame.iloc[54:60].reset_index(drop=True)

        result = lstm_forecast(
            history,
            validation,
            full_frame,
            pd.Timestamp("2023-01-03 06:00:00"),
            6,
            context_hours=12,
            step_horizon=3,
            exog_cols=[],
            hidden_size=8,
            num_layers=1,
            epochs=2,
            learning_rate=0.01,
            batch_size=8,
            diurnal_blend_alpha=0.0,
            device="cpu",
            roll_actuals=False,
            seed=7,
        )

        self.assertEqual(len(result), 6)
        self.assertTrue(result["predicted_kwh"].notna().all())
        self.assertTrue((result["predicted_kwh"] >= 0).all())
        self.assertTrue((result["q10_kwh"] <= result["predicted_kwh"]).all())
        self.assertTrue((result["q90_kwh"] >= result["predicted_kwh"]).all())


class SelectionTests(unittest.TestCase):
    def test_builds_price_comparison_fields_and_summary(self):
        fields = price_comparison_fields(1.0, 10)
        self.assertEqual(fields["adjusted_service_price"], 1.1)
        self.assertEqual(fields["adjusted_minus_actual_service_price"], 0.1)
        self.assertEqual(fields["adjusted_vs_actual_pct"], 10.0)
        self.assertNotIn("abs_adjusted_minus_actual_service_price", fields)

        summary = build_price_comparison_summary(
            pd.DataFrame(
                [
                    {
                        "zone_id": "102",
                        "category": "User-selected",
                        "actual_service_price": 1.0,
                        "adjusted_service_price": 1.05,
                        "adjusted_minus_actual_service_price": 0.05,
                        "adjusted_vs_actual_pct": 5.0,
                    },
                    {
                        "zone_id": "102",
                        "category": "User-selected",
                        "actual_service_price": 2.0,
                        "adjusted_service_price": 1.8,
                        "adjusted_minus_actual_service_price": -0.2,
                        "adjusted_vs_actual_pct": -10.0,
                    },
                ]
            )
        )
        self.assertNotIn("ALL", summary["zone_id"].tolist())
        self.assertNotIn("avg_abs_adjusted_minus_actual_service_price", summary.columns)
        zone_summary = summary[summary["zone_id"] == "102"].iloc[0]
        self.assertEqual(zone_summary["price_windows"], 2)
        self.assertEqual(zone_summary["price_error_threshold_pct"], 8.0)
        self.assertEqual(zone_summary["price_pass_windows"], 1)
        self.assertAlmostEqual(zone_summary["price_accuracy"], 0.5)
        self.assertAlmostEqual(zone_summary["avg_actual_service_price"], 1.5)
        self.assertAlmostEqual(zone_summary["avg_adjusted_service_price"], 1.425)
        self.assertAlmostEqual(zone_summary["avg_adjusted_minus_actual_service_price"], -0.075)
        self.assertAlmostEqual(zone_summary["avg_adjusted_vs_actual_pct"], -2.5)

    def test_builds_hourly_agent_context_and_three_hour_windows(self):
        hourly = pd.DataFrame(
            {
                "time": pd.date_range("2022-09-09", periods=6, freq="h"),
                "predicted_kwh": [10, 12, 14, 8, 6, 7],
                "actual_kwh": [14, 14, 14, 9, 5, 8],
                "s_price": [0.7, 0.7, 0.8, 0.8, 0.6, 0.6],
                "e_price": [1.0, 1.1, 1.2, 0.9, 0.8, 0.8],
                "occupancy": [0.2, 0.3, 0.4, 0.2, 0.1, 0.1],
                "T": [20, 21, 22, 20, 19, 18],
                "U": [60, 61, 62, 63, 64, 65],
                "nRAIN": [0, 0, 1, 0, 0, 0],
                "error_kwh": [1, -1, -1, 1, -1, 1],
                "abs_pct_error": [10, 8.3, 7.1, 12.5, 16.7, 14.3],
            }
        )
        hourly_context = build_agent_hourly_data(hourly)
        averages = build_hourly_averages(hourly)
        windows = build_pricing_windows_3h(
            hourly,
            stress_thresholds={"available": True, "q50": 25.0, "q80": 35.0, "q95": 40.0},
        )
        self.assertEqual(len(hourly_context), 6)
        self.assertEqual(hourly_context[0]["time"], "2022-09-09 00:00:00")
        self.assertEqual(averages["mean_predicted_kwh"], 9.5)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0]["sum_predicted_kwh"], 36.0)
        self.assertEqual(windows[0]["load_stress_level"], "High")
        self.assertEqual(windows[0]["sum_actual_kwh"], 42.0)
        self.assertEqual(windows[0]["actual_load_stress_level"], "Extreme High")
        self.assertFalse(windows[0]["stress_correct"])
        self.assertTrue(windows[0]["stress_missed"])
        summary = apply_load_quantile_stress(
            {"forecast_peak_kwh": 14.0},
            {"available": True, "q50": 25.0, "q80": 35.0, "q95": 40.0},
            windows,
        )
        self.assertEqual(summary["grid_stress_level"], "High")
        self.assertEqual(summary["actual_grid_stress_level"], "Extreme High")
        self.assertEqual(summary["stress_eval_windows"], 2)
        self.assertEqual(summary["stress_miss_count"], 1)
        self.assertAlmostEqual(summary["stress_accuracy"], 0.5)
        self.assertAlmostEqual(summary["miss_stress_rate"], 0.5)

    def test_caches_zone_three_hour_load_quantiles_from_volume_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            cache_dir = root / "cache"
            data_dir.mkdir()
            times = pd.date_range("2022-01-01", periods=12, freq="h")
            pd.DataFrame(
                {
                    "time": times,
                    "102": [float(value) for value in range(1, 13)],
                    "104": [2.0] * 12,
                }
            ).to_csv(data_dir / "volume.csv", index=False)

            quantiles = build_zone_3h_load_quantiles(data_dir, cache_dir)
            row = quantiles.set_index("zone_id").loc["102"]
            self.assertTrue((cache_dir / "zone_3h_load_quantiles.csv").exists())
            self.assertEqual(row["stress_source_file"], "volume.csv")
            self.assertEqual(row["stress_window_hours"], 3)
            self.assertAlmostEqual(row["load_3h_q50_kwh"], 19.5)
            self.assertAlmostEqual(row["load_3h_q80_kwh"], 27.6)

    def test_pipeline_load_source_uses_volume_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            times = pd.date_range("2022-01-01", periods=2, freq="h")
            pd.DataFrame({"time": times, "102": [10.0, 11.0]}).to_csv(data_dir / "volume.csv", index=False)
            pd.DataFrame({"time": times, "102": [1000.0, 1001.0]}).to_csv(
                data_dir / "volume-11kW.csv",
                index=False,
            )
            pd.DataFrame({"time": times, "102": [0.7, 0.8]}).to_csv(data_dir / "s_price.csv", index=False)
            pd.DataFrame({"time": times, "102": [1.0, 1.1]}).to_csv(data_dir / "e_price.csv", index=False)
            pd.DataFrame({"time": times, "102": [0.2, 0.3]}).to_csv(data_dir / "occupancy.csv", index=False)
            pd.DataFrame({"time": times, "T": [20.0, 21.0], "U": [60.0, 61.0], "nRAIN": [0.0, 0.0]}).to_csv(
                data_dir / "weather_airport.csv",
                index=False,
            )
            profiles = pd.DataFrame({"zone_id": ["102"]})

            self.assertEqual(available_zone_ids(data_dir), ["102"])
            pipeline_data = load_pipeline_data(data_dir, profiles, ["102"])
            self.assertEqual(pipeline_data.load["102"].tolist(), [10.0, 11.0])

    def test_classifies_grid_stress_from_zone_three_hour_quantiles(self):
        thresholds = {"available": True, "q50": 50.0, "q80": 80.0, "q95": 95.0}
        self.assertEqual(classify_load_stress(96, thresholds), "Extreme High")
        self.assertEqual(classify_load_stress(90, thresholds), "High")
        self.assertEqual(classify_load_stress(60, thresholds), "Medium")
        self.assertEqual(classify_load_stress(50, thresholds), "Low")

    def test_forecast_output_dir_uses_model_subfolder(self):
        self.assertEqual(forecast_output_dir(Path("output"), "chronos"), Path("output") / "chronos")
        self.assertEqual(forecast_output_dir(Path("output"), "lstm"), Path("output") / "lstm")
        self.assertEqual(forecast_output_dir(Path("output"), "timesfm"), Path("output") / "timesfm")
        self.assertEqual(forecast_output_dir(Path("output"), "AR"), Path("output") / "AR")

    def test_experiment_matrix_writes_isolated_outputs_and_summaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            calls = []

            def fake_run_pipeline(**kwargs):
                calls.append(kwargs)
                run_dir = forecast_output_dir(kwargs["output_dir"], kwargs["forecast_model"])
                run_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = run_dir / "forecast_metrics.csv"
                price_path = run_dir / "price_comparison_summary.csv"
                rationale_path = run_dir / "rationale_trace.csv"
                pd.DataFrame(
                    [
                        {
                            "zone_id": "102",
                            "forecast_model": kwargs["forecast_model"],
                            "diurnal_blend_alpha": kwargs.get("timesfm_diurnal_blend_alpha"),
                            "forecast_start": kwargs["forecast_start"],
                            "MAE": 1.0,
                        },
                        {
                            "zone_id": "105",
                            "forecast_model": kwargs["forecast_model"],
                            "diurnal_blend_alpha": kwargs.get("timesfm_diurnal_blend_alpha"),
                            "forecast_start": kwargs["forecast_start"],
                            "MAE": 2.0,
                        },
                    ]
                ).to_csv(metrics_path, index=False)
                pd.DataFrame(
                    [
                        {"zone_id": "102", "category": "User-selected", "price_accuracy": 0.5},
                        {"zone_id": "105", "category": "User-selected", "price_accuracy": 0.75},
                    ]
                ).to_csv(price_path, index=False)
                pd.DataFrame(
                    [
                        {"zone_id": "102", "source": "agent"},
                        {"zone_id": "105", "source": "agent"},
                    ]
                ).to_csv(rationale_path, index=False)
                return {
                    "forecast_metrics_csv": metrics_path,
                    "price_comparison_summary_csv": price_path,
                    "rationale_trace_csv": rationale_path,
                }

            with patch("orchestrator.run_pipeline", side_effect=fake_run_pipeline):
                outputs = run_experiment_matrix(
                    data_dir=root / "data",
                    output_dir=root / "output",
                    config_path=Path("config.yaml"),
                    dry_run=True,
                    forecast_starts=["2022-09-09 00:00:00", "2022-10-14 00:00:00"],
                    forecast_models=["timesfm", "AR"],
                    zone_ids=["102", "105"],
                )

            self.assertEqual(len(calls), 4)
            self.assertEqual(calls[0]["cache_dir"], root / "output" / "cache")
            run_dirs = {
                forecast_output_dir(call["output_dir"], call["forecast_model"])
                for call in calls
            }
            self.assertEqual(len(run_dirs), 4)
            self.assertIn(
                root / "output" / "experiments" / "zones_102_105_2starts" / "2022-09-09_000000" / "timesfm",
                run_dirs,
            )
            self.assertIn(
                root / "output" / "experiments" / "zones_102_105_2starts" / "2022-10-14_000000" / "timesfm",
                run_dirs,
            )

            runs = pd.read_csv(outputs["experiment_runs_csv"])
            metrics = pd.read_csv(outputs["experiment_forecast_metrics_csv"])
            prices = pd.read_csv(outputs["experiment_price_comparison_summary_csv"])
            rationales = pd.read_csv(outputs["experiment_rationale_trace_csv"])
            self.assertEqual(len(runs), 4)
            self.assertTrue((runs["status"] == "success").all())
            self.assertEqual(len(metrics), 8)
            self.assertEqual(len(prices), 8)
            self.assertEqual(len(rationales), 8)
            self.assertEqual(set(metrics["zone_id"].astype(str)), {"102", "105"})
            self.assertIn("run_output_dir", metrics.columns)
            forecast_summary = pd.read_csv(outputs["experiment_forecast_summary_csv"])
            self.assertIn("mean", forecast_summary.columns)

    def test_experiment_matrix_expands_seeds_agent_modes_and_blends(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            calls = []

            def fake_run_pipeline(**kwargs):
                calls.append(kwargs)
                run_dir = forecast_output_dir(kwargs["output_dir"], kwargs["forecast_model"])
                run_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = run_dir / "forecast_metrics.csv"
                price_path = run_dir / "price_comparison_summary.csv"
                rationale_path = run_dir / "rationale_trace.csv"
                explainability_path = run_dir / "explainability_review_packet.csv"
                pd.DataFrame(
                    [
                        {
                            "zone_id": "102",
                            "forecast_model": kwargs["forecast_model"],
                            "MAE": float(kwargs.get("lstm_seed", 0) or 0) + 1.0,
                            "diurnal_blend_alpha": kwargs.get("lstm_diurnal_blend_alpha"),
                        }
                    ]
                ).to_csv(metrics_path, index=False)
                pd.DataFrame(
                    [
                        {
                            "zone_id": "102",
                            "category": "User-selected",
                            "price_accuracy": 0.5 if kwargs["agent_mode"] == "rules" else 0.75,
                            "avg_adjusted_minus_actual_service_price": 0.1,
                            "avg_adjusted_vs_actual_pct": 5.0,
                        }
                    ]
                ).to_csv(price_path, index=False)
                pd.DataFrame(
                    [
                        {
                            "zone_id": "102",
                            "source": kwargs["agent_mode"],
                            "stress_accuracy": 0.8,
                            "miss_stress_rate": 0.1,
                            "suggested_price_shift_pct": 3,
                        }
                    ]
                ).to_csv(rationale_path, index=False)
                pd.DataFrame([{"zone_id": "102", "rationale": "specific rationale"}]).to_csv(
                    explainability_path,
                    index=False,
                )
                return {
                    "forecast_metrics_csv": metrics_path,
                    "price_comparison_summary_csv": price_path,
                    "rationale_trace_csv": rationale_path,
                    "explainability_review_packet_csv": explainability_path,
                }

            with patch("orchestrator.run_pipeline", side_effect=fake_run_pipeline):
                outputs = run_experiment_matrix(
                    data_dir=root / "data",
                    output_dir=root / "output",
                    config_path=Path("config.yaml"),
                    dry_run=True,
                    forecast_starts=["2022-09-09 00:00:00"],
                    forecast_models=["lstm"],
                    zone_ids=["102"],
                    experiment_seeds=[7, 42],
                    agent_modes=["agents", "rules"],
                    diurnal_blend_alphas=[0.0, 0.3],
                )

            self.assertEqual(len(calls), 8)
            self.assertEqual({call["lstm_seed"] for call in calls}, {7, 42})
            self.assertEqual({call["agent_mode"] for call in calls}, {"agents", "rules"})
            self.assertEqual({call["lstm_diurnal_blend_alpha"] for call in calls}, {0.0, 0.3})
            self.assertTrue(all("seed_" in str(call["output_dir"]) for call in calls))
            self.assertTrue(all("agent_" in str(call["output_dir"]) for call in calls))
            self.assertTrue(all("blend_" in str(call["output_dir"]) for call in calls))
            runs = pd.read_csv(outputs["experiment_runs_csv"])
            self.assertEqual(len(runs), 8)
            decision_summary = pd.read_csv(outputs["experiment_decision_quality_summary_csv"])
            self.assertIn("price_accuracy", set(decision_summary["metric"]))

    def test_experiment_matrix_writes_detailed_error_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            def fake_run_pipeline(**kwargs):
                raise AgentStageError(
                    stage="agent.grid",
                    zone_id="102",
                    agent="Grid Analyst",
                    original=TypeError("'NoneType' object is not subscriptable"),
                )

            with patch("orchestrator.run_pipeline", side_effect=fake_run_pipeline):
                outputs = run_experiment_matrix(
                    data_dir=root / "data",
                    output_dir=root / "output",
                    config_path=Path("config.yaml"),
                    forecast_starts=["2022-09-09 00:00:00"],
                    forecast_models=["timesfm"],
                    zone_ids=["102"],
                    agent_modes=["agents"],
                    diurnal_blend_alphas=[0.3],
                )

            runs = pd.read_csv(outputs["experiment_runs_csv"])
            self.assertEqual(runs.loc[0, "status"], "failed")
            self.assertEqual(runs.loc[0, "error_code"], "MAPF-E210")
            self.assertEqual(runs.loc[0, "error_stage"], "agent.grid")
            self.assertEqual(str(runs.loc[0, "error_zone_id"]), "102")
            self.assertEqual(runs.loc[0, "error_agent"], "Grid Analyst")
            self.assertEqual(runs.loc[0, "error_type"], "TypeError")
            traceback_file = Path(runs.loc[0, "traceback_file"])
            self.assertTrue(traceback_file.exists())
            self.assertIn("TRACEBACK", traceback_file.read_text(encoding="utf-8"))

    def test_selects_representative_zone_ids_beyond_category_five(self):
        profiles = pd.DataFrame(
            {
                "zone_id": [str(i) for i in range(1, 9)],
                "poi_business_density": np.linspace(1, 8, 8),
                "morning_ratio": np.linspace(1, 2, 8),
                "noon_ratio": np.linspace(2, 1, 8),
                "night_ratio": np.linspace(1, 3, 8),
                "poi_food_density": np.linspace(3, 1, 8),
                "poi_lifestyle_density": np.linspace(1, 3, 8),
                "charge_count": np.arange(10, 18),
                "burstiness_p99_mean": np.linspace(1, 2, 8),
                "load_cv": np.linspace(0.1, 0.8, 8),
                "peak_load_kwh": np.linspace(20, 80, 8),
                "mean_load_kwh": np.linspace(5, 40, 8),
                "weekend_ratio": np.linspace(0.8, 1.2, 8),
                "peak_capacity_ratio": np.linspace(0.1, 0.4, 8),
                "longitude": 0,
                "latitude": 0,
                "station_count": 1,
                "capacity_kw_proxy": 1,
                "morning_ratio": np.linspace(1, 2, 8),
                "evening_ratio": np.linspace(1, 2, 8),
                "poi_food": 0,
                "poi_business": 0,
                "poi_lifestyle": 0,
                "poi_total": 0,
                "mean_service_price": 1.0,
            }
        )
        selected = select_representative_zone_ids(profiles, count=7)
        self.assertEqual(len(selected), 7)
        self.assertEqual(len(set(selected)), 7)

    def test_normalizes_requested_zone_ids(self):
        zone_ids = normalize_zone_ids(["102,104", " 108 ", "104"])
        self.assertEqual(zone_ids, ["102", "104", "108"])

    def test_selects_requested_zones_in_user_order(self):
        profiles = pd.DataFrame(
            {
                "zone_id": ["101", "102", "104"],
                "longitude": [0.0, 1.0, 2.0],
                "latitude": [0.0, 1.0, 2.0],
                "station_count": [1, 2, 3],
                "charge_count": [10, 20, 30],
                "capacity_kw_proxy": [110.0, 220.0, 330.0],
                "mean_load_kwh": [5.0, 6.0, 7.0],
                "peak_load_kwh": [8.0, 9.0, 10.0],
                "peak_capacity_ratio": [0.1, 0.2, 0.3],
                "load_cv": [0.1, 0.2, 0.3],
                "burstiness_p99_mean": [1.1, 1.2, 1.3],
                "morning_ratio": [1.0, 1.0, 1.0],
                "noon_ratio": [1.0, 1.0, 1.0],
                "evening_ratio": [1.0, 1.0, 1.0],
                "night_ratio": [1.0, 1.0, 1.0],
                "weekend_ratio": [1.0, 1.0, 1.0],
                "poi_food": [0, 0, 0],
                "poi_business": [0, 0, 0],
                "poi_lifestyle": [0, 0, 0],
                "poi_total": [0, 0, 0],
                "mean_service_price": [0.7, 0.8, 0.9],
            }
        )
        selected = select_requested_zones(profiles, ["104", "102"])
        self.assertEqual(selected["zone_id"].tolist(), ["104", "102"])
        self.assertEqual(selected["category"].tolist(), ["User-selected", "User-selected"])

    def test_selects_five_unique_categories(self):
        profiles = pd.DataFrame(
            {
                "zone_id": ["1", "2", "3", "4", "5", "6"],
                "poi_business_density": [10, 100, 5, 3, 2, 1],
                "poi_food_density": [5, 2, 8, 100, 1, 1],
                "poi_lifestyle_density": [5, 3, 8, 80, 1, 1],
                "morning_ratio": [1.5, 1.0, 1.0, 1.0, 0.8, 0.7],
                "noon_ratio": [1.4, 1.0, 1.0, 1.0, 0.9, 0.8],
                "evening_ratio": [0.8, 1.5, 1.0, 1.4, 1.0, 0.8],
                "night_ratio": [0.7, 1.7, 1.0, 1.0, 1.0, 0.8],
                "weekend_ratio": [1.0, 1.0, 1.1, 1.8, 1.0, 0.9],
                "mean_load_kwh": [100, 120, 200, 150, 180, 90],
                "peak_load_kwh": [180, 200, 500, 250, 220, 100],
                "charge_count": [20, 30, 400, 40, 300, 10],
                "burstiness_p99_mean": [1.5, 1.6, 5.0, 2.0, 1.2, 1.0],
                "load_cv": [0.5, 0.6, 2.0, 0.8, 0.1, 0.4],
                "longitude": [0] * 6,
                "latitude": [0] * 6,
                "station_count": [1] * 6,
                "capacity_kw_proxy": [220, 330, 4400, 440, 3300, 110],
                "peak_capacity_ratio": [0.8, 0.6, 0.1, 0.5, 0.07, 0.9],
                "poi_food": [0] * 6,
                "poi_business": [0] * 6,
                "poi_lifestyle": [0] * 6,
                "poi_total": [0] * 6,
                "mean_service_price": [0.76] * 6,
            }
        )
        selected = select_zone_categories(profiles)
        self.assertEqual(len(selected), 5)
        self.assertEqual(selected["zone_id"].nunique(), 5)


if __name__ == "__main__":
    unittest.main()
