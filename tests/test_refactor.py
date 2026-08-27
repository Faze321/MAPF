import json
import asyncio
import copy
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from dataset_adapter import (
    DatasetSpec,
    build_price_change_reference,
    load_canonical_dataset,
    split_cache_key,
)
from config import AgentConfig, RunConfig, normalize_agent_mode
from global_forecaster import GlobalForecaster, NativeForecasterArtifact
from forecasting import chronos_forecast, lstm_forecast, timesfm_forecast
from main import build_parser
from prompts import agent_safe_context
from reporting import write_agent_outputs
from orchestrator import (
    control_attempt_snapshot,
    energy_price_with_predicted_windows,
    forecast_output_dir,
    revise_control_reports,
    run_control_loop,
    run_experiment_matrix,
    run_pipeline,
)
from unittest.mock import patch
from agents import (
    AgentChatClient,
    agent_call_usage_record,
    normalize_price_windows,
    run_zone_chain,
    summarize_agent_call_usage,
)
from time_utils import parse_datetime_24h


def workspace_temporary_directory() -> Path:
    path = Path(__file__).resolve().parents[1] / ".tmp_tests" / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


def discussion_test_context() -> dict:
    return {
        "zone_id": "a",
        "category": "test",
        "forecast_start": "2024-01-01 00:00:00",
        "forecast_end": "2024-01-01 02:00:00",
        "forecast_horizon_days": 1,
        "forecast_total_kwh": 30.0,
        "forecast_peak_kwh": 12.0,
        "predicted_change_pct": 5.0,
        "grid_stress_level": "High",
        "weather": {"rain_hours": 1},
        "hourly_shape": {
            "night_20_6": 10.0,
            "morning_7_10": 8.0,
            "evening_17_22": 11.0,
        },
        "profile": {"poi_total": 3},
        "pricing_windows_3h": [
            {
                "window_start": "2024-01-01 00:00:00",
                "window_end": "2024-01-01 02:00:00",
                "sum_predicted_kwh": 30.0,
                "mean_predicted_kwh": 10.0,
                "mean_energy_price": 1.0,
                "mean_service_price": 0.5,
                "load_range_position_pct": 85.0,
                "load_stress_level": "High",
            }
        ],
    }


class OutputPathTests(unittest.TestCase):
    def test_agent_provider_profiles_are_selected_independently(self):
        temporary = workspace_temporary_directory()
        config_path = temporary / "config.yaml"
        config_path.write_text(
            """
agent:
  multi_agent:
    api_key: multi-key
    base_url: https://multi.example/v1
    model: multi-model
    timeout_seconds: 31
    max_concurrent_requests: 3
  single_agent:
    api_key: single-key
    base_url: https://single.example/v1
    model: single-model
    timeout_seconds: 47
    max_concurrent_requests: 1
""".strip(),
            encoding="utf-8",
        )

        multi = AgentConfig.from_file(
            config_path,
            agent_mode="multi_agent_discussion_3rounds",
        )
        single = AgentConfig.from_file(
            config_path,
            agent_mode="single_agent_full_retry",
        )

        self.assertEqual("multi_agent", multi.profile)
        self.assertEqual("multi-key", multi.api_key)
        self.assertEqual("https://multi.example/v1", multi.base_url)
        self.assertEqual("multi-model", multi.model)
        self.assertEqual(31, multi.timeout_seconds)
        self.assertEqual(3, multi.max_concurrent_requests)
        self.assertEqual("single_agent", single.profile)
        self.assertEqual("single-key", single.api_key)
        self.assertEqual("https://single.example/v1", single.base_url)
        self.assertEqual("single-model", single.model)
        self.assertEqual(47, single.timeout_seconds)
        self.assertEqual(1, single.max_concurrent_requests)

    def test_only_active_agent_profile_expands_environment_variables(self):
        temporary = workspace_temporary_directory()
        config_path = temporary / "config.yaml"
        config_path.write_text(
            """
agent:
  multi_agent:
    api_key: multi-key
  single_agent:
    api_key: "${MAPF_TEST_MISSING_SINGLE_AGENT_KEY}"
""".strip(),
            encoding="utf-8",
        )

        multi = AgentConfig.from_file(
            config_path,
            agent_mode="multi_agent_economist_retry",
        )
        self.assertEqual("multi-key", multi.api_key)
        with self.assertRaisesRegex(ValueError, "MAPF_TEST_MISSING_SINGLE_AGENT_KEY"):
            AgentConfig.from_file(
                config_path,
                agent_mode="single_agent_price_retry",
            )

    def test_legacy_flat_agent_config_still_selects_single_model(self):
        temporary = workspace_temporary_directory()
        config_path = temporary / "config.yaml"
        config_path.write_text(
            """
agent:
  api_key: shared-key
  base_url: https://shared.example/v1
  model: legacy-multi-model
  single_agent_model: legacy-single-model
""".strip(),
            encoding="utf-8",
        )

        single = AgentConfig.from_file(
            config_path,
            agent_mode="single_agent_price_retry",
        )
        overridden = AgentConfig.from_file(
            config_path,
            agent_mode="single_agent_price_retry",
            model="cli-override",
        )
        self.assertEqual("shared-key", single.api_key)
        self.assertEqual("https://shared.example/v1", single.base_url)
        self.assertEqual("legacy-single-model", single.model)
        self.assertEqual("cli-override", overridden.model)

    def test_config_separates_output_folder_and_experiment_name(self):
        configured = RunConfig.from_mapping(
            {"output_folder": "generated", "experiment_name": "trial_01"}
        )
        self.assertEqual("generated", configured.output_folder)
        self.assertEqual("trial_01", configured.experiment_name)

        legacy = RunConfig.from_mapping({"output_dir": "legacy-output"})
        self.assertEqual("legacy-output", legacy.output_folder)
        self.assertIsNone(legacy.experiment_name)

    def test_three_round_discussion_mode_and_aliases_are_supported(self):
        canonical = "multi_agent_discussion_3rounds"
        self.assertEqual(canonical, normalize_agent_mode(canonical))
        self.assertEqual(canonical, normalize_agent_mode("multi-agent-discussion"))
        self.assertEqual(
            canonical,
            normalize_agent_mode("multi_agent_communication_3rounds"),
        )

    def test_parser_separates_output_folder_and_experiment_name(self):
        args = build_parser().parse_args(
            ["--output-folder", "generated", "--experiment-name", "trial_01"]
        )
        self.assertEqual("generated", args.output_folder)
        self.assertEqual("trial_01", args.experiment_name)

        legacy = build_parser().parse_args(["--output-dir", "legacy-output"])
        self.assertEqual("legacy-output", legacy.output_folder)

    def test_single_run_output_remains_model_named(self):
        root = Path("output")
        self.assertEqual(root / "AR", forecast_output_dir(root, "AR"))

    def test_matrix_name_defaults_to_existing_slug_and_can_be_overridden(self):
        temporary = workspace_temporary_directory()
        data_dir = temporary / "dataset"
        data_dir.mkdir()
        output_folder = temporary / "output"
        with patch("orchestrator.run_pipeline", return_value={}):
            automatic = run_experiment_matrix(
                data_dir=data_dir,
                output_dir=output_folder,
                forecast_starts=["2024-01-01"],
                forecast_models=["AR"],
                zone_ids=["a"],
            )
            custom = run_experiment_matrix(
                data_dir=data_dir,
                output_dir=output_folder,
                experiment_name="named experiment",
                forecast_starts=["2024-01-01"],
                forecast_models=["AR"],
                zone_ids=["a"],
            )
        self.assertEqual(
            output_folder / "1zonesx1timesx1modes_1blends",
            automatic["experiment_dir"],
        )
        self.assertEqual(
            output_folder / "named_experiment",
            custom["experiment_dir"],
        )
        self.assertTrue(automatic["experiment_agent_attempt_usage_csv"].is_file())
        self.assertTrue(
            automatic["experiment_agent_step_token_usage_csv"].is_file()
        )
        self.assertTrue(automatic["experiment_control_attempt_trace_csv"].is_file())


class DatasetCacheTests(unittest.TestCase):
    def make_dataset(self, root: Path) -> Path:
        dataset = root / "dataset"
        dataset.mkdir()
        times = pd.date_range("2024-01-01", periods=12, freq="h")
        pd.DataFrame(
            {
                "when": np.repeat(times, 2),
                "area": ["a", "b"] * len(times),
                "demand": np.arange(len(times) * 2, dtype=float) + 1,
                "tariff": [1.0, 2.0] * len(times),
                "temperature": np.repeat(np.linspace(10, 15, len(times)), 2),
            }
        ).to_csv(dataset / "series.csv", index=False)
        return dataset

    def spec(self, dataset: Path) -> DatasetSpec:
        return DatasetSpec(
            path=dataset,
            adapter="long_format",
            timeseries_file="series.csv",
            column_mapping={
                "timestamp": "when",
                "zone_id": "area",
                "load_kwh": "demand",
                "energy_price": "tariff",
            },
        )

    def test_default_cache_is_dataset_local_and_hits(self):
        temporary = workspace_temporary_directory()
        dataset = self.make_dataset(temporary)
        first = load_canonical_dataset(self.spec(dataset))
        cache = dataset / "cache" / "datasets" / first.dataset_fingerprint
        marker_mtime = (cache / "canonical_timeseries.csv.gz").stat().st_mtime_ns
        second = load_canonical_dataset(self.spec(dataset))
        self.assertEqual(first.dataset_fingerprint, second.dataset_fingerprint)
        self.assertEqual(marker_mtime, (cache / "canonical_timeseries.csv.gz").stat().st_mtime_ns)
        self.assertTrue((cache / "canonical_schema.json").is_file())
        self.assertTrue((cache / "poi_zone_counts.csv").is_file())

    def test_source_change_and_split_policy_invalidate_keys(self):
        temporary = workspace_temporary_directory()
        dataset = self.make_dataset(temporary)
        first = load_canonical_dataset(self.spec(dataset))
        series = pd.read_csv(dataset / "series.csv")
        series.loc[0, "demand"] += 1
        series.to_csv(dataset / "series.csv", index=False)
        second = load_canonical_dataset(self.spec(dataset))
        self.assertNotEqual(first.dataset_fingerprint, second.dataset_fingerprint)
        key_a = split_cache_key(second.dataset_fingerprint, "2024-01-02", window_hours=3, policy_id="a")
        key_b = split_cache_key(second.dataset_fingerprint, "2024-01-03", window_hours=3, policy_id="a")
        key_c = split_cache_key(second.dataset_fingerprint, "2024-01-02", window_hours=3, policy_id="b")
        self.assertEqual(3, len({key_a, key_b, key_c}))

    def test_corrupt_cache_is_rebuilt(self):
        temporary = workspace_temporary_directory()
        dataset = self.make_dataset(temporary)
        first = load_canonical_dataset(self.spec(dataset))
        artifact = dataset / "cache" / "datasets" / first.dataset_fingerprint / "feature_manifest.json"
        artifact.write_text("not json", encoding="utf-8")
        rebuilt = load_canonical_dataset(self.spec(dataset))
        self.assertEqual(first.dataset_fingerprint, rebuilt.dataset_fingerprint)
        self.assertEqual(1, json.loads(artifact.read_text(encoding="utf-8"))["schema_version"])

    def test_explicit_mapping_and_unit_conversion(self):
        temporary = workspace_temporary_directory()
        dataset = self.make_dataset(temporary)
        spec = self.spec(dataset)
        spec = DatasetSpec(**{**spec.__dict__, "unit_conversions": {"load_kwh": 0.001}})
        canonical = load_canonical_dataset(spec)
        self.assertAlmostEqual(0.001, canonical.timeseries.loc[0, "load_kwh"])

    def test_long_format_adapter_normalizes_24_hour_timestamp(self):
        temporary = workspace_temporary_directory()
        dataset = temporary / "dataset"
        dataset.mkdir()
        pd.DataFrame(
            {
                "timestamp": [
                    "2022-09-09 23:00:00",
                    "2022-09-09 24:00:00",
                ],
                "zone": ["a", "a"],
                "load": [1.0, 2.0],
                "energy_price": [1.0, 1.1],
            }
        ).to_csv(dataset / "series.csv", index=False)
        canonical = load_canonical_dataset(
            DatasetSpec(
                path=dataset,
                adapter="long_format",
                timeseries_file="series.csv",
            )
        )
        self.assertEqual(
            pd.Timestamp("2022-09-10 00:00:00"),
            canonical.timeseries["timestamp"].max(),
        )


class GlobalForecasterTests(unittest.TestCase):
    def frames(self):
        times = pd.date_range("2024-01-01", periods=80, freq="h")
        rows = []
        for zone_index, zone in enumerate(("a", "b")):
            for index, timestamp in enumerate(times):
                price = 1.0 + (index % 6) * 0.05
                rows.append(
                    {
                        "timestamp": timestamp,
                        "zone": zone,
                        "load": 10 + zone_index * 4 + np.sin(index / 4) - price,
                        "energy_price": price,
                        "temperature": 15 + np.cos(index / 8),
                        "zone_type": "residential" if zone == "a" else "commercial",
                    }
                )
        frame = pd.DataFrame(rows)
        return frame[frame.timestamp < times[64]], frame[frame.timestamp >= times[64]]

    def test_fit_predict_is_cross_zone_fixed_origin_and_reusable(self):
        history, validation = self.frames()
        artifact = GlobalForecaster("AR").fit(history, validation)
        known = validation.drop(columns=["load"])
        first = artifact.predict(known)
        changed_schedule = known[["timestamp", "zone", "energy_price"]].copy()
        changed_schedule["energy_price"] *= 1.2
        second = artifact.predict(known, changed_schedule)
        self.assertEqual({"a", "b"}, set(first.hourly.zone))
        self.assertIn("zone_type", artifact.categorical_levels)
        self.assertEqual(len(known), len(first.hourly))
        self.assertFalse(first.hourly.predicted_load.equals(second.hourly.predicted_load))
        self.assertTrue(all(timestamp < artifact.forecast_origin for timestamp in sum(artifact.history_timestamps.values(), ())))

    def test_artifact_round_trip(self):
        history, validation = self.frames()
        artifact = GlobalForecaster("lstm").fit(history, validation)
        path = workspace_temporary_directory() / "artifact.json"
        artifact.save(path)
        loaded = type(artifact).load(path)
        self.assertEqual(artifact.backend, loaded.backend)
        self.assertEqual(artifact.zones, loaded.zones)

    def test_native_artifact_calls_original_backend_with_new_prices(self):
        future_times = pd.date_range("2024-01-05", periods=3, freq="h")

        class Result:
            reusable_forecaster = {
                "zone": "a",
                "backend": "AR",
                "forecast_start": future_times[0].isoformat(),
                "horizon_hours": 3,
                "history": pd.DataFrame(
                    {
                        "time": pd.date_range("2024-01-01", periods=24, freq="h"),
                        "actual_kwh": np.arange(24, dtype=float),
                        "e_price": 1.0,
                    }
                ),
                "validation": pd.DataFrame(
                    {
                        "time": pd.date_range("2024-01-02", periods=3, freq="h"),
                        "actual_kwh": [10.0, 11.0, 12.0],
                        "e_price": 1.0,
                    }
                ),
                "known_future": pd.DataFrame(
                    {
                        "time": future_times,
                        "e_price": [1.0, 1.0, 1.0],
                        "T": [10.0, 11.0, 12.0],
                        "temp_price_idx": [10.0, 11.0, 12.0],
                    }
                ),
                "forecast_parameters": {
                    "timesfm_repo": "timesfm",
                    "timesfm_context_hours": 24,
                    "timesfm_step_horizon": 3,
                    "timesfm_exog_cols": ["e_price"],
                    "timesfm_diurnal_blend_alpha": 0.0,
                    "ar_diurnal_blend_alpha": 0.0,
                    "chronos_repo": "chronos",
                    "chronos_context_hours": 24,
                    "chronos_step_horizon": 3,
                    "chronos_exog_cols": ["e_price"],
                    "chronos_diurnal_blend_alpha": 0.0,
                    "chronos_device": "cpu",
                    "lstm_context_hours": 24,
                    "lstm_step_horizon": 3,
                    "lstm_exog_cols": ["e_price"],
                    "lstm_hidden_size": 4,
                    "lstm_num_layers": 1,
                    "lstm_epochs": 1,
                    "lstm_learning_rate": 0.001,
                    "lstm_batch_size": 2,
                    "lstm_diurnal_blend_alpha": 0.0,
                    "lstm_device": "cpu",
                    "lstm_seed": 42,
                },
                "fitted_state": {
                    "backend": "AR",
                    "energy_price_response": {"enabled": False},
                    "calibration": {"enabled": True, "bias_mean": 1.0},
                },
            }

        def fake_forecast_load(history, validation, known_future, *args, **kwargs):
            self.assertNotIn("actual_kwh", known_future.columns)
            self.assertEqual([2.0, 2.0, 2.0], known_future.e_price.tolist())
            self.assertEqual([20.0, 22.0, 24.0], known_future.temp_price_idx.tolist())
            self.assertEqual(Result.reusable_forecaster["fitted_state"], kwargs["fitted_state"])
            return pd.DataFrame(
                {"time": known_future.time, "predicted_kwh": known_future.e_price * 7}
            )

        artifact = NativeForecasterArtifact.from_forecast_result(Result())
        schedule = pd.DataFrame({"time": future_times, "a": [2.0, 2.0, 2.0]})
        with patch("forecasting.forecast_load", side_effect=fake_forecast_load) as backend:
            batch = artifact.predict(schedule)
        self.assertEqual(1, backend.call_count)
        self.assertEqual([14.0, 14.0, 14.0], batch.hourly.predicted_load.tolist())
        self.assertTrue(batch.metadata["artifact_reused"])
        self.assertFalse(batch.metadata["model_retrained"])
        self.assertFalse(batch.metadata["approximation_used"])
        restored = NativeForecasterArtifact.from_dict(artifact.to_dict())
        self.assertEqual(artifact.zone, restored.zone)
        self.assertNotIn("slope_kwh_per_energy_price", json.dumps(artifact.to_dict()))

    def test_lstm_reforecast_restores_weights_without_retraining(self):
        times = pd.date_range("2024-01-01", periods=60, freq="h")
        frame = pd.DataFrame(
            {
                "time": times,
                "actual_kwh": 10 + np.sin(np.arange(60) / 3),
                "e_price": 1 + (np.arange(60) % 4) * 0.1,
            }
        )
        history = frame.iloc[:48].copy()
        validation = frame.iloc[48:54].copy()
        known_future = frame.iloc[54:].drop(columns=["actual_kwh"]).copy()
        parameters = dict(
            context_hours=8,
            step_horizon=3,
            exog_cols=["e_price"],
            hidden_size=4,
            num_layers=1,
            epochs=1,
            learning_rate=0.001,
            batch_size=8,
            diurnal_blend_alpha=0.0,
            device="cpu",
            roll_actuals=False,
            seed=42,
        )
        initial = lstm_forecast(
            history,
            validation,
            known_future,
            known_future.time.iloc[0],
            len(known_future),
            **parameters,
        )
        fitted_state = initial.attrs["reusable_forecaster_state"]
        with patch(
            "forecasting.train_lstm_model",
            side_effect=AssertionError("reforecast must not retrain"),
        ):
            repeated = lstm_forecast(
                history,
                validation,
                known_future,
                known_future.time.iloc[0],
                len(known_future),
                **parameters,
                fitted_state=fitted_state,
            )
        np.testing.assert_allclose(initial.predicted_kwh, repeated.predicted_kwh, rtol=1e-6)

    def test_pretrained_backends_reuse_fixed_context_and_new_covariates(self):
        times = pd.date_range("2024-01-01", periods=30, freq="h")
        frame = pd.DataFrame(
            {
                "time": times,
                "actual_kwh": 10 + np.sin(np.arange(30) / 3),
                "e_price": 1.0,
            }
        )
        history = frame.iloc[:24].copy()
        validation = frame.iloc[24:27].copy()
        known_future = frame.iloc[27:].drop(columns=["actual_kwh"]).copy()
        changed_future = known_future.copy()
        changed_future["e_price"] = 2.0

        def fake_timesfm(
            model,
            context_load,
            exog_context,
            exog_horizon,
            horizon,
            context_hours,
            exog_cols,
        ):
            point = 12.0 - exog_horizon[:horizon, 0]
            return point, point - 1.0, point + 1.0

        with patch("forecasting.load_timesfm_model", return_value=object()), patch(
            "forecasting.run_timesfm_prediction", side_effect=fake_timesfm
        ):
            initial_timesfm = timesfm_forecast(
                history,
                validation,
                known_future,
                known_future.time.iloc[0],
                3,
                repo="timesfm",
                context_hours=24,
                step_horizon=3,
                exog_cols=["e_price"],
                diurnal_blend_alpha=0.0,
                roll_actuals=False,
            )
            repeated_timesfm = timesfm_forecast(
                history,
                validation,
                changed_future,
                changed_future.time.iloc[0],
                3,
                repo="timesfm",
                context_hours=24,
                step_horizon=3,
                exog_cols=["e_price"],
                diurnal_blend_alpha=0.0,
                roll_actuals=False,
                fitted_state=initial_timesfm.attrs["reusable_forecaster_state"],
            )
        self.assertFalse(
            initial_timesfm.predicted_kwh.equals(repeated_timesfm.predicted_kwh)
        )

        def fake_chronos(
            pipeline,
            context_load,
            horizon,
            context_hours,
            *,
            exog_context,
            exog_horizon,
            exog_cols,
        ):
            point = 12.0 - exog_horizon[:horizon, 0]
            return point, point - 1.0, point + 1.0

        with patch("forecasting.load_chronos_model", return_value=object()), patch(
            "forecasting.run_chronos_prediction", side_effect=fake_chronos
        ):
            initial_chronos = chronos_forecast(
                history,
                validation,
                known_future,
                known_future.time.iloc[0],
                3,
                repo="chronos",
                context_hours=24,
                step_horizon=3,
                exog_cols=["e_price"],
                diurnal_blend_alpha=0.0,
                device="cpu",
                roll_actuals=False,
            )
            repeated_chronos = chronos_forecast(
                history,
                validation,
                changed_future,
                changed_future.time.iloc[0],
                3,
                repo="chronos",
                context_hours=24,
                step_horizon=3,
                exog_cols=["e_price"],
                diurnal_blend_alpha=0.0,
                device="cpu",
                roll_actuals=False,
                fitted_state=initial_chronos.attrs["reusable_forecaster_state"],
            )
        self.assertFalse(
            initial_chronos.predicted_kwh.equals(repeated_chronos.predicted_kwh)
        )


class ControlOutputTests(unittest.TestCase):
    def test_agent_http_client_is_closed_once_on_own_event_loop(self):
        class FakeAsyncOpenAI:
            def __init__(self):
                self.close_loops = []

            async def close(self):
                self.close_loops.append(asyncio.get_running_loop())

        transport = FakeAsyncOpenAI()
        client = object.__new__(AgentChatClient)
        client._client = transport
        client._closed = False

        async def exercise():
            active_loop = asyncio.get_running_loop()
            await client.aclose()
            await client.aclose()
            self.assertIs(active_loop, transport.close_loops[0])

        asyncio.run(exercise())
        self.assertEqual(1, len(transport.close_loops))

    def test_forecaster_to_control_dry_run_handoff(self):
        root = workspace_temporary_directory()
        dataset = root / "dataset"
        dataset.mkdir()
        times = pd.date_range("2024-01-01", periods=240, freq="h")
        index = np.arange(len(times))
        pd.DataFrame(
            {
                "timestamp": times,
                "zone": "a",
                "load": 20 + 3 * np.sin(index * 2 * np.pi / 24),
                "energy_price": 1 + 0.1 * (index % 6),
                "service_price": 0.5,
                "occupancy": 0.4,
                "temperature": 15 + np.sin(index * 2 * np.pi / 24),
                "humidity": 60,
                "rain": 0,
            }
        ).to_csv(dataset / "series.csv", index=False)
        common = dict(
            data_dir=dataset,
            output_dir=root / "output",
            config_path=root / "missing.yaml",
            dataset_spec=DatasetSpec(
                path=dataset,
                adapter="long_format",
                timeseries_file="series.csv",
            ),
            dry_run=True,
            forecast_start="2024-01-09 00:00:00",
            horizon_days=1,
            history_days=5,
            validation_days=1,
            zone_ids=["a"],
            forecast_model="AR",
        )
        forecaster_outputs = run_pipeline(
            **common,
            pipeline_stage="forecaster",
            agent_mode="single_agent_price_retry",
        )
        outputs = run_pipeline(
            **common,
            pipeline_stage="agent",
            forecaster_output_dir=forecaster_outputs["forecaster_output_dir"],
            agent_mode="single_agent_price_retry",
        )
        payload = json.loads(outputs["control_results_json"].read_text(encoding="utf-8"))
        self.assertIn(payload["status"], {"success", "fail"})
        self.assertEqual(1, len(payload["zones"]))
        self.assertTrue(forecaster_outputs["forecaster_artifact_json"].is_file())
        artifact_payload = json.loads(
            forecaster_outputs["forecaster_artifact_json"].read_text(encoding="utf-8")
        )
        self.assertEqual("native_backend_reforecast", artifact_payload["artifact_type"])
        self.assertEqual(
            "native_backend_reforecast",
            artifact_payload["artifacts"]["a"]["artifact_type"],
        )
        self.assertTrue(
            all(
                "actual_kwh" not in row and "load" not in row
                for row in artifact_payload["artifacts"]["a"]["known_future"]
            )
        )
        self.assertFalse(
            payload["zones"][0]["price_conditioned_forecast_metadata"][
                "approximation_used"
            ]
        )
        self.assertFalse(payload["agent_cumulative_usage"]["agent_invoked"])
        self.assertEqual(0, payload["agent_cumulative_usage"]["total_tokens"])
        usage_frame = pd.read_csv(outputs["agent_attempt_usage_csv"])
        self.assertTrue((usage_frame["agent_call_count"] == 0).all())
        self.assertTrue(usage_frame["token_usage_complete"].all())

    def test_supported_modes_use_expected_retry_chain(self):
        async def fake_chain(context, **kwargs):
            return {"zone_id": context["zone_id"], "source": "full"}

        async def fake_price(context, previous, **kwargs):
            return {"zone_id": context["zone_id"], "source": "price"}

        jobs = [({"zone_id": "a"}, {"zone_id": "a"}, {("s", "e")})]
        expected = {
            "multi_agent_economist_retry": (0, 1),
            "multi_agent_full_retry": (1, 0),
            "multi_agent_discussion_3rounds": (1, 0),
            "single_agent_price_retry": (0, 1),
            "single_agent_full_retry": (1, 0),
        }
        for mode, counts in expected.items():
            with self.subTest(mode=mode), patch(
                "orchestrator.run_zone_chain", side_effect=fake_chain
            ) as chain, patch(
                "orchestrator.retry_zone_prices", side_effect=fake_price
            ) as retry:
                result = asyncio.run(
                    revise_control_reports(
                        jobs,
                        client=None,
                        temperature=0.2,
                        heuristic_source="test",
                        agent_mode=mode,
                    )
                )
                self.assertEqual(counts, (chain.call_count, retry.call_count))
                self.assertEqual("a", result[0][0]["zone_id"])
                if mode == "multi_agent_discussion_3rounds":
                    self.assertEqual(
                        "multi_agent_discussion",
                        chain.call_args.kwargs["chain_mode"],
                    )

    def test_usage_marks_missing_provider_tokens_incomplete(self):
        complete = agent_call_usage_record(
            stage="agent.grid",
            agent="Grid Analyst",
            zone_id="a",
            token_usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        )
        incomplete = agent_call_usage_record(
            stage="agent.economist",
            agent="Market Economist",
            zone_id="a",
            token_usage={},
        )
        summary = summarize_agent_call_usage([complete, incomplete])
        self.assertEqual(2, summary["agent_call_count"])
        self.assertEqual(15, summary["total_tokens"])
        self.assertFalse(summary["token_usage_complete"])
        self.assertFalse(incomplete["token_usage_complete"])
        self.assertNotIn("elapsed_seconds", complete)
        self.assertNotIn("elapsed_seconds", summary)

    def test_three_agents_exchange_information_for_three_rounds(self):
        class FakeDiscussionClient:
            def __init__(self):
                self.prompts = []

            async def complete_json(self, prompt, *, temperature):
                self.prompts.append(prompt)
                call_index = len(self.prompts) - 1
                discussion_round = call_index // 3 + 1
                role_index = call_index % 3
                usage = {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "token_usage_complete": True,
                }
                if role_index == 0:
                    response = {
                        "reasoning_summary": f"grid round {discussion_round}",
                        "adjustment_needed": True,
                        "confidence": "medium",
                        "message_to_other_agents": (
                            f"grid message round {discussion_round}"
                        ),
                        "window_assessments": [
                            {
                                "window_start": "2024-01-01 00:00:00",
                                "window_end": "2024-01-01 02:00:00",
                                "predicted_load_kwh": 30.0,
                                "load_range_position_pct": 85.0,
                                "grid_stress_level": "High",
                                "adjustment_needed": True,
                                "reasoning_summary": "high forecast window",
                            }
                        ],
                    }
                elif role_index == 1:
                    response = {
                        "reasoning_summary": f"behavior round {discussion_round}",
                        "demand_drivers": ["weather", "occupancy"],
                        "elasticity_factor": 0.2,
                        "window_elasticities": [0.2],
                        "confidence": "medium",
                        "message_to_other_agents": (
                            f"behavior message round {discussion_round}"
                        ),
                    }
                else:
                    response = {
                        "reasoning_summary": f"economist round {discussion_round}",
                        "suggested_price_shift_pct": 10.0 + discussion_round,
                        "action_label": "Raise energy price",
                        "price_rationale": (
                            f"shared conclusion round {discussion_round}"
                        ),
                        "message_to_other_agents": (
                            f"economist message round {discussion_round}"
                        ),
                        "price_change_windows_3h": [
                            {
                                "window_start": "2024-01-01 00:00:00",
                                "window_end": "2024-01-01 02:00:00",
                                "suggested_price_shift_pct": 10.0
                                + discussion_round,
                                "proposed_energy_price": 1.1
                                + discussion_round * 0.01,
                                "action_label": "Raise energy price",
                                "price_rationale": (
                                    f"round {discussion_round} joint decision"
                                ),
                            }
                        ],
                    }
                response["_agent_completion_usage"] = usage
                return response

        context = discussion_test_context()
        client = FakeDiscussionClient()
        report = asyncio.run(
            run_zone_chain(
                context,
                client=client,
                chain_mode="multi_agent_discussion",
            )
        )
        self.assertEqual(9, len(client.prompts))
        self.assertEqual("multi_agent_discussion_3rounds", report["source"])
        self.assertEqual(3, report["agent_discussion_round_count"])
        self.assertEqual(3, len(report["agent_discussion_rounds"]))
        self.assertEqual(9, report["agent_call_count"])
        self.assertEqual(135, report["agent_total_tokens"])
        self.assertIn("economist round 1", client.prompts[3])
        for prompt_index in (3, 6):
            compact_prompt = client.prompts[prompt_index]
            self.assertIn('"conclusion_summary"', compact_prompt)
            self.assertIn('"disagreements"', compact_prompt)
            self.assertIn('"key_decisions"', compact_prompt)
            self.assertNotIn('"grid_output"', compact_prompt)
            self.assertNotIn('"behavior_output"', compact_prompt)
            self.assertNotIn('"economist_output"', compact_prompt)
        self.assertEqual(
            "economist round 3",
            report["economist_reasoning_summary"],
        )
        self.assertEqual(
            [3, 3, 3],
            [
                item["agent_usage"]["agent_call_count"]
                for item in report["agent_discussion_rounds"]
            ],
        )
        self.assertFalse(report["agent_discussion_converged"])
        self.assertEqual("max_rounds_reached", report["agent_discussion_stop_reason"])
        self.assertTrue(
            all(
                len(item["agent_call_usage"]) == 3
                for item in report["agent_discussion_rounds"]
            )
        )
        self.assertEqual(
            {"conclusion_summary", "disagreements", "key_decisions"},
            set(
                report["agent_discussion_rounds"][0][
                    "next_round_handoff"
                ]
            ),
        )

    def test_discussion_stops_after_equal_first_and_second_decisions(self):
        class StableDiscussionClient:
            def __init__(self):
                self.prompts = []

            async def complete_json(self, prompt, *, temperature):
                self.prompts.append(prompt)
                call_index = len(self.prompts) - 1
                discussion_round = call_index // 3 + 1
                role_index = call_index % 3
                if role_index == 0:
                    response = {
                        "reasoning_summary": f"grid explanation {discussion_round}",
                        "adjustment_needed": True,
                        "window_assessments": [
                            {
                                "window_start": "2024-01-01 00:00:00",
                                "window_end": "2024-01-01 02:00:00",
                                "predicted_load_kwh": 30.0,
                                "load_range_position_pct": 85.0,
                                "grid_stress_level": "High",
                                "adjustment_needed": True,
                                "reasoning_summary": (
                                    f"grid window explanation {discussion_round}"
                                ),
                            }
                        ],
                    }
                elif role_index == 1:
                    response = {
                        "reasoning_summary": (
                            f"behavior explanation {discussion_round}"
                        ),
                        "demand_drivers": ["weather"],
                        "elasticity_factor": 0.2,
                        "window_elasticities": [0.2],
                    }
                else:
                    response = {
                        "reasoning_summary": (
                            f"economist explanation {discussion_round}"
                        ),
                        "suggested_price_shift_pct": 10.0,
                        "action_label": "Raise energy price",
                        "price_rationale": f"rationale {discussion_round}",
                        "price_change_windows_3h": [
                            {
                                "window_start": "2024-01-01 00:00:00",
                                "window_end": "2024-01-01 02:00:00",
                                "suggested_price_shift_pct": 10.0,
                                "proposed_energy_price": 1.1,
                                "action_label": "Raise energy price",
                                "price_rationale": (
                                    f"window rationale {discussion_round}"
                                ),
                            }
                        ],
                    }
                response["_agent_completion_usage"] = {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10,
                    "token_usage_complete": True,
                }
                return response

        client = StableDiscussionClient()
        report = asyncio.run(
            run_zone_chain(
                discussion_test_context(),
                client=client,
                chain_mode="multi_agent_discussion",
            )
        )
        self.assertEqual(6, len(client.prompts))
        self.assertEqual(2, report["agent_discussion_round_count"])
        self.assertTrue(report["agent_discussion_converged"])
        self.assertEqual(
            "round_2_matches_round_1",
            report["agent_discussion_stop_reason"],
        )
        self.assertFalse(
            report["agent_discussion_rounds"][0]["matches_previous_round"]
        )
        self.assertTrue(
            report["agent_discussion_rounds"][1]["matches_previous_round"]
        )
        self.assertEqual(6, report["agent_call_count"])
        self.assertEqual(60, report["agent_total_tokens"])

    def test_provider_retry_token_usage_is_marked_incomplete(self):
        class FakeCompletions:
            def __init__(self):
                self.calls = 0

            async def create(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise json.JSONDecodeError("truncated", "{", 1)
                return {
                    "choices": [{"message": {"content": "{}"}}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                }

        completions = FakeCompletions()
        client = object.__new__(AgentChatClient)
        client.config = SimpleNamespace(
            model="test-model",
            reasoning_effort=None,
            provider_json_retries=1,
            provider_json_retry_backoff_seconds=0.0,
            max_concurrent_requests=1,
        )
        client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        response = asyncio.run(client.complete_json("test", temperature=0.0))
        usage = response["_agent_completion_usage"]
        self.assertEqual(2, usage["provider_attempt_count"])
        self.assertFalse(usage["token_usage_complete"])

    def test_control_attempt_snapshot_tracks_price_deltas_and_zero_previous(self):
        report = {
            "zone_id": "a",
            "control_status": "fail",
            "failed_window_count": 1,
            "agent_call_usage": [],
            "price_change_windows_3h": [
                {
                    "window_start": "2024-01-01 00:00:00",
                    "window_end": "2024-01-01 02:00:00",
                    "mean_energy_price": 1.0,
                    "proposed_energy_price": 1.2,
                    "suggested_price_shift_pct": 20.0,
                    "price_conditioned_baseline_load_kwh": 20.0,
                    "price_conditioned_load_range_position_pct": 85.0,
                    "price_conditioned_load_stress_level": "High",
                    "control_success": False,
                    "control_failure_reason": "price_conditioned_load_is_High",
                    "price_rationale": "initial",
                }
            ],
        }
        first = control_attempt_snapshot(report, 1)
        first_window = first["windows"][0]
        self.assertEqual("initial", first_window["proposal_status"])
        self.assertEqual(1.0, first_window["previous_proposed_energy_price"])
        self.assertEqual(0.2, first_window["price_change_amount"])
        self.assertEqual(20.0, first_window["price_change_pct_vs_previous"])

        second = control_attempt_snapshot(
            report,
            2,
            previous_snapshot=first,
            revised_keys={("2024-01-01 00:00:00", "2024-01-01 02:00:00")},
            proposal_phase="economist_retry",
            triggered_by_attempt=1,
        )
        second_window = second["windows"][0]
        self.assertEqual("revised", second_window["proposal_status"])
        self.assertFalse(second_window["price_changed"])
        self.assertEqual(
            "price_conditioned_load_is_High",
            second_window["trigger_failure_reason"],
        )

        zero_previous = copy.deepcopy(first)
        zero_previous["windows"][0]["proposed_energy_price"] = 0.0
        third_report = copy.deepcopy(report)
        third_report["price_change_windows_3h"][0]["proposed_energy_price"] = 1.0
        third = control_attempt_snapshot(
            third_report,
            3,
            previous_snapshot=zero_previous,
            revised_keys={("2024-01-01 00:00:00", "2024-01-01 02:00:00")},
            proposal_phase="economist_retry",
            triggered_by_attempt=2,
        )
        self.assertIsNone(third["windows"][0]["price_change_pct_vs_previous"])

    def test_agent_cannot_rewrite_forecaster_window_identity_to_24_hour(self):
        fallback = [
            {
                "window_start": "2022-09-09 21:00:00",
                "window_end": "2022-09-09 23:00:00",
                "mean_energy_price": 1.0,
                "sum_predicted_kwh": 10.0,
                "load_stress_level": "High",
            }
        ]
        model_output = [
            {
                "window_start": "2022-09-09 21:00:00",
                "window_end": "2022-09-09 24:00:00",
                "suggested_price_shift_pct": 10.0,
                "proposed_energy_price": 1.1,
                "action_label": "Raise energy price",
                "price_rationale": "high load",
            }
        ]
        normalized = normalize_price_windows(model_output, fallback)
        self.assertEqual("2022-09-09 23:00:00", normalized[0]["window_end"])

    def test_control_boundary_accepts_external_24_hour_timestamp(self):
        self.assertEqual(
            pd.Timestamp("2022-09-10 00:00:00"),
            parse_datetime_24h("2022-09-09 24:00:00"),
        )
        frame = pd.DataFrame(
            {
                "time": pd.date_range("2022-09-09 21:00:00", periods=4, freq="h"),
                "a": [1.0, 1.0, 1.0, 1.0],
            }
        )
        updated = energy_price_with_predicted_windows(
            frame,
            "a",
            [
                {
                    "window_start": "2022-09-09 21:00:00",
                    "window_end": "2022-09-09 24:00:00",
                    "proposed_energy_price": 2.0,
                }
            ],
        )
        self.assertEqual([2.0, 2.0, 2.0, 2.0], updated["a"].tolist())

    def test_control_loop_accumulates_usage_and_keeps_frozen_window_trace(self):
        window = {
            "window_start": "2024-01-01 00:00:00",
            "window_end": "2024-01-01 02:00:00",
            "mean_energy_price": 1.0,
            "proposed_energy_price": 1.1,
            "predicted_energy_price": 1.1,
            "suggested_price_shift_pct": 10.0,
            "price_valid": True,
            "price_rationale": "initial proposal",
        }

        def usage(zone_id, stage, total):
            return {
                "stage": stage,
                "agent": stage,
                "zone_id": zone_id,
                "prompt_tokens": total - 2,
                "completion_tokens": 2,
                "total_tokens": total,
                "token_usage_complete": True,
                "provider_attempt_count": 1,
            }

        reports = [
            {
                "zone_id": zone_id,
                "category": "test",
                "price_change_windows_3h": [copy.deepcopy(window)],
                "agent_call_usage": [usage(zone_id, "agent.grid", 15)],
            }
            for zone_id in ("a", "b")
        ]
        contexts = [
            {"zone_id": zone_id, "pricing_windows_3h": [copy.deepcopy(window)]}
            for zone_id in ("a", "b")
        ]
        forecast_round = {"value": 0}

        def fake_reforecast(*, reports, **kwargs):
            forecast_round["value"] += 1
            current_round = forecast_round["value"]
            updated_reports = []
            for report in reports:
                updated = copy.deepcopy(report)
                for item in updated["price_change_windows_3h"]:
                    stress = (
                        "High"
                        if current_round == 1 and report["zone_id"] == "b"
                        else "Medium"
                    )
                    item["price_conditioned_baseline_load_kwh"] = 20.0
                    item["price_conditioned_load_range_position_pct"] = (
                        85.0 if stress == "High" else 50.0
                    )
                    item["price_conditioned_load_stress_level"] = stress
                    item["price_conditioned_baseline_source"] = "test_forecaster"
                updated_reports.append(updated)
            return updated_reports

        async def fake_revise(jobs, **kwargs):
            revised = []
            for _, previous, failed_keys in jobs:
                revision = copy.deepcopy(previous)
                revision["agent_call_usage"] = [
                    usage(previous["zone_id"], "agent.economist_retry", 10)
                ]
                for item in revision["price_change_windows_3h"]:
                    item["proposed_energy_price"] = 1.3
                    item["predicted_energy_price"] = 1.3
                    item["suggested_price_shift_pct"] = 30.0
                    item["price_rationale"] = "retry proposal"
                revised.append((revision, failed_keys))
            return revised

        with patch(
            "orchestrator.apply_price_conditioned_baseline_forecasts",
            side_effect=fake_reforecast,
        ), patch(
            "orchestrator.revise_control_reports",
            side_effect=fake_revise,
        ), patch("orchestrator.price_conditioned_parameters", return_value={}):
            controlled = asyncio.run(
                run_control_loop(
                    reports=reports,
                    contexts=contexts,
                    client=object(),
                    heuristic_source="test",
                    agent_mode="multi_agent_economist_retry",
                    temperature=0.2,
                    pipeline_data=object(),
                    zone_load_thresholds=pd.DataFrame(),
                    price_change_reference=None,
                    forecast_parameters={},
                    scenario_artifacts={},
                )
            )

        by_zone = {report["zone_id"]: report for report in controlled}
        self.assertEqual(2, len(by_zone["a"]["control_attempt_trace"]))
        self.assertEqual(
            "frozen",
            by_zone["a"]["control_attempt_trace"][1]["windows"][0][
                "proposal_status"
            ],
        )
        self.assertFalse(
            by_zone["a"]["control_attempt_trace"][1]["agent_usage"][
                "agent_invoked"
            ]
        )
        retry = by_zone["b"]["control_attempt_trace"][1]
        self.assertEqual("revised", retry["windows"][0]["proposal_status"])
        self.assertEqual(1.1, retry["windows"][0]["previous_proposed_energy_price"])
        self.assertEqual(1.3, retry["windows"][0]["proposed_energy_price"])
        self.assertEqual(0.2, retry["windows"][0]["price_change_amount"])
        self.assertEqual(2, by_zone["b"]["agent_cumulative_usage"]["agent_call_count"])
        self.assertEqual(25, by_zone["b"]["agent_total_tokens"])

        temporary = workspace_temporary_directory()
        outputs = write_agent_outputs(
            output_dir=temporary,
            reports=controlled,
            forecast_model="AR",
            agent_mode="multi_agent_economist_retry",
        )
        payload = json.loads(
            outputs["control_results_json"].read_text(encoding="utf-8")
        )
        self.assertEqual(2, len(payload["agent_round_usage"]))
        self.assertEqual(40, payload["agent_cumulative_usage"]["total_tokens"])
        self.assertEqual(40, payload["agent_token_totals"]["total_tokens"])
        self.assertEqual(3, len(payload["agent_step_token_usage"]))
        self.assertNotIn("elapsed", json.dumps(payload))
        self.assertNotIn("wall_time", json.dumps(payload))
        usage_frame = pd.read_csv(outputs["agent_attempt_usage_csv"])
        self.assertEqual(6, len(usage_frame))
        self.assertNotIn("agent_elapsed_seconds", usage_frame.columns)
        self.assertNotIn("agent_batch_wall_time_seconds", usage_frame.columns)
        step_frame = pd.read_csv(outputs["agent_step_token_usage_csv"])
        self.assertEqual(3, len(step_frame))
        self.assertEqual(40, int(step_frame.total_tokens.sum()))
        self.assertEqual(
            ["agent.grid", "agent.grid", "agent.economist_retry"],
            step_frame.stage.tolist(),
        )
        self.assertEqual(
            1,
            int(
                usage_frame[
                    (usage_frame.record_scope == "global")
                    & (usage_frame.attempt == 2)
                ].iloc[0].agent_invoked_zone_count
            ),
        )
        attempt_frame = pd.read_csv(outputs["control_attempt_trace_csv"])
        self.assertEqual(4, len(attempt_frame))
        self.assertIn("previous_proposed_energy_price", attempt_frame.columns)
        self.assertIn("proposal_status", attempt_frame.columns)

    def test_price_reference_uses_nonoverlapping_three_hour_changes(self):
        times = pd.date_range("2024-01-01", periods=9, freq="h")
        frame = pd.DataFrame(
            {
                "timestamp": times,
                "zone_id": ["a"] * 9,
                "load_kwh": [1.0] * 9,
                "energy_price": [1.0] * 3 + [2.0] * 3 + [1.0] * 3,
            }
        )
        reference = build_price_change_reference(frame)
        self.assertEqual((50.0, 100.0), reference.values_pct)
        self.assertEqual(100.0, reference.percentile(100))
        self.assertGreater(reference.p95_pct, 90)

    def test_agent_allowlist_drops_evaluation(self):
        safe = agent_safe_context(
            {
                "zone_id": "a",
                "forecast_total_kwh": 10,
                "actual_total_kwh": 11,
                "metrics": {"MAE": 1},
                "pricing_windows_3h": [
                    {
                        "window_start": "2024-01-01 00:00:00",
                        "window_end": "2024-01-01 02:00:00",
                        "sum_predicted_kwh": 10,
                        "sum_actual_kwh": 11,
                    }
                ],
            }
        )
        encoded = json.dumps(safe)
        self.assertNotIn("actual", encoded)
        self.assertNotIn("MAE", encoded)

    def test_authoritative_control_outputs_are_versioned(self):
        report = {
            "zone_id": "a",
            "category": "test",
            "control_status": "success",
            "attempts_used": 1,
            "failed_window_count": 0,
            "predicted_load_kwh": 10,
            "grid_stress_level": "Medium",
            "price_rationale": "stay in Medium",
            "control_attempt_trace": [{"attempt": 1, "control_status": "success", "windows": []}],
            "price_change_windows_3h": [
                {
                    "window_start": "2024-01-01 00:00:00",
                    "window_end": "2024-01-01 02:00:00",
                    "sum_predicted_kwh": 10,
                    "mean_energy_price": 1.0,
                    "proposed_energy_price": 1.1,
                    "suggested_price_shift_pct": 10,
                    "price_valid": True,
                    "price_conditioned_baseline_load_kwh": 9,
                    "price_conditioned_load_stress_level": "Medium",
                    "control_success": True,
                }
            ],
        }
        temporary = workspace_temporary_directory()
        outputs = write_agent_outputs(
                output_dir=temporary,
                reports=[report],
                forecast_model="AR",
                agent_mode="single_agent_price_retry",
                manifest={
                    "dataset_fingerprint": "abc",
                    "cache_keys": {"split": "def"},
                    "feature_manifest": {"dynamic_features": ["temperature"]},
                    "forecast_origin": "2024-01-01T00:00:00",
                },
            )
        payload = json.loads(outputs["control_results_json"].read_text(encoding="utf-8"))
        self.assertEqual(3, payload["schema_version"])
        self.assertEqual("success", payload["status"])
        self.assertEqual("abc", payload["dataset_fingerprint"])
        self.assertIn("agent_round_usage", payload)
        self.assertIn("agent_cumulative_usage", payload)
        self.assertIn("agent_token_totals", payload)
        self.assertTrue(outputs["agent_attempt_usage_csv"].is_file())
        self.assertTrue(outputs["agent_step_token_usage_csv"].is_file())
        self.assertNotIn("actual", json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
