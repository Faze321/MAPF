"""Offline training resolution and one-time experiment preparation contracts."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from config import AppConfig, RunConfig, TrainingConfig
from main import build_parser, main
from orchestrator import make_forecast_parameters, forecast_runtime_parameters, run_experiment_matrix, run_training_stage
from orchestrator import load_forecaster_bundle, FORECASTER_MANIFEST_SCHEMA_VERSION
from orchestrator import build_contexts, PipelineStageError
from global_forecaster import NATIVE_ARTIFACT_SCHEMA_VERSION


class TrainingConfigTests(unittest.TestCase):
    def test_fixed_defaults_use_explicit_window_independent_hyperparameters(self):
        config = TrainingConfig()
        lstm = config.resolve("dataset", "001", "lstm")
        xgb = config.resolve("dataset", "001", "xgboost")
        self.assertEqual((lstm["lstm_hidden_size"], lstm["lstm_num_layers"], lstm["lstm_batch_size"]), (32, 1, 64))
        self.assertEqual((lstm["lstm_epochs"], lstm["lstm_learning_rate"]), (100, .001))
        self.assertEqual((xgb["xgboost_n_estimators"], xgb["xgboost_max_depth"], xgb["xgboost_n_jobs"]), (800, 3, 2))
        self.assertNotIn("lstm_context_hours", lstm)
        self.assertNotIn("xgboost_lags", xgb)
        self.assertEqual((config.train_ratio, config.validation_ratio, config.sample_stride_hours, config.min_samples), (.8, .2, 1, 512))

    def test_precedence_and_dataset_zone_isolation(self):
        config = TrainingConfig(seed=7, defaults={"lstm_hidden_size": 8},
                                dataset_overrides={"dataset": {"lstm_hidden_size": 16}},
                                zone_overrides={"dataset": {"001": {"lstm_hidden_size": 24}}})
        self.assertEqual(config.resolve("other", "001", "lstm")["lstm_hidden_size"], 8)
        self.assertEqual(config.resolve("dataset", "002", "lstm")["lstm_hidden_size"], 16)
        self.assertEqual(config.resolve(Path("dataset").resolve(), "001", "lstm")["lstm_hidden_size"], 24)
        actual = config.resolve("dataset", "001", "lstm", {"lstm_hidden_size": 32, "lstm_seed": 99})
        self.assertEqual(actual["lstm_hidden_size"], 32)
        self.assertEqual(actual["lstm_seed"], 99)
        self.assertEqual(config.resolve("dataset", "001", "xgboost")["xgboost_seed"], 7)

    def test_invalid_configuration_is_rejected(self):
        cases = ({"lstm_epochs": 0}, {"lstm_num_layers": True}, {"lstm_learning_rate": float("nan")},
                 {"lstm_dropout": 1.0}, {"xgboost_subsample": 0.0}, {"xgboost_lags": []},
                 {"xgboost_lags": [1, 1]}, {"lstm_exog_cols": ["actual_kwh"]},
                 {"xgboost_device": "cuda:bad"}, {"lstm_device": "cuda:bad"}, {"typo": 1})
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                TrainingConfig(defaults=values)
        for values in ({"train_ratio": .7}, {"train_ratio": 1, "validation_ratio": 0},
                       {"train_ratio": float("nan")}, {"sample_stride_hours": 0}, {"min_samples": 0},
                       {"validation_days": 14}, {"min_reference_samples": 512}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                TrainingConfig.from_mapping(values)
        with self.assertRaises(ValueError):
            TrainingConfig().resolve("dataset", "001", "lstm", {"misspelt": 1})

    def test_training_workers_accept_any_positive_integer(self):
        for workers in (1, 2, 3, 4, 16):
            config = TrainingConfig(lstm_workers=workers, xgboost_workers=workers)
            self.assertEqual((config.lstm_workers, config.xgboost_workers), (workers, workers))
        for key in ("lstm_workers", "xgboost_workers"):
            for value in (0, -1, True, 1.5, "3"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    TrainingConfig(**{key: value})

    def test_cpu_and_explicit_batch_overrides_preserve_precedence(self):
        config = TrainingConfig(defaults={"lstm_device": "cpu"})
        self.assertEqual(config.resolve("dataset", "001", "lstm")["lstm_batch_size"], 64)
        resolved = config.resolve("dataset", "001", "lstm", {"lstm_batch_size": 16})
        self.assertEqual((resolved["lstm_device"], resolved["lstm_batch_size"]), ("cpu", 16))

    def test_cuda_defaults_and_explicit_cpu_or_gpu_index_overrides(self):
        config = TrainingConfig()
        for backend in ("lstm", "xgboost"):
            key = backend + "_device"
            self.assertEqual(config.resolve("dataset", "001", backend)[key], "cuda")
            for device in ("cpu", "cuda", "cuda:0", "cuda:2"):
                with self.subTest(backend=backend, device=device):
                    self.assertEqual(config.resolve("dataset", "001", backend, {key: device})[key], device)
            for device in ("cuda:", "cuda:-1", "cuda:1.5", "cuda:bad", "gpu"):
                with self.subTest(backend=backend, device=device), self.assertRaises(ValueError):
                    TrainingConfig(defaults={key: device})
        self.assertEqual(config.resolve("dataset", "001", "xgboost")["xgboost_tree_method"], "hist")

    def test_training_keys_cannot_remain_under_run(self):
        for key in ("lstm_epochs", "xgboost_lags", "lstm_unrecognized"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "training.defaults"):
                RunConfig.from_mapping({key: 1})
        self.assertEqual(RunConfig.from_mapping({"lstm_diurnal_blend_alphas": [0]}).lstm_diurnal_blend_alphas, [0.0])
        self.assertFalse(hasattr(RunConfig(), "lstm_epochs"))

    def test_cli_exposes_training_parameters_without_abbreviations(self):
        args = build_parser().parse_args(["--stage", "train", "--lstm-epochs", "3", "--xgboost-n-estimators", "4",
                                          "--training-train-ratio", ".7", "--training-validation-ratio", ".3",
                                          "--training-seed", "9", "--training-lstm-workers", "3",
                                          "--training-xgboost-workers", "4", "--force-train"])
        self.assertEqual(args.lstm_epochs, 3)
        self.assertEqual(args.xgboost_n_estimators, 4)
        self.assertEqual((args.training_train_ratio, args.training_validation_ratio), (.7, .3))
        self.assertEqual(args.training_seed, 9)
        self.assertEqual((args.training_lstm_workers, args.training_xgboost_workers), (3, 4))
        self.assertTrue(args.force_train)
        self.assertFalse(build_parser().allow_abbrev)

    def test_train_cli_forwards_selection_and_windows_without_agent_or_dates(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.yaml"
            path.write_text("run:\n  data_dir: dataset\n  pipeline_stage: train\n"
                            "  zones: ['001']\n  experiment_zone_selection: random\n"
                            "  experiment_zone_seed: 19\n  experiment_zone_count: 2\n"
                            "training:\n  defaults:\n    lstm_epochs: 7\n", encoding="utf-8")
            config = AppConfig.from_file(path, load_agent=False)
            self.assertEqual(config.training.defaults["lstm_epochs"], 7)
            with patch("main.resolve_dataset_spec", side_effect=lambda spec: spec), \
                 patch("main.run_training_stage", return_value={}) as train, \
                 patch("main.run_experiment_matrix") as matrix, \
                 patch("main.default_forecast_start") as dates, redirect_stdout(io.StringIO()):
                main(["--config", str(path), "--lstm-epochs", "2", "--force-train",
                      "--zones", "002", "--history-days", "7", "--horizon-days", "2",
                      "--training-train-ratio", "0.7", "--training-validation-ratio", "0.3",
                      "--training-min-samples", "10", "--training-sample-stride-hours", "2"])
            self.assertEqual(train.call_args.kwargs["backends"], ["lstm", "xgboost"])
            self.assertEqual(train.call_args.kwargs["training_cli_overrides"], {"lstm_epochs": 2})
            self.assertTrue(train.call_args.kwargs["force_train"])
            self.assertEqual(train.call_args.kwargs["zone_ids"], ["002"])
            self.assertEqual(train.call_args.kwargs["history_days"], 7)
            self.assertEqual(train.call_args.kwargs["horizon_days"], 2)
            self.assertEqual(train.call_args.kwargs["experiment_zone_selection"], "random")
            self.assertEqual(train.call_args.kwargs["experiment_zone_count"], 2)
            self.assertEqual(train.call_args.kwargs["experiment_zone_seed"], 19)
            policies = train.call_args.kwargs["training_config"]
            self.assertEqual((policies.train_ratio, policies.validation_ratio, policies.min_samples,
                              policies.sample_stride_hours), (0.7, 0.3, 10, 2))
            matrix.assert_not_called()
            dates.assert_not_called()

    def test_forecast_identity_contains_model_ids_without_training_defaults(self):
        parameters = make_forecast_parameters(forecast_origin="2026-01-01", backend="lstm",
                                              history_days=30, validation_days=5, horizon_days=1,
                                              diurnal_blend_alpha=0, model_ids={"001": "m1"})
        self.assertEqual(parameters["model_ids"], {"001": "m1"})
        self.assertFalse(any(key.startswith(("lstm_", "xgboost_")) for key in parameters))
        self.assertNotIn("model_ids", forecast_runtime_parameters(parameters))

    def test_train_stage_honors_custom_cache_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "custom"
            dataset = SimpleNamespace(cache_dir=cache)
            with patch("orchestrator.load_canonical_dataset", return_value=dataset), \
                 patch("orchestrator.eligible_training_zone_ids", return_value=["001"]), \
                 patch("orchestrator.prepare_offline_models", return_value={}) as prepare:
                outputs = run_training_stage(data_dir=Path("dataset"), cache_dir=cache, backends=["lstm"],
                                             zone_ids=["001"], history_days=7, horizon_days=2, force_train=True)
            self.assertEqual(outputs["lstm_training_index"], cache / "weight" / "lstm" / "index.json")
            self.assertTrue(prepare.call_args.kwargs["force_train"])
            self.assertEqual(prepare.call_args.kwargs["data_dir"], Path("dataset"))
            self.assertEqual(prepare.call_args.kwargs["zone_ids"], ["001"])
            self.assertEqual((prepare.call_args.kwargs["history_days"], prepare.call_args.kwargs["horizon_days"]), (7, 2))

    def test_train_stage_selects_random_sites_before_preparing_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = SimpleNamespace(cache_dir=Path(temporary))
            events = []
            def assess(*args, **kwargs):
                events.append("assess")
                return ["003", "001", "002"]
            def prepare(*args, **kwargs):
                events.append("prepare")
                return {}
            with patch("orchestrator.load_canonical_dataset", return_value=dataset), \
                 patch("orchestrator.eligible_training_zone_ids", side_effect=assess), \
                 patch("orchestrator.prepare_offline_models", side_effect=prepare) as train:
                for _ in range(2):
                    run_training_stage(data_dir=Path("dataset"), backends=["lstm", "xgboost"],
                                       experiment_zone_selection="random", experiment_zone_count=1,
                                       experiment_zone_seed=7)
            self.assertEqual(events, ["assess", "prepare", "assess", "prepare"])
            selected = [call.kwargs["zone_ids"] for call in train.call_args_list]
            self.assertEqual(selected, [["002"], ["002"]])
            self.assertTrue(all(call.args[1] == ["lstm", "xgboost"] for call in train.call_args_list))

    def test_ineligible_explicit_training_site_fails_without_substitution(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = SimpleNamespace(cache_dir=Path(temporary))
            report = {"eligible_zone_ids": ["001"], "zones": {"002": {"reason": "insufficient_samples"}}}
            with patch("orchestrator.load_canonical_dataset", return_value=dataset), \
                 patch("zone_training.assess_dataset_training", return_value=report), \
                 patch("orchestrator.prepare_offline_models") as train:
                with self.assertRaisesRegex(ValueError, "002.*insufficient_samples"):
                    run_training_stage(data_dir=Path("dataset"), zone_ids=["002"])
            train.assert_not_called()

    def test_matrix_prepares_weights_once_and_shares_raw_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = SimpleNamespace(cache_dir=root / "cache")
            references = {"lstm": {"001": {"model_id": "m1"}}}
            with patch("orchestrator.load_canonical_dataset", return_value=dataset), \
                 patch("orchestrator.eligible_training_zone_ids", return_value=["001"]), \
                 patch("orchestrator.prepare_offline_models", return_value=references) as prepare, \
                 patch("orchestrator.run_pipeline", return_value={}) as run:
                run_experiment_matrix(data_dir=root, output_dir=root / "output", zone_ids=["001"],
                                      forecast_starts=["2026-01-01", "2026-01-02"], forecast_models=["lstm"],
                                      agent_modes=["single_agent_full_retry", "multi_agent_full_retry"],
                                      experiment_seeds=[1, 2], diurnal_blend_alphas=[0, .5], force_train=True,
                                      history_days=7, horizon_days=2)
            prepare.assert_called_once()
            self.assertEqual(prepare.call_args.kwargs["zone_ids"], ["001"])
            self.assertEqual((prepare.call_args.kwargs["history_days"], prepare.call_args.kwargs["horizon_days"]), (7, 2))
            self.assertEqual(run.call_count, 16)
            caches = [call.kwargs["raw_prediction_cache"] for call in run.call_args_list]
            self.assertTrue(all(cache is caches[0] for cache in caches))
            for call in run.call_args_list:
                self.assertIs(call.kwargs["zone_model_references"], references)
                self.assertFalse(call.kwargs["force_train"])
                self.assertNotIn("lstm_seed", call.kwargs)

    def test_matrix_intersects_eligibility_before_training_only_selected_sites(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = SimpleNamespace(cache_dir=root / "cache")
            report = {"eligible_zone_ids": ["002", "003"], "cache_path": root / "eligibility.json",
                      "csv_path": root / "eligibility.csv"}
            references = {backend: {"002": {"model_id": backend + "-002"}} for backend in ("lstm", "xgboost")}
            with patch("orchestrator.load_canonical_dataset", return_value=dataset), \
                 patch("orchestrator.eligible_training_zone_ids", return_value=["001", "002"]), \
                 patch("orchestrator.load_zone_eligibility", return_value=report), \
                 patch("orchestrator.prepare_offline_models", return_value=references) as prepare, \
                 patch("orchestrator.run_pipeline", return_value={}) as run:
                run_experiment_matrix(data_dir=root, output_dir=root / "output",
                                      forecast_starts=["2026-01-01", "2026-01-02"],
                                      forecast_models=["lstm", "xgboost"], experiment_zone_count=1,
                                      experiment_zone_selection="random", pipeline_stage="forecaster",
                                      diurnal_blend_alphas=[0, 0.5])
            prepare.assert_called_once()
            self.assertEqual(prepare.call_args.kwargs["zone_ids"], ["002"])
            self.assertEqual(run.call_count, 8)
            self.assertTrue(all(call.kwargs["zone_ids"] == ["002"] for call in run.call_args_list))

    def test_agent_matrix_never_assesses_or_prepares_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("orchestrator.eligible_training_zone_ids") as assess, \
                 patch("orchestrator.prepare_offline_models") as prepare, \
                 patch("orchestrator.run_pipeline", return_value={}) as run:
                run_experiment_matrix(data_dir=root, output_dir=root / "output", zone_ids=["001"],
                                      forecast_starts=["2026-01-01"], forecast_models=["lstm"],
                                      pipeline_stage="agent")
            run.assert_called_once()
            assess.assert_not_called()
            prepare.assert_not_called()

    def test_direct_pipeline_prepares_only_selected_zones_before_loading_inputs(self):
        from orchestrator import run_pipeline
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profiles = pd.DataFrame({"zone_id": ["001", "002", "003"]})
            dataset = SimpleNamespace(dataset_fingerprint="test-data", wide_frame=lambda *args: pd.DataFrame())
            events = []
            def prepare(*args, **kwargs):
                events.append(("prepare", kwargs["zone_ids"]))
                return {"lstm": {"003": {"model_id": "m003"}}}
            def load_inputs(*args, **kwargs):
                events.append(("load", args[2]))
                raise ValueError("stop after selection and preparation")
            with patch("orchestrator.load_canonical_dataset", return_value=dataset), \
                 patch("orchestrator.build_zone_profiles_from_canonical", return_value=profiles), \
                 patch("orchestrator.build_zone_3h_load_policy_thresholds", return_value=profiles), \
                 patch("orchestrator.eligible_training_zone_ids", return_value=["003"]) as assess, \
                 patch("orchestrator.prepare_offline_models", side_effect=prepare) as train, \
                 patch("orchestrator.load_pipeline_data", side_effect=load_inputs):
                with self.assertRaisesRegex(PipelineStageError, "stop after selection and preparation"):
                    run_pipeline(data_dir=root, output_dir=root / "output", backend="lstm",
                                 pipeline_stage="forecaster", zone_ids=["003"],
                                 forecast_origin="2026-01-01", history_days=7, horizon_days=2)
            self.assertEqual(events, [("prepare", ["003"]), ("load", ["003"])])
            self.assertEqual(assess.call_args.kwargs["zone_ids"], ["003"])
            self.assertEqual((train.call_args.kwargs["history_days"], train.call_args.kwargs["horizon_days"]), (7, 2))

    def test_agent_replay_requires_the_exact_saved_zone_model_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parameters = make_forecast_parameters(forecast_origin="2026-01-01", backend="lstm",
                                                  history_days=30, validation_days=5, horizon_days=1,
                                                  diurnal_blend_alpha=0, model_ids={"001": "expected"})
            manifest = {"schema_version": FORECASTER_MANIFEST_SCHEMA_VERSION, "stage": "forecaster",
                        "forecast_parameters": parameters, "model_ids": {"001": "expected"}, "zone_ids": ["001"],
                        "artifacts": {"selected_zones": "selected.csv", "context_snippets": "context.json",
                                      "forecaster_artifact_json": "artifact.json"}}
            (root / "forecaster_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "selected.csv").write_text("zone_id\n001\n", encoding="utf-8")
            (root / "context.json").write_text(json.dumps([{"zone_id": "001"}]), encoding="utf-8")
            (root / "artifact.json").write_text(json.dumps({"schema_version": NATIVE_ARTIFACT_SCHEMA_VERSION,
                "artifact_type": "native_backend_reforecast", "artifacts": {"001": {}}}), encoding="utf-8")
            for model_id in ("expected", "wrong"):
                artifact = SimpleNamespace(forecast_parameters={"diurnal_blend_alpha": 0},
                    metadata=lambda: {"backend": "lstm", "forecast_origin": "2026-01-01", "model_id": model_id})
                with patch("orchestrator.NativeForecasterArtifact.from_dict", return_value=artifact):
                    if model_id == "expected":
                        self.assertEqual(load_forecaster_bundle(root).forecast_parameters["model_ids"], {"001": "expected"})
                    else:
                        with self.assertRaisesRegex(ValueError, "model ID differs"):
                            load_forecaster_bundle(root)

    def test_missing_selected_zone_weights_fail_without_local_training_fallback(self):
        parameters = make_forecast_parameters(forecast_origin="2026-01-01", backend="xgboost",
                                              history_days=30, validation_days=5, horizon_days=1,
                                              diurnal_blend_alpha=0)
        zones = pd.DataFrame([{"zone_id": "001", "category": "test"}])
        with patch("orchestrator.load_stress_thresholds", return_value={}), \
             patch("orchestrator.forecast_zone") as predict:
            with self.assertRaisesRegex(PipelineStageError, "No usable offline xgboost weights for zone 001"):
                build_contexts(pipeline_data=SimpleNamespace(profiles=zones), selected_zones=zones,
                               zone_load_thresholds=zones, zone_model_references={},
                               **forecast_runtime_parameters(parameters))
        predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
