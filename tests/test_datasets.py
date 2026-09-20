from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from config import DataConfig, RunConfig
from dataset_adapter import (DatasetSpec, load_canonical_dataset, validate_forecaster_dataset_source,
                             MPEVDataDatasetAdapter, tou_schedule, resolve_dataset_spec, merge_cache_manifest,
                             replace_cache_file)
from reporting import output_lock, scoped_output
from forecasting import summarize_weather
from global_forecaster import scenario_price_series
from main import main
from orchestrator import run_dataset_batch


def tariff_frame(rows):
    return pd.DataFrame(rows, columns=["Month", "Period", "Electricity Price(RMB)", "Service Price(RMB)"])


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def charged(self):
        folder = self.root / "LOA"
        folder.mkdir()
        times = pd.date_range("2023-04-01", periods=48, freq="h")
        pd.DataFrame({"Unnamed: 0": times, "001": np.arange(48), "002": 5}).to_csv(folder / "volume.csv", index=False)
        pd.DataFrame({"time": times, "001": 0.4, "002": 0.0}).to_csv(folder / "e_price.csv", index=False)
        pd.DataFrame({"site_id": ["001", "002"], "longitude": [1, 2], "latitude": [1, 2],
                      "total_volume": [100000, 100000], "avg_power": [999, 999]}).to_csv(folder / "sites.csv", index=False)
        return DatasetSpec(folder, adapter="charged")

    def test_charged_filters_zero_price_and_preserves_ids_and_sources(self):
        spec = self.charged()
        path = spec.path / "volume.csv"
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        data = load_canonical_dataset(spec)
        self.assertEqual(set(data.timeseries.zone_id), {"001"})
        self.assertEqual(data.timeseries.load_kwh.sum(), sum(range(48)))
        self.assertIn("002", data.feature_manifest["excluded_sites"])
        self.assertNotIn("total_volume", data.static_zone_features)
        self.assertNotIn("avg_power", data.static_zone_features)
        cached = load_canonical_dataset(spec)
        self.assertEqual(set(cached.timeseries.zone_id), {"001"})
        self.assertEqual(data.dataset_fingerprint, cached.dataset_fingerprint)
        self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_charged_rejects_misaligned_hours(self):
        spec = self.charged()
        path = spec.path / "e_price.csv"
        pd.read_csv(path).iloc[:-1].to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "identical hours"):
            load_canonical_dataset(spec)

    def test_charged_site_identifier_variant(self):
        spec = self.charged()
        path = spec.path / "sites.csv"
        pd.read_csv(path, dtype={"site_id": str}).rename(columns={"site_id": "site"}).to_csv(path, index=False)
        data = load_canonical_dataset(spec)
        self.assertEqual(data.static_zone_features.iloc[0].zone_id, "001")
        self.assertEqual(data.static_zone_features.iloc[0].longitude, 1)

    def test_tariff_peak_precedence_and_midnight(self):
        table = tariff_frame([(1, "Sharp(19:00-21:00)", 1.4, 0.4),
                              (1, "Peak(08:00-11:00,18:00-21:00)", 1.1, 0.2),
                              (1, "Off-peak(22:00-00:00,00:00-06:00)", 0.3, 0.6)])
        rates = tou_schedule(table).set_index(["month", "hour"])
        self.assertAlmostEqual(rates.loc[(1, 19), "energy_price"], 1.4)
        self.assertAlmostEqual(rates.loc[(1, 18), "energy_price"], 1.1)
        self.assertAlmostEqual(rates.loc[(1, 23), "energy_price"], 0.3)
        self.assertAlmostEqual(rates.loc[(1, 0), "energy_price"], 0.3)
        self.assertNotIn((1, 21), rates.index)

    def test_conflicting_tariffs_fail(self):
        with self.assertRaisesRegex(ValueError, "overlapping"):
            tou_schedule(tariff_frame([(1, "Peak(08:00-11:00)", 1, 0), (1, "Shoulder(10:00-12:00)", 0.5, 0)]))

    def test_mp_evdata_real_workbook_layout(self):
        meta = pd.DataFrame({"Charging Prototypes": ["Taxi", "Bus", "Swap"], "ID": ["A1", "A4", "A7"],
                             "Single-pile Power": [80, 120, 60], "Total Orders": [999, 999, 999],
                             "Capacity": [5000, 3000, 1000], "Equipment Type": ["DC", "DC", "Swap"], "Price": ["TOU", "Free", "TOU"]})
        with pd.ExcelWriter(self.root / MPEVDataDatasetAdapter.load_filename) as writer:
            meta.to_excel(writer, sheet_name="metadata", index=False)
            for site, column in [("A1", "power"), ("A4", "power"), ("A7", "session_count")]:
                pd.DataFrame({"datetime": pd.date_range("2024-12-31", periods=25, freq="h"), column: 12.5}).to_excel(writer, sheet_name=site, index=False)
        with pd.ExcelWriter(self.root / "price.xlsx") as writer:
            meta.to_excel(writer, sheet_name="metadata", index=False)
            tariff_frame([(12, "Shoulder(00:00-00:00)", 0.6, 0.3)]).to_excel(writer, sheet_name="A1", index=False)
            tariff_frame([(12, "Shoulder(00:00-00:00)", np.nan, np.nan)]).to_excel(writer, sheet_name="A4", index=False)
        data = load_canonical_dataset(DatasetSpec(self.root, adapter="mp_evdata"))
        self.assertEqual(set(data.timeseries.zone_id), {"A1"})
        self.assertEqual(len(data.timeseries), 24)
        self.assertEqual(data.timeseries.load_kwh.sum(), 300)
        self.assertEqual(data.feature_manifest["trimmed_outside_2024"]["A1"], 1)
        self.assertIn("A4", data.feature_manifest["excluded_sites"])
        self.assertIn("A7", data.feature_manifest["excluded_sites"])
        self.assertNotIn("Total Orders", data.static_zone_features)

    def test_folder_config_and_automatic_defaults(self):
        folder = str(self.root / "folder with spaces,commas")
        self.assertEqual(RunConfig.from_mapping({"data_dir": folder}).data_dir, folder)
        run = RunConfig.from_mapping({"data_dir": [folder, str(self.root / "other")]})
        self.assertEqual(len(run.data_dir), 2)
        self.assertEqual(DataConfig.from_mapping({}).adapter, "auto")
        self.assertIsNone(run.forecast_start)
        self.assertIsNone(run.zone_ids)
        self.assertEqual(run.lstm_device, "auto")
        for value in [[], "", 4, [None], [folder, folder]]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                RunConfig.from_mapping({"data_dir": value})
        for value in [0, -1, 1.5, True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                RunConfig.from_mapping({"max_parallel_datasets": value})

    def test_outputs_are_isolated_even_for_identically_named_folders(self):
        root = self.root / "output"
        paths = {scoped_output(root, name, self.root / folder)
                 for name, folder in [("urbanev", "UrbanEV"), ("charged", "one/JHB"), ("charged", "two/JHB"), ("mp_evdata", "MP-EVData")]}
        self.assertEqual(len(paths), 4)
        self.assertNotIn(root, paths)
        self.assertEqual(scoped_output(root, "charged", self.root / "one/JHB"),
                         scoped_output(root, "charged", self.root / "one/JHB/../JHB"))

    def test_run_lock_blocks_other_process_and_releases(self):
        code = "from pathlib import Path; from reporting import output_lock; import sys\nwith output_lock(Path(sys.argv[1])): pass"
        with output_lock(self.root):
            result = subprocess.run([sys.executable, "-c", code, str(self.root)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"Another run", result.stderr)
        result = subprocess.run([sys.executable, "-c", code, str(self.root)], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_concurrent_cache_index_updates_keep_all_entries(self):
        path = self.root / "cache_manifest.json"
        code = ("from pathlib import Path; from dataset_adapter import merge_cache_manifest; import sys\n"
                "for i in range(20): merge_cache_manifest(Path(sys.argv[1]), {'datasets': {sys.argv[2] + str(i): {}}})")
        workers = [subprocess.Popen([sys.executable, "-c", code, str(path), str(i)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE) for i in range(3)]
        try:
            results = [worker.communicate(timeout=30) for worker in workers]
        finally:
            for worker in workers:
                if worker.poll() is None:
                    worker.kill()
                    worker.communicate()
        for worker, (_, stderr) in zip(workers, results):
            self.assertEqual(worker.returncode, 0, stderr)
        self.assertEqual(len(json.loads(path.read_text())["datasets"]), 60)
        with patch("dataset_adapter.atomic_write_json") as write:
            merge_cache_manifest(path, {"datasets": {"00": {}}})
        write.assert_not_called()

    def test_cache_replace_retries_only_transient_windows_file_locks(self):
        error = PermissionError("file in use")
        error.winerror = 32
        with patch("dataset_adapter.os.name", "nt"), patch("dataset_adapter.os.replace", side_effect=[error, None]) as replace, patch("dataset_adapter.time.sleep"):
            replace_cache_file("source", "target")
        self.assertEqual(replace.call_count, 2)
        with patch("dataset_adapter.os.replace", side_effect=PermissionError("denied")), self.assertRaises(PermissionError):
            replace_cache_file("source", "target")

    def test_missing_weather_is_not_reported_as_zero_weather(self):
        times = pd.date_range("2024-01-01", periods=24, freq="h")
        summary = summarize_weather(pd.DataFrame({"time": times}), times.min(), times.max())
        self.assertTrue(all(value is None for value in summary.values()))

    def test_auto_detection_uses_files_not_folder_name(self):
        spec = self.charged()
        folder = spec.path.with_name("arbitrary folder")
        spec.path.rename(folder)
        self.assertEqual(resolve_dataset_spec(DatasetSpec(folder, adapter="auto")).adapter, "charged")
        (folder / "inf.csv").write_text("TAZID\n1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unambiguously"):
            resolve_dataset_spec(DatasetSpec(folder, adapter="auto"))
        (folder / "sites.csv").unlink()
        self.assertEqual(resolve_dataset_spec(DatasetSpec(folder, adapter="auto")).adapter, "urbanev")
        for name in [MPEVDataDatasetAdapter.load_filename, "price.xlsx"]:
            (self.root / name).touch()
        self.assertEqual(resolve_dataset_spec(DatasetSpec(self.root, adapter="auto")).adapter, "mp_evdata")

    def test_config_only_entrypoint_resolves_matrix_date_from_data(self):
        spec = self.charged()
        config = self.root / "config.yaml"
        config.write_text(yaml.safe_dump({"agent": {}, "run": {
            "data_dir": str(spec.path), "output_folder": str(self.root / "out"),
            "forecast_models": ["AR", "lstm"], "horizon_days": 1, "dry_run": True,
        }}), encoding="utf-8")
        with patch("main.run_experiment_matrix", return_value={}) as run:
            main(["--config", str(config)])
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["dataset_spec"].adapter, "charged")
        self.assertEqual(kwargs["forecast_starts"], ["2023-04-02T00:00:00"])
        self.assertEqual(kwargs["forecast_models"], ["AR", "lstm"])
        self.assertIsNone(kwargs["zone_ids"])
        self.assertEqual(kwargs["lstm_device"], "auto")

    def test_parallel_forwards_config_and_reports_each_failure(self):
        folders = [self.root / "one", self.root / "two"]
        for folder in folders:
            folder.mkdir()
        argv = ["--config", "config with spaces.yaml", "--forecast-model", "lstm"]
        def execute(command, **kwargs):
            self.assertEqual(command[2:6], argv)
            self.assertEqual(command[6], "--data-dir")
            self.assertNotIn("--device", command)
            return subprocess.CompletedProcess(command, 1 if str(folders[0]) in command else 0)
        with patch("orchestrator.subprocess.run", side_effect=execute), self.assertRaisesRegex(RuntimeError, "dataset runs failed"):
            run_dataset_batch(folders, argv=argv, output_dir=self.root / "out", max_workers=2)
        manifest = json.loads(next((self.root / "out").glob("*/batch_manifest.json")).read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(sorted(job["returncode"] for job in manifest["jobs"]), [0, 1])
        self.assertEqual(len({job["log"] for job in manifest["jobs"]}), 2)

    def test_config_list_dispatch_and_single_folder_override(self):
        spec = self.charged()
        config = self.root / "config.yaml"
        content = {"agent": {}, "run": {
            "data_dir": [str(spec.path)], "output_folder": str(self.root / "out"),
            "forecast_model": "AR", "pipeline_stage": "forecaster", "max_parallel_datasets": 3,
        }}
        config.write_text(yaml.safe_dump(content), encoding="utf-8")
        with patch("main.run_dataset_batch", return_value={}) as batch:
            main(["--config", str(config)])
        self.assertEqual(batch.call_args.args[0], [spec.path])
        self.assertEqual(batch.call_args.kwargs["max_workers"], 3)
        self.assertFalse(batch.call_args.kwargs["reuse_outputs"])
        with patch("main.run_pipeline", return_value={}) as run, patch("main.run_dataset_batch") as batch:
            main(["--config", str(config), "--data-dir", str(spec.path)])
        batch.assert_not_called()
        self.assertEqual(run.call_args.kwargs["pipeline_stage"], "forecaster")
        with patch("main.run_dataset_batch", return_value={}) as batch:
            main(["--config", str(config), "--stage", "agent"])
        self.assertTrue(batch.call_args.kwargs["reuse_outputs"])
        content["run"]["forecaster_output_dir"] = "shared"
        config.write_text(yaml.safe_dump(content), encoding="utf-8")
        with patch("main.run_dataset_batch") as batch, self.assertRaisesRegex(ValueError, "forecaster_output_dir"):
            main(["--config", str(config)])
        batch.assert_not_called()

    def test_parallel_agent_reuses_the_selected_output_root(self):
        folder = self.root / "dataset"
        folder.mkdir()
        output = self.root / "previous-batch"
        with patch("orchestrator.subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run:
            result = run_dataset_batch([folder], argv=[], output_dir=output, max_workers=1, reuse_outputs=True)
        self.assertEqual(run.call_args.args[0][-2:], ["--output-folder", str(output.resolve())])
        manifest = json.loads(result["batch_manifest_json"].read_text())
        self.assertEqual(manifest["status"], "success")

    def test_forecaster_handoff_rejects_wrong_dataset_and_city(self):
        spec = DatasetSpec(self.root / "JHB", adapter="charged")
        with self.assertRaisesRegex(ValueError, "adapter"):
            validate_forecaster_dataset_source(spec, {"adapter": "mp_evdata"})
        with self.assertRaisesRegex(ValueError, "different dataset directory"):
            validate_forecaster_dataset_source(spec, {"adapter": "charged", "data_dir": str(self.root / "LOA")})
        validate_forecaster_dataset_source(spec, {"adapter": "charged", "data_dir": str(spec.path)})
        validate_forecaster_dataset_source(DatasetSpec(self.root, adapter="mp-evdata"), {"adapter": "mp_evdata", "data_dir": str(self.root)})

    def test_late_opening_station_only_requires_forecast_prices(self):
        times = pd.date_range("2024-11-01", periods=3, freq="h")
        schedule = pd.DataFrame({"time": times, "A10": [np.nan, 0.6, 0.7]})
        prices = scenario_price_series(schedule, "A10", timestamps=times[1:])
        self.assertEqual(prices.tolist(), [0.6, 0.7])
        with self.assertRaisesRegex(ValueError, "non-numeric"):
            scenario_price_series(schedule, "A10", timestamps=times)


if __name__ == "__main__":
    unittest.main()
