"""Dataset-specific automatic zone counts preserve global and CLI overrides."""
from __future__ import annotations

import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import yaml

from config import RunConfig
from main import main
from orchestrator import run_dataset_batch


class DatasetZoneCountConfigTests(unittest.TestCase):
    def test_original_global_count_remains_the_default(self):
        self.assertEqual(RunConfig.from_mapping({}).zone_count_for("data/UrbanEV"), 12)
        for overrides in ({}, {"experiment_zone_counts": None}, {"experiment_zone_counts": {}}):
            with self.subTest(overrides=overrides):
                config = RunConfig.from_mapping({"experiment_zone_count": 7, **overrides})
                self.assertEqual(config.zone_count_for("data/UrbanEV"), 7)
                self.assertEqual(config.zone_count_for("data/CHARGED/JHB"), 7)

    def test_each_dataset_override_is_independent_and_other_folders_fall_back(self):
        config = RunConfig.from_mapping({
            "experiment_zone_count": 4,
            "experiment_zone_counts": {"data/UrbanEV": 8, "data/CHARGED/JHB": 3,
                                       "data/MP-EVData": 2},
        })
        for folder, expected in (("data/UrbanEV", 8), ("data/CHARGED/JHB", 3),
                                 ("data/MP-EVData", 2), ("data/CHARGED/LOA", 4)):
            with self.subTest(folder=folder):
                self.assertEqual(config.zone_count_for(folder), expected)

    def test_relative_absolute_and_normalized_paths_match(self):
        folder = Path("data/CHARGED/JHB")
        config = RunConfig.from_mapping({"experiment_zone_counts": {str(folder): 3}})
        for equivalent in (folder, folder.resolve(), folder / ".." / "JHB",
                           str(folder.resolve()).replace("\\", "/")):
            with self.subTest(path=str(equivalent)):
                self.assertEqual(config.zone_count_for(equivalent), 3)
        absolute = RunConfig.from_mapping({"experiment_zone_counts": {str(folder.resolve()): 9}})
        self.assertEqual(absolute.zone_count_for(folder), 9)
        if os.name == "nt":
            self.assertEqual(config.zone_count_for(str(folder.resolve()).swapcase()), 3)

    def test_equivalent_duplicate_paths_fail_even_when_counts_agree(self):
        folder = "data/CHARGED/JHB"
        for duplicate in (str(Path(folder).resolve()), "data/CHARGED/LOA/../JHB"):
            with self.subTest(duplicate=duplicate), self.assertRaisesRegex(ValueError, "experiment_zone_counts"):
                RunConfig.from_mapping({"experiment_zone_counts": {folder: 3, duplicate: 3}})

    def test_override_values_must_be_positive_integers(self):
        for count in (0, -1, True, False, 1.5, "3", None, [], {}):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "experiment_zone_counts"):
                RunConfig.from_mapping({"experiment_zone_counts": {"data/UrbanEV": count}})

    def test_invalid_mapping_and_folder_keys_fail(self):
        for value in ([], [3], "data/UrbanEV", 3, False, {None: 3}, {4: 3}, {"": 3}, {"  ": 3}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "experiment_zone_counts"):
                RunConfig.from_mapping({"experiment_zone_counts": value})


class DatasetZoneCountEntrypointTests(unittest.TestCase):
    def setUp(self):
        # Keep fixtures on the project drive so relative-path matching works on Windows.
        temporary_root = Path(__file__).resolve().parents[1] / ".tmp_tests"
        temporary_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="zone_counts_", dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.folders = [self.root / "CHARGED" / name for name in ("JHB", "LOA", "MEL")]
        for folder in self.folders:
            folder.mkdir(parents=True)
            (folder / "volume.csv").write_text("time,001\n2023-04-01 00:00:00,1\n", encoding="utf-8")
            (folder / "e_price.csv").write_text("time,001\n2023-04-01 00:00:00,0.4\n", encoding="utf-8")
            (folder / "sites.csv").write_text("site_id,longitude,latitude\n001,1,1\n", encoding="utf-8")
        self.config_path = self.root / "settings.yaml"
        self.settings = {
            "data_dir": str(self.folders[0]), "output_folder": str(self.root / "output"),
            "experiment_name": "counts", "forecast_models": ["AR"], "forecast_starts": ["2023-04-02"],
            "experiment_zone_count": 5,
            "experiment_zone_counts": {os.path.relpath(self.folders[0]): 2,
                                       os.path.relpath(self.folders[1]): 3},
        }

    def write_config(self):
        self.config_path.write_text(yaml.safe_dump({"run": self.settings, "agent": {}}), encoding="utf-8")

    def capture_matrix(self, *args):
        self.write_config()
        with redirect_stdout(io.StringIO()), patch("main.run_experiment_matrix", return_value={}) as matrix:
            main(["--config", str(self.config_path), *args])
        matrix.assert_called_once()
        return matrix.call_args.kwargs

    def test_main_resolves_dataset_count_for_random_and_representative_selection(self):
        for selection in ("random", "representative"):
            with self.subTest(selection=selection):
                self.settings["experiment_zone_selection"] = selection
                received = self.capture_matrix()
                self.assertEqual(received["experiment_zone_count"], 2)
                self.assertEqual(received["experiment_zone_selection"], selection)

    def test_cli_data_folder_changes_the_selected_override(self):
        received = self.capture_matrix("--data-dir", str(self.folders[1]))
        self.assertEqual(received["data_dir"], self.folders[1])
        self.assertEqual(received["experiment_zone_count"], 3)
        self.assertEqual(self.capture_matrix("--data-dir", str(self.folders[2]))["experiment_zone_count"], 5)

    def test_dataset_error_limit_reaches_the_matrix_from_yaml(self):
        self.assertEqual(self.capture_matrix()["max_dataset_errors"], 3)
        for limit in (1, 5):
            with self.subTest(limit=limit):
                self.settings["max_dataset_errors"] = limit
                self.assertEqual(self.capture_matrix()["max_dataset_errors"], limit)

    def test_cli_count_takes_precedence_over_dataset_and_global_counts(self):
        for folder in self.folders:
            with self.subTest(folder=folder):
                received = self.capture_matrix("--data-dir", str(folder), "--experiment-zone-count", "7")
                self.assertEqual(received["experiment_zone_count"], 7)

    def test_override_enables_automatic_selection_for_an_unnamed_single_model_run(self):
        self.settings.pop("experiment_name")
        received = self.capture_matrix()
        self.assertEqual(received["experiment_zone_count"], 2)
        self.assertEqual(received["forecast_models"], ["AR"])

    def test_original_single_pipeline_still_receives_global_count_without_overrides(self):
        self.settings.pop("experiment_name")
        self.settings.pop("experiment_zone_counts")
        self.write_config()
        with redirect_stdout(io.StringIO()), patch("main.run_experiment_matrix") as matrix, \
                patch("main.run_pipeline", return_value={}) as pipeline:
            main(["--config", str(self.config_path)])
        matrix.assert_not_called()
        self.assertEqual(pipeline.call_args.kwargs["experiment_zone_count"], 5)

    def test_agent_replay_keeps_single_pipeline_with_existing_forecaster_artifacts(self):
        self.settings["pipeline_stage"] = "agent"
        self.settings["forecaster_output_dir"] = str(self.root / "forecaster")
        self.write_config()
        with redirect_stdout(io.StringIO()), patch("main.run_experiment_matrix") as matrix, \
                patch("main.run_pipeline", return_value={}) as pipeline:
            main(["--config", str(self.config_path)])
        matrix.assert_not_called()
        self.assertEqual(pipeline.call_args.kwargs["experiment_zone_count"], 2)
        self.assertEqual(pipeline.call_args.kwargs["forecaster_output_dir"], self.settings["forecaster_output_dir"])

    def test_batch_absolute_child_paths_retain_dataset_specific_counts_and_cli_priority(self):
        self.settings["data_dir"] = [os.path.relpath(folder) for folder in self.folders]
        self.write_config()
        for cli_args, expected in (([], [2, 3, 5]), (["--experiment-zone-count", "6"], [6, 6, 6])):
            with self.subTest(cli_args=cli_args), redirect_stdout(io.StringIO()), \
                    patch("orchestrator.subprocess.run", return_value=subprocess.CompletedProcess([], 0)):
                outputs = run_dataset_batch(
                    [Path(folder) for folder in self.settings["data_dir"]],
                    argv=["--config", str(self.config_path), *cli_args],
                    output_dir=self.root / "output", experiment_name="counts", max_workers=2,
                )
                jobs = json.loads(outputs["batch_manifest_json"].read_text(encoding="utf-8"))["jobs"]
                received = []
                for job in jobs:
                    command = job["command"]
                    self.assertTrue(Path(command[command.index("--data-dir") + 1]).is_absolute())
                    with patch("main.run_experiment_matrix", return_value={}) as matrix:
                        main(command[2:])
                    received.append(matrix.call_args.kwargs["experiment_zone_count"])
                self.assertEqual(received, expected)

    def test_explicit_config_and_cli_zones_are_not_truncated_by_override(self):
        self.settings["zones"] = ["001", "002", "003", "004"]
        self.settings["experiment_zone_selection"] = "random"
        self.settings["forecast_starts"] = ["2023-04-02", "2023-04-03"]
        self.write_config()
        for cli_args, expected in (([], self.settings["zones"]),
                                   (["--zones", "007", "008", "009"], ["007", "008", "009"])):
            with self.subTest(cli_args=cli_args), redirect_stdout(io.StringIO()), \
                    patch("orchestrator.run_pipeline", return_value={}) as pipeline, \
                    patch("orchestrator.load_canonical_dataset") as load_dataset:
                main(["--config", str(self.config_path), *cli_args])
            load_dataset.assert_not_called()
            self.assertEqual(pipeline.call_count, 2)
            for call in pipeline.call_args_list:
                self.assertEqual(call.kwargs["zone_ids"], expected)


if __name__ == "__main__":
    unittest.main()
