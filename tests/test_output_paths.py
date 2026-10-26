"""Readable, persistent and isolated per-dataset experiment directories."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from reporting import scoped_output


class DatasetOutputPathTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.experiment = self.root / "output" / "example_experiment"

    def test_default_names_are_one_level_below_experiment(self):
        for adapter, folder, expected in (
            ("urbanev", "data/UrbanEV", "UrbanEV"),
            ("charged", "data/CHARGED/JHB", "CHARGED_JHB"),
            ("mp_evdata", "data/MP-EVData", "MP-EVData"),
        ):
            with self.subTest(adapter=adapter):
                result = scoped_output(self.experiment, adapter, self.root / folder)
                self.assertEqual(result, self.experiment / expected)

    def test_custom_folder_keeps_its_name_and_adapter(self):
        result = scoped_output(self.experiment, "urbanev", self.root / "data" / "city sample")
        self.assertEqual(result.name, "UrbanEV_city_sample")

    def test_same_named_datasets_remain_isolated_and_bindings_are_reused(self):
        first_data = self.root / "first" / "JHB"
        second_data = self.root / "second" / "JHB"
        first = scoped_output(self.experiment, "charged", first_data)
        second = scoped_output(self.experiment, "charged", second_data)
        self.assertEqual(first.name, "CHARGED_JHB")
        self.assertRegex(second.name, r"^CHARGED_JHB_[0-9a-f]{10}$")
        self.assertNotEqual(first, second)
        second.mkdir()
        (second / "results.csv").write_text("existing results", encoding="utf-8")
        # Reload in reverse order, including an equivalent non-normalized path.
        self.assertEqual(scoped_output(self.experiment, "charged", second_data / ".." / "JHB"), second)
        self.assertEqual(scoped_output(self.experiment, "charged", first_data), first)
        self.assertEqual((second / "results.csv").read_text(encoding="utf-8"), "existing results")

    def test_names_differing_only_in_case_cannot_share_a_directory(self):
        first = scoped_output(self.experiment, "charged", self.root / "one" / "JHB")
        second = scoped_output(self.experiment, "charged", self.root / "two" / "jhb")
        self.assertNotEqual(first.name.casefold(), second.name.casefold())
        self.assertRegex(second.name, r"^CHARGED_jhb_[0-9a-f]{10}$")

    def test_unregistered_existing_directory_is_not_reused(self):
        existing = self.experiment / "CHARGED_JHB"
        existing.mkdir(parents=True)
        (existing / "results.csv").write_text("unregistered results", encoding="utf-8")
        result = scoped_output(self.experiment, "charged", self.root / "data" / "JHB")
        self.assertNotEqual(result, existing)
        self.assertEqual((existing / "results.csv").read_text(encoding="utf-8"), "unregistered results")

    def test_corrupt_registry_is_reported_without_overwriting_it(self):
        self.experiment.mkdir(parents=True)
        registry = self.experiment / ".dataset_directories.json"
        for invalid in ("not json", '{"schema_version": 99, "datasets": {}}',
                        '{"schema_version": 1, "datasets": {"bad": {}}}'):
            with self.subTest(invalid=invalid):
                registry.write_text(invalid, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "registry"):
                    scoped_output(self.experiment, "charged", self.root / "JHB")
                self.assertEqual(registry.read_text(encoding="utf-8"), invalid)

    def test_registry_rejects_directory_traversal_and_duplicate_names(self):
        scoped_output(self.experiment, "charged", self.root / "one" / "JHB")
        scoped_output(self.experiment, "charged", self.root / "two" / "JHB")
        registry = self.experiment / ".dataset_directories.json"
        original = registry.read_text(encoding="utf-8")
        for invalid_directory in ("../outside", "charged_jhb"):
            with self.subTest(directory=invalid_directory):
                payload = json.loads(original)
                entries = list(payload["datasets"].values())
                entries[1]["directory"] = invalid_directory
                registry.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "binding"):
                    scoped_output(self.experiment, "charged", self.root / "three" / "JHB")

    def test_concurrent_allocations_preserve_all_bindings(self):
        code = (
            "from pathlib import Path; import sys; from reporting import scoped_output\n"
            "print(scoped_output(Path(sys.argv[1]), 'charged', Path(sys.argv[2])))"
        )
        paths = [self.root / "one" / "JHB", self.root / "two" / "JHB", self.root / "one" / "JHB"]
        workers = [subprocess.Popen(
            [sys.executable, "-c", code, str(self.experiment), str(path)],
            cwd=Path(__file__).resolve().parents[1], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        ) for path in paths]
        try:
            results = [worker.communicate(timeout=45) for worker in workers]
        finally:
            for worker in workers:
                if worker.poll() is None:
                    worker.kill()
                    worker.communicate()
        for worker, (_, errors) in zip(workers, results):
            self.assertEqual(worker.returncode, 0, errors)
        outputs = [Path(output.strip()) for output, _ in results]
        self.assertEqual(outputs[0], outputs[2])
        self.assertNotEqual(outputs[0], outputs[1])
        self.assertIn("CHARGED_JHB", [output.name for output in outputs])
        registry = json.loads((self.experiment / ".dataset_directories.json").read_text(encoding="utf-8"))
        self.assertEqual(len(registry["datasets"]), 2)
        for path, output in zip(paths, outputs):
            self.assertEqual(scoped_output(self.experiment, "charged", path), output)


if __name__ == "__main__":
    unittest.main()
