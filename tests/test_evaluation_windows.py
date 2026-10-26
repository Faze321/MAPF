import copy
import json
import unittest

import pandas as pd

from evaluation_windows import build_window_tables, extract_window_rounds, summarize_windows


def window(stress="Medium", *, initial="High", price=2, load=30, index=0, **extra):
    return {"window_start": str(index), "window_end": str(index + 1),
            "load_stress_level": initial, "sum_predicted_kwh": 40,
            "mean_energy_price": 1, "proposed_energy_price": price,
            "reforecast_load_kwh": load, "reforecast_load_stress_level": stress,
            "control_success": True, **extra}


def control(states=("Medium", "High", "Medium"), *, initial="High", limit=3, status="fail"):
    return {"schema_version": 3, "status": status, "attempts_used": len(states),
            "attempt_limit": limit, "zones": [{"zone_id": "001", "attempts_used": len(states),
                "attempt_trace": [{"attempt": index, "windows": [window(state, initial=initial)]}
                                  for index, state in enumerate(states, 1)],
                "final_windows": [window(states[-1], initial=initial)]}]}


def select(rows, phase, number=None):
    return [row for row in rows if row["phase"] == phase and (number is None or row["round"] == number)]


class WindowExtractionTests(unittest.TestCase):
    def test_revenue_uses_window_load_and_actual_price_without_extra_hours(self):
        payload = control()
        payload["zones"][0]["final_windows"][0]["suggested_price_shift_pct"] = 999
        original = copy.deepcopy(payload)
        final = select(extract_window_rounds(payload), "final")[0]
        self.assertEqual(final["baseline_revenue"], 40)
        self.assertEqual(final["revenue"], 60)
        self.assertEqual(final["revenue_change"], 20)
        self.assertEqual(final["revenue_change_pct"], 50)
        self.assertEqual(payload, original)

    def test_new_medium_is_relative_to_baseline_not_adjacent_round(self):
        rows = extract_window_rounds(control())
        second = summarize_windows(select(rows, "round", 2))
        third = summarize_windows(select(rows, "round", 3))
        self.assertEqual(second["new_medium_count"], 0)
        self.assertEqual(second["exited_medium_count"], 1)
        self.assertEqual(second["net_medium_count"], 0)
        self.assertEqual(second["adjacent_net_medium_count"], -1)
        self.assertEqual(third["new_medium_count"], 1)
        self.assertEqual(third["gained_medium_count"], 1)
        self.assertEqual(third["first_entry_count"], 0)
        self.assertEqual(select(rows, "final")[0]["first_entry_round"], 1)

    def test_original_medium_is_not_a_new_entry_and_control_success_is_independent(self):
        payload = control(("Medium",), initial="Medium")
        payload["zones"][0]["final_windows"][0]["control_success"] = False
        final = select(extract_window_rounds(payload), "final")
        self.assertIs(final[0]["medium"], True)
        self.assertIs(final[0]["control_success"], False)
        self.assertEqual(final[0]["first_entry_round"], 0)
        self.assertEqual(summarize_windows(final)["new_medium_count"], 0)

    def test_missing_round_unknown_and_no_false_claim_of_first_entry(self):
        payload = control(("Low", "High", "Medium"))
        payload["zones"][0]["attempt_trace"].pop(1)
        rows = extract_window_rounds(payload)
        missing = select(rows, "round", 2)[0]
        last = select(rows, "round", 3)[0]
        self.assertIsNone(missing["medium"])
        self.assertIsNone(missing["revenue"])
        self.assertEqual(missing["observation_source"], "missing")
        self.assertIsNone(last["first_entry_round"])
        self.assertIs(last["first_entry_known"], False)
        self.assertIs(last["ever_newly_medium"], True)
        summary = summarize_windows([last])
        self.assertEqual(summary["adjacent_unknown_transition_count"], 1)
        self.assertEqual(summary["gained_medium_count"], 0)

    def test_global_successful_early_stop_only_carries_future_rounds(self):
        payload = control(("Medium",), status="success", limit=5)
        rows = extract_window_rounds(payload)
        for number in range(2, 6):
            row = select(rows, "round", number)[0]
            self.assertEqual(row["observation_source"], "early_stop_carry")
            self.assertIs(row["medium"], True)
            self.assertEqual(row["revenue"], 60)
        self.assertEqual(select(rows, "final")[0]["round"], 1)
        payload["status"] = "fail"
        self.assertIsNone(select(extract_window_rounds(payload), "round", 2)[0]["medium"])

    def test_early_stop_does_not_fill_missing_intermediate_round(self):
        payload = control(("Low", "Medium"), status="success", limit=4)
        payload["zones"][0]["attempt_trace"].pop(0)
        rows = extract_window_rounds(payload)
        self.assertIsNone(select(rows, "round", 1)[0]["medium"])
        self.assertEqual(select(rows, "round", 3)[0]["observation_source"], "early_stop_carry")

    def test_frozen_window_uses_latest_recorded_forecast(self):
        payload = control(("Medium", "High"), limit=2)
        payload["zones"][0]["attempt_trace"][1]["windows"][0].update(proposal_status="frozen", reforecast_load_kwh=90)
        row = select(extract_window_rounds(payload), "round", 2)[0]
        self.assertEqual(row["load_kwh"], 90)
        self.assertIs(row["medium"], False)
        self.assertEqual(row["proposal_status"], "frozen")

    def test_ever_success_then_final_exit_and_first_entry(self):
        final = select(extract_window_rounds(control(("High", "Medium", "High"))), "final")
        summary = summarize_windows(final)
        self.assertEqual(final[0]["first_entry_round"], 2)
        self.assertEqual(summary["ever_newly_medium_count"], 1)
        self.assertEqual(summary["ever_medium_final_exit_count"], 1)
        self.assertEqual(summary["ever_newly_medium_final_exit_count"], 1)
        self.assertEqual(summary["medium_count"], 0)

    def test_old_load_fields_timestamps_and_schema_two(self):
        payload = control(("Medium",), limit=1)
        payload["schema_version"] = 2
        attempt = payload["zones"][0]["attempt_trace"][0]["windows"][0]
        final = payload["zones"][0]["final_windows"][0]
        for item in (attempt, final):
            item["price_conditioned_baseline_load_kwh"] = item.pop("reforecast_load_kwh")
            item["price_conditioned_load_stress_level"] = item.pop("reforecast_load_stress_level")
        attempt.update(window_start="2026-01-01 00:00:00", window_end="2026-01-01T03:00:00Z")
        final.update(window_start="2026-01-01T01:00:00+01:00", window_end="2026-01-01 03:00:00")
        rows = extract_window_rounds(payload)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["window_start"], "2026-01-01T00:00:00")
        self.assertEqual(rows[-1]["revenue"], 60)

    def test_invalid_price_and_load_and_zero_baseline(self):
        for extra in ({"proposed_energy_price": -1}, {"proposed_energy_price": float("inf")},
                      {"proposed_energy_price": True}, {"reforecast_load_kwh": float("nan")},
                      {"price_valid": False}):
            with self.subTest(extra=extra):
                payload = control(("Medium",))
                payload["zones"][0]["final_windows"][0].update(extra)
                row = select(extract_window_rounds(payload), "final")[0]
                self.assertIsNone(row["revenue"])
                self.assertEqual(row["baseline_revenue"], 40)
                json.dumps(row, allow_nan=False)
        payload = control(("Medium",))
        payload["zones"][0]["final_windows"][0].update(mean_energy_price=0, proposed_energy_price=0)
        row = select(extract_window_rounds(payload), "final")[0]
        self.assertEqual(row["revenue"], 0)
        self.assertEqual(row["baseline_revenue"], 0)
        self.assertEqual(row["revenue_change"], 0)
        self.assertIsNone(row["revenue_change_pct"])

    def test_duplicate_window_or_round_rejected(self):
        payload = control()
        payload["zones"][0]["final_windows"] *= 2
        with self.assertRaisesRegex(ValueError, "Duplicate window"):
            extract_window_rounds(payload)
        payload = control()
        payload["zones"][0]["attempt_trace"].append(payload["zones"][0]["attempt_trace"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate control attempt"):
            extract_window_rounds(payload)

    def test_unknown_stress_does_not_become_non_medium(self):
        payload = control(("unknown",), initial="unknown", limit=1)
        row = select(extract_window_rounds(payload), "final")[0]
        summary = summarize_windows([row])
        self.assertIsNone(row["medium"])
        self.assertIsNone(row["baseline_medium"])
        self.assertEqual(summary["medium_known_count"], 0)
        self.assertIsNone(summary["medium_rate_pct"])
        self.assertEqual(summary["new_medium_count"], 0)
        self.assertEqual(summary["stress_distribution"], {"Unknown": 1})

    def test_overflow_revenue_is_unknown_and_rows_remain_json_serializable(self):
        payload = control(("Medium",), limit=1)
        payload["zones"][0]["final_windows"][0].update(proposed_energy_price=1e308, reforecast_load_kwh=30)
        final = select(extract_window_rounds(payload), "final")
        self.assertIsNone(final[0]["revenue"])
        json.dumps(final, allow_nan=False)
        final[0].update(revenue=1e308, baseline_revenue=1e-300)
        summary = summarize_windows(final * 2, include_revenue=True)
        self.assertIsNone(summary["revenue"])
        self.assertIsNone(summary["revenue_change_pct"])
        json.dumps(summary, allow_nan=False)


class WindowSummaryTests(unittest.TestCase):
    def test_partial_revenue_is_not_full_total_and_different_units_are_rejected(self):
        rows = select(extract_window_rounds(control(("Medium",))), "final")
        rows.append({**rows[0], "window_start": "other", "revenue": None})
        summary = summarize_windows(rows, include_revenue=True)
        self.assertIsNone(summary["revenue"])
        self.assertEqual(summary["known_revenue"], 60)
        self.assertEqual(summary["revenue_known_count"], 1)
        self.assertEqual(summary["revenue_unknown_count"], 1)
        self.assertIsNone(summary["revenue_change"])
        self.assertEqual(summary["known_revenue_change"], 20)
        rows[0]["revenue_unit"], rows[1]["revenue_unit"] = "CNY", "ZAR"
        with self.assertRaisesRegex(ValueError, "revenue_unit"):
            summarize_windows(rows, include_revenue=True)

    def test_full_region_and_run_require_all_expected_windows(self):
        payload = control(("Medium",), limit=1)
        payload["zones"].append({**copy.deepcopy(payload["zones"][0]), "zone_id": "002"})
        rows = select(extract_window_rounds(payload), "final")
        summary = summarize_windows(rows)
        self.assertEqual(summary["zone_all_medium_count"], 2)
        self.assertEqual(summary["run_all_medium_count"], 1)
        partial = summarize_windows(rows[:1])
        self.assertEqual(partial["zone_all_medium_count"], 1)
        self.assertEqual(partial["run_complete_count"], 0)
        rows[1]["medium"] = None
        missing = summarize_windows(rows)
        self.assertEqual(missing["zone_all_medium_count"], 1)
        self.assertEqual(missing["zone_unknown_count"], 1)
        self.assertEqual(missing["run_unknown_count"], 1)

    def test_tables_separate_batches_datasets_units_and_do_not_sum_revenue_over_rounds(self):
        rows = []
        for dataset, unit in (("UrbanEV", "CNY"), ("JHB", "ZAR")):
            for row in extract_window_rounds(control(("Medium",), limit=1)):
                rows.append({**row, "batch": "27B", "dataset": dataset, "dataset_identity": dataset,
                             "revenue_unit": unit, "backend": "lstm", "agent_mode": "single",
                             "forecast_origin": "t1", "run_id": dataset})
        tables = build_window_tables(rows)
        self.assertEqual(len(tables["round_overall"]), 3)
        self.assertNotIn("revenue", tables["round_overall"][0])
        final = [row for row in tables["revenue_by_dataset_model_mode"] if row["phase"] == "final"]
        self.assertEqual(len(final), 2)
        self.assertTrue(all(row["revenue"] == 60 for row in final))
        self.assertEqual({row["revenue_unit"] for row in final}, {"CNY", "ZAR"})
        self.assertEqual(summarize_windows([])["medium_rate_pct"], None)
        json.dumps(tables, allow_nan=False)


class HourlyRevenueBaselineTests(unittest.TestCase):
    def setUp(self):
        self.payload = control(("Medium",), limit=1)
        zone = self.payload["zones"][0]
        for item in (zone["final_windows"][0], zone["attempt_trace"][0]["windows"][0]):
            item.update(window_start="2026-01-01T00:00:00", window_end="2026-01-01T02:00:00",
                        mean_energy_price=2, sum_predicted_kwh=60,
                        proposed_energy_price=2.5, reforecast_load_kwh=60)
        self.hourly = [{"zone_id": "001", "time": f"2026-01-01T0{hour}:00:00",
                        "e_price": price, "predicted_kwh": load}
                       for hour, price, load in ((0, 1, 10), (1, 2, 20), (2, 3, 30))]

    def final(self, source=None):
        rows = extract_window_rounds(self.payload, baseline_hourly=self.hourly if source is None else source)
        return select(rows, "final")[0]

    def test_hourly_product_differs_from_mean_price_product_without_replacing_old_fields(self):
        original_payload, original_hourly = copy.deepcopy(self.payload), copy.deepcopy(self.hourly)
        rows = extract_window_rounds(self.payload, baseline_hourly=self.hourly)
        final = select(rows, "final")[0]
        self.assertEqual(final["baseline_revenue"], 120)
        self.assertEqual(final["baseline_hourly_revenue"], 140)
        self.assertEqual(final["revenue"], 150)
        self.assertEqual(final["revenue_change"], 30)
        self.assertEqual(final["revenue_change_pct"], 25)
        self.assertEqual(final["revenue_change_vs_hourly"], 10)
        self.assertAlmostEqual(final["revenue_change_vs_hourly_pct"], 100 / 14)
        baseline = select(rows, "baseline")[0]
        self.assertEqual(baseline["revenue"], 120)
        self.assertEqual(baseline["revenue_change"], 0)
        self.assertEqual(baseline["revenue_change_vs_hourly"], -20)
        self.assertEqual(self.payload, original_payload)
        self.assertEqual(self.hourly, original_hourly)

    def test_missing_source_or_hour_is_unknown_with_no_approximation(self):
        final = select(extract_window_rounds(self.payload), "final")[0]
        self.assertIsNone(final["baseline_hourly_revenue"])
        self.assertIsNone(final["revenue_change_vs_hourly"])
        for source in ([], self.hourly[:2], [self.hourly[0], self.hourly[2]]):
            with self.subTest(source=source):
                row = self.final(source)
                self.assertIsNone(row["baseline_hourly_revenue"])
                self.assertEqual(row["baseline_revenue"], 120)
                self.assertEqual(row["revenue"], 150)

    def test_duplicate_hour_is_unknown_even_when_identical(self):
        self.hourly.append(dict(self.hourly[1]))
        self.assertIsNone(self.final()["baseline_hourly_revenue"])

    def test_zero_hourly_baseline_is_known_and_percentage_is_unknown(self):
        for row in self.hourly:
            row["e_price"] = 0
        final = self.final()
        self.assertEqual(final["baseline_hourly_revenue"], 0)
        self.assertEqual(final["revenue_change_vs_hourly"], 150)
        self.assertIsNone(final["revenue_change_vs_hourly_pct"])
        summary = summarize_windows([final], include_revenue=True)
        self.assertEqual(summary["baseline_hourly_revenue"], 0)
        self.assertIs(summary["baseline_hourly_revenue_complete"], True)
        self.assertEqual(summary["baseline_hourly_revenue_coverage_pct"], 100)
        self.assertIsNone(summary["revenue_change_vs_hourly_pct"])

    def test_invalid_hourly_values_and_explicit_validity_flags_are_unknown(self):
        for extra in ({"e_price": -1}, {"e_price": None}, {"e_price": float("nan")},
                      {"predicted_kwh": float("inf")}, {"predicted_kwh": -3},
                      {"predicted_kwh": True}, {"e_price_valid": False}, {"price_valid": False}):
            with self.subTest(extra=extra):
                records = copy.deepcopy(self.hourly)
                records[1].update(extra)
                self.assertIsNone(self.final(records)["baseline_hourly_revenue"])

    def test_zone_and_timestamp_normalization_accepts_dataframe_and_generator(self):
        for index, row in enumerate(self.hourly):
            row["zone_id"] = " 001 "
            row["time"] = f"2026-01-01T0{index + 1}:00:00+01:00"
        self.assertEqual(self.final(pd.DataFrame(self.hourly))["baseline_hourly_revenue"], 140)
        self.assertEqual(self.final(iter(self.hourly))["baseline_hourly_revenue"], 140)
        # Numeric IDs normalize without collapsing meaningful leading zero IDs.
        self.payload["zones"][0]["zone_id"] = 1
        for row in self.hourly:
            row["zone_id"] = 1.0
        self.assertEqual(self.final()["baseline_hourly_revenue"], 140)
        self.payload["zones"][0]["zone_id"] = "001"
        self.assertIsNone(self.final()["baseline_hourly_revenue"])

    def test_duplicate_times_in_different_formats_and_irregular_hours_are_unknown(self):
        records = self.hourly + [{**self.hourly[1], "time": "2026-01-01T02:00:00+01:00"}]
        self.assertIsNone(self.final(records)["baseline_hourly_revenue"])
        records = copy.deepcopy(self.hourly)
        records[1]["time"] = "2026-01-01T01:30:00"
        self.assertIsNone(self.final(records)["baseline_hourly_revenue"])
        records[1]["time"] = "not-a-date"
        self.assertIsNone(self.final(records)["baseline_hourly_revenue"])

    def test_other_zones_and_outside_hours_do_not_change_complete_window(self):
        records = self.hourly + [
            {**self.hourly[0], "zone_id": "002", "e_price": 999},
            {**self.hourly[0], "time": "2026-01-01T03:00:00", "e_price": -1},
        ]
        self.assertEqual(self.final(records)["baseline_hourly_revenue"], 140)

    def test_overflow_in_hourly_product_or_sum_is_unknown(self):
        records = copy.deepcopy(self.hourly)
        records[0]["e_price"] = 1e308
        self.assertIsNone(self.final(records)["baseline_hourly_revenue"])
        for row in records:
            row.update(e_price=1e308, predicted_kwh=1)
        final = self.final(records)
        self.assertIsNone(final["baseline_hourly_revenue"])
        json.dumps(final, allow_nan=False)

    def test_summary_full_and_partial_hourly_comparisons_keep_window_denominator(self):
        first = self.final()
        second = {**first, "window_start": "other", "baseline_hourly_revenue": 100}
        summary = summarize_windows([first, second], include_revenue=True)
        self.assertEqual(summary["baseline_hourly_revenue"], 240)
        self.assertEqual(summary["known_baseline_hourly_revenue"], 240)
        self.assertEqual(summary["baseline_hourly_revenue_known_count"], 2)
        self.assertEqual(summary["revenue_hourly_comparable_count"], 2)
        self.assertEqual(summary["revenue_change_vs_hourly"], 60)
        self.assertEqual(summary["revenue_change_vs_hourly_pct"], 25)
        second["baseline_hourly_revenue"] = None
        partial = summarize_windows([first, second], include_revenue=True)
        self.assertIsNone(partial["baseline_hourly_revenue"])
        self.assertEqual(partial["known_baseline_hourly_revenue"], 140)
        self.assertEqual(partial["baseline_hourly_revenue_known_count"], 1)
        self.assertEqual(partial["baseline_hourly_revenue_unknown_count"], 1)
        self.assertEqual(partial["baseline_hourly_revenue_coverage_pct"], 50)
        self.assertEqual(partial["revenue_hourly_comparable_count"], 1)
        self.assertEqual(partial["known_revenue_change_vs_hourly"], 10)
        self.assertIsNone(partial["revenue_change_vs_hourly"])
        self.assertIsNone(partial["revenue_change_vs_hourly_pct"])
        no_coverage = summarize_windows([second], include_revenue=True)
        self.assertIsNone(no_coverage["known_revenue_change_vs_hourly"])
        self.assertEqual(no_coverage["revenue_hourly_comparable_count"], 0)

    def test_tables_export_both_baselines_without_summing_rounds(self):
        rows = [dict(row, batch="sample", dataset="UrbanEV", dataset_identity="urban",
                     revenue_unit="CNY", backend="lstm", agent_mode="single", run_id="run")
                for row in extract_window_rounds(self.payload, baseline_hourly=self.hourly)]
        tables = build_window_tables(rows)
        table = tables["revenue_by_dataset_model_mode"]
        self.assertEqual(len(table), 3)
        self.assertTrue(all(row["baseline_hourly_revenue"] == 140 for row in table))
        self.assertTrue(all(row["baseline_revenue"] == 120 for row in table))
        self.assertEqual([row["revenue_change_vs_hourly"] for row in table], [-20, 10, 10])


if __name__ == "__main__":
    unittest.main()
