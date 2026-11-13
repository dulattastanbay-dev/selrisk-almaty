"""Contract fixtures are artificial unit tests, never observational data."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from selrisk.scientific import FEATURES, METADATA, main, validate_panel


def fixture():
    dates = pd.to_datetime(["2020-01-02", "2021-07-27", "2023-07-21"])
    df = pd.DataFrame({name: np.zeros(3) for name in FEATURES})
    df["latitude"], df["longitude"] = 43.1, 77.25
    df["soil_moisture_lag1_m3_m3"] = 0.3
    df["temperature_min_lag1_c"] = -1
    df["temperature_max_lag1_c"] = 1
    for name in ["earthquake_max_magnitude_prev7d", "earthquake_nearest_prev7d_km"]:
        df[name] = np.nan
    df["day_of_year_sin"] = np.sin(2*np.pi*(dates.dayofyear-1)/366)
    df["day_of_year_cos"] = np.cos(2*np.pi*(dates.dayofyear-1)/366)
    df["date"] = dates.strftime("%Y-%m-%d")
    df["location_id"] = "test_point"
    df["feature_window_end_local"] = (dates-pd.Timedelta(days=1)).strftime("%Y-%m-%dT23:59:59+06:00")
    df["split"] = ["train", "validation", "test"]
    df["target"] = [np.nan, 1, 1]
    df["label_status"] = ["unlabeled", "confirmed_basin_event", "confirmed_basin_event"]
    df["event_id"] = ["", "test1", "test2"]
    df["event_type"] = ["", "debris_flow", "debris_flow"]
    df["event_source_url"] = ["", "https://example.org/1", "https://example.org/2"]
    return df


class PanelTests(unittest.TestCase):
    def test_positive_unlabeled_preserved(self):
        df = fixture()
        original = df.copy(deep=True)
        report = validate_panel(df)
        self.assertEqual(report["unlabeled"], 1)
        self.assertEqual(report["positive_basin_days"], 2)
        self.assertEqual(report["training_status"], "blocked_insufficient_ground_truth")
        self.assertTrue(set(FEATURES).isdisjoint(METADATA))
        pd.testing.assert_frame_equal(df, original)

    def test_invalid_contracts(self):
        mutations = {
            "negative_target": lambda d: d.assign(target=[0,1,1]),
            "missing_target_metadata": lambda d: d.assign(event_source_url=""),
            "unknown_target": lambda d: d.assign(target=[2,1,1]),
            "infinity": lambda d: d.assign(latitude=np.inf),
            "missing_feature": lambda d: d.drop(columns="rain_lag1_mm"),
            "extra_leakage": lambda d: d.assign(trigger_prob_true=0.5),
            "duplicate": lambda d: pd.concat([d,d.iloc[:1]], ignore_index=True),
            "missing_day": lambda d: d.assign(date=[None,"2021-07-27","2023-07-21"]),
            "wrong_cutoff": lambda d: d.assign(feature_window_end_local="2023-07-21T23:59:59+06:00"),
            "split_leak": lambda d: d.assign(split="train"),
            "bad_coordinate": lambda d: d.assign(latitude=91),
            "negative_rain": lambda d: d.assign(rain_lag1_mm=-1),
            "soil": lambda d: d.assign(soil_moisture_lag1_m3_m3=1.5),
            "missing_soil": lambda d: d.assign(soil_moisture_lag1_m3_m3=np.nan),
            "temperature": lambda d: d.assign(temperature_min_lag1_c=2),
            "count": lambda d: d.assign(earthquake_count_prev7d=0.5),
            "missing_eq_summary": lambda d: d.assign(earthquake_count_prev7d=1),
            "false_eq_summary": lambda d: d.assign(earthquake_max_magnitude_prev7d=0),
            "nested_rain": lambda d: d.assign(precipitation_lag1_mm=10),
            "season": lambda d: d.assign(day_of_year_sin=10),
            "bad_status": lambda d: d.assign(label_status="unlabeled"),
            "landslide": lambda d: d.assign(event_type=["","landslide","debris_flow"]),
            "url": lambda d: d.assign(event_source_url=["","not-a-url","https://example.org"]),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_panel(mutate(fixture()))

    def test_missing_input_blocks_with_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["--dataset",str(path/"absent.csv"),"--report",str(path/"report.json")])
            self.assertEqual(code,2)
            self.assertEqual(json.loads((path/"report.json").read_text())["schema_status"],"invalid_or_missing")

    def test_csv_cli_and_stale_report_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            fixture().to_csv(path/"panel.csv",index=False)
            (path/"report.json").write_text('{"training_status":"ready"}')
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["--dataset",str(path/"panel.csv"),"--report",str(path/"report.json")])
            report = json.loads((path/"report.json").read_text())
            self.assertEqual(code,2)
            self.assertEqual(report["schema_status"],"valid")
            self.assertEqual(len(report["dataset_sha256"]),64)
            self.assertEqual(report["training_status"],"blocked_insufficient_ground_truth")

    def test_entrypoints_no_demo_fallback(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            for name,function in [("train","cli"),("run_all","main")]:
                spec=importlib.util.spec_from_file_location("entry_"+name,root/"scripts"/(name+".py"))
                module=importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                with contextlib.redirect_stdout(io.StringIO()):
                    code=getattr(module,function)(["--dataset",str(Path(tmp)/"missing.csv"),
                        "--report",str(Path(tmp)/(name+".json"))])
                self.assertEqual(code,2)
                self.assertFalse((Path(tmp)/"models").exists())

    def test_demo_rejects_real_data_flags(self):
        root = Path(__file__).resolve().parents[1]
        for name,function in [("train","cli"),("run_all","main")]:
            spec=importlib.util.spec_from_file_location("entry_"+name,root/"scripts"/(name+".py"))
            module=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                getattr(module,function)(["--demo","--dataset","anything.csv"])
            self.assertEqual(caught.exception.code,2)


if __name__ == "__main__":
    unittest.main()
