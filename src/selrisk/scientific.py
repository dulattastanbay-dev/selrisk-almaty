"""Fail-closed validation of the point-day positive/unlabeled panel.

This checks data contracts, not provenance authenticity or model validity.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = [
    "latitude", "longitude", "temperature_mean_lag1_c", "temperature_min_lag1_c",
    "temperature_max_lag1_c", "temperature_mean_prev7d_c", "precipitation_lag1_mm",
    "precipitation_prev3d_mm", "precipitation_prev7d_mm", "precipitation_prev14d_mm",
    "rain_lag1_mm", "snowfall_lag1_cm", "soil_moisture_lag1_m3_m3", "snow_depth_lag1_m",
    "snow_depth_change_prev1d_m", "earthquake_count_prev7d",
    "earthquake_max_magnitude_prev7d", "earthquake_nearest_prev7d_km",
    "day_of_year_sin", "day_of_year_cos",
]
METADATA = ["date", "location_id", "feature_window_end_local", "split", "target",
            "label_status", "event_id", "event_type", "event_source_url"]
OPTIONAL_MISSING = ["earthquake_max_magnitude_prev7d", "earthquake_nearest_prev7d_km"]


def validate_panel(frame: pd.DataFrame) -> dict:
    """Validate without dropping rows, imputing labels or fitting a model."""
    df = frame.copy()
    required = FEATURES + METADATA
    if len(df) == 0:
        raise ValueError("Dataset is empty")
    if df.columns.duplicated().any() or set(df.columns) != set(required):
        raise ValueError("Expected exactly the 29 version-0.1 panel columns")
    dates = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="raise")
    if dates.isna().any() or not df["date"].astype(str).eq(dates.dt.strftime("%Y-%m-%d")).all():
        raise ValueError("Dates must be nonempty ISO calendar dates")
    ids = df["location_id"]
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
        raise ValueError("Missing location_id")
    if df.duplicated(["date", "location_id"]).any():
        raise ValueError("Duplicate location-day")
    numeric = df[FEATURES].apply(pd.to_numeric, errors="raise")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Infinite features")
    if numeric.drop(columns=OPTIONAL_MISSING).isna().any().any():
        raise ValueError("Missing mandatory features; no synthetic fallback allowed")
    if not numeric.latitude.between(-90, 90).all() or not numeric.longitude.between(-180, 180).all():
        raise ValueError("Invalid coordinates")
    for name in ["precipitation_lag1_mm", "precipitation_prev3d_mm", "precipitation_prev7d_mm",
                 "precipitation_prev14d_mm", "rain_lag1_mm", "snowfall_lag1_cm", "snow_depth_lag1_m"]:
        if numeric[name].lt(0).any():
            raise ValueError("Negative water/snow feature: " + name)
    if not numeric.soil_moisture_lag1_m3_m3.between(0, 1).all():
        raise ValueError("Invalid volumetric soil moisture")
    if (numeric.temperature_min_lag1_c > numeric.temperature_mean_lag1_c).any() or (numeric.temperature_mean_lag1_c > numeric.temperature_max_lag1_c).any():
        raise ValueError("Inconsistent temperature range")
    count = numeric.earthquake_count_prev7d
    if count.lt(0).any() or not count.eq(np.floor(count)).all():
        raise ValueError("Earthquake count must be a nonnegative integer")
    for name in OPTIONAL_MISSING:
        if not numeric[name].isna().eq(count.eq(0)).all():
            raise ValueError("Earthquake summaries must be null iff count is zero")
    if numeric.earthquake_nearest_prev7d_km.dropna().lt(0).any():
        raise ValueError("Negative epicentral distance")
    if (numeric.precipitation_lag1_mm > numeric.precipitation_prev3d_mm + 1e-5).any() or (numeric.precipitation_prev3d_mm > numeric.precipitation_prev7d_mm + 1e-5).any() or (numeric.precipitation_prev7d_mm > numeric.precipitation_prev14d_mm + 1e-5).any():
        raise ValueError("Inconsistent nested precipitation windows")
    for name, function in [("day_of_year_sin", np.sin), ("day_of_year_cos", np.cos)]:
        expected = function(2 * np.pi * (dates.dt.dayofyear - 1) / 366)
        if not np.allclose(numeric[name], expected, atol=1e-6, rtol=0):
            raise ValueError("Incorrect seasonal encoding")
    cutoff = pd.to_datetime(df["feature_window_end_local"], utc=True, errors="raise")
    expected_cutoff = dates.dt.tz_localize("Etc/GMT-6").dt.tz_convert("UTC") - pd.Timedelta(seconds=1)
    if cutoff.isna().any() or not cutoff.eq(expected_cutoff).all():
        raise ValueError("Feature cutoff must be the end of D-1 at UTC+6")
    if not dates.between(pd.Timestamp("2016-01-01"), pd.Timestamp("2023-12-31")).all():
        raise ValueError("This v0.1 contract covers 2016-2023 only")
    expected_split = np.where(dates.dt.year <= 2020, "train", np.where(dates.dt.year == 2021, "validation", "test"))
    if not df["split"].eq(expected_split).all():
        raise ValueError("Invalid chronological split")
    target = pd.to_numeric(df["target"], errors="raise")
    if not target.dropna().eq(1).all():
        raise ValueError("v0.1 labels must be confirmed positives or null; negatives require a new reviewed contract")
    positive = target.eq(1).fillna(False)
    expected_status = np.where(positive, "confirmed_basin_event", "unlabeled")
    if not df["label_status"].eq(expected_status).all():
        raise ValueError("Label status disagrees with target")
    for name in ["event_id", "event_type", "event_source_url"]:
        present = df[name].fillna("").astype(str).str.strip().ne("")
        if not present.eq(positive).all():
            raise ValueError("Event metadata must exist exactly for positive labels: " + name)
    if not df.loc[positive, "event_source_url"].str.match(r"https?://[^\s/]+").all():
        raise ValueError("Positive labels require source URLs")
    if not df.loc[positive, "event_type"].isin(["debris_flow", "mudslide"]).all():
        raise ValueError("Landslides must not silently become debris-flow labels")
    per_split = {}
    for split in ["train", "validation", "test"]:
        labels = target[df["split"].eq(split)]
        per_split[split] = {"rows": int(len(labels)), "positives": int(labels.eq(1).sum()),
                            "unlabeled": int(labels.isna().sum())}
    return {"schema_status": "valid", "training_status": "blocked_insufficient_ground_truth",
        "rows": len(df), "columns": len(df.columns), "locations": int(ids.nunique()),
        "positive_basin_days": int(positive.sum()), "positive_dates": int(dates[positive].nunique()),
        "confirmed_negatives": 0, "unlabeled": int(target.isna().sum()), "splits": per_split,
        "allowed_features": list(FEATURES), "excluded_metadata": list(METADATA),
        "limitations": ["No verified negative labels; null is not zero.",
            "Point weather is not a catchment average or a terrain susceptibility dataset.",
            "Three basin positives on one date are not three independent storms.",
            "Lagged retrospective reanalysis is not proof of real-time availability.",
            "Schema checks do not authenticate event reports, hashes or upstream feature calculations.",
            "No model, calibration, AUC, probability or safety claim is produced."]}


def load_panel(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Dataset not found: {path}. Supply the previously generated CSV or Parquet; no demo fallback.")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise ValueError("Parquet requires pyarrow; install it or use the CSV") from exc
    raise ValueError("Only CSV and Parquet are supported")


def main(argv=None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "data/real/model_dataset.csv")
    parser.add_argument("--report", type=Path, default=root / "data/outputs/scientific/validation.json")
    args = parser.parse_args(argv)
    try:
        panel = load_panel(args.dataset)
        report = validate_panel(panel)
        report["dataset_sha256"] = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
        report["dataset_path"] = str(args.dataset.resolve())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        report = {"schema_status": "invalid_or_missing", "training_status": "blocked", "error": str(exc)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 2
