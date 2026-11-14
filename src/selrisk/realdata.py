"""Versioned, hash-checked v0.2 data. No synthetic or network fallback."""
from __future__ import annotations
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
KINDS = {"power": "regional_power", "stations": "regional_stations"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ensure_dataset(destination=None, archive=None) -> Path:
    manifest = json.loads((ROOT / "datasets/manifest.json").read_text(encoding="utf-8"))
    archive = Path(archive or ROOT / manifest["archive"])
    destination = Path(destination or ROOT / "data/real/v0.2")
    if not archive.is_file() or sha256(archive) != manifest["sha256"]:
        raise ValueError("Missing or modified v0.2 archive; no demo fallback is permitted")
    if (destination / "checksums.sha256").is_file():
        return destination
    if destination.exists():
        raise ValueError("Incomplete dataset directory; move it aside before retrying")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix="selrisk-", dir=destination.parent))
    try:
        with zipfile.ZipFile(archive) as z:
            if sum(i.file_size for i in z.infolist()) > 250_000_000:
                raise ValueError("Archive exceeds extraction size limit")
            for info in z.infolist():
                p = Path(info.filename)
                if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0] != manifest["prefix"]:
                    raise ValueError("Unsafe archive member")
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive symlinks are forbidden")
            z.extractall(temp)
        (temp / manifest["prefix"]).rename(destination)
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    return destination


def verify_file(root, relative):
    root = Path(root)
    manifest = json.loads((ROOT / "datasets/manifest.json").read_text(encoding="utf-8"))
    if sha256(root / "checksums.sha256") != manifest["checksums_sha256"]:
        raise ValueError("Modified dataset checksum manifest")
    name = Path(relative)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError("Unsafe dataset path")
    hashes = {}
    for line in (root / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        hashes[name] = digest
    path = root / relative
    if relative not in hashes or not path.is_file() or sha256(path) != hashes[relative]:
        raise ValueError(f"Dataset checksum mismatch: {relative}")
    return path


def feature_names(kind="power", root=None):
    if kind not in KINDS:
        raise ValueError(f"Unknown dataset kind: {kind}")
    root = Path(root or ensure_dataset())
    p = verify_file(root, "metadata/feature_allowlist.json")
    return json.loads(p.read_text())[KINDS[kind]]


def validate_regional(df, features, kind="power"):
    if kind not in KINDS:
        raise ValueError("Unsupported regional dataset")
    required = {"date", "target", "label_status", "split", "feature_complete",
                "date_uncertainty_overlap"} | set(features)
    if df.empty or df.columns.duplicated().any() or not required <= set(df):
        raise ValueError("Empty table or missing/duplicate required columns")
    d = pd.to_datetime(df.date, format="%Y-%m-%d", errors="raise")
    if d.isna().any() or not df.date.astype(str).eq(d.dt.strftime("%Y-%m-%d")).all():
        raise ValueError("Dates must be nonempty ISO days")
    if d.duplicated().any() or not d.is_monotonic_increasing:
        raise ValueError("Regional days must be unique and chronological")
    if (d.diff().dropna() != pd.Timedelta(days=1)).any():
        raise ValueError("Regional daily timeline has gaps")
    target = pd.to_numeric(df.target, errors="raise")
    if not target.dropna().eq(1).all():
        raise ValueError("Strict v0.2 outcomes are positive or unknown, never inferred negatives")
    if not df.label_status.eq(np.where(target.eq(1), "confirmed_catalogue_positive", "unlabeled")).all():
        raise ValueError("Outcome and label status disagree")
    expected_features = feature_names(kind)
    if list(features) != expected_features:
        raise ValueError("Features must match the reviewed allowlist in order")
    for col in ["feature_complete", "date_uncertainty_overlap"]:
        if df[col].isna().any() or not df[col].isin([True, False]).all():
            raise ValueError(f"Invalid boolean field: {col}")
    x = df[features].apply(pd.to_numeric, errors="raise")
    if np.isinf(x.to_numpy(dtype=float)).any():
        raise ValueError("Infinite feature")
    if not df.feature_complete.eq(x.notna().all(axis=1)).all():
        raise ValueError("feature_complete disagrees with actual missingness")
    rain = [c for c in features if "precipitation" in c or "degree_days" in c]
    if x[rain].lt(0).any().any():
        raise ValueError("Negative precipitation / positive-degree-day feature")
    for col in [c for c in features if c.startswith("day_of_year")]:
        fn = np.sin if col.endswith("sin") else np.cos
        expected = fn(2*np.pi*(d.dt.day_of_year-1)/365.2425)
        if not np.allclose(x[col], expected, atol=1e-6):
            raise ValueError("Invalid seasonal encoding")
    years = (1981, 2020, 2005, 2012) if kind == "power" else (1937, 1991, 1973, 1981)
    if not d.between(pd.Timestamp(years[0],1,1),pd.Timestamp(years[1],12,31)).all():
        raise ValueError("Dates outside the reviewed dataset period")
    expected = np.where(d.dt.year <= years[2], "train", np.where(d.dt.year <= years[3], "validation", "test"))
    if not df.split.eq(expected).all():
        raise ValueError("Invalid chronological split")
    if kind == "power":
        cutoff = pd.to_datetime(df.feature_window_end_utc, utc=True, errors="raise")
        if not cutoff.eq(d.dt.tz_localize("UTC")-pd.Timedelta(days=1)).all():
            raise ValueError("Invalid POWER feature cutoff")
        if (x.temperature_min_lag2utc_c > x.temperature_mean_lag2utc_c).any():
            raise ValueError("Invalid temperature range")
        if (x.temperature_mean_lag2utc_c > x.temperature_max_lag2utc_c).any():
            raise ValueError("Invalid temperature range")
        for a,b in [("precipitation_lag2utc_mm","precipitation_prev3d_mm"),
                    ("precipitation_prev3d_mm","precipitation_prev7d_mm"),
                    ("precipitation_prev7d_mm","precipitation_prev14d_mm")]:
            if (x[a] > x[b]+1e-5).any():
                raise ValueError("Invalid nested rainfall windows")
    else:
        cutoff = pd.to_datetime(df.feature_window_last_observation_date, errors="raise")
        if not cutoff.eq(d-pd.Timedelta(days=2)).all():
            raise ValueError("Invalid NOAA feature cutoff")
        for station in ("mynzhylki","bao","gorelnik","almaty"):
            for a,b in ((1,3),(3,7),(7,14)):
                if (x[f"{station}_precipitation_prev{a}d_lag2_mm"] >
                    x[f"{station}_precipitation_prev{b}d_lag2_mm"]+1e-5).any():
                    raise ValueError("Invalid station rainfall windows")
    for split in ["train","validation","test"]:
        sub = df[df.split.eq(split)]
        if sub.empty or not sub.target.eq(1).any():
            raise ValueError(f"No positive examples in {split}")
    return {"schema_status":"valid","rows":len(df),"positive_days":int(target.eq(1).sum()),
            "unknown_days":int(target.isna().sum()),"verified_negative_days":0}


def load_regional(kind="power"):
    root = ensure_dataset()
    features = feature_names(kind, root)
    p = verify_file(root, f"data/model_{KINDS[kind]}.csv")
    df = pd.read_csv(p)
    validate_regional(df, features, kind)
    return df, features, root
