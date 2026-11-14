"""Real-data RF terrain ranking + histogram gradient boosting weather trigger.

No synthetic features. Spatial labels describe INVENTORY MEMBERSHIP, not safety.
Temporal zeroes are the user's separate 300-day assumption, never source labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from .realdata import ROOT, ensure_dataset, load_regional, sha256, verify_file
from .research import OUTPUT, design, ranking_metrics, write_json

TERRAIN_FEATURES = ["elevation_m", "slope_deg", "aspect_sin", "aspect_cos"]
EXPORT_NAMES = ("model_rf.json", "model_boosting.json", "terrain_grid.json",
                "two_model_report.json", "assumed_negative_days.csv")


def compact_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                    separators=(",", ":")) + "\n", encoding="utf-8")


def load_spatial():
    root = ensure_dataset()
    path = verify_file(root, "data/spatial_pu_grid.csv")
    grid = pd.read_csv(path)
    needed = TERRAIN_FEATURES + ["latitude", "longitude"]
    if (grid.empty or grid.sample_id.duplicated().any()
            or not np.isfinite(grid[needed].to_numpy(float)).all()
            or not grid.within_mapped_mudflow_center.isin([0, 1]).all()
            or not grid.target.dropna().eq(1).all()
            or not grid.target.eq(1).equals(grid.within_mapped_mudflow_center.eq(1))
            or grid.spatial_block_id.isna().any()):
        raise ValueError("Invalid real spatial inventory; no synthetic fallback")
    return grid, path


def bounds(frame, features):
    low, high = frame[features].min().tolist(), frame[features].max().tolist()
    for i, key in enumerate(features):
        if key.startswith("day_of_year_"):
            low[i], high[i] = -1., 1.
    return {"feature_min": low, "feature_max": high}


def forest_trees(model):
    result = []
    for estimator in model.estimators_:
        t = estimator.tree_
        # sklearn forest inference converts inputs to float32, unlike HistGB.
        values = t.value[:, 0, :]
        p = values[:, 1] / values.sum(axis=1)
        result.append([[int(t.children_left[i]), int(t.children_right[i]),
                        int(t.feature[i]) if t.children_left[i] != -1 else -1,
                        float(t.threshold[i]) if t.children_left[i] != -1 else 0.,
                        float(p[i]) if t.children_left[i] == -1 else 0.]
                       for i in range(t.node_count)])
    return result


def boosting_trees(model):
    result = []
    # This private sklearn representation is version-pinned and parity-tested.
    # Predictor leaf values ALREADY contain the learning rate. Do not multiply twice.
    for iteration in model._predictors:
        if len(iteration) != 1:
            raise ValueError("Only binary numeric HistGB is supported")
        nodes = iteration[0].nodes
        if nodes["is_categorical"].any():
            raise ValueError("Categorical tree export is unsupported")
        result.append([[-1 if n["is_leaf"] else int(n["left"]),
                        -1 if n["is_leaf"] else int(n["right"]),
                        -1 if n["is_leaf"] else int(n["feature_idx"]),
                        0. if n["is_leaf"] else float(n["num_threshold"]),
                        float(n["value"]) if n["is_leaf"] else 0.]
                       for n in nodes])
    return result


def _build(output, web_output):
    grid, spatial_path = load_spatial()
    frame, features, root = load_regional("power")
    d = design(frame)
    train = d[d.split.eq("train") & d.evaluation_eligible]
    positives = train[train.target.eq(1)]
    pool = train[train.research_role.eq("unlabeled_background")]
    negatives = pool.sample(n=300, random_state=42).sort_values("date")
    assumptions = pd.DataFrame({
        "date": negatives.date, "source_target": negatives.target, "assumed_target": 0,
        "label_basis": "user_assumption_no_catalogue_record",
        "verified_negative": False, "split": "train"})
    x = pd.concat([positives, negatives])
    y = np.r_[np.ones(len(positives)), np.zeros(len(negatives))]
    weights = np.r_[np.ones(len(positives)), np.full(300, len(pool) / 300)]

    # Same algorithm families and original tree hyperparameters; only real,
    # available predictors are used. No invented lithology, snow or lake distances.
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=8,
                                class_weight="balanced", random_state=42, n_jobs=1)
    spatial_y = grid.within_mapped_mudflow_center.to_numpy(int)
    oof = np.full(len(grid), np.nan)
    folds = []
    for fit, held in GroupKFold(n_splits=5).split(grid, spatial_y, grid.spatial_block_id):
        model = clone(rf).fit(grid.iloc[fit][TERRAIN_FEATURES], spatial_y[fit])
        oof[held] = model.predict_proba(grid.iloc[held][TERRAIN_FEATURES])[:, 1]
        folds.append({
            "fit_blocks": sorted(grid.iloc[fit].spatial_block_id.unique().tolist()),
            "heldout_blocks": sorted(grid.iloc[held].spatial_block_id.unique().tolist()),
            "heldout_rows": len(held)})
    rf.fit(grid[TERRAIN_FEATURES], spatial_y)
    gb = HistGradientBoostingClassifier(
        learning_rate=.06, max_iter=400, max_depth=4, l2_regularization=1.,
        random_state=42, early_stopping=False)
    gb.fit(x[features], y, sample_weight=weights)

    common = {
        "version": "0.3", "dataset_version": "0.2",
        "node_layout": ["left", "right", "feature", "threshold", "leaf_value"],
        "source_labels_modified": False, "synthetic_features": False,
        "seed": 42, "calibration": "not_independently_validated"}
    rf_artifact = {
        **common, "model_type": "random_forest_inventory_background",
        "estimator": "sklearn.ensemble.RandomForestClassifier",
        "features": TERRAIN_FEATURES, **bounds(grid, TERRAIN_FEATURES),
        "input_dtype": "float32", "trees": forest_trees(rf),
        "trained_rows": len(grid), "mapped_inventory_samples": int(spatial_y.sum()),
        "unlabeled_background_samples": int((spatial_y == 0).sum()),
        "training_label": "within_mapped_mudflow_center_NOT_event_absence",
        "output": "terrain_inventory_similarity_not_calibrated_susceptibility",
        "cv": "5-fold GroupKFold by pre-existing 5km blocks; final refit on inventory",
        "inventory_temporal_cutoff": "undated static inventory; not historical as-of geography",
        "source_sha256": sha256(spatial_path)}
    domain = d[d.split.eq("train") & d.feature_complete]
    gb_artifact = {
        **common, "model_type": "hist_gradient_boosting_assumed_negative_trigger",
        "estimator": "sklearn.ensemble.HistGradientBoostingClassifier",
        "features": features, **bounds(domain, features),
        "input_dtype": "float64", "trees": boosting_trees(gb),
        "baseline": float(gb._baseline_prediction[0, 0]),
        "leaf_values_include_learning_rate": True,
        "train_start": str(x.date.min()), "train_end": str(x.date.max()),
        "positive_training_dates": len(positives), "assumed_negative_days": 300,
        "verified_negative_days": 0, "negative_sample_weight": len(pool) / 300,
        "assumed_negative_population": len(pool), "negative_sampling_seed": 42,
        "validation_or_test_used_for_fit": False,
        "weather_training_source": "NASA POWER regional point 43.1,77.0",
        "weather_live_source": "Open-Meteo; provider transfer not validated",
        "weather_cutoff": "D-15 through D-2 UTC",
        "output": "conditional_trigger_score_under_assumed_negative_labels",
        "source_sha256": sha256(verify_file(root, "data/model_regional_power.csv"))}
    report = {
        "version": "0.3", "active_model_count": 2,
        "models": [rf_artifact["estimator"], gb_artifact["estimator"]],
        "fusion": "RF terrain inventory similarity * GB regional weather trigger",
        "fusion_interpretation": "relative conditional index, NOT calibrated local event probability",
        "spatial": {
            "samples": len(grid), "inventory_positive": int(spatial_y.sum()),
            "background": int((spatial_y == 0).sum()), "verified_absences": 0,
            "features": TERRAIN_FEATURES, "dem_resolution_m": 90, "sample_spacing_m": 450,
            "cv_folds": folds,
            "oof_origin_discrimination": ranking_metrics(spatial_y, oof),
            "final_refit_on_entire_static_inventory": True,
            "not_as_of_historical_forecast": True},
        "temporal": {
            "train_positive_dates": len(positives), "assumed_negative_days": 300,
            "verified_negative_days": 0, "negative_population": len(pool),
            "negative_sample_weight": len(pool) / 300,
            "split": "train 1981-2005; validation 2006-2012; test 2013-2020",
            "fit_uses_validation_or_test": False, "early_stopping": False,
            "hyperparameters": gb.get_params(),
            "evaluation": {}},
        "limitations": [
            "Spatial background is not observed safety. RF learns inventory membership.",
            "300 temporal zeroes are user-authorized assumptions; source targets stay 1/NULL.",
            "One regional weather signal is shared by all terrain cells.",
            "No joint spatiotemporal event-probability calibration or prospective validation.",
            "Existing test years were previously inspected in research comparisons.",
            "Current Open-Meteo vs training NASA POWER provider shift is unvalidated."]}
    for split in ("validation", "test"):
        rows = d[d.split.eq(split) & d.evaluation_eligible]
        p = gb.predict_proba(rows[features])[:, 1]
        report["temporal"]["evaluation"][split] = ranking_metrics(rows.sample_origin, p)

    terrain = {
        "version": "0.3", "sample_spacing_m": 450, "dem_resolution_m": 90,
        "geometry": "regular terrain sample locations, NOT catchment boundaries",
        "source_sha256": sha256(spatial_path),
        "columns": ["sample_id", "latitude", "longitude"] + TERRAIN_FEATURES,
        "data": grid[["sample_id", "latitude", "longitude"] + TERRAIN_FEATURES].values.tolist()}
    output, web_output = Path(output), Path(web_output)
    output.mkdir(parents=True, exist_ok=True)
    web_output.mkdir(parents=True, exist_ok=True)
    objects = {"model_rf.json": rf_artifact, "model_boosting.json": gb_artifact,
               "terrain_grid.json": terrain, "two_model_report.json": report}
    for name, value in objects.items():
        compact_json(output / name, value)
        compact_json(web_output / name, value)
    for destination in (output, web_output):
        assumptions.to_csv(destination / "assumed_negative_days.csv", index=False)
    joblib.dump(rf, output / "model_rf.joblib", compress=3)
    joblib.dump(gb, output / "model_boosting.joblib", compress=3)
    ledger = grid[["sample_id", "spatial_block_id", "target", "label_status",
                   "within_mapped_mudflow_center"]].copy()
    ledger["training_sample_origin"] = spatial_y
    ledger.to_csv(output / "spatial_training_ledger.csv", index=False)
    write_json(output / "training_manifest.json", {
        "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
        "source_files": {"spatial": rf_artifact["source_sha256"],
                         "weather": gb_artifact["source_sha256"]},
        "sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()
                   and p.name != "training_manifest.json"},
        "rf_hyperparameters": rf.get_params(), "boosting_hyperparameters": gb.get_params(),
        "source_labels_modified": False, "joblib_load_trusted_files_only": True})
    print(json.dumps({"active_models": 2, "spatial_samples": len(grid),
                      "temporal_positive_dates": len(positives),
                      "assumed_negative_days": 300, "output": str(output)}))
    return report


def build_two_model_map(output=None, web_output=None):
    # Avoid severe OpenMP oversubscription on high-core build/CI machines.
    with threadpool_limits(limits=1):
        return _build(output or OUTPUT / "two_model_map", web_output or ROOT / "web/data")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT / "two_model_map")
    parser.add_argument("--web-output", type=Path, default=ROOT / "web/data")
    args = parser.parse_args()
    build_two_model_map(args.output, args.web_output)


if __name__ == "__main__":
    main()
