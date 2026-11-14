"""Experimental map model with explicitly assumed, not verified, negative days."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .realdata import ROOT, load_regional
from .research import OUTPUT, design, model_artifact, write_json
from .persistence import save_pipeline


def center_key(value):
    return re.split(r"\s*\(", str(value).strip())[0].strip().casefold()


def build_map_model(output=None, web_output=None):
    frame, features, _ = load_regional("power")
    d = design(frame)
    centers = json.loads((ROOT/"web/data/centers.geojson").read_text())
    events = pd.read_csv(ROOT/"web/data/event_catalogue.csv")
    mapping = {center_key(f["properties"]["name"]): f["properties"]["center_id"]
               for f in centers["features"]}
    events = events[events.exact_date_eligible & events.dedup_keep
                    & ~events.date_year_conflict].copy()
    events["center_id"] = events.center_name.map(center_key).map(mapping)
    train_days = set(d.loc[d.split.eq("train") & d.evaluation_eligible & d.target.eq(1), "date"])
    mapped = events[events.date_start.isin(train_days) & events.center_id.notna()]
    positive_dates = set(mapped.date_start)
    positives = d[d.split.eq("train") & d.evaluation_eligible & d.target.eq(1) & d.date.isin(positive_dates)]
    pool = d[d.split.eq("train") & d.evaluation_eligible
             & d.research_role.eq("unlabeled_background")]
    if len(pool) < 300 or len(positives) < 10:
        raise ValueError("Insufficient data for experimental map")
    negatives = pool.sample(n=300, random_state=42).sort_values("date").copy()
    # Source labels are immutable; assumptions live in a separately exported table.
    assumptions = pd.DataFrame({
        "date": negatives.date, "source_target": negatives.target,
        "assumed_target": 0, "label_basis": "user_assumption_no_catalogue_record",
        "verified_negative": False, "split": "train"})
    x = pd.concat([positives, negatives])
    y = np.r_[np.ones(len(positives)), np.zeros(300)]
    # Correct the 300-day subsampling within this assumed-negative population.
    weights = np.r_[np.ones(len(positives)), np.full(300, len(pool)/300)]
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.3, max_iter=4000, random_state=42))
    model.fit(x[features], y, standardscaler__sample_weight=weights,
              logisticregression__sample_weight=weights)
    artifact = model_artifact(model, features, x, "power", .3)
    # Domain support comes from the entire observed train-period feature population,
    # not the accidental extrema of a 300-day sample; no later weather is used.
    domain = d[d.split.eq("train") & d.feature_complete]
    artifact["feature_min"] = domain[features].min().tolist()
    artifact["feature_max"] = domain[features].max().tolist()
    for i, key in enumerate(features):
        if key.startswith("day_of_year_"):
            artifact["feature_min"][i], artifact["feature_max"][i] = -1., 1.
    artifact.update(
        model_type="experimental_assumed_negative_logistic",
        output="conditional_probability_under_assumed_labels_not_validated_event_probability",
        assumed_negative_days=300, verified_negative_days=0,
        assumed_negative_population=len(pool), positive_training_dates=len(positives),
        negative_sampling_seed=42, negative_sample_weight=len(pool)/300,
        domain_policy="observed_train_period_weather_bounds_and_analytic_season_bounds",
        source_labels_modified=False, weather_training_source="NASA POWER",
        weather_live_source="Open-Meteo; cross-provider transfer is not validated",
        validation_or_test_used_for_fit=False, calibration="not_independently_validated",
        warning="Conditional experimental model: absence of a catalogue record is an assumption, not verified absence.")
    output = Path(output) if output is not None else OUTPUT/"map300"
    web_output = Path(web_output) if web_output is not None else ROOT/"web/data"
    web_output.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"model_power.json", artifact)
    save_pipeline(model, output, "power")
    assumptions.to_csv(output/"assumed_negative_days.csv", index=False)
    write_json(web_output/"model_map300.json", artifact)
    assumptions.to_csv(web_output/"assumed_negative_days.csv", index=False)
    # Allocate a regional Poisson intensity using train-only center-day frequency.
    # This is an explicit spatial assumption, not a validated local-risk model.
    counts = mapped.drop_duplicates(["date_start", "center_id"]).center_id.value_counts()
    total = float(sum(int(counts.get(f["properties"]["center_id"], 0))+1
                      for f in centers["features"]))
    spatial = {
        "method": "train_center_day_frequency_plus_one_poisson_allocation_assumption",
        "train_end": artifact["train_end"], "test_used": False,
        "geometry": "published_mudflow_centers_not_catchments",
        "matched_training_records": len(mapped), "conditional_not_validated": True,
        "centers": {f["properties"]["center_id"]: {
            "count": int(counts.get(f["properties"]["center_id"], 0)),
            "weight": (int(counts.get(f["properties"]["center_id"], 0))+1)/total}
            for f in centers["features"]}}
    write_json(web_output/"spatial_prior.json", spatial)
    write_json(output/"spatial_prior.json", spatial)
    print(json.dumps({"model": "map300", "assumed_negative_days": 300,
                      "verified_negative_days": 0, "positive_training_dates": len(positives),
                      "output": str(output)}))
    return artifact


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    build_map_model()


if __name__ == "__main__":
    main()
