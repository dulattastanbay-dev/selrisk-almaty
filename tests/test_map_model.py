import json
import subprocess
import joblib
import numpy as np
import pandas as pd
import pytest
from selrisk.realdata import ROOT, load_regional
from selrisk.research import design, score_rows
from selrisk.map_model import build_map_model
from selrisk.reproducibility import compare_model


def test_map_assumptions_are_separate_300_train_only():
    a = pd.read_csv(ROOT/"web/data/assumed_negative_days.csv")
    d, _, _ = load_regional("power")
    assert len(a) == a.date.nunique() == 300
    assert a.assumed_target.eq(0).all() and a.source_target.isna().all()
    assert not a.verified_negative.any() and a.split.eq("train").all()
    selected = design(d).set_index("date").loc[a.date]
    assert selected.target.isna().all()
    assert selected.split.eq("train").all()
    assert selected.research_role.eq("unlabeled_background").all()
    assert d.target.eq(0).sum() == 0


def test_map_training_persistence_and_reproducibility(tmp_path):
    artifact = build_map_model(tmp_path/"model", tmp_path/"export")
    expected = json.loads((ROOT/"web/data/model_map300.json").read_text())
    compare_model(expected, artifact)
    assert artifact["source_labels_modified"] is False
    assert artifact["verified_negative_days"] == 0
    assert artifact["validation_or_test_used_for_fit"] is False
    assert artifact["assumed_negative_days"] == 300
    assert artifact["negative_sample_weight"] > 1
    pipeline = joblib.load(tmp_path/"model/model_power.joblib")
    d, features, _ = load_regional("power")
    scores, status = score_rows(artifact, d)
    use = status == "available"
    np.testing.assert_allclose(pipeline.decision_function(d.loc[use, features]),
                               scores[use], atol=2e-12, rtol=0)
    for name in ["assumed_negative_days.csv", "spatial_prior.json"]:
        assert (tmp_path/"export"/name).read_bytes() == (ROOT/"web/data"/name).read_bytes()


def test_spatial_prior_only_sourced_centers_and_normalized_weights():
    prior = json.loads((ROOT/"web/data/spatial_prior.json").read_text())
    centers = json.loads((ROOT/"web/data/centers.geojson").read_text())
    assert set(prior["centers"]) == {f["properties"]["center_id"] for f in centers["features"]}
    assert not prior["test_used"] and prior["train_end"] <= "2005-12-31"
    weights = [p["weight"] for p in prior["centers"].values()]
    assert all(w > 0 for w in weights) and sum(weights) == pytest.approx(1)


def test_map_js_weather_window_and_two_model_fusion():
    script = r"""
const assert=require('node:assert/strict');
const api=require('./web/map-app.js');
const time=Array.from({length:14},(_,i)=>`2020-01-${String(i+1).padStart(2,'0')}`);
const daily={time,precipitation_sum:Array(14).fill(1),temperature_2m_mean:Array(14).fill(5),
temperature_2m_min:Array(14).fill(2),temperature_2m_max:Array(14).fill(8)};
const r=api.featuresFromDaily(daily,'2020-01-16');
assert.equal(r.precipitation_prev3d_mm,3);assert.equal(r.precipitation_prev7d_mm,7);
assert.equal(r.precipitation_prev14d_mm,14);assert.equal(r.positive_degree_days_prev7d,35);
assert.throws(()=>api.featuresFromDaily({...daily,time:time.slice(1)},'2020-01-16'));
assert.throws(()=>api.featuresFromDaily({...daily,precipitation_sum:Array(14).fill(null)},'2020-01-16'));
assert.equal(api.localRisk(0,.5),0);assert.equal(api.localRisk(1,.5),.5);
assert.equal(api.localRisk(.2,.3),.06);
assert.throws(()=>api.localRisk(NaN,.5));
assert.throws(()=>api.featuresFromDaily({...daily,time:[...time.slice(0,5),'2020-01-05',...time.slice(6)]},'2020-01-16'));
assert.notEqual(api.color(.000001),api.color(.01));
console.log('map weather/probability checks passed');
"""
    subprocess.run(["node", "-e", script], cwd=ROOT, check=True, capture_output=True, text=True)


def test_active_map_never_loads_legacy_logistic_or_frequency_prior():
    script = (ROOT/"web/map-app.js").read_text()
    assert "model_boosting.json" in script and "model_rf.json" in script
    assert "trees.fuse(layer.spatialScore,current.p)" in script
    assert "spatial_prior" not in script and "model_map300" not in script
