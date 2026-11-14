import json
import subprocess

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from threadpoolctl import threadpool_limits

from selrisk.realdata import ROOT, load_regional
from selrisk.two_model_map import EXPORT_NAMES, TERRAIN_FEATURES, build_two_model_map, load_spatial
from selrisk.reproducibility import compare_tree_model, compare_tree_report


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    directory = tmp_path_factory.mktemp("two-models")
    model, export = directory/"native", directory/"web"
    report = build_two_model_map(model, export)
    return model, export, report


def js_scores(model, rows):
    code = """
const t=require('./web/tree-model.js'),fs=require('node:fs');
const p=JSON.parse(fs.readFileSync(0,'utf8'));t.validate(p.model);
console.log(JSON.stringify(p.rows.map(r=>t.predict(p.model,r))));
"""
    run = subprocess.run(["node", "-e", code], cwd=ROOT, check=True, text=True,
                         input=json.dumps({"model": model, "rows": rows}),
                         capture_output=True)
    return json.loads(run.stdout)


def test_real_algorithms_saved_and_reproducible(trained):
    directory, export, report = trained
    assert isinstance(joblib.load(directory/"model_rf.joblib"), RandomForestClassifier)
    assert isinstance(joblib.load(directory/"model_boosting.joblib"), HistGradientBoostingClassifier)
    assert report["active_model_count"] == 2
    assert report["temporal"]["train_positive_dates"] == 88
    assert not report["temporal"]["fit_uses_validation_or_test"]
    for name in EXPORT_NAMES:
        expected, actual = ROOT/"web/data"/name, export/name
        if name.startswith("model_"):
            compare_tree_model(json.loads(expected.read_text()), json.loads(actual.read_text()))
        elif name == "two_model_report.json":
            compare_tree_report(json.loads(expected.read_text()), json.loads(actual.read_text()))
        else:
            assert expected.read_bytes() == actual.read_bytes()
    manifest = json.loads((directory/"training_manifest.json").read_text())
    assert set(manifest["sha256"]) >= {"model_rf.joblib", "model_boosting.joblib",
                                       "model_rf.json", "model_boosting.json"}
    assert manifest["source_labels_modified"] is False


def test_spatial_labels_and_group_cv_are_honest(trained):
    directory, _, report = trained
    ledger = pd.read_csv(directory/"spatial_training_ledger.csv")
    assert ledger.target.eq(1).sum() == 300
    assert ledger.target.eq(0).sum() == 0 and ledger.target.isna().sum() == 8268
    assert ledger.training_sample_origin.equals(ledger.within_mapped_mudflow_center)
    assert report["spatial"]["features"] == TERRAIN_FEATURES
    assert report["spatial"]["verified_absences"] == 0
    assert report["spatial"]["final_refit_on_entire_static_inventory"] is True
    held = []
    for fold in report["spatial"]["cv_folds"]:
        assert not set(fold["fit_blocks"]) & set(fold["heldout_blocks"])
        held.extend(fold["heldout_blocks"])
    assert len(held) == len(set(held)) == 84


@pytest.mark.parametrize("kind", ["rf", "boosting"])
def test_full_browser_python_parity_and_domain(trained, kind):
    directory, _, _ = trained
    artifact = json.loads((directory/f"model_{kind}.json").read_text())
    estimator = joblib.load(directory/f"model_{kind}.joblib")
    if kind == "rf":
        rows, _ = load_spatial()
    else:
        rows, _, _ = load_regional("power")
    features = artifact["features"]
    values = rows[features].to_numpy(float)
    available = (np.isfinite(values).all(axis=1)
                 & (values >= np.asarray(artifact["feature_min"])-1e-7).all(axis=1)
                 & (values <= np.asarray(artifact["feature_max"])+1e-7).all(axis=1))
    use = rows.loc[available, features]
    # All supported weather days / all 8,568 terrain samples, not two cherry-picked cases.
    actual = js_scores(artifact, use.to_dict("records"))
    assert all(v["status"] == "available" for v in actual)
    with threadpool_limits(limits=1):
        expected = estimator.predict_proba(use)[:, 1]
    np.testing.assert_allclose([v["value"] for v in actual], expected, atol=2e-12, rtol=0)
    row = use.iloc[0].to_dict()
    missing = {**row, features[0]: None}
    extreme = {**row, features[0]: artifact["feature_max"][0]+1000}
    invalid = js_scores(artifact, [missing, extreme])
    assert invalid == [{"value": None, "status": "missing_features"},
                       {"value": None, "status": "out_of_training_domain"}]


def test_rf_float32_split_boundary_parity(trained):
    directory, _, _ = trained
    model = joblib.load(directory/"model_rf.joblib")
    artifact = json.loads((directory/"model_rf.json").read_text())
    grid, _ = load_spatial()
    rows = []
    for tree in model.estimators_[::10]:
        feature = tree.tree_.feature[0]
        threshold = tree.tree_.threshold[0]
        for delta in [-1e-10, 0, 1e-10]:
            row = grid[TERRAIN_FEATURES].iloc[0].to_dict()
            row[TERRAIN_FEATURES[feature]] = threshold+delta
            rows.append(row)
    actual = js_scores(artifact, rows)
    with threadpool_limits(limits=1):
        expected = model.predict_proba(pd.DataFrame(rows)[TERRAIN_FEATURES])[:, 1]
    np.testing.assert_allclose([v["value"] for v in actual], expected, atol=2e-12, rtol=0)


def test_malformed_trees_rejected():
    code = """
const t=require('./web/tree-model.js'),a=require('node:assert/strict'),fs=require('node:fs');
const rf=JSON.parse(fs.readFileSync('web/data/model_rf.json'));
a.throws(()=>t.validate({...rf,model_type:'logistic'}));
const bad=structuredClone(rf);bad.trees[0][0][0]=0;a.throws(()=>t.validate(bad));
a.throws(()=>t.fuse(null,.5));a.throws(()=>t.fuse(.5,2));
t.validate(rf);console.log('tree schema checks passed');
"""
    subprocess.run(["node", "-e", code], cwd=ROOT, check=True, capture_output=True)
