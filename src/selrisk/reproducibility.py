"""Strict data equality, narrowly scoped float tolerance for model computation."""
from __future__ import annotations
import io
import json
import subprocess
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
from .realdata import ROOT

ATOL=2e-8
MODEL_FLOATS={"coefficients","intercept","mean","scale"}
SCORE_COLS={"retrospective_raw_score","research_score"}


def compare_model(expected, actual):
    if set(expected)!=set(actual):
        raise AssertionError("Model keys differ")
    strict_left={k:v for k,v in expected.items() if k not in MODEL_FLOATS}
    strict_right={k:v for k,v in actual.items() if k not in MODEL_FLOATS}
    if json.dumps(strict_left,sort_keys=True,allow_nan=False)!=json.dumps(strict_right,sort_keys=True,allow_nan=False):
        raise AssertionError("Model metadata differs")
    for k in expected:
        if k in MODEL_FLOATS:
            left=np.asarray(expected[k],float);right=np.asarray(actual[k],float)
            if left.shape!=right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
                raise AssertionError(f"Invalid numeric model field: {k}")
            np.testing.assert_allclose(left,right,atol=ATOL,rtol=0,err_msg=k)


def compare_predictions(expected, actual):
    left=pd.read_csv(io.StringIO(expected),dtype=str,keep_default_na=False)
    right=pd.read_csv(io.StringIO(actual),dtype=str,keep_default_na=False)
    if left.columns.tolist()!=right.columns.tolist() or left.shape!=right.shape:
        raise AssertionError("Prediction CSV structure differs")
    for key in left:
        if key not in SCORE_COLS:
            if not left[key].equals(right[key]):
                raise AssertionError(f"Strict prediction field differs: {key}")
        else:
            if not left[key].eq("").equals(right[key].eq("")):
                raise AssertionError(f"Prediction missingness differs: {key}")
            a=pd.to_numeric(left[key].replace("",np.nan)).to_numpy(float)
            b=pd.to_numeric(right[key].replace("",np.nan)).to_numpy(float)
            if np.isinf(a).any() or np.isinf(b).any():
                raise AssertionError("Infinite prediction score")
            np.testing.assert_allclose(a,b,atol=ATOL,rtol=0,equal_nan=True,err_msg=key)


def compare_examples(expected, actual):
    left, right = deepcopy(expected), deepcopy(actual)
    # Only derived scores/thresholds may have cross-CPU rounding differences.
    for a, b in [(left, right)] + list(zip(left["cases"], right["cases"])):
        for key in ("threshold", "score"):
            if key in a:
                if not np.isfinite([a[key], b[key]]).all():
                    raise AssertionError("Nonfinite presentation score")
                np.testing.assert_allclose(a[key], b[key], atol=ATOL, rtol=0)
                a[key] = b[key] = 0
    if left != right:
        raise AssertionError("Strict presentation metadata differs")


def compare_tree_model(expected, actual):
    left, right = deepcopy(expected), deepcopy(actual)
    a, b = left.pop("trees"), right.pop("trees")
    if left != right or len(a) != len(b):
        raise AssertionError("Tree model metadata/ensemble size differs")
    for ta, tb in zip(a, b):
        x, y = np.asarray(ta, float), np.asarray(tb, float)
        if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise AssertionError("Tree shape or finiteness differs")
        np.testing.assert_array_equal(x[:, :3], y[:, :3])  # topology/features: exact
        np.testing.assert_allclose(x[:, 3:], y[:, 3:], atol=ATOL, rtol=0)


def compare_tree_report(expected, actual):
    left, right = deepcopy(expected), deepcopy(actual)
    metrics = ["proxy_average_precision", "proxy_roc_auc",
               "catalogue_positive_capture_top_decile", "actual_selected_fraction"]
    pairs = [(left["spatial"]["oof_origin_discrimination"],
              right["spatial"]["oof_origin_discrimination"])]
    pairs += [(left["temporal"]["evaluation"][s], right["temporal"]["evaluation"][s])
              for s in ("validation", "test")]
    for a, b in pairs:
        for key in metrics:
            np.testing.assert_allclose(a[key], b[key], atol=ATOL, rtol=0)
            a[key] = b[key] = 0
    if left != right:
        raise AssertionError("Two-model report metadata differs")


def verify_export():
    tracked=subprocess.check_output(["git","-C",str(ROOT),"ls-files","web/data"],text=True).splitlines()
    actual={p.relative_to(ROOT).as_posix() for p in (ROOT/"web/data").iterdir() if p.is_file()}
    if actual!=set(tracked):
        raise AssertionError("Export file set differs from the committed reference")
    numerical=[];exact=[]
    for name in tracked:
        reference=subprocess.check_output(["git","-C",str(ROOT),"show",f"HEAD:{name}"])
        current=(ROOT/name).read_bytes()
        if reference==current:
            exact.append(name);continue
        base=Path(name).name
        if base in {"model_power.json","model_stations.json","model_map300.json"}:
            compare_model(json.loads(reference),json.loads(current))
            numerical.append(name)
        elif base in {"predictions_power.csv","predictions_stations.csv"}:
            compare_predictions(reference.decode(),current.decode())
            numerical.append(name)
        elif base == "presentation_examples.json":
            compare_examples(json.loads(reference),json.loads(current))
            numerical.append(name)
        elif base in {"model_rf.json", "model_boosting.json"}:
            compare_tree_model(json.loads(reference),json.loads(current))
            numerical.append(name)
        elif base == "two_model_report.json":
            compare_tree_report(json.loads(reference),json.loads(current))
            numerical.append(name)
        else:
            raise AssertionError(f"Strict data/report export differs: {name}")
    print(json.dumps({"exact_files":len(exact),"numeric_equivalent_files":numerical,
                      "absolute_tolerance":ATOL,"relative_tolerance":0,
                      "strict_fields":"All outcomes, dates, statuses, features, model metadata and reports"}))
    return 0
