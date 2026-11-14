import hashlib
import json
from copy import deepcopy
import joblib
import numpy as np
import pandas as pd
import pytest
from selrisk.realdata import ROOT, load_regional
from selrisk.research import train_research, score_rows, design
from selrisk.persistence import load_artifact
from selrisk.presentation import presentation_examples
from selrisk.reproducibility import compare_examples


@pytest.mark.parametrize("kind", ["power", "stations"])
def test_saved_pipeline_and_json_roundtrip(kind, tmp_path):
    train_research(kind, tmp_path)
    artifact = load_artifact(tmp_path / f"model_{kind}.json", kind)
    frame, features, _ = load_regional(kind)
    frame = design(frame)
    available, status = score_rows(artifact, frame)
    use = frame[status == "available"]
    pipeline = joblib.load(tmp_path / f"model_{kind}.joblib")  # own file only
    np.testing.assert_allclose(pipeline.decision_function(use[features]),
                               available[status == "available"], atol=2e-12, rtol=0)
    assert pipeline[0].n_samples_seen_ == int(
        (frame.split.eq("train") & frame.evaluation_eligible).sum())
    manifest = json.loads((tmp_path / f"training_{kind}.json").read_text())
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    assert manifest["contains_fitted_preprocessing"]
    assert not manifest["final_refit_on_all_data"]


@pytest.mark.parametrize("mutation", ["scale", "features", "nan", "kind"])
def test_saved_json_rejects_invalid_model(tmp_path, mutation):
    artifact = json.loads((ROOT / "web/data/model_power.json").read_text())
    if mutation == "scale":
        artifact["scale"][0] = 0
    elif mutation == "features":
        artifact["features"].reverse()
    elif mutation == "nan":
        artifact["coefficients"][0] = float("nan")
    else:
        artifact["dataset_kind"] = "stations"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError):
        load_artifact(path, "power")


def presentation_inputs():
    return (pd.read_csv(ROOT / "web/data/predictions_power.csv"),
            json.loads((ROOT / "web/data/model_power.json").read_text()),
            pd.read_csv(ROOT / "web/data/event_catalogue.csv"))


def test_cases_are_real_heldout_in_domain_and_validation_threshold_only():
    predictions, model, events = presentation_inputs()
    result = presentation_examples(predictions, model, events)
    assert [c["date"] for c in result["cases"]] == ["2013-06-06", "2015-07-05"]
    val = predictions[predictions.split.eq("validation")
                      & predictions.evaluation_eligible
                      & predictions.prediction_status.eq("available")]
    assert result["threshold"] == np.quantile(val.research_score, .9)
    for case in result["cases"]:
        row = predictions[predictions.date.eq(case["date"])].iloc[0]
        event = events[events.event_id.eq(case["event_id"])].iloc[0]
        assert row.split == "test" and row.target == 1 and row.evaluation_eligible
        assert row.prediction_status == "available"
        assert case["score"] >= result["threshold"]
        assert pd.Timestamp(case["feature_window_end_utc"]) < pd.Timestamp(case["date"], tz="UTC")
        assert event.date_start == event.date_end == case["date"]
        assert event.exact_date_eligible and event.dedup_keep and not event.date_year_conflict
        assert case["source_url"] == event.source_url
        assert int(case["source_fid"]) == int(event.source_fid)
    # Test scores cannot affect the numeric validation threshold.
    changed = predictions.copy()
    changed.loc[changed.split.eq("test"), "research_score"] += .01
    assert presentation_examples(changed, model, events)["threshold"] == result["threshold"]
    summary = result["test_summary"]
    assert summary["catalogue_positive_days"] == 33
    assert summary["positive_hits"] + summary["positive_not_selected"] == 33
    compare_examples(result, json.loads((ROOT / "web/data/presentation_examples.json").read_text()))


def test_no_fabricated_successes_if_missing_evidence():
    predictions, model, events = presentation_inputs()
    predictions.loc[predictions.split.eq("test"), "research_score"] = -1e6
    with pytest.raises(ValueError, match="Fewer than two"):
        presentation_examples(predictions, model, events)


def test_case_comparison_preserves_sources_and_summary():
    data = json.loads((ROOT / "web/data/presentation_examples.json").read_text())
    tiny = deepcopy(data)
    tiny["cases"][0]["score"] += 1e-12
    compare_examples(data, tiny)
    for key in ("event_id", "source_url"):
        bad = deepcopy(data)
        bad["cases"][0][key] = "fabricated"
        with pytest.raises(AssertionError):
            compare_examples(data, bad)
    bad = deepcopy(data)
    bad["test_summary"]["positive_hits"] += 1
    with pytest.raises(AssertionError):
        compare_examples(data, bad)


def test_saved_model_cli_never_fits(monkeypatch, capsys):
    import sys
    import selrisk.research as research
    from selrisk.cli import entry
    monkeypatch.setattr(research, "fit_model", lambda *a, **k: pytest.fail("Inference fitted a model"))
    monkeypatch.setattr(sys, "argv", ["predict.py", "--kind", "power", "--model",
                        str(ROOT / "web/data/model_power.json"), "--date", "2013-06-06"])
    entry("predict", lambda: pytest.fail("Legacy called"))
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "available" and result["probability"] is None
    assert result["score"] > 1.7
