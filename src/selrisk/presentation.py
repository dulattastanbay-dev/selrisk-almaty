"""Transparent retrospective case selection, not prospective warning validation."""
from __future__ import annotations
import numpy as np
import pandas as pd


def presentation_examples(predictions, model, events):
    available = predictions.prediction_status.eq("available")
    eligible = predictions.evaluation_eligible
    validation = predictions[eligible & available & predictions.split.eq("validation")]
    if validation.empty:
        raise ValueError("No validation days for presentation threshold")
    threshold = float(np.quantile(validation.research_score, 0.9))
    test = predictions[eligible & predictions.split.eq("test")]
    selected = test.prediction_status.eq("available") & test.research_score.ge(threshold)
    positives = test.target.eq(1)
    hits = test[selected & positives].sort_values("date")
    # Date-only chronological choice, one case per year; never maximize test scores.
    cases = []
    years = set()
    for row in hits.to_dict("records"):
        year = row["date"][:4]
        exact = events[
            events.date_start.eq(row["date"]) & events.date_end.eq(row["date"])
            & events.exact_date_eligible.eq(True) & events.dedup_keep.eq(True)
            & ~events.date_year_conflict.eq(True)]
        if year in years or exact.empty:
            continue
        source = exact.sort_values("event_id").iloc[0]
        cases.append({
            "date": row["date"], "split": "test", "score": float(row["research_score"]),
            "threshold": threshold, "threshold_exceeded": True,
            "feature_window_end_utc": row["feature_window_end_utc"],
            "event_id": source.event_id, "basin": source.basin,
            "center_name": source.center_name, "event_type": source.event_type,
            "source_url": source.source_url, "source_fid": int(source.source_fid),
            "source_date": source.source_date,
            "features": {key: float(row[key]) for key in model["features"]},
        })
        years.add(year)
        if len(cases) == 2:
            break
    if len(cases) != 2:
        raise ValueError("Fewer than two supported held-out cases; do not fabricate examples")
    return {
        "version": "0.2", "dataset_kind": "power",
        "interpretation": "Retrospective threshold hits, NOT forecasts issued before the events.",
        "selection": "First chronological test hits with exact catalogue dates, one per year; selected after test inspection.",
        "threshold_rule": "90th percentile of eligible in-domain validation scores, inclusive >=; no test threshold tuning.",
        "threshold": threshold, "validation_threshold_days": len(validation),
        "test_summary": {
            "eligible_days": len(test), "catalogue_positive_days": int(positives.sum()),
            "positive_hits": int((selected & positives).sum()),
            "positive_not_selected": int((~selected & positives).sum()),
            "positive_withheld": int((positives & ~test.prediction_status.eq("available")).sum()),
            "selected_days": int(selected.sum()),
            "selected_unknown_days": int((selected & ~positives).sum()),
            "unavailable_days": int((~test.prediction_status.eq("available")).sum()),
        }, "cases": cases,
        "limitations": "Selected unknown days are NOT proven false alarms. Two selected hits do not measure accuracy; reanalysis availability and catalogue bias remain unvalidated.",
    }
