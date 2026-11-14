"""Presence-background ranking, NOT calibrated debris-flow probabilities.

Strict outcomes remain 1/NULL. A separate sampled-origin indicator distinguishes
catalogued days from background days; background is never called a true negative.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

from .realdata import ROOT, load_regional, validate_regional

OUTPUT = ROOT / "data/outputs/research"
C_GRID = (0.03, 0.3, 3.0)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n",encoding="utf-8")


def design(frame):
    """Deterministic background, split-contained embargo; no target mutation."""
    df = frame.copy()
    dates = pd.to_datetime(df.date)
    positive = df.target.eq(1)
    near = positive.rolling(15, center=True, min_periods=1).max().astype(bool)
    background = ~positive & ~near & ~df.date_uncertainty_overlap
    eligible = df.feature_complete.copy()
    # Remove the first 16 days of later splits: no overlapping 14-day weather windows.
    for name in ("validation", "test"):
        start = dates[df.split.eq(name)].min()
        eligible &= ~((df.split.eq(name)) & (dates < start+pd.Timedelta(days=16)))
    df["research_role"] = np.where(positive, "catalogue_positive",
                                  np.where(background, "unlabeled_background", "excluded_uncertain_or_near_event"))
    df["evaluation_eligible"] = eligible & (positive | background)
    # Deliberately not called target, event, absence or ground_truth.
    df["sample_origin"] = positive.astype(int)
    return df


def ranking_metrics(origin, scores):
    y=np.asarray(origin,int); score=np.asarray(scores,float)
    if len(y)==0 or len(set(y))<2 or not np.isfinite(score).all():
        raise ValueError("Ranking evaluation requires finite scores and both sample origins")
    # Ties at the selection boundary are all retained; actual budget is reported.
    boundary=float(np.quantile(score,0.9))
    selected=score>=boundary
    return {"n_days":len(y),"catalogue_positive_days":int(y.sum()),
        "unlabeled_background_days":int((y==0).sum()),
        "proxy_average_precision":float(average_precision_score(y,score)),
        "proxy_roc_auc":float(roc_auc_score(y,score)),
        "catalogue_positive_capture_top_decile":float(selected[y==1].mean()),
        "actual_selected_fraction":float(selected.mean()),
        "interpretation":"Origin-discrimination metrics only; background may contain unreported events."}


def fit_model(frame, features, C):
    train=frame[frame.split.eq("train") & frame.evaluation_eligible]
    if train.sample_origin.nunique()!=2:
        raise ValueError("Insufficient training sample origins")
    model=make_pipeline(StandardScaler(),LogisticRegression(C=C,class_weight="balanced",max_iter=4000,random_state=42))
    return model.fit(train[features],train.sample_origin)


def model_artifact(model, features, train, kind, C):
    sc,lr=model[0],model[-1]
    return dict(version="0.2",dataset_kind=kind,
        model_type="presence_background_logistic_ranking",
        output="uncalibrated_decision_score_not_event_probability",
        features=features,mean=sc.mean_.tolist(),scale=sc.scale_.tolist(),
        coefficients=lr.coef_[0].tolist(),intercept=float(lr.intercept_[0]),
        feature_min=train[features].min().tolist(),feature_max=train[features].max().tolist(),
        train_start=str(train.date.min()),train_end=str(train.date.max()),C=C,
        trained_rows=len(train),verified_negative_days=0,
        warning="No class-prior or registration-bias correction. No operational or safety interpretation.")


def score_rows(artifact, rows):
    """Reject missing, infinite and out-of-training-domain input. No clipping."""
    features=artifact["features"]
    matrix=rows[features].apply(pd.to_numeric,errors="raise").to_numpy(float)
    if np.isinf(matrix).any():
        raise ValueError("Infinite scenario input")
    finite=np.isfinite(matrix).all(axis=1)
    bounds=(matrix>=np.asarray(artifact["feature_min"])-1e-7)&(matrix<=np.asarray(artifact["feature_max"])+1e-7)
    domain=finite & bounds.all(axis=1)
    # Scores on historical held-out rows can be evaluated as extrapolation diagnostics.
    # The user-facing prediction API withholds them outside the train domain.
    scores=np.full(len(rows),np.nan)
    scores[domain]=((matrix[domain]-artifact["mean"])/artifact["scale"])@artifact["coefficients"]+artifact["intercept"]
    return scores, np.where(~finite,"missing_features",np.where(~domain,"out_of_training_domain","available"))


def lookup_date(frame, artifact, date):
    date=pd.Timestamp(date)
    if pd.isna(date) or date != date.normalize():
        raise ValueError("Expected one calendar day")
    row=frame[pd.to_datetime(frame.date).eq(date)]
    if row.empty:
        return {"date":date.strftime("%Y-%m-%d"),"status":"no_data","score":None,
                "ground_truth":"unknown","probability":None}
    scores,status=score_rows(artifact,row)
    return {"date":date.strftime("%Y-%m-%d"),"status":str(status[0]),
            "score":float(scores[0]) if np.isfinite(scores[0]) else None,
            "ground_truth":"catalogue_positive" if row.target.eq(1).iloc[0] else "unknown",
            "probability":None}


def train_research(kind="power", output=OUTPUT):
    frame,features,root=load_regional(kind)
    contract=validate_regional(frame,features,kind)
    d=design(frame)
    masks={s:d.split.eq(s)&d.evaluation_eligible for s in ["train","validation","test"]}
    trials=[]
    # Neither StandardScaler nor model fitting sees validation or test rows.
    for C in C_GRID:
        m=fit_model(d,features,C)
        v=d[masks["validation"]]
        trials.append((float(average_precision_score(v.sample_origin,m.decision_function(v[features]))),C,m))
    _,C,model=max(trials,key=lambda t:(t[0],-t[1]))
    artifact=model_artifact(model,features,d[masks["train"]],kind,C)
    scores=np.full(len(d),np.nan)
    complete=d.feature_complete.to_numpy(bool)
    scores[complete]=model.decision_function(d.loc[complete,features])
    baseline_features=[c for c in features if "prev3d" in c or "prev3d_lag2" in c]
    # A mean across available selected stations; all model features must still be complete.
    rain=d[baseline_features].mean(axis=1).to_numpy()
    seasonal_features=["day_of_year_sin","day_of_year_cos"]
    seasonal=fit_model(d,seasonal_features,0.3)
    season=np.full(len(d),np.nan)
    season[complete]=seasonal.decision_function(d.loc[complete,seasonal_features])
    metrics={}
    for name,score in [("presence_background",scores),("rainfall_3day",rain),("season_only",season)]:
        metrics[name]={s:ranking_metrics(d.loc[mask,"sample_origin"],score[mask.to_numpy()])
                       for s,mask in masks.items()}
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    write_json(output/f"model_{kind}.json",artifact)
    d["retrospective_raw_score"]=scores
    ui_score,status=score_rows(artifact,d)
    d["research_score"]=ui_score; d["prediction_status"]=status
    # Official outcomes are unchanged, including NULL.
    d.to_csv(output/f"predictions_{kind}.csv",index=False,float_format="%.10g")
    report=dict(version="0.2",kind=kind,status="research_ranking_only",contract=contract,
       dataset_sha256=hashlib.sha256((root/f"data/model_regional_{kind if kind=='power' else 'stations'}.csv").read_bytes()).hexdigest(),
       hyperparameter_selection="validation_origin_AP_only_test_never_used_for_selection",
       validation_trials=[{"C":c,"proxy_average_precision":ap} for ap,c,_ in trials],
       selected_C=C,preprocessing_fit_split="train_only",final_refit_on_all_data=False,
       exclusion_counts=d.research_role.value_counts().to_dict(),
       excluded_incomplete_positive_days=int((d.target.eq(1)&~d.feature_complete).sum()),
       domain_withheld_days=int((status!="available").sum()),
       metrics=metrics,
       limitations=["No verified negatives; no precision, specificity, false-alarm or calibrated event-probability claim.",
          "Proxy AP/AUC measure catalogue-positive versus chosen background discrimination, not true event accuracy.",
          "Unknown registration bias; SCAR and the event class prior are not established.",
          "Grid weather does not resolve local convective rainfall in mountain gullies.",
          "All mechanisms are pooled; rainfall-only associations do not validate a GLOF model.",
          "Retrospective reanalysis latency has not been validated for live deployment.",
          "Warm-season and temporal dependence may inflate proxy separation; compare season-only baseline.",
          "Out-of-domain historical scores are included only in retrospective diagnostics; UI withholds them."],
       references=["https://www.nature.com/articles/s44304-024-00050-7",
          "https://link.springer.com/article/10.1007/s10994-020-05877-5",
          "https://scikit-learn.org/stable/common_pitfalls.html"])
    write_json(output/f"report_{kind}.json",report)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind",choices=["power","stations","all"],default="all")
    p.add_argument("--output",type=Path,default=OUTPUT)
    args=p.parse_args(argv)
    for kind in (["power","stations"] if args.kind=="all" else [args.kind]):
        report=train_research(kind,args.output)
        print(json.dumps({"kind":kind,"status":report["status"],"contract":report["contract"],
                          "output":str(args.output)},ensure_ascii=False))
    return 0
