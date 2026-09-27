"""Эксперименты для научной строгости: бейзлайны, абляция, разброс по seed, калибровка.

Отвечают на вопросы, которые задаёт жюри:
  * лучше ли ML простых бейзлайнов (логрег, только ID-порог)?
  * какой вклад дают отдельные группы признаков (сейсмика, озёра, осадки)?
  * устойчивы ли метрики к выбору случайного seed (mean ± std)?
  * откалиброваны ли предсказанные вероятности (reliability, Brier)?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import load_config
from .dataset import build_terrain_grid, generate_meteo, make_susceptibility_training
from .models.susceptibility import spatial_blocks, train_susceptibility
from .models.triggering import build_rainfall_events, fit_id_threshold, train_triggering

SEEDS = [42, 1, 7, 13, 21, 99, 2024, 77]


def _susc_logreg(train: pd.DataFrame) -> dict:
    feats = load_config()["models"]["susceptibility"]["features"]
    X, y = train[feats].to_numpy(), train["label"].to_numpy()
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced"))
    oof = cross_val_predict(clf, X, y, cv=GroupKFold(5), groups=spatial_blocks(train),
                            method="predict_proba", n_jobs=-1)[:, 1]
    return {"roc_auc": float(roc_auc_score(y, oof)),
            "ap": float(average_precision_score(y, oof))}


def _trig_logreg(meteo: pd.DataFrame, seed: int) -> dict:
    trg = load_config()["models"]["triggering"]
    feats = trg["features"]
    df = meteo.dropna(subset=feats + ["event"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    ts = pd.Timestamp(trg["test_start"])
    tr, te = df[df["date"] < ts], df[df["date"] >= ts]
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(tr[feats].to_numpy(), tr["event"].to_numpy())
    p = clf.predict_proba(te[feats].to_numpy())[:, 1]
    y = te["event"].to_numpy()
    return {"roc_auc": float(roc_auc_score(y, p)),
            "ap": float(average_precision_score(y, p))}


def run_all(n_seeds: int = 8) -> dict:
    cfg = load_config()
    seed0 = cfg["project"]["random_seed"]
    grid = build_terrain_grid()
    feats_s = cfg["models"]["susceptibility"]["features"]
    feats_t = cfg["models"]["triggering"]["features"]

    train0 = make_susceptibility_training(grid, seed0)
    meteo0 = generate_meteo(grid, seed0, save=False)

    # --- Бейзлайны и абляция (seed0) ---
    rf = train_susceptibility(train0, seed0)
    lr_s = _susc_logreg(train0)
    abl_s = train_susceptibility(train0, seed0,
                                 features=[f for f in feats_s if "lake" not in f])

    gbm = train_triggering(meteo0, seed0)                      # с важностями + калибровкой
    lr_t = _trig_logreg(meteo0, seed0)
    events = build_rainfall_events(meteo0)
    idt = fit_id_threshold(events)
    abl_eq = train_triggering(meteo0, seed0, compute_importance=False, fit_full=False,
                              features=[f for f in feats_t if not f.startswith("eq_")])
    abl_rain = train_triggering(meteo0, seed0, compute_importance=False, fit_full=False,
                                features=[f for f in feats_t
                                          if not f.startswith("rain") and f != "api"])

    # --- Разброс по seed ---
    rows = []
    for s in SEEDS[:n_seeds]:
        tr = make_susceptibility_training(grid, s)
        su = train_susceptibility(tr, s)
        mt = generate_meteo(grid, s, save=False)
        tg = train_triggering(mt, s, compute_importance=False, fit_full=False)
        rows.append({"seed": s, "susc_auc": su["cv_auc"], "susc_ap": su["cv_ap"],
                     "trig_auc": tg["test_auc"], "trig_ap": tg["test_ap"]})
    variance = pd.DataFrame(rows)

    # --- Калибровка (GBM, holdout) ---
    frac_pos, mean_pred = calibration_curve(gbm["test_y"], gbm["test_proba"],
                                            n_bins=8, strategy="quantile")
    brier = float(brier_score_loss(gbm["test_y"], gbm["test_proba"]))

    return {
        "baselines_susceptibility": {
            "RandomForest": {"roc_auc": rf["cv_auc"], "ap": rf["cv_ap"]},
            "LogReg": lr_s,
        },
        "baselines_triggering": {
            "GradientBoosting": {"roc_auc": gbm["test_auc"], "ap": gbm["test_ap"]},
            "LogReg": lr_t,
            "ID_threshold": {"recall": idt["recall"], "precision": idt["precision"]},
        },
        "ablation_susceptibility": {
            "full": {"roc_auc": rf["cv_auc"], "ap": rf["cv_ap"]},
            "no_glacial_lakes": {"roc_auc": abl_s["cv_auc"], "ap": abl_s["cv_ap"]},
        },
        "ablation_triggering": {
            "full": {"roc_auc": gbm["test_auc"], "ap": gbm["test_ap"]},
            "no_seismic": {"roc_auc": abl_eq["test_auc"], "ap": abl_eq["test_ap"]},
            "no_rainfall": {"roc_auc": abl_rain["test_auc"], "ap": abl_rain["test_ap"]},
        },
        "seed_variance": {
            "n_seeds": int(len(rows)),
            "susc_auc_mean": float(variance["susc_auc"].mean()),
            "susc_auc_std": float(variance["susc_auc"].std()),
            "susc_ap_mean": float(variance["susc_ap"].mean()),
            "susc_ap_std": float(variance["susc_ap"].std()),
            "trig_auc_mean": float(variance["trig_auc"].mean()),
            "trig_auc_std": float(variance["trig_auc"].std()),
            "trig_ap_mean": float(variance["trig_ap"].mean()),
            "trig_ap_std": float(variance["trig_ap"].std()),
            "table": variance.round(4).to_dict(orient="records"),
        },
        "calibration": {
            "brier_score": brier,
            "fraction_positive": frac_pos.tolist(),
            "mean_predicted": mean_pred.tolist(),
        },
        "_variance_df": variance,      # для графиков (не сериализуется в JSON)
        "_calib": (frac_pos, mean_pred, brier),
    }
