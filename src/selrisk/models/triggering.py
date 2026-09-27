"""Триггерная модель (triggering): при каких условиях сход происходит.

Два подхода:
  1. Порог «интенсивность–длительность» осадков (Caine 1980; Guzzetti 2008) —
     классический интерпретируемый метод, нижняя огибающая триггерных дождей.
  2. Градиентный бустинг по гидрометеорологическим и сейсмическим признакам —
     суточная вероятность триггера. Оценка на временном holdout (обучение на
     ранних годах, тест на поздних) — без утечки во времени.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from ..config import load_config


# --- 1. Порог интенсивность–длительность -----------------------------------
def build_rainfall_events(meteo: pd.DataFrame, wet_min: float | None = None) -> pd.DataFrame:
    """Выделяет дождевые эпизоды (серии дождливых дней) и метит их триггерными."""
    if wet_min is None:
        wet_min = load_config()["models"]["id_threshold"]["wet_day_min_mm"]
    events = []
    for basin, g in meteo.groupby("basin"):
        g = g.sort_values("date").reset_index(drop=True)
        wet = (g["precip_mm"].to_numpy() >= wet_min)
        ev = g["event"].to_numpy()
        p = g["precip_mm"].to_numpy()
        dts = g["date"].to_numpy()
        n = len(g)
        i = 0
        while i < n:
            if wet[i]:
                j = i
                while j + 1 < n and wet[j + 1]:
                    j += 1
                E = float(p[i:j + 1].sum())
                D = j - i + 1
                triggered = int(ev[i:min(j + 2, n)].max())  # сход в эпизоде или на след. день
                events.append((basin, dts[i], E, D, E / D, triggered))
                i = j + 1
            else:
                i += 1
    return pd.DataFrame(events, columns=["basin", "start", "E_mm", "D_days", "I_mm_day", "triggered"])


def fit_id_threshold(events: pd.DataFrame, envelope_q: float = 0.10) -> dict:
    """Подбирает степенной порог I = a * D^(-b) как нижнюю огибающую триггеров."""
    trig = events[events["triggered"] == 1]
    x = np.log10(trig["D_days"].to_numpy())
    y = np.log10(trig["I_mm_day"].to_numpy())
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    c_thr = intercept + np.quantile(resid, envelope_q)
    a = 10 ** c_thr
    b = -slope

    D = events["D_days"].to_numpy()
    I = events["I_mm_day"].to_numpy()
    pred = (I >= a * D ** (-b)).astype(int)
    tp = int(((pred == 1) & (events["triggered"] == 1)).sum())
    fp = int(((pred == 1) & (events["triggered"] == 0)).sum())
    fn = int(((pred == 0) & (events["triggered"] == 1)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    return {"a": float(a), "b": float(b), "precision": precision, "recall": recall,
            "n_triggered": int(trig.shape[0])}


# --- 2. Вероятностная модель на временном holdout ---------------------------
def train_triggering(meteo: pd.DataFrame, seed: int | None = None,
                     features: list[str] | None = None,
                     compute_importance: bool = True, fit_full: bool = True) -> dict:
    """Обучает бустинг на временном holdout.

    features — подмножество признаков (для абляции). compute_importance/fit_full
    можно отключить для ускорения (например, при прогоне по многим seed).
    """
    cfg = load_config()
    trg = cfg["models"]["triggering"]
    seed = seed if seed is not None else cfg["project"]["random_seed"]
    feats = features if features is not None else trg["features"]

    df = meteo.dropna(subset=feats + ["event"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    test_start = pd.Timestamp(trg["test_start"])
    train = df[df["date"] < test_start]
    test = df[df["date"] >= test_start]

    Xtr, ytr = train[feats].to_numpy(), train["event"].to_numpy()
    Xte, yte = test[feats].to_numpy(), test["event"].to_numpy()

    model = HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=400, max_depth=4,
        l2_regularization=1.0, random_state=seed,
    )
    model.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))

    p_te = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p_te) if yte.sum() > 0 else float("nan")
    ap = average_precision_score(yte, p_te) if yte.sum() > 0 else float("nan")

    # порог по F1 на обучающем периоде
    p_tr = model.predict_proba(Xtr)[:, 1]
    prec, rec, thr = precision_recall_curve(ytr, p_tr)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    threshold = float(thr[int(np.argmax(f1[:-1]))]) if len(thr) else 0.5
    cm = confusion_matrix(yte, (p_te >= threshold).astype(int)).tolist()

    # важность признаков через перестановки (permutation importance) на holdout —
    # честно показывает вклад каждого признака, в т.ч. вторичность сейсмики
    importances = None
    if compute_importance:
        pi = permutation_importance(model, Xte, yte, n_repeats=15,
                                    random_state=seed, scoring="average_precision")
        importances = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False)

    # финальная модель на всех данных — для прогноза/демо
    model_all = None
    if fit_full:
        model_all = HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=400, max_depth=4,
            l2_regularization=1.0, random_state=seed,
        )
        model_all.fit(df[feats].to_numpy(), df["event"].to_numpy(),
                      sample_weight=compute_sample_weight("balanced", df["event"].to_numpy()))

    return {
        "model_holdout": model,
        "model_all": model_all,
        "features": feats,
        "test_auc": float(auc),
        "test_ap": float(ap),
        "threshold": threshold,
        "confusion": cm,
        "importances": importances,
        "test_dates": test["date"].to_numpy(),
        "test_proba": p_te,
        "test_y": yte,
        "n_events_total": int(df["event"].sum()),
        "n_events_test": int(yte.sum()),
    }


def predict_trigger_prob(model, meteo_feat: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict_proba(meteo_feat[features].to_numpy())[:, 1]
