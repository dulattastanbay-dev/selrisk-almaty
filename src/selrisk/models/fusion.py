"""Legacy DEMO product index. Not a validated probability or expected loss.

Kept for compatibility and missing-data regression tests; not used in v0.2 UI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config


def classify_risk(prob: np.ndarray) -> np.ndarray:
    """Класс риска по порогам из config.yaml (низкий/умеренный/высокий/экстремальный)."""
    classes = load_config()["risk_classes"]
    names = np.array([c["name"] for c in classes])
    uppers = [c["max"] for c in classes][:-1]
    idx = np.digitize(np.asarray(prob), uppers)
    result = names[np.clip(idx, 0, len(names) - 1)].astype(object)
    p = np.asarray(prob, dtype=float)
    return np.where(np.isfinite(p) & (p>=0) & (p<=1), result, "нет данных")


def risk_color_map() -> dict[str, str]:
    return {c["name"]: c["color"] for c in load_config()["risk_classes"]}


def trigger_level(prob) -> np.ndarray:
    """Уровень опасности по откалиброванной суточной вероятности схода в бассейне
    (пороги models.calibrated_trigger.levels: 1 / 5 / 20 %)."""
    levels = load_config()["models"]["calibrated_trigger"]["levels"]
    names = np.array([lv["name"] for lv in levels])
    idx = np.digitize(np.asarray(prob), [lv["max"] for lv in levels][:-1])
    result = names[np.clip(idx, 0, len(names) - 1)].astype(object)
    p = np.asarray(prob, dtype=float)
    return np.where(np.isfinite(p) & (p>=0) & (p<=1), result, "нет данных")


def trigger_prob_on_date(meteo_pred: pd.DataFrame, date) -> dict[str, float]:
    """Словарь {бассейн: вероятность триггера} на заданную дату."""
    d = meteo_pred[pd.to_datetime(meteo_pred["date"]) == pd.Timestamp(date)]
    if d.basin.duplicated().any():
        raise ValueError("Duplicate basin/date predictions")
    return dict(zip(d["basin"], d["trigger_prob"]))


def daily_risk(grid_susc: pd.DataFrame, meteo_pred: pd.DataFrame, date) -> pd.DataFrame:
    """Карта риска по ячейкам на дату. grid_susc содержит basin и susceptibility."""
    tp = trigger_prob_on_date(meteo_pred, date)
    trig = grid_susc["basin"].map(tp).to_numpy(dtype=float)
    risk = grid_susc["susceptibility"].to_numpy() * trig
    out = grid_susc.copy()
    out["trigger_prob"] = trig
    out["risk"] = risk
    out["risk_class"] = classify_risk(risk)
    return out
