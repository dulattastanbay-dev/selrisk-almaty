"""Слияние: итоговый риск = подверженность (где) × вероятность триггера (когда).

Подверженность задаётся по ячейкам, вероятность триггера — по бассейну и дате.
Для каждой ячейки риск в заданную дату = susc[cell] × trigger[basin(cell), date].
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
    return names[np.clip(idx, 0, len(names) - 1)]


def risk_color_map() -> dict[str, str]:
    return {c["name"]: c["color"] for c in load_config()["risk_classes"]}


def trigger_prob_on_date(meteo_pred: pd.DataFrame, date) -> dict[str, float]:
    """Словарь {бассейн: вероятность триггера} на заданную дату."""
    d = meteo_pred[meteo_pred["date"] == pd.Timestamp(date)]
    return dict(zip(d["basin"], d["trigger_prob"]))


def daily_risk(grid_susc: pd.DataFrame, meteo_pred: pd.DataFrame, date) -> pd.DataFrame:
    """Карта риска по ячейкам на дату. grid_susc содержит basin и susceptibility."""
    tp = trigger_prob_on_date(meteo_pred, date)
    trig = grid_susc["basin"].map(tp).fillna(0.0).to_numpy()
    risk = grid_susc["susceptibility"].to_numpy() * trig
    out = grid_susc.copy()
    out["trigger_prob"] = trig
    out["risk"] = risk
    out["risk_class"] = classify_risk(risk)
    return out
