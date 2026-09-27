"""Инженерия признаков триггерной (временной) модели.

Функция add_antecedent_features применяется одинаково при обучении и при
прогнозе, поэтому переход на реальные данные ERA5/USGS не требует изменений.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config


def antecedent_precip_index(precip: np.ndarray, decay: float) -> np.ndarray:
    """API_t = decay * API_{t-1} + precip_t (рекурсивный индекс увлажнения)."""
    api = np.zeros_like(precip, dtype=float)
    acc = 0.0
    for i, p in enumerate(precip):
        acc = decay * acc + p
        api[i] = acc
    return api


def add_antecedent_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет накопленные осадки, индекс увлажнения, градусо-дни и сезонность.

    Ожидает колонки: date, basin, precip_mm, tmean_c, а также (посуточно, общие
    для AOI) eq_max_mag_3d, eq_energy_7d. Сортирует по basin+date.
    """
    cfg = load_config()["models"]["triggering"]
    df = df.sort_values(["basin", "date"]).reset_index(drop=True)

    precip = df.groupby("basin")["precip_mm"]
    df["rain_1d"] = df["precip_mm"]
    df["rain_3d"] = precip.transform(lambda s: s.rolling(3, min_periods=1).sum())
    df["rain_7d"] = precip.transform(lambda s: s.rolling(7, min_periods=1).sum())

    # индекс увлажнения (рекурсивный) — считаем по каждому бассейну отдельно
    api = np.zeros(len(df))
    p_all = df["precip_mm"].to_numpy()
    for _, pos in df.groupby("basin").indices.items():
        api[pos] = antecedent_precip_index(p_all[pos], cfg["api_decay"])
    df["api"] = api

    df["_pdd"] = np.maximum(df["tmean_c"] - cfg["melt_base_temp_c"], 0.0)
    df["pdd_3d"] = df.groupby("basin")["_pdd"].transform(
        lambda s: s.rolling(3, min_periods=1).sum()
    )
    df = df.drop(columns="_pdd")

    doy = df["date"].dt.dayofyear.to_numpy()
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df.reset_index(drop=True)
