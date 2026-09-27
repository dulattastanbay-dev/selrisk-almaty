"""Загрузка каталога землетрясений из USGS FDSN Event API (открыт, без ключа).

Пример реального запроса — работает при наличии интернета:
    df = fetch_earthquakes()               # весь период из config.yaml
    df = fetch_earthquakes("2015-01-01", "2015-12-31")
"""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import requests

from ..aoi import aoi_center
from ..config import load_config

_TIMEOUT = 60


def fetch_earthquakes(start: str | None = None, end: str | None = None,
                      save: bool = True) -> pd.DataFrame:
    """Возвращает каталог землетрясений вокруг AOI за период.

    Радиус, минимальная магнитуда и период по умолчанию берутся из config.yaml.
    Результат: колонки time (datetime, UTC), lon, lat, depth_km, mag.
    """
    cfg = load_config()
    src = cfg["data_sources"]["earthquakes"]
    lon0, lat0 = aoi_center()
    params = {
        "format": "csv",
        "starttime": start or src["start"],
        "endtime": end or src["end"],
        "latitude": lat0,
        "longitude": lon0,
        "maxradiuskm": src["radius_km"],
        "minmagnitude": src["min_magnitude"],
        "orderby": "time-asc",
    }
    resp = requests.get(src["url"], params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    raw = pd.read_csv(io.StringIO(resp.text))

    df = pd.DataFrame(
        {
            "time": pd.to_datetime(raw["time"], utc=True),
            "lon": raw["longitude"].astype(float),
            "lat": raw["latitude"].astype(float),
            "depth_km": raw["depth"].astype(float),
            "mag": raw["mag"].astype(float),
        }
    ).dropna(subset=["mag"]).sort_values("time").reset_index(drop=True)

    if save:
        out = cfg.path("raw") / "earthquakes_usgs.csv"
        df.to_csv(out, index=False)
        print(f"[earthquakes] {len(df)} событий сохранено -> {out}")
    return df


def energy_from_magnitude(mag) -> np.ndarray:
    """Сейсмическая энергия (условные единицы) из магнитуды: log10(E) = 1.5*M + 4.8.

    Возвращается нормированная величина 10**(1.5*M) — для суммирования энергии
    нескольких толчков в скользящем окне.
    """
    return np.power(10.0, 1.5 * np.asarray(mag, dtype=float))


if __name__ == "__main__":
    try:
        fetch_earthquakes()
    except Exception as exc:  # noqa: BLE001
        print(f"Не удалось загрузить каталог USGS ({exc}). "
              "Проверьте интернет или используйте сгенерированную выборку "
              "(scripts/make_sample_data.py).")
