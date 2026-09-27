"""Область исследования (AOI): сетка ячеек и привязка к бассейнам."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config

# Средний радиус Земли, км — для перевода градусов в метры
_EARTH_R_KM = 6371.0


def aoi_bbox() -> tuple[float, float, float, float]:
    return tuple(load_config()["aoi"]["bbox"])  # type: ignore[return-value]


def aoi_center() -> tuple[float, float]:
    lon0, lat0, lon1, lat1 = aoi_bbox()
    return (lon0 + lon1) / 2.0, (lat0 + lat1) / 2.0


def basins() -> pd.DataFrame:
    return pd.DataFrame(load_config()["basins"])


def haversine_km(lon1, lat1, lon2, lat2):
    """Расстояние по большому кругу, км. Аргументы — скаляры или массивы."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * _EARTH_R_KM * np.arcsin(np.sqrt(a))


def make_grid() -> pd.DataFrame:
    """Регулярная сетка ячеек по AOI с координатами центров и id."""
    cfg = load_config()
    lon0, lat0, lon1, lat1 = cfg["aoi"]["bbox"]
    res = cfg["aoi"]["grid_res_deg"]

    lons = np.arange(lon0 + res / 2, lon1, res)
    lats = np.arange(lat0 + res / 2, lat1, res)
    grid_lon, grid_lat = np.meshgrid(lons, lats)

    df = pd.DataFrame(
        {
            "lon": grid_lon.ravel(),
            "lat": grid_lat.ravel(),
        }
    )
    df.insert(0, "cell_id", np.arange(len(df)))
    df["row"] = np.repeat(np.arange(len(lats)), len(lons))
    df["col"] = np.tile(np.arange(len(lons)), len(lats))
    df.attrs["nlon"] = len(lons)
    df.attrs["nlat"] = len(lats)
    df.attrs["res"] = res
    return df


def assign_basin(df: pd.DataFrame) -> pd.Series:
    """Назначает каждой ячейке ближайший по центроиду бассейн."""
    b = basins()
    d = np.stack(
        [haversine_km(df["lon"].values, df["lat"].values, r.lon, r.lat) for r in b.itertuples()],
        axis=1,
    )
    return pd.Series(b["name"].values[d.argmin(axis=1)], index=df.index, name="basin")
