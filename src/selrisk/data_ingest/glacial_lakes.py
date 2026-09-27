"""Инвентаризация ледниковых (моренных) озёр и признаки близости к ним.

Прорывы моренно-подпрудных озёр (GLOF) — ведущий механизм селей Алматы,
поэтому близость и объём вышележащих озёр важнее сейсмики. Координаты в
seed-инвентаре ПРИБЛИЗИТЕЛЬНЫЕ и служат шаблоном: замените их данными
GLIMS / ICIMOD / собственной инвентаризации по спутниковым снимкам.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..aoi import haversine_km
from ..config import load_config


def load_lakes() -> pd.DataFrame:
    """Читает data/events/glacial_lakes.csv (создаётся seed-функцией при отсутствии)."""
    cfg = load_config()
    path = cfg.path("events") / "glacial_lakes.csv"
    if not path.exists():
        write_seed_inventory()
    return pd.read_csv(path)


def write_seed_inventory() -> None:
    """Записывает стартовую (приблизительную) инвентаризацию озёр."""
    cfg = load_config()
    path = cfg.path("events") / "glacial_lakes.csv"
    lakes = pd.DataFrame(
        [
            # name, lon, lat, elev_m, area_km2, basin, coords_approx
            ("оз. № 6 (М. Алматинка)", 77.08, 43.055, 3400, 0.030, "Малая Алматинка", True),
            ("оз. Маншук Маметовой",   77.10, 43.040, 3560, 0.018, "Малая Алматинка", True),
            ("Большое Алматинское",    76.985, 43.045, 2510, 0.380, "Большая Алматинка", True),
            ("оз. в верховьях БАО",    76.96, 43.020, 3510, 0.040, "Большая Алматинка", True),
            ("оз. Жарсай (тип Иссык)", 77.28, 43.060, 3300, 0.055, "Талгар", True),
            ("моренное оз. Талгар-1",  77.26, 43.035, 3620, 0.025, "Талгар", True),
        ],
        columns=["name", "lon", "lat", "elev_m", "area_km2", "basin", "coords_approx"],
    )
    lakes.to_csv(path, index=False)
    print(f"[glacial_lakes] seed-инвентаризация ({len(lakes)} озёр) -> {path}")


def lake_features(cells: pd.DataFrame, lakes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Для каждой ячейки: расстояние до ближайшего озера и площадь вышележащих озёр.

    cells: DataFrame с колонками lon, lat, elevation.
    Возвращает DataFrame с dist_lake_m и upstream_lake_area_km2 (индекс как у cells).
    """
    if lakes is None:
        lakes = load_lakes()

    lon = cells["lon"].values[:, None]
    lat = cells["lat"].values[:, None]
    elev = cells["elevation"].values[:, None]
    lk_lon = lakes["lon"].values[None, :]
    lk_lat = lakes["lat"].values[None, :]
    lk_elev = lakes["elev_m"].values[None, :]
    lk_area = lakes["area_km2"].values[None, :]

    dist_km = haversine_km(lon, lat, lk_lon, lk_lat)          # (n_cells, n_lakes)
    dist_lake_m = dist_km.min(axis=1) * 1000.0

    # "вышележащее" озеро: выше ячейки и в пределах 8 км
    upstream = (lk_elev > elev) & (dist_km < 8.0)
    upstream_area = np.where(upstream, lk_area, 0.0).sum(axis=1)

    return pd.DataFrame(
        {"dist_lake_m": dist_lake_m, "upstream_lake_area_km2": upstream_area},
        index=cells.index,
    )
