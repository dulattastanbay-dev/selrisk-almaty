"""Экспорт результатов в ГИС-форматы (без GDAL): GeoJSON и ESRI ASCII Grid.

ESRI ASCII (.asc) читается QGIS/ArcGIS напрямую и одной командой gdal_translate
конвертируется в GeoTIFF. GeoJSON — точки-ячейки со значением и классом.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .aoi import aoi_bbox
from .config import load_config


def _grid_2d(grid: pd.DataFrame, values) -> np.ndarray:
    nlat = int(grid["row"].max()) + 1
    nlon = int(grid["col"].max()) + 1
    arr = np.full((nlat, nlon), np.nan)
    arr[grid["row"].to_numpy(), grid["col"].to_numpy()] = np.asarray(values)
    return arr


def to_esri_ascii(grid: pd.DataFrame, values, path) -> str:
    """Записывает регулярный растр в формате ESRI ASCII Grid (.asc)."""
    lon0, lat0, _, _ = aoi_bbox()
    res = load_config()["aoi"]["grid_res_deg"]
    arr = _grid_2d(grid, values)
    north_top = np.flipud(arr)                     # ESRI: первая строка — север
    nlat, nlon = north_top.shape
    lines = [
        f"ncols {nlon}", f"nrows {nlat}",
        f"xllcorner {lon0}", f"yllcorner {lat0}",
        f"cellsize {res}", "NODATA_value -9999",
    ]
    for r in north_top:
        lines.append(" ".join("-9999" if np.isnan(v) else f"{v:.4f}" for v in r))
    with open(path, "w", encoding="ascii") as fh:
        fh.write("\n".join(lines) + "\n")
    return str(path)


def to_geojson(grid: pd.DataFrame, value_col: str, path, extra_cols=("basin",)) -> str:
    """Записывает точки-ячейки со значением value_col в GeoJSON (EPSG:4326)."""
    feats = []
    for row in grid.itertuples():
        props = {value_col: round(float(getattr(row, value_col)), 4)}
        for c in extra_cols:
            if hasattr(row, c):
                props[c] = getattr(row, c)
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(row.lon, 5), round(row.lat, 5)]},
            "properties": props,
        })
    fc = {"type": "FeatureCollection",
          "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
          "features": feats}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(fc, fh, ensure_ascii=False)
    return str(path)


def build_map_payload(grid_susc: pd.DataFrame, meteo_pred: pd.DataFrame, dates) -> dict:
    """Компактная нагрузка для интерактивной карты Leaflet.

    cells: [lon, lat, susceptibility, basin_index]; triggers: {date: [tp по бассейнам]}.
    Риск считается на клиенте: susc × trigger[basin].
    """
    basins = sorted(grid_susc["basin"].unique())
    b_index = {b: i for i, b in enumerate(basins)}
    cells = [
        [round(r.lon, 5), round(r.lat, 5), round(float(r.susceptibility), 4), b_index[r.basin]]
        for r in grid_susc.itertuples()
    ]
    triggers = {}
    for d in dates:
        sub = meteo_pred[meteo_pred["date"] == pd.Timestamp(d)]
        tp = dict(zip(sub["basin"], sub["trigger_prob"]))
        triggers[str(pd.Timestamp(d).date())] = [round(float(tp.get(b, 0.0)), 4) for b in basins]
    return {
        "bbox": list(aoi_bbox()),
        "res": load_config()["aoi"]["grid_res_deg"],
        "basins": basins,
        "cells": cells,
        "triggers": triggers,
        "classes": load_config()["risk_classes"],
    }
