"""Загрузка реального каталога сходов из NASA COOLR / Global Landslide Catalog.

Источник: Cooperative Open Online Landslide Repository (COOLR), NASA GPM.
ArcGIS REST: https://maps.nccs.nasa.gov/mapping/rest/services/COOLR/COOLR_Events_Point

Замечание о покрытии: GLC собирается преимущественно из англоязычных СМИ, поэтому
по Центральной Азии он разрежен — для Алматы это, как правило, единицы событий.
Модуль реализует настоящую интеграцию (запрос, пагинация, пространственный фильтр,
приведение к нашей схеме) и корректно откатывается к курируемому CSV при
недоступности сервиса.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from ..aoi import assign_basin
from ..config import load_config

_LAYER = ("https://maps.nccs.nasa.gov/mapping/rest/services/COOLR/"
          "COOLR_Events_Point/MapServer/0/query")


def _get(params: dict, timeout: int = 120) -> dict:
    url = _LAYER + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def fetch_coolr(bbox=None, retries: int = 3, page: int = 500) -> pd.DataFrame:
    """Возвращает события COOLR в пределах bbox (lon_min,lat_min,lon_max,lat_max).

    Колонки: date, lon, lat, trigger_type, severity, fatalities, title, source.
    Бросает RuntimeError, если сервис недоступен после повторов.
    """
    cfg = load_config()
    if bbox is None:
        bbox = cfg["aoi"]["bbox"]
    lon0, lat0, lon1, lat1 = bbox
    base = {
        "where": "1=1",
        "geometry": f"{lon0},{lat0},{lon1},{lat1}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326", "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": ("ev_date,event_date,landslide_trigger,landslide_category,"
                      "landslide_size,fatality_count,event_title,source_name"),
        "returnGeometry": "true",
        "f": "json",
    }

    rows = []
    offset = 0
    while True:
        params = dict(base, resultOffset=offset, resultRecordCount=page)
        last_err = None
        for attempt in range(retries):
            try:
                data = _get(params)
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(2 * (attempt + 1))
        if last_err is not None:
            raise RuntimeError(f"COOLR недоступен: {last_err}")

        feats = data.get("features", [])
        for ft in feats:
            a = {k.lower(): v for k, v in ft.get("attributes", {}).items()}
            g = ft.get("geometry") or {}
            ts = a.get("ev_date") or a.get("event_date")
            date = (pd.to_datetime(ts, unit="ms", errors="coerce")
                    if isinstance(ts, (int, float)) else pd.to_datetime(ts, errors="coerce"))
            rows.append({
                "date": date,
                "lon": g.get("x"), "lat": g.get("y"),
                "trigger_type": a.get("landslide_trigger"),
                "severity": a.get("landslide_size"),
                "fatalities": a.get("fatality_count"),
                "title": a.get("event_title"),
                "source": a.get("source_name") or "NASA COOLR/GLC",
            })
        if len(feats) < page:
            break
        offset += page

    df = pd.DataFrame(rows).dropna(subset=["lon", "lat"])
    return df.reset_index(drop=True)


def coolr_to_events_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит выгрузку COOLR к схеме data/events/historical_events.csv."""
    if df.empty:
        return df
    inside = df.copy()
    inside["basin"] = assign_basin(inside)
    trig_map = {"downpour": "rain", "rain": "rain", "continuous_rain": "rain",
                "snowfall_snowmelt": "glof", "flooding": "rain"}
    return pd.DataFrame({
        "date": pd.to_datetime(inside["date"]).dt.date,
        "basin": inside["basin"],
        "trigger_type": inside["trigger_type"].map(lambda t: trig_map.get(str(t).lower(), "unknown")),
        "severity": inside["severity"].fillna("unknown"),
        "source_lon": inside["lon"].round(4),
        "source_lat": inside["lat"].round(4),
        "coords_approx": False,
        "confidence": "coolr",
        "notes": inside["title"].fillna("") + " [NASA COOLR/GLC]",
    }).dropna(subset=["date"])
