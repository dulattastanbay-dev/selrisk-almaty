"""Экспорт модели и данных для веб-калькулятора (папка web/).

    python scripts/export_web.py        # после run_all.py

Сайт показывает откалиброванную вероятность триггера — логистическую регрессию
из selrisk.models.triggering.train_calibrated_trigger (та же модель, что в
Streamlit-демо). Она гладкая, монотонная и точно переносится в JavaScript
(коэффициенты + нормировка); параметры и уровни — в config.yaml
(models.calibrated_trigger).

Пишет web/data/selrisk_data.js  (window.SELRISK_DATA = {...}).
"""
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
for _s in (sys.stdout, sys.stderr):          # корректный вывод кириллицы в Windows-консоли
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd

from selrisk.aoi import aoi_center, basins, haversine_km
from selrisk.config import PROJECT_ROOT, load_config
from selrisk.data_ingest.earthquakes import energy_from_magnitude
from selrisk.models.triggering import predict_calibrated_trigger, train_calibrated_trigger

ERA5_RES = 0.1         # шаг сетки ERA5-Land, градусы


def era5_points() -> dict[str, list[list[float]]]:
    """Узлы сетки ERA5-Land в AOI, назначенные бассейнам так же, как в era5.py."""
    lon0, lat0, lon1, lat1 = load_config()["aoi"]["bbox"]
    lats = np.round(np.arange(np.ceil(lat0 / ERA5_RES - 1e-6) * ERA5_RES, lat1 + 1e-6, ERA5_RES), 1)
    lons = np.round(np.arange(np.ceil(lon0 / ERA5_RES - 1e-6) * ERA5_RES, lon1 + 1e-6, ERA5_RES), 1)
    LON, LAT = np.meshgrid(lons, lats)
    b = basins()
    dist = np.stack([haversine_km(LON, LAT, r.lon, r.lat) for r in b.itertuples()])
    idx = dist.argmin(axis=0)
    return {r.name: [[float(la), float(lo)] for la, lo in zip(LAT[idx == i], LON[idx == i])]
            for i, r in enumerate(b.itertuples())}


def main() -> None:
    cfg = load_config()
    trg = cfg["models"]["triggering"]
    cal_cfg = cfg["models"]["calibrated_trigger"]
    odir = cfg.path("outputs")
    meteo = pd.read_csv(odir / "meteo_with_trigger.csv", parse_dates=["date"])
    grid = pd.read_csv(odir / "grid_with_susceptibility.csv")
    names = list(basins()["name"])

    # --- откалиброванная модель: holdout 2020–2023 + финальная на всём периоде ---
    cal = train_calibrated_trigger(meteo)
    metrics = cal["metrics"]
    print("[export_web] holdout 2020–2023: " + ", ".join(f"{k}={v:.4g}" for k, v in metrics.items()))
    sc, lr = cal["model"][0], cal["model"][-1]
    p_all = predict_calibrated_trigger(cal, meteo)
    meteo["p_web"] = p_all
    levels = cal_cfg["levels"]
    share = np.bincount(np.digitize(p_all, [lv["max"] for lv in levels][:-1]),
                        minlength=len(levels)) / len(p_all)
    print("[export_web] доля дней по уровням: "
          + ", ".join(f"{lv['name']}={s:.1%}" for lv, s in zip(levels, share)))

    model = {
        "features": cal["features"],
        "raw_features": cal["raw_features"],
        "basins": names,
        "mean": sc.mean_.tolist(),
        "scale": sc.scale_.tolist(),
        "coef": lr.coef_[0].tolist(),
        "intercept": float(lr.intercept_[0]),
        "clip_min": np.round(cal["clip_min"], 4).tolist(),
        "clip_max": np.round(cal["clip_max"], 4).tolist(),
        "api_decay": trg["api_decay"],
        "melt_base_temp_c": trg["melt_base_temp_c"],
        "base_rate": cal["base_rate"],
        "n_events": int(meteo["event"].sum()),
        "n_days": int(meteo["date"].nunique()),
        "period": [str(meteo["date"].min().date()), str(meteo["date"].max().date())],
        "levels": levels,
        "level_share": share.round(4).tolist(),
        "C": cal_cfg["C"],
        "metrics": metrics,
    }

    # --- сетка подверженности (индекс = row * nlon + col) ---
    nlon, nlat = int(grid["col"].max()) + 1, int(grid["row"].max()) + 1
    g = grid.sort_values(["row", "col"])
    assert len(g) == nlon * nlat
    bidx = g["basin"].map({b: i for i, b in enumerate(names)}).astype(int)
    lon0, lat0, lon1, lat1 = cfg["aoi"]["bbox"]
    grid_out = {
        "lon0": lon0, "lat0": lat0, "res": cfg["aoi"]["grid_res_deg"], "nlon": nlon, "nlat": nlat,
        "susc": (g["susceptibility"].to_numpy() * 1000).round().astype(int).tolist(),
        "basin": "".join(map(str, bidx.tolist())),
    }

    # --- архив 2016–2023: исходные суточные ряды (признаки считает JS) ---
    start = meteo["date"].min()
    arch = {"start": str(start.date()), "n": int(meteo["date"].nunique()), "basins": {}}
    for b in names:
        s = meteo[meteo["basin"] == b].sort_values("date")
        arch["basins"][b] = {
            "precip": s["precip_mm"].round(2).tolist(),
            "tmean": s["tmean_c"].round(2).tolist(),
            "swe": s["swe_mm"].round(2).tolist(),
            "events": [int(i) for i in np.flatnonzero(s["event"].to_numpy())],
            "p_py": s["p_web"].round(6).tolist(),           # для проверки паритета JS
        }
    eq_path = cfg.path("raw") / "earthquakes_usgs.csv"
    eq_days = {}
    if eq_path.exists():
        cat = pd.read_csv(eq_path)
        t = (pd.to_datetime(cat["time"], utc=True, format="ISO8601")
             .dt.tz_localize(None).dt.normalize())
        for day, grp in cat.assign(day=t).groupby("day"):
            k = (day - start).days
            if 0 <= k < arch["n"]:
                eq_days[k] = [float(grp["mag"].max()),
                              float(energy_from_magnitude(grp["mag"]).sum())]
    arch["eq"] = eq_days

    # климатология для ручного режима: средние осадки, мм/сут, по месяцам
    clim = meteo.groupby(meteo["date"].dt.month)["precip_mm"].mean().round(2)

    lakes = pd.read_csv(cfg.path("events") / "glacial_lakes.csv")
    hist = pd.read_csv(cfg.path("events") / "historical_events.csv")
    top = (meteo.sort_values("p_web", ascending=False)
           .drop_duplicates("date").head(6)[["date", "basin", "p_web", "event"]])
    susc_metrics = json.loads((odir / "metrics.json").read_text(encoding="utf-8"))["susceptibility"]
    lon_c, lat_c = aoi_center()
    pts = era5_points()
    print("[export_web] узлы ERA5 по бассейнам: " + ", ".join(f"{b}={len(p)}" for b, p in pts.items()))

    payload = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "model": model,
        "susc_metrics": {"cv_roc_auc": susc_metrics["cv_roc_auc"],
                         "cv_ap": susc_metrics["cv_average_precision"]},
        "grid": grid_out,
        "basins": [{"name": r.name, "lon": r.lon, "lat": r.lat, "points": pts[r.name]}
                   for r in basins().itertuples()],
        "aoi": {"bbox": cfg["aoi"]["bbox"], "center": [lat_c, lon_c],
                "eq_radius_km": cfg["data_sources"]["earthquakes"]["radius_km"],
                "eq_min_mag": cfg["data_sources"]["earthquakes"]["min_magnitude"]},
        "archive": arch,
        "top_days": [{"date": str(r.date.date()), "basin": r.basin, "p": round(float(r.p_web), 4),
                      "event": int(r.event)} for r in top.itertuples()],
        "climatology_mm_day": {int(k): float(v) for k, v in clim.items()},
        "lakes": lakes[["name", "lon", "lat", "area_km2", "basin"]].to_dict(orient="records"),
        "history": hist[["date", "basin", "trigger_type", "severity", "source_lon", "source_lat", "notes"]]
                   .to_dict(orient="records"),
    }

    out = PROJECT_ROOT / "web" / "data" / "selrisk_data.js"
    out.parent.mkdir(parents=True, exist_ok=True)
    js = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    out.write_text("// Сгенерировано scripts/export_web.py — не редактировать вручную.\n"
                   f"window.SELRISK_DATA = {js};\n", encoding="utf-8")
    print(f"[export_web] -> {out} ({out.stat().st_size / 1024:.0f} КБ)")


if __name__ == "__main__":
    main()
