"""ERA5-Land — суточные метеоряды через Copernicus Climate Data Store (CDS).

Загрузка требует бесплатной регистрации на https://cds.climate.copernicus.eu
и файла ~/.cdsapirc с ключом, а также пакетов `cdsapi` и `xarray`+`netCDF4`
(не входят в базовые зависимости проекта — ставятся отдельно).

Схема на выходе (data/processed/meteo_daily.csv), которую ожидают модели:
    date, basin, precip_mm, tmin_c, tmax_c, tmean_c, swe_mm

Без ключа CDS используйте сгенерированную выборку того же формата
(scripts/make_sample_data.py) — весь остальной пайплайн идентичен.
"""
from __future__ import annotations

from pathlib import Path

from ..aoi import aoi_bbox
from ..config import load_config


def build_cds_request(year: int, month: int) -> dict:
    """Запрос к reanalysis-era5-land за ОДИН месяц (почасово, 3 переменные).

    Помесячно — потому что запрос за целый год почасово превышает лимит CDS
    на размер (cost limits exceeded).
    """
    from calendar import monthrange

    lon0, lat0, lon1, lat1 = aoi_bbox()
    ndays = monthrange(year, month)[1]
    return {
        "variable": [
            "2m_temperature",
            "total_precipitation",
            "snow_depth_water_equivalent",
        ],
        "year": str(year),
        "month": f"{month:02d}",
        "day": [f"{d:02d}" for d in range(1, ndays + 1)],
        "time": [f"{h:02d}:00" for h in range(24)],
        # область: [север, запад, юг, восток]
        "area": [lat1, lon0, lat0, lon1],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def fetch_era5(years: range | None = None, months: range | None = None) -> Path:
    """Скачивает почасовой ERA5-Land ПОМЕСЯЧНО в data/raw (era5_land_YYYY_MM.nc).

    Возобновляемо: уже скачанные месяцы пропускаются. Требует cdsapi и ключа CDS.
    `months` ограничивает месяцы (напр. range(4,10) — только сель-сезон апрель–сентябрь).
    """
    try:
        import cdsapi  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Пакет cdsapi не установлен. Установите: pip install cdsapi "
            "и настройте ключ CDS (~/.cdsapirc). Либо используйте "
            "сгенерированную выборку scripts/make_sample_data.py."
        ) from exc

    import cdsapi

    cfg = load_config()
    src = cfg["data_sources"]["weather"]
    if years is None:
        years = range(int(src["start"][:4]), int(src["end"][:4]) + 1)
    if months is None:
        months = range(1, 13)

    client = cdsapi.Client()
    raw_dir = cfg.path("raw")
    for year in years:
        for month in months:
            target = raw_dir / f"era5_land_{year}_{month:02d}.nc"
            if target.exists():
                continue
            print(f"[era5] загрузка {year}-{month:02d} ...")
            client.retrieve(src["dataset"], build_cds_request(year, month), str(target))
    return raw_dir


def aggregate_to_daily(save: bool = True):
    """Агрегация почасового ERA5-Land до суточного ряда по бассейнам.

    Читает data/raw/era5_land_*.nc, назначает каждую ячейку сетки ERA5 ближайшему
    бассейну и усредняет по нему. Возвращает DataFrame со схемой:
        date, basin, precip_mm, tmin_c, tmax_c, tmean_c, swe_mm
    Требует xarray и движок чтения NetCDF (netCDF4 или h5netcdf).
    """
    import os
    import shutil
    import tempfile

    import numpy as np
    import pandas as pd
    import xarray as xr

    from ..aoi import basins, haversine_km

    cfg = load_config()
    raw = cfg.path("raw")
    files = sorted(raw.glob("era5_land_*.nc"))
    if not files:
        raise FileNotFoundError("Нет файлов era5_land_*.nc в data/raw — сначала fetch_era5().")

    # netCDF4 (C-библиотека) не открывает пути с не-ASCII символами (папка «РКНП»),
    # поэтому копируем файлы во временный ASCII-путь перед чтением.
    _tmpdir = tempfile.mkdtemp(prefix="era5_")
    _tmp = [os.path.join(_tmpdir, f"era5_{i}.nc") for i in range(len(files))]
    for _f, _t in zip(files, _tmp):
        shutil.copyfile(_f, _t)

    # Открываем помесячные файлы и грузим в память (без dask), затем объединяем.
    parts_ds = []
    for _t in _tmp:
        with xr.open_dataset(_t) as _d:
            parts_ds.append(_d.load())
    shutil.rmtree(_tmpdir, ignore_errors=True)          # данные уже в памяти

    tname = next((c for c in ("valid_time", "time")
                  if c in parts_ds[0].coords or c in parts_ds[0].dims), None)
    if tname is None:
        raise KeyError(f"Не найдена временная координата в {list(parts_ds[0].coords)}")
    ds = xr.concat(parts_ds, dim=tname) if len(parts_ds) > 1 else parts_ds[0]
    ds = ds.sortby(tname)

    def pick(*names):
        for n in names:
            if n in ds:
                return ds[n]
        raise KeyError(f"Нет ни одной из переменных {names}; есть: {list(ds.data_vars)}")

    t2m = pick("t2m", "2m_temperature")
    tp = pick("tp", "total_precipitation")
    swe = pick("sd", "swe", "snow_depth_water_equivalent")

    lat = ds["latitude"].values
    lon = ds["longitude"].values
    LON, LAT = np.meshgrid(lon, lat)
    b = basins()
    dist = np.stack([haversine_km(LON, LAT, r.lon, r.lat) for r in b.itertuples()], axis=0)
    basin_idx = dist.argmin(axis=0)                       # (nlat, nlon)

    tmin = t2m.resample({tname: "1D"}).min()
    tmax = t2m.resample({tname: "1D"}).max()
    tmean = t2m.resample({tname: "1D"}).mean()
    # tp в ERA5-Land накапливается с 00 UTC, и значение в 00:00 — это сумма за
    # ПРЕДЫДУЩИЕ сутки. Сдвиг на −1 ч относит часы 01…24 к одной дате; без него
    # max() по суткам брал бы вчерашнюю сумму и дождь «копировался» на след. день.
    tp_shift = tp.assign_coords({tname: tp[tname] - np.timedelta64(1, "h")})
    tp_d = tp_shift.resample({tname: "1D"}).max().reindex({tname: tmean[tname]})  # м/сут
    swe_d = swe.resample({tname: "1D"}).mean()
    days = pd.to_datetime(tp_d[tname].values)

    def bmean(da, bi):
        arr = np.asarray(da.values)                       # (day, lat, lon)
        mask = basin_idx == bi
        if not mask.any():
            r0, c0 = np.unravel_index(dist[bi].argmin(), dist[bi].shape)
            return arr[:, r0, c0]
        return np.nanmean(arr[:, mask], axis=1)

    parts = []
    for bi, row in enumerate(b.itertuples()):
        parts.append(pd.DataFrame({
            "date": days,
            "basin": row.name,
            "precip_mm": bmean(tp_d, bi) * 1000.0,
            "tmin_c": bmean(tmin, bi) - 273.15,
            "tmax_c": bmean(tmax, bi) - 273.15,
            "tmean_c": bmean(tmean, bi) - 273.15,
            "swe_mm": bmean(swe_d, bi) * 1000.0,
        }))
    out = pd.concat(parts, ignore_index=True).sort_values(["basin", "date"]).reset_index(drop=True)

    if save:
        dst = cfg.path("processed") / "meteo_weather_era5.csv"
        out.to_csv(dst, index=False)
        print(f"[era5] суточные ряды {len(out)} строк ({out['date'].min().date()}"
              f"…{out['date'].max().date()}) -> {dst}")
    return out
