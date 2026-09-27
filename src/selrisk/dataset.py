"""Сборка обучающих таблиц из данных о рельефе, погоде, сейсмике и озёрах.

Всё, что помечено как «генерация выборки», физически обосновано, но синтетично
и служит для запуска пайплайна без закрытых данных. Реальная инвентаризация
сходов (Казселезащита) и ERA5/USGS подставляются в те же схемы без изменения
моделей. Метки подверженности порождаются из явной генеративной модели —
это позволяет честно измерять, восстанавливает ли классификатор зависимость.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .aoi import assign_basin, basins, make_grid
from .config import load_config
from .data_ingest.dem import (
    cellsize_m,
    d8_accumulation,
    make_synthetic_dem,
    terrain_derivatives,
    topographic_wetness_index,
    _gaussian_blur,
)
from .data_ingest.earthquakes import energy_from_magnitude
from .data_ingest.glacial_lakes import lake_features, load_lakes
from .features import add_antecedent_features


# ---------------------------------------------------------------------------
#  Статическая сетка признаков рельефа
# ---------------------------------------------------------------------------
def build_terrain_grid(seed: int | None = None, dem: np.ndarray | None = None) -> pd.DataFrame:
    """Сетка ячеек AOI со всеми статическими признаками для модели подверженности."""
    cfg = load_config()
    seed = seed if seed is not None else cfg["project"]["random_seed"]
    grid = make_grid()
    nlat = int(grid["row"].max()) + 1
    nlon = int(grid["col"].max()) + 1
    res = cfg["aoi"]["grid_res_deg"]
    dx, dy = cellsize_m(res, float(grid["lat"].mean()))

    if dem is None:
        cache = cfg.path("raw") / "dem_real.npy"
        if cache.exists():
            arr = np.load(cache)
            if arr.shape == (nlat, nlon):
                dem = arr
        if dem is None:
            dem = make_synthetic_dem(nlat, nlon, seed=seed)
    der = terrain_derivatives(dem, dx, dy)
    twi = topographic_wetness_index(dem, dx, dy, der["slope_rad"])
    acc = d8_accumulation(dem, dx, dy)

    r = grid["row"].to_numpy()
    c = grid["col"].to_numpy()
    grid["elevation"] = dem[r, c]
    grid["slope_deg"] = der["slope_deg"][r, c]
    grid["aspect_sin"] = der["aspect_sin"][r, c]
    grid["aspect_cos"] = der["aspect_cos"][r, c]
    grid["plan_curv"] = der["plan_curv"][r, c]
    grid["prof_curv"] = der["prof_curv"][r, c]
    grid["twi"] = twi[r, c]

    # дренажная сеть = ячейки с высоким накоплением стока; расстояние до неё
    grid_acc = acc[r, c]
    is_channel = grid_acc > np.quantile(grid_acc, 0.985)
    ch = grid.loc[is_channel, ["lon", "lat"]].to_numpy()
    from .aoi import haversine_km
    dkm = haversine_km(
        grid["lon"].to_numpy()[:, None], grid["lat"].to_numpy()[:, None],
        ch[:, 0][None, :], ch[:, 1][None, :],
    )
    grid["dist_drainage_m"] = dkm.min(axis=1) * 1000.0

    # эродируемость пород — пространственно-коррелированное поле 0..1
    rng = np.random.default_rng(seed + 7)
    lith = _gaussian_blur(rng.standard_normal((nlat, nlon)), 5.0)
    lith = (lith - lith.min()) / (lith.max() - lith.min() + 1e-9)
    grid["lithology_erodibility"] = lith[r, c]

    # класс земного покрова по высоте: 4 ледник/скалы, 3 альпика, 2 лес, 1 долина/город
    elev = grid["elevation"].to_numpy()
    grid["landcover_code"] = np.select(
        [elev > 3500, elev > 2500, elev > 1600], [4, 3, 2], default=1
    )

    grid["basin"] = assign_basin(grid)
    grid = grid.join(lake_features(grid, load_lakes()))
    return grid


def cache_real_dem(zoom: int = 11):
    """Скачивает реальную ЦМР (AWS Terrain Tiles) для AOI и кэширует в data/raw."""
    from .data_ingest.dem import fetch_real_dem

    cfg = load_config()
    grid = make_grid()
    nlat = int(grid["row"].max()) + 1
    nlon = int(grid["col"].max()) + 1
    dem = fetch_real_dem(cfg["aoi"]["bbox"], nlat, nlon, zoom)
    out = cfg.path("raw") / "dem_real.npy"
    np.save(out, dem)
    print(f"[dem] реальная ЦМР {dem.shape}, высоты {dem.min():.0f}–{dem.max():.0f} м -> {out}")
    return out, dem


# ---------------------------------------------------------------------------
#  Латентная (генеративная) подверженность и обучающая выборка меток
# ---------------------------------------------------------------------------
def _z(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-9)


def latent_score(grid: pd.DataFrame) -> np.ndarray:
    """Стандартизованный «истинный» сигнал подверженности (~N(0,1)).

    Максимален на крутых собирающих склонах зоны формирования (2100–3500 м),
    вблизи русел и ниже моренных озёр, при высокой эродируемости пород.
    """
    slope = grid["slope_deg"].to_numpy()
    slope_opt = np.exp(-((slope - 38.0) ** 2) / (2 * 12.0 ** 2))
    terms = (
        1.4 * _z(slope_opt)
        + 1.1 * _z(np.exp(-grid["dist_drainage_m"] / 250.0))
        + 1.3 * _z(np.log1p(grid["upstream_lake_area_km2"]))
        + 0.7 * _z(np.exp(-grid["dist_lake_m"] / 1500.0))
        + 0.8 * _z(grid["lithology_erodibility"])
        + 0.6 * _z(-grid["prof_curv"])
        + 0.5 * _z(grid["twi"])
        + 0.9 * _z(np.exp(-((grid["elevation"] - 2800.0) ** 2) / (2 * 700.0 ** 2)))
    )
    return _z(terms)


def latent_susceptibility(grid: pd.DataFrame) -> np.ndarray:
    """«Истинная» подверженность в [0,1] (для карт и слияния)."""
    return 1.0 / (1.0 + np.exp(-0.9 * latent_score(grid)))


def make_susceptibility_training(grid: pd.DataFrame, seed: int | None = None,
                                 n_total: int = 1600) -> pd.DataFrame:
    """Размеченная «инвентаризация»: source-ячейки (1) и устойчивые склоны (0).

    Имитирует полевую/дистанционную инвентаризацию с реалистичным шумом:
    метка наблюдается не как жёсткая функция подверженности, а как исход
    Бернулли от неё с шумом, поэтому классы перекрываются (иначе ROC-AUC
    искусственно близок к 1). Для достаточного числа очагов верхняя и нижняя
    доли латентной подверженности берутся поровну.
    """
    cfg = load_config()
    seed = seed if seed is not None else cfg["project"]["random_seed"]
    score = latent_score(grid)
    rng = np.random.default_rng(seed + 11)
    n = len(grid)
    order = np.argsort(score)
    high = order[-int(n * 0.40):]
    low = order[: int(n * 0.60)]
    take_high = rng.choice(high, size=min(n_total // 2, len(high)), replace=False)
    take_low = rng.choice(low, size=min(n_total // 2, len(low)), replace=False)
    idx = np.concatenate([take_high, take_low])

    logit = 1.9 * score[idx] + rng.normal(0.0, 0.7, len(idx))     # шумная инвентаризация
    p = 1.0 / (1.0 + np.exp(-logit))
    label = (rng.random(len(idx)) < p).astype(int)

    train = grid.iloc[idx].copy()
    train["label"] = label
    return train.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def basin_susceptibility(grid: pd.DataFrame, latent: np.ndarray | None = None) -> pd.Series:
    """Скалярная подверженность бассейна = 90-й перцентиль латентной подверженности."""
    if latent is None:
        latent = latent_susceptibility(grid)
    tmp = grid[["basin"]].copy()
    tmp["_lat"] = latent
    return tmp.groupby("basin")["_lat"].quantile(0.90)


# ---------------------------------------------------------------------------
#  Суточный сейсмический ряд (реальный USGS при наличии кэша, иначе синтетика)
# ---------------------------------------------------------------------------
def daily_eq_series(dates: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    cfg = load_config()
    path = cfg.path("raw") / "earthquakes_usgs.csv"
    idx = pd.DatetimeIndex(dates)
    day = idx.normalize()

    if path.exists():
        cat = pd.read_csv(path)
        t = (pd.to_datetime(cat["time"], utc=True, format="ISO8601")
             .dt.tz_localize(None).dt.normalize())
        mag = cat["mag"].astype(float)
        daily_max = mag.groupby(t).max()
        daily_energy = pd.Series(energy_from_magnitude(mag)).groupby(t).sum()
    else:
        rng = np.random.default_rng(seed + 3)
        n = len(idx)
        counts = rng.poisson(0.15, n)
        mags = np.where(counts > 0, rng.uniform(3.0, 4.5, n), 0.0)
        strong = rng.random(n) < 0.002
        mags = np.where(strong, rng.uniform(5.0, 6.3, n), mags)
        daily_max = pd.Series(mags, index=day)
        daily_energy = pd.Series(np.where(mags > 0, energy_from_magnitude(mags), 0.0), index=day)

    dm = daily_max.reindex(day).fillna(0.0)
    de = daily_energy.reindex(day).fillna(0.0)
    eq_max_3d = pd.Series(dm.to_numpy(), index=idx).rolling(3, min_periods=1).max()
    eq_energy_7d = pd.Series(de.to_numpy(), index=idx).rolling(7, min_periods=1).sum()
    return pd.DataFrame(
        {
            "date": idx,
            "eq_max_mag_3d": eq_max_3d.to_numpy(),
            "eq_energy_7d": np.log1p(eq_energy_7d.to_numpy()),
        }
    )


# ---------------------------------------------------------------------------
#  Суточные метеоряды по бассейнам + генерация событий
# ---------------------------------------------------------------------------
def _weather_one_basin(basin: str, dates: pd.DatetimeIndex, rng) -> pd.DataFrame:
    n = len(dates)
    doy = dates.dayofyear.to_numpy()

    tmean = 5.0 + 14.0 * np.sin(2 * np.pi * (doy - 110) / 365.25) + rng.normal(0, 2.5, n)
    tmin = tmean - 6.0 - rng.random(n) * 2
    tmax = tmean + 6.0 + rng.random(n) * 2

    wet_p = 0.22 + 0.22 * np.clip(np.sin(2 * np.pi * (doy - 120) / 365.25), 0, 1)
    wet = rng.random(n) < wet_p
    amount = rng.gamma(0.8, 6.0, n)
    summer = (doy >= 150) & (doy <= 230)
    amount *= np.where(summer, 1.8, 1.0)
    extreme = summer & (rng.random(n) < 0.010)
    amount = amount + np.where(extreme, rng.gamma(3.0, 25.0, n), 0.0)
    precip_total = np.where(wet, amount, 0.0)

    rain = np.where(tmean >= 1.0, precip_total, 0.0)       # жидкие осадки
    snowfall = np.where(tmean < 1.0, precip_total, 0.0)

    swe = np.zeros(n)
    melt = np.zeros(n)
    s = 0.0
    for i in range(n):
        s += snowfall[i]
        m = min(max(tmean[i] - 0.0, 0.0) * 3.0, s)
        s -= m
        swe[i] = s
        melt[i] = m

    return pd.DataFrame(
        {
            "date": dates,
            "basin": basin,
            "precip_mm": rain,
            "tmin_c": tmin,
            "tmax_c": tmax,
            "tmean_c": tmean,
            "swe_mm": swe,
            "snowmelt_mm": melt,
        }
    )


def generate_meteo(grid: pd.DataFrame | None = None, seed: int | None = None,
                   n_events_target: int | None = None, save: bool = True,
                   use_real_weather: bool = True) -> pd.DataFrame:
    """Суточные ряды по бассейнам + метка схода `event` (физически согласованная).

    Если есть реальные суточные ряды ERA5 (data/processed/meteo_weather_era5.csv),
    они используются как погодная основа (осадки/температура/снег), снеготаяние
    считается по убыли SWE; иначе погода синтезируется. Сейсмика — реальная USGS
    при наличии кэша. Метки событий порождаются от воды/увлажнения/подверженности.
    """
    cfg = load_config()
    seed = seed if seed is not None else cfg["project"]["random_seed"]
    if grid is None:
        grid = build_terrain_grid(seed)

    latent = latent_susceptibility(grid)
    bsus = basin_susceptibility(grid, latent)

    rw_path = cfg.path("processed") / "meteo_weather_era5.csv"
    if use_real_weather and rw_path.exists():
        meteo = pd.read_csv(rw_path, parse_dates=["date"]).sort_values(["basin", "date"])
        melt = -meteo.groupby("basin")["swe_mm"].diff()
        meteo["snowmelt_mm"] = melt.clip(lower=0).fillna(0.0)
        dates = pd.DatetimeIndex(sorted(meteo["date"].unique()))
        print(f"[dataset] погода: реальный ERA5 ({rw_path.name})")
    else:
        dates = pd.date_range(cfg["data_sources"]["weather"]["start"],
                              cfg["data_sources"]["weather"]["end"], freq="D")
        parts = []
        for i, row in enumerate(basins().itertuples()):
            rng_b = np.random.default_rng(seed + 100 + i)
            parts.append(_weather_one_basin(row.name, dates, rng_b))
        meteo = pd.concat(parts, ignore_index=True)

    if n_events_target is None:                         # ~5 селеопасных дней в год
        n_years = max(1.0, (dates.max() - dates.min()).days / 365.25)
        n_events_target = max(30, int(round(5.0 * n_years)))

    eq = daily_eq_series(dates, seed)
    meteo = meteo.merge(eq, on="date", how="left")
    meteo = add_antecedent_features(meteo)

    # --- вероятностная генерация событий (Бернулли) ---
    # Линейный предиктор зависит от воды, увлажнения, сейсмики и подверженности
    # бассейна; заметный шумовой член делает связь неполной, поэтому модель по
    # наблюдаемым признакам не может предсказывать идеально (реалистичный AUC).
    rng = np.random.default_rng(seed + 999)
    snowmelt_3d = (
        meteo.groupby("basin")["snowmelt_mm"]
        .transform(lambda s: s.rolling(3, min_periods=1).sum())
    )
    water_3d = (meteo["rain_3d"] + snowmelt_3d).to_numpy()
    b = meteo["basin"].map(bsus).to_numpy()
    lin = (
        0.8 * _z(b)                            # подверженность бассейна («где»)
        + 1.8 * _z(meteo["rain_3d"])           # осадки — главный наблюдаемый фактор
        + 1.0 * _z(snowmelt_3d)
        + 1.1 * _z(meteo["api"])
        + 0.25 * _z(meteo["eq_energy_7d"])     # сейсмика слабая — проверяем гипотезу
        + rng.normal(0.0, 0.5, len(meteo))     # неустранимая случайность
    )
    eligible = water_3d > 10.0                 # без достаточной воды сель не сходит
    lin = np.where(eligible, lin, -30.0)

    # калибруем свободный член так, чтобы ожидаемое число событий ≈ целевому
    def _expected(c):
        return float(np.sum(1.0 / (1.0 + np.exp(-(c + lin)))))

    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _expected(mid) > n_events_target:
            hi = mid
        else:
            lo = mid
    prob = 1.0 / (1.0 + np.exp(-((lo + hi) / 2 + lin)))
    meteo["trigger_prob_true"] = prob
    meteo["event"] = (rng.random(len(meteo)) < prob).astype(int)

    # --- зафиксировать реальные события (курируемые + COOLR при наличии) ---
    ev_files = [cfg.path("events") / "historical_events.csv",
                cfg.path("events") / "coolr_events.csv"]
    for ev_path in ev_files:
        if not ev_path.exists():
            continue
        hist = pd.read_csv(ev_path, parse_dates=["date"])
        for h in hist.itertuples():
            m = (meteo["date"] == pd.Timestamp(h.date)) & (meteo["basin"] == h.basin)
            meteo.loc[m, "event"] = 1

    if save:
        out = cfg.path("sample") / "meteo_daily.csv"
        meteo.to_csv(out, index=False)
        print(f"[dataset] метеоряд {len(meteo)} строк, событий={int(meteo['event'].sum())} -> {out}")
    return meteo
