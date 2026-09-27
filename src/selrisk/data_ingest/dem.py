"""Рельеф: производные (уклон, экспозиция, кривизна, TWI) из ЦМР.

Функции terrain_derivatives / d8_accumulation работают на любом массиве высот —
как на реальной ЦМР (Copernicus GLO-30 / SRTM), так и на синтетической,
которая используется, когда реальный растр не скачан. Формулы стандартные
(Horn 1981 для уклона/экспозиции, Zevenbergen–Thorne 1987 для кривизны,
Beven & Kirkby 1979 для TWI), поэтому переход на реальную ЦМР не меняет модели.
"""
from __future__ import annotations

import io
import math
import urllib.request

import numpy as np

_EPS = 1e-9
_M_PER_DEG = 111_320.0  # метров в одном градусе широты

# Открытый набор AWS Terrain Tiles (Terrarium), без ключа. Высота кодируется в
# RGB: elev = R*256 + G + B/256 - 32768 (метры).
_TERRARIUM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


def cellsize_m(res_deg: float, lat_center: float) -> tuple[float, float]:
    """Размер ячейки в метрах: (dx по долготе, dy по широте)."""
    dy = res_deg * _M_PER_DEG
    dx = res_deg * _M_PER_DEG * np.cos(np.radians(lat_center))
    return dx, dy


def _gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Разделимое гауссово сглаживание (только numpy), отражающее края."""
    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    k /= k.sum()

    def conv(arr, axis):
        return np.apply_along_axis(
            lambda m: np.convolve(np.pad(m, radius, mode="reflect"), k, mode="valid"),
            axis, arr,
        )

    return conv(conv(a, 0), 1)


def _deg2num(lon: float, lat: float, z: int) -> tuple[float, float]:
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def fetch_real_dem(bbox, nlat: int, nlon: int, zoom: int = 11) -> np.ndarray:
    """Реальная ЦМР для bbox из открытых AWS Terrain Tiles, ресемпл до (nlat,nlon).

    bbox = (lon_min, lat_min, lon_max, lat_max). Возвращает высоты (м), row 0 —
    южный край (как у синтетической ЦМР), col 0 — западный. Требует Pillow.
    """
    from PIL import Image

    lon0, lat0, lon1, lat1 = bbox
    x0f, y_north = _deg2num(lon0, lat1, zoom)      # северо-запад
    x1f, y_south = _deg2num(lon1, lat0, zoom)      # юго-восток
    xt0, xt1 = int(math.floor(x0f)), int(math.floor(x1f))
    yt0, yt1 = int(math.floor(y_north)), int(math.floor(y_south))

    mosaic = np.zeros(((yt1 - yt0 + 1) * 256, (xt1 - xt0 + 1) * 256), dtype=np.float32)
    for ty in range(yt0, yt1 + 1):
        for tx in range(xt0, xt1 + 1):
            url = _TERRARIUM.format(z=zoom, x=tx, y=ty)
            try:
                data = urllib.request.urlopen(url, timeout=60).read()
                rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.float32)
                elev = rgb[:, :, 0] * 256.0 + rgb[:, :, 1] + rgb[:, :, 2] / 256.0 - 32768.0
            except Exception:                       # noqa: BLE001 (океан/отсутствие тайла)
                elev = np.zeros((256, 256), dtype=np.float32)
            r = (ty - yt0) * 256
            c = (tx - xt0) * 256
            mosaic[r:r + 256, c:c + 256] = elev

    # обрезка мозаики точно по bbox (в пикселях)
    left = int(round((x0f - xt0) * 256))
    right = int(round((x1f - xt0) * 256))
    top = int(round((y_north - yt0) * 256))
    bottom = int(round((y_south - yt0) * 256))
    crop = mosaic[top:bottom, left:right]           # row 0 = север

    img = Image.fromarray(crop, mode="F").resize((nlon, nlat), Image.BILINEAR)
    dem_north_top = np.asarray(img, dtype=np.float32)
    return np.flipud(dem_north_top)                 # row 0 -> юг


def make_synthetic_dem(nlat: int, nlon: int, seed: int = 42) -> np.ndarray:
    """Физически правдоподобная синтетическая ЦМР Заилийского Алатау.

    Высота растёт к югу (к хребту), поверх наложены многомасштабные гребни и
    долины. row=0 соответствует южному (высокогорному) краю AOI.
    Возвращает массив высот (nlat, nlon) в метрах.
    """
    rng = np.random.default_rng(seed)
    rows = np.linspace(0, 1, nlat)[:, None]          # 0 = юг (высоко), 1 = север (город)
    base = 4100.0 - 3000.0 * rows                     # 4100 м -> 1100 м
    base = np.repeat(base, nlon, axis=1)

    # многомасштабный фрактальный шум -> гребни и долины
    noise = np.zeros((nlat, nlon))
    for sigma, amp in [(1.2, 120.0), (3.0, 260.0), (7.0, 420.0)]:
        noise += amp * _gaussian_blur(rng.standard_normal((nlat, nlon)), sigma)

    # врезанные долины вдоль трёх основных бассейнов (синусоидальные ложбины)
    cols = np.linspace(0, 1, nlon)[None, :]
    valleys = np.zeros((nlat, nlon))
    for cpos in (0.30, 0.55, 0.82):
        valleys += -350.0 * np.exp(-((cols - cpos) ** 2) / (2 * 0.02 ** 2))

    dem = base + noise + valleys * (1 - rows)          # долины глубже к северу
    return _gaussian_blur(dem, 0.8)


def terrain_derivatives(dem: np.ndarray, dx: float, dy: float) -> dict[str, np.ndarray]:
    """Уклон (град), sin/cos экспозиции, план- и профиль-кривизна из ЦМР."""
    gy, gx = np.gradient(dem, dy, dx)                  # dz/dy (строки), dz/dx (столбцы)
    slope_rad = np.arctan(np.hypot(gx, gy))
    slope_deg = np.degrees(slope_rad)

    aspect = np.arctan2(gy, -gx)                        # радианы, 0 = восток
    aspect_sin, aspect_cos = np.sin(aspect), np.cos(aspect)

    # вторые производные для кривизны (Zevenbergen–Thorne)
    zxx = np.gradient(gx, dx, axis=1)
    zyy = np.gradient(gy, dy, axis=0)
    zxy = np.gradient(gx, dy, axis=0)
    p, q = gx, gy
    denom_prof = (p ** 2 + q ** 2) * (1 + p ** 2 + q ** 2) ** 1.5 + _EPS
    denom_plan = (p ** 2 + q ** 2) ** 1.5 + _EPS
    prof_curv = -2 * (zxx * p ** 2 + 2 * zxy * p * q + zyy * q ** 2) / denom_prof
    plan_curv = 2 * (zxx * q ** 2 - 2 * zxy * p * q + zyy * p ** 2) / denom_plan

    return {
        "elevation": dem,
        "slope_deg": slope_deg,
        "slope_rad": slope_rad,
        "aspect_sin": aspect_sin,
        "aspect_cos": aspect_cos,
        "plan_curv": plan_curv,
        "prof_curv": prof_curv,
    }


def d8_accumulation(dem: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Накопление стока по алгоритму D8 (число вышележащих ячеек + 1)."""
    nlat, nlon = dem.shape
    flat = dem.ravel()
    n = flat.size

    # смещения 8 соседей и расстояния до них
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    dist = [np.hypot(dx, dy), dy, np.hypot(dx, dy), dx, dx,
            np.hypot(dx, dy), dy, np.hypot(dx, dy)]

    receiver = np.full(n, -1, dtype=np.int64)
    rr, cc = np.divmod(np.arange(n), nlon)
    best_slope = np.zeros(n)
    for (dr, dc), dd in zip(offs, dist):
        nr, nc = rr + dr, cc + dc
        valid = (nr >= 0) & (nr < nlat) & (nc >= 0) & (nc < nlon)
        nidx = np.where(valid, nr * nlon + nc, 0)
        drop = (flat - flat[nidx]) / dd               # >0, если сосед ниже
        better = valid & (drop > best_slope)
        best_slope = np.where(better, drop, best_slope)
        receiver = np.where(better, nidx, receiver)

    acc = np.ones(n)
    order = np.argsort(flat)[::-1]                     # от высоких к низким
    for idx in order:
        r = receiver[idx]
        if r >= 0:
            acc[r] += acc[idx]
    return acc.reshape(nlat, nlon)


def topographic_wetness_index(dem: np.ndarray, dx: float, dy: float,
                              slope_rad: np.ndarray | None = None) -> np.ndarray:
    """TWI = ln(a / tan(beta)); a — удельная площадь водосбора."""
    if slope_rad is None:
        gy, gx = np.gradient(dem, dy, dx)
        slope_rad = np.arctan(np.hypot(gx, gy))
    acc = d8_accumulation(dem, dx, dy)
    spec_area = acc * ((dx + dy) / 2.0)                # площадь на ширину контура
    tan_beta = np.tan(np.maximum(slope_rad, np.radians(0.5)))
    return np.log(spec_area / tan_beta + _EPS)
