"""Загрузка и агрегация реальных данных ERA5-Land (после принятия лицензии CDS).

Предпосылки:
  1) настроен ~/.cdsapirc с ключом (url: https://cds.climate.copernicus.eu/api);
  2) на сайте CDS приняты условия датасета reanalysis-era5-land:
     https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download#manage-licences
  3) установлены пакеты: pip install cdsapi xarray netCDF4

    python scripts/fetch_era5.py                 # весь период из config.yaml (медленно!)
    python scripts/fetch_era5.py 2019 2023        # только 2019–2023

После завершения запустите `python scripts/run_all.py` — пайплайн автоматически
использует реальную погоду (data/processed/meteo_weather_era5.csv).
Замечание: ERA5-Land почасовой за много лет качается долго (очередь CDS).
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from selrisk.config import load_config
from selrisk.data_ingest.era5 import aggregate_to_daily, fetch_era5


def main() -> None:
    cfg = load_config()
    src = cfg["data_sources"]["weather"]
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else int(src["start"][:4])
    y1 = int(sys.argv[2]) if len(sys.argv) > 2 else int(src["end"][:4])
    print(f"[fetch_era5] загрузка {y0}–{y1} (это может занять время — очередь CDS)…")
    try:
        fetch_era5(range(y0, y1 + 1))
    except Exception as exc:  # noqa: BLE001
        print(f"[fetch_era5] ошибка загрузки: {exc}")
        if "licence" in str(exc).lower() or "403" in str(exc):
            print("  Примите условия датасета ERA5-Land на сайте CDS (ссылка в шапке файла).")
        return
    print("[fetch_era5] агрегация до суточных рядов по бассейнам…")
    aggregate_to_daily()
    print("[fetch_era5] готово. Теперь: python scripts/run_all.py")


if __name__ == "__main__":
    main()
