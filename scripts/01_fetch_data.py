"""Загрузка РЕАЛЬНЫХ открытых данных (по возможности).

  * Каталог землетрясений USGS — скачивается сразу (открытый API, без ключа).
  * ERA5-Land — печатает инструкцию (нужны учётка Copernicus CDS и cdsapi).
  * ЦМР — печатает инструкцию (Copernicus GLO-30 / SRTM).

После загрузки сейсмики скрипт train.py автоматически использует реальный
каталог (data/raw/earthquakes_usgs.csv) вместо синтетического.
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
from selrisk.data_ingest.earthquakes import fetch_earthquakes
from selrisk.data_ingest.coolr import coolr_to_events_schema, fetch_coolr


def main() -> None:
    cfg = load_config()

    print("== 1. Землетрясения (USGS FDSN, открытый API) ==")
    try:
        fetch_earthquakes()
    except Exception as exc:  # noqa: BLE001
        print(f"   Не удалось загрузить (нет сети?): {exc}")
        print("   Пайплайн будет использовать синтетический сейсморяд.")

    print("\n== 1b. Каталог сходов NASA COOLR / GLC (открытый ArcGIS) ==")
    try:
        regional = fetch_coolr(bbox=(74, 41, 82, 45))       # Тянь-Шань / ЮВ Казахстан
        regional.to_csv(cfg.path("events") / "coolr_regional.csv", index=False)
        aoi = fetch_coolr()                                  # bbox AOI
        events = coolr_to_events_schema(aoi)
        if len(events):
            events.to_csv(cfg.path("events") / "coolr_events.csv", index=False)
        print(f"   COOLR: {len(regional)} событий в регионе, {len(events)} в AOI "
              f"(записаны в data/events/).")
        if len(regional) == 0:
            print("   Прим.: покрытие Центральной Азии в GLC разреженное — это ожидаемо.")
    except Exception as exc:  # noqa: BLE001
        print(f"   COOLR недоступен ({exc}).")
        print("   Используется курируемый data/events/historical_events.csv.")

    print("\n== 2. ERA5-Land (осадки/температура/снег) ==")
    print("   1) ~/.cdsapirc с ключом; 2) примите лицензию датасета на сайте CDS:")
    print("      https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download#manage-licences")
    print("   3) pip install cdsapi xarray netCDF4; затем: python scripts/fetch_era5.py [год0 год1]")

    print("\n== 3. Цифровая модель рельефа (ЦМР) ==")
    print("   Copernicus GLO-30: https://dataspace.copernicus.eu  (или SRTM 1 arc-sec)")
    print("   Положите GeoTIFF в data/raw и подключите через data_ingest/dem.py")

    print(f"\nОбласть исследования (bbox): {cfg['aoi']['bbox']}")
    print("Готово. Далее: python scripts/run_all.py")


if __name__ == "__main__":
    main()
