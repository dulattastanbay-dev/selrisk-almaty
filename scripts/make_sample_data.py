"""Генерирует физически обоснованную демонстрационную выборку (без API-ключей).

Пишет в data/sample:
  grid.csv                 — сетка ячеек AOI со статическими признаками рельефа
  susceptibility_train.csv — размеченная «инвентаризация» очагов (1) и склонов (0)
  meteo_daily.csv          — суточные метео/сейсмо-ряды по бассейнам + метка event
Реальный пайплайн пишет файлы той же схемы в data/processed.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
for _s in (sys.stdout, sys.stderr):          # корректный вывод кириллицы в Windows-консоли
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from selrisk.config import load_config, seed_everything
from selrisk.dataset import build_terrain_grid, generate_meteo, make_susceptibility_training


def main() -> None:
    cfg = load_config()
    seed = seed_everything()
    print(f"[make_sample_data] seed={seed}")

    grid = build_terrain_grid(seed)
    grid.to_csv(cfg.path("sample") / "grid.csv", index=False)
    print(f"[make_sample_data] сетка: {len(grid)} ячеек -> data/sample/grid.csv")

    train = make_susceptibility_training(grid, seed)
    train.to_csv(cfg.path("sample") / "susceptibility_train.csv", index=False)
    print(f"[make_sample_data] метки подверженности: {len(train)} "
          f"(положительных={int(train['label'].sum())}) -> data/sample/susceptibility_train.csv")

    generate_meteo(grid, seed)   # сохраняет meteo_daily.csv сам


if __name__ == "__main__":
    main()
