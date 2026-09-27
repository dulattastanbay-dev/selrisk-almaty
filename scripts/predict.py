"""Демонстрация: оценка риска селей на заданную дату + карта и текстовый отчёт.

    python scripts/predict.py --date 2017-06-15
    python scripts/predict.py                 # дата максимального риска
"""
import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
for _s in (sys.stdout, sys.stderr):          # корректный вывод кириллицы в Windows-консоли
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd

from selrisk.config import load_config
from selrisk.models.fusion import daily_risk
from selrisk import viz


def main() -> None:
    cfg = load_config()
    odir = cfg.path("outputs")
    grid_susc = pd.read_csv(odir / "grid_with_susceptibility.csv")
    meteo = pd.read_csv(odir / "meteo_with_trigger.csv", parse_dates=["date"])

    ap = argparse.ArgumentParser(description="Оценка риска селей над Алматы на дату")
    ap.add_argument("--date", help="дата YYYY-MM-DD (по умолчанию — день максимального риска)")
    args = ap.parse_args()

    if args.date:
        date = pd.Timestamp(args.date)
        if date not in set(meteo["date"]):
            raise SystemExit(f"Нет данных на {date.date()}. "
                             f"Диапазон: {meteo['date'].min().date()} … {meteo['date'].max().date()}")
    else:
        date = meteo.loc[meteo["trigger_prob"].idxmax(), "date"]

    risk = daily_risk(grid_susc, meteo, date)

    print("=" * 64)
    print(f"  ОЦЕНКА РИСКА СЕЛЕЙ — {pd.Timestamp(date).date()}")
    print("=" * 64)
    by_basin = (
        risk.groupby("basin")
        .agg(trigger_prob=("trigger_prob", "first"),
             max_risk=("risk", "max"),
             mean_susc=("susceptibility", "mean"))
        .sort_values("max_risk", ascending=False)
    )
    for b, row in by_basin.iterrows():
        top = risk[risk["basin"] == b].nlargest(1, "risk").iloc[0]
        print(f"  {b:20s}  P(триггер)={row.trigger_prob:0.2f}  "
              f"макс.риск={row.max_risk:0.2f} [{top.risk_class}]")
    print("-" * 64)
    worst = by_basin.index[0]
    lvl = risk[risk["basin"] == worst]["risk"].max()
    verdict = ("ЭКСТРЕМАЛЬНЫЙ" if lvl > 0.7 else "ВЫСОКИЙ" if lvl > 0.4
               else "УМЕРЕННЫЙ" if lvl > 0.15 else "НИЗКИЙ")
    print(f"  Наибольший риск: {worst} — уровень {verdict} ({lvl:.2f}).")
    print(f"  Ячеек класса «высокий/экстремальный»: "
          f"{int((risk['risk'] > 0.4).sum())} из {len(risk)}.")
    print("=" * 64)

    out_png = odir / f"predicted_risk_{pd.Timestamp(date).date()}.png"
    viz.plot_risk_map(risk, grid_susc, date, out_png)
    print(f"Карта риска: {out_png}")


if __name__ == "__main__":
    main()
