"""Строит все карты и графики результатов в data/outputs."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
for _s in (sys.stdout, sys.stderr):          # корректный вывод кириллицы в Windows-консоли
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import pandas as pd

from selrisk.config import load_config
from selrisk.dataset import build_terrain_grid
from selrisk.models.fusion import daily_risk
from selrisk.models.triggering import build_rainfall_events, fit_id_threshold
from selrisk import viz


def main() -> None:
    cfg = load_config()
    odir = cfg.path("outputs")
    grid = build_terrain_grid()
    grid_susc = pd.read_csv(odir / "grid_with_susceptibility.csv")
    meteo = pd.read_csv(odir / "meteo_with_trigger.csv", parse_dates=["date"])
    susc = joblib.load(cfg.path("models") / "susceptibility_result.joblib")
    trg = joblib.load(cfg.path("models") / "triggering_result.joblib")

    # статические поля
    dem_title = ("Высота (Copernicus / AWS Terrain Tiles)"
                 if (cfg.path("raw") / "dem_real.npy").exists()
                 else "Высота (синтетическая ЦМР)")
    viz.map_field(grid, grid["elevation"], dem_title,
                  odir / "01_elevation.png", cmap="terrain", label="м")
    viz.map_field(grid, grid["slope_deg"], "Уклон",
                  odir / "02_slope.png", cmap="magma", label="градусы")
    viz.map_field(grid_susc, grid_susc["susceptibility"],
                  "Подверженность склонов сходам (RF)",
                  odir / "03_susceptibility.png", cmap="YlOrRd", label="P(очаг)")

    # оценка подверженности
    area_frac, cum_pos = susc["success_curve"]
    viz.plot_success_curve(area_frac, cum_pos, susc["success_rate_auc"],
                           odir / "04_success_curve.png")
    viz.plot_feature_importance(susc["importances"], odir / "05_importance_susc.png",
                                "Важность признаков — подверженность")

    # оценка триггера
    viz.plot_roc_pr(trg["test_y"], trg["test_proba"], trg["test_auc"], trg["test_ap"],
                    odir / "06_trigger_roc_pr.png")
    viz.plot_feature_importance(trg["importances"], odir / "06b_importance_trigger.png",
                                "Важность признаков — триггер (перестановки)")
    events = build_rainfall_events(meteo)
    idt = fit_id_threshold(events)
    viz.plot_id_threshold(events, idt["a"], idt["b"], odir / "07_id_threshold.png")
    viz.plot_event_timeline(meteo, odir / "08_event_timeline.png")

    # показательная карта: день с наибольшим числом сработавших бассейнов
    # (широкий дождевой сход — карта информативна по всем бассейнам, а не по одному)
    piv = meteo.pivot_table(index="date", columns="basin",
                            values="trigger_prob", aggfunc="first").fillna(0.0)
    show_score = (piv > 0.3).sum(axis=1) * 10 + piv.sum(axis=1)
    show_date = show_score.idxmax()
    risk = daily_risk(grid_susc, meteo, show_date)
    viz.plot_risk_map(risk, grid_susc, show_date, odir / "09_risk_map_peakday.png")
    viz.plot_risk_map(risk, grid_susc, show_date, odir / "predicted_risk_showcase.png")
    n_hot = int((piv.loc[show_date] > 0.3).sum())
    print(f"[evaluate] показательная карта на {pd.Timestamp(show_date).date()} "
          f"(сработало бассейнов: {n_hot}/3, макс. риск={risk['risk'].max():.2f})")
    print(f"[evaluate] графики сохранены в {odir}")


if __name__ == "__main__":
    main()
