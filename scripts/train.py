"""Обучение обеих моделей и сохранение прогнозов, моделей и метрик."""
import json
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

from selrisk.config import load_config, seed_everything
from selrisk.dataset import build_terrain_grid
from selrisk.models.susceptibility import predict_susceptibility, train_susceptibility
from selrisk.models.triggering import (
    build_rainfall_events,
    fit_id_threshold,
    predict_calibrated_trigger,
    predict_trigger_prob,
    train_calibrated_trigger,
    train_triggering,
)


def _data_dir(cfg):
    """Реальные данные (processed) в приоритете, иначе демонстрационная выборка."""
    proc = cfg.path("processed")
    if (proc / "meteo_daily.csv").exists() and (proc / "grid.csv").exists():
        return proc
    return cfg.path("sample")


def main() -> None:
    cfg = load_config()
    seed = seed_everything()
    ddir = _data_dir(cfg)
    print(f"[train] источник данных: {ddir}")

    # --- Подверженность ---
    grid = build_terrain_grid(seed)
    train_df = pd.read_csv(ddir / "susceptibility_train.csv")
    susc = train_susceptibility(train_df, seed)
    grid["susceptibility"] = predict_susceptibility(susc["model"], grid, susc["features"])
    print(f"[train] подверженность: CV ROC-AUC={susc['cv_auc']:.3f}, "
          f"AP={susc['cv_ap']:.3f}, success-AUC={susc['success_rate_auc']:.3f}")

    # --- Триггер ---
    meteo = pd.read_csv(ddir / "meteo_daily.csv", parse_dates=["date"])
    trg = train_triggering(meteo, seed)
    meteo["trigger_prob"] = predict_trigger_prob(trg["model_all"], meteo, trg["features"])
    print(f"[train] триггер: holdout ROC-AUC={trg['test_auc']:.3f}, AP={trg['test_ap']:.3f}, "
          f"событий всего={trg['n_events_total']} (в тесте={trg['n_events_test']})")

    # --- Откалиброванная вероятность (сайт и Streamlit показывают её в %) ---
    cal = train_calibrated_trigger(meteo)
    meteo["trigger_prob_cal"] = predict_calibrated_trigger(cal, meteo)
    cm = cal["metrics"]
    print(f"[train] калибр. триггер: holdout ROC-AUC={cm['holdout_roc_auc']:.3f}, "
          f"AP={cm['holdout_ap']:.3f}, ожидалось событий {cm['holdout_expected_events']:.1f} "
          f"при {cm['holdout_events']} фактических")

    # --- Порог интенсивность–длительность ---
    events = build_rainfall_events(meteo)
    idt = fit_id_threshold(events)
    print(f"[train] ID-порог: I={idt['a']:.1f}·D^(-{idt['b']:.2f}), "
          f"recall={idt['recall']:.2f}, precision={idt['precision']:.2f}")

    # --- Сохранение ---
    mdir = cfg.path("models")
    joblib.dump(susc, mdir / "susceptibility_result.joblib")
    joblib.dump(trg, mdir / "triggering_result.joblib")
    joblib.dump(susc["model"], mdir / "susceptibility_model.joblib")
    joblib.dump(trg["model_all"], mdir / "triggering_model.joblib")
    joblib.dump(cal, mdir / "trigger_calibrated.joblib")

    odir = cfg.path("outputs")
    grid.to_csv(odir / "grid_with_susceptibility.csv", index=False)
    meteo.to_csv(odir / "meteo_with_trigger.csv", index=False)
    events.to_csv(odir / "rainfall_events.csv", index=False)

    metrics = {
        "susceptibility": {
            "cv_roc_auc": susc["cv_auc"],
            "cv_average_precision": susc["cv_ap"],
            "success_rate_auc": susc["success_rate_auc"],
            "feature_importances": susc["importances"].round(4).to_dict(),
        },
        "triggering": {
            "holdout_roc_auc": trg["test_auc"],
            "holdout_average_precision": trg["test_ap"],
            "decision_threshold": trg["threshold"],
            "confusion_matrix_test": trg["confusion"],
            "permutation_importances": trg["importances"].round(4).to_dict(),
            "n_events_total": trg["n_events_total"],
            "n_events_test": trg["n_events_test"],
        },
        "trigger_calibrated": cal["metrics"],
        "id_threshold": idt,
    }
    with open(odir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    print(f"[train] метрики -> {odir / 'metrics.json'}")


if __name__ == "__main__":
    main()
