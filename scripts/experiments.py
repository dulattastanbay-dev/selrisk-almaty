"""Научная строгость: бейзлайны, абляция, разброс по seed, калибровка.

    python scripts/experiments.py

Результаты: data/outputs/experiments.json и графики 10..13_*.png.
"""
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from selrisk.config import load_config, seed_everything
from selrisk.experiments import run_all
from selrisk import viz


def main() -> None:
    cfg = load_config()
    seed_everything()
    odir = cfg.path("outputs")
    print("[experiments] прогон (бейзлайны, абляция, разброс по seed, калибровка)…")
    res = run_all(n_seeds=8)

    # графики
    viz.plot_metric_bars(res["baselines_susceptibility"],
                         "Подверженность: RF vs логрег", odir / "10_baseline_susc.png")
    viz.plot_metric_bars({k: v for k, v in res["baselines_triggering"].items()
                          if k != "ID_threshold"},
                         "Триггер: бустинг vs логрег", odir / "11_baseline_trigger.png")
    viz.plot_metric_bars(res["ablation_triggering"],
                         "Абляция триггера (вклад групп признаков)",
                         odir / "12_ablation_trigger.png")
    viz.plot_seed_variance(res["_variance_df"], odir / "13_seed_variance.png")
    frac, mean_pred, brier = res["_calib"]
    viz.plot_reliability(frac, mean_pred, brier, odir / "14_calibration.png")

    # JSON (без несериализуемых полей)
    out = {k: v for k, v in res.items() if not k.startswith("_")}
    with open(odir / "experiments.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    sv = res["seed_variance"]
    ab = res["ablation_triggering"]
    print(f"[experiments] подверженность: RF ROC-AUC={res['baselines_susceptibility']['RandomForest']['roc_auc']:.3f}"
          f" vs логрег {res['baselines_susceptibility']['LogReg']['roc_auc']:.3f}")
    print(f"[experiments] триггер по seed: ROC-AUC={sv['trig_auc_mean']:.3f}±{sv['trig_auc_std']:.3f}, "
          f"AP={sv['trig_ap_mean']:.3f}±{sv['trig_ap_std']:.3f}")
    print(f"[experiments] абляция триггера: full AP={ab['full']['ap']:.3f}, "
          f"без сейсмики AP={ab['no_seismic']['ap']:.3f}, без осадков AP={ab['no_rainfall']['ap']:.3f}")
    print(f"[experiments] калибровка Brier={brier:.4f}")
    print(f"[experiments] -> {odir / 'experiments.json'}")


if __name__ == "__main__":
    main()
