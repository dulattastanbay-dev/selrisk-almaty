"""Legacy training is demo-only. Real panels undergo fail-closed validation."""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


def main(*, demo=False):
    if not demo:
        from selrisk.scientific import main as validate
        raise SystemExit(validate([]))
    import joblib
    import pandas as pd
    from selrisk.config import load_config, seed_everything
    from selrisk.dataset import build_terrain_grid
    from selrisk.models.susceptibility import predict_susceptibility, train_susceptibility
    from selrisk.models.triggering import (
        build_rainfall_events, fit_id_threshold, predict_calibrated_trigger,
        predict_trigger_prob, train_calibrated_trigger, train_triggering,
    )
    cfg = load_config()
    seed = seed_everything()
    ddir = cfg.path("sample")
    required = ["meteo_daily.csv", "susceptibility_train.csv"]
    missing = [name for name in required if not (ddir / name).is_file()]
    if missing:
        raise ValueError("Missing demo inputs: " + ", ".join(missing) + "; run scripts/run_all.py --demo")
    print(f"DEMO ONLY: {ddir}; synthetic labels are not real validation")
    grid = build_terrain_grid(seed)
    train_df = pd.read_csv(ddir / "susceptibility_train.csv")
    susc = train_susceptibility(train_df, seed)
    grid["susceptibility"] = predict_susceptibility(susc["model"], grid, susc["features"])
    meteo = pd.read_csv(ddir / "meteo_daily.csv", parse_dates=["date"])
    trg = train_triggering(meteo, seed)
    meteo["trigger_prob"] = predict_trigger_prob(trg["model_all"], meteo, trg["features"])
    cal = train_calibrated_trigger(meteo)
    meteo["trigger_prob_cal"] = predict_calibrated_trigger(cal, meteo)
    events = build_rainfall_events(meteo)
    idt = fit_id_threshold(events)
    mdir = cfg.path("models")
    odir = cfg.path("outputs")
    if mdir.name != "demo":
        mdir = mdir / "demo"
    if odir.name != "demo":
        odir = odir / "demo"
    mdir.mkdir(parents=True, exist_ok=True)
    odir.mkdir(parents=True, exist_ok=True)
    for name, obj in [("susceptibility_result", susc), ("triggering_result", trg),
                      ("susceptibility_model", susc["model"]), ("triggering_model", trg["model_all"]),
                      ("trigger_calibrated", cal)]:
        joblib.dump(obj, mdir / (name + ".joblib"))
    grid.to_csv(odir / "grid_with_susceptibility.csv", index=False)
    meteo.to_csv(odir / "meteo_with_trigger.csv", index=False)
    events.to_csv(odir / "rainfall_events.csv", index=False)
    metrics = {
        "data_status": "demo_synthetic_or_mixed_not_scientifically_validated",
        "susceptibility": {"cv_roc_auc": susc["cv_auc"], "cv_average_precision": susc["cv_ap"],
            "success_rate_auc": susc["success_rate_auc"],
            "feature_importances": susc["importances"].round(4).to_dict()},
        "triggering": {"holdout_roc_auc": trg["test_auc"], "holdout_average_precision": trg["test_ap"],
            "decision_threshold": trg["threshold"], "confusion_matrix_test": trg["confusion"],
            "permutation_importances": trg["importances"].round(4).to_dict(),
            "n_events_total": trg["n_events_total"], "n_events_test": trg["n_events_test"]},
        "trigger_calibrated": cal["metrics"], "id_threshold": idt,
    }
    (odir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Demo artifacts: {mdir}; {odir}")


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    args, rest = parser.parse_known_args(argv)
    if not args.demo:
        from selrisk.scientific import main as validate
        return validate(rest)
    if rest:
        parser.error("--demo cannot be combined with real-data arguments")
    main(demo=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
