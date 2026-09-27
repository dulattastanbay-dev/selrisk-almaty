"""Быстрые проверки инвариантов пайплайна (pytest)."""
import numpy as np
import pandas as pd
import pytest

from selrisk.aoi import make_grid, assign_basin
from selrisk.dataset import build_terrain_grid, generate_meteo, make_susceptibility_training
from selrisk.features import add_antecedent_features
from selrisk.models.fusion import classify_risk, daily_risk, trigger_level
from selrisk.models.susceptibility import train_susceptibility
from selrisk.models.triggering import (
    build_rainfall_events,
    fit_id_threshold,
    predict_calibrated_trigger,
    train_calibrated_trigger,
    trigger_contributions,
)


@pytest.fixture(scope="module")
def grid():
    return build_terrain_grid(seed=1)


def test_grid_geometry():
    g = make_grid()
    assert len(g) > 1000
    assert {"cell_id", "lon", "lat", "row", "col"} <= set(g.columns)
    assert g["cell_id"].is_unique


def test_terrain_features_valid(grid):
    for col in ["elevation", "slope_deg", "twi", "dist_drainage_m", "dist_lake_m"]:
        assert col in grid.columns
        assert grid[col].notna().all()
    assert (grid["slope_deg"] >= 0).all() and (grid["slope_deg"] <= 90).all()
    assert (grid["dist_lake_m"] >= 0).all()
    assert grid["basin"].nunique() == 3


def test_antecedent_features_monotone():
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    df = pd.DataFrame({
        "date": list(dates) * 2,
        "basin": ["A"] * 40 + ["B"] * 40,
        "precip_mm": np.r_[np.random.default_rng(0).gamma(1, 4, 40),
                           np.random.default_rng(1).gamma(1, 4, 40)],
        "tmean_c": 5.0,
        "eq_max_mag_3d": 0.0,
        "eq_energy_7d": 0.0,
    })
    out = add_antecedent_features(df)
    assert (out["rain_7d"] >= out["rain_3d"] - 1e-9).all()
    assert (out["rain_3d"] >= out["rain_1d"] - 1e-9).all()
    assert (out["api"] >= 0).all()


def test_classify_risk_thresholds():
    cls = classify_risk(np.array([0.05, 0.30, 0.55, 0.90]))
    assert list(cls) == ["низкий", "умеренный", "высокий", "экстремальный"]


def test_susceptibility_trains(grid):
    train = make_susceptibility_training(grid, seed=1, n_total=600)
    assert set(train["label"].unique()) <= {0, 1}
    assert train["label"].sum() > 20                 # есть положительные примеры
    res = train_susceptibility(train, seed=1)
    assert 0.5 <= res["cv_auc"] <= 1.0               # лучше случайного
    assert len(res["importances"]) == len(res["features"])


def test_id_threshold_and_fusion(grid):
    meteo = generate_meteo(grid, seed=1, save=False)
    assert meteo["event"].sum() > 0
    events = build_rainfall_events(meteo)
    idt = fit_id_threshold(events)
    # a=10^intercept всегда > 0; b может быть любым конечным (на реальных данных
    # порог I–D нередко вырождается — это ожидаемо и обсуждается в статье)
    assert idt["a"] > 0 and np.isfinite(idt["b"])
    assert 0.0 <= idt["recall"] <= 1.0

    # слияние: риск ограничен [0,1]
    grid2 = grid.copy()
    grid2["susceptibility"] = np.linspace(0, 1, len(grid2))
    meteo["trigger_prob"] = meteo["trigger_prob_true"]
    date = meteo["date"].iloc[100]
    risk = daily_risk(grid2, meteo, date)
    assert (risk["risk"] >= 0).all() and (risk["risk"] <= 1).all()


def test_calibrated_trigger(grid):
    meteo = generate_meteo(grid, seed=1, save=False)
    res = train_calibrated_trigger(meteo)
    p = predict_calibrated_trigger(res, meteo)
    assert ((p >= 0) & (p <= 1)).all()
    # логрег без взвешивания калиброван «в среднем»: сумма вероятностей ≈ число событий
    assert abs(p.sum() - meteo["event"].sum()) < 1.0
    # вклады групп в сумме с опорным логитом дают ту же вероятность
    lr = res["model"][-1]
    logit = lr.intercept_[0] + trigger_contributions(res, meteo.head(50)).sum(axis=1)
    assert np.allclose(1 / (1 + np.exp(-logit)), p[:50])
    # значения выше наблюдавшихся не экстраполируются
    cols = ["rain_1d", "rain_3d", "rain_7d", "api"]
    wet, at_max = meteo.head(1).copy(), meteo.head(1).copy()
    wet[cols] = 1e4
    for f in cols:
        at_max[f] = res["clip_max"][res["raw_features"].index(f)]
    assert np.isclose(predict_calibrated_trigger(res, wet)[0], predict_calibrated_trigger(res, at_max)[0])
    assert list(trigger_level([0.001, 0.02, 0.1, 0.5])) == ["низкий", "умеренный", "высокий", "экстремальный"]
