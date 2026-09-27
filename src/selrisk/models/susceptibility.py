"""Модель подверженности (susceptibility): где в принципе возможен сход.

Random Forest по статическим признакам рельефа и близости к руслам/озёрам.
Оценка — блочная (пространственная) кросс-валидация: ячейки одного
пространственного блока не попадают одновременно в train и test, иначе
пространственная автокорреляция завышает метрики.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score

from ..config import load_config


def spatial_blocks(df: pd.DataFrame, n_blocks_axis: int = 5) -> np.ndarray:
    """Идентификатор пространственного блока по квантильной решётке lon×lat."""
    def binned(col):
        edges = np.quantile(df[col], np.linspace(0, 1, n_blocks_axis + 1)[1:-1])
        return np.digitize(df[col], edges)

    return binned("lon") * (n_blocks_axis + 1) + binned("lat")


def train_susceptibility(train: pd.DataFrame, seed: int | None = None,
                         features: list[str] | None = None) -> dict:
    """Обучает RF и оценивает блочной CV. Возвращает модель, метрики, важности.

    features — необязательное подмножество признаков (для экспериментов абляции);
    по умолчанию берётся полный список из config.yaml.
    """
    cfg = load_config()
    mcfg = cfg["models"]["susceptibility"]
    seed = seed if seed is not None else cfg["project"]["random_seed"]
    feats = features if features is not None else mcfg["features"]

    X = train[feats].to_numpy()
    y = train["label"].to_numpy()
    groups = spatial_blocks(train)

    model = RandomForestClassifier(
        n_estimators=mcfg["n_estimators"],
        max_depth=mcfg["max_depth"],
        min_samples_leaf=mcfg["min_samples_leaf"],
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )

    cv = GroupKFold(n_splits=mcfg["n_spatial_folds"])
    oof = cross_val_predict(model, X, y, cv=cv, groups=groups,
                            method="predict_proba", n_jobs=-1)[:, 1]
    auc = roc_auc_score(y, oof)
    ap = average_precision_score(y, oof)

    # успех-рейт кривая: доля реальных очагов, покрытых верхними долями карты
    order = np.argsort(oof)[::-1]
    cum_pos = np.cumsum(y[order]) / y.sum()
    area_frac = np.arange(1, len(y) + 1) / len(y)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))  # numpy<2.0: trapz
    success_auc = _trapz(cum_pos, area_frac)

    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)

    return {
        "model": model,
        "features": feats,
        "cv_auc": float(auc),
        "cv_ap": float(ap),
        "success_rate_auc": float(success_auc),
        "oof": oof,
        "success_curve": (area_frac, cum_pos),
        "importances": importances,
    }


def predict_susceptibility(model: RandomForestClassifier, grid: pd.DataFrame,
                           features: list[str]) -> np.ndarray:
    """Вероятность подверженности для каждой ячейки сетки (0..1)."""
    return model.predict_proba(grid[features].to_numpy())[:, 1]
