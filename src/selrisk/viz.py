"""Построение карт и графиков результатов (matplotlib, backend Agg)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .aoi import aoi_bbox, basins
from .config import load_config
from .data_ingest.glacial_lakes import load_lakes

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def _grid_to_2d(grid: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    nlat = int(grid["row"].max()) + 1
    nlon = int(grid["col"].max()) + 1
    arr = np.full((nlat, nlon), np.nan)
    arr[grid["row"].to_numpy(), grid["col"].to_numpy()] = values
    return arr


def _overlay_context(ax, lakes=True, events=True):
    b = basins()
    ax.scatter(b["lon"], b["lat"], marker="^", s=70, c="k", zorder=5)
    for r in b.itertuples():
        ax.annotate(r.name, (r.lon, r.lat), fontsize=8, xytext=(4, 4),
                    textcoords="offset points", zorder=6)
    if lakes:
        lk = load_lakes()
        ax.scatter(lk["lon"], lk["lat"], marker="o", s=30, facecolors="none",
                   edgecolors="cyan", linewidths=1.4, zorder=5, label="ледниковые озёра")
    if events:
        cfg = load_config()
        hist = pd.read_csv(cfg.path("events") / "historical_events.csv")
        ax.scatter(hist["source_lon"], hist["source_lat"], marker="x", s=60,
                   c="magenta", linewidths=1.6, zorder=7, label="ист. сходы")


def map_field(grid, values, title, path, cmap="viridis", label="", context=True):
    lon0, lat0, lon1, lat1 = aoi_bbox()
    arr = _grid_to_2d(grid, values)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    im = ax.imshow(arr, origin="lower", extent=[lon0, lon1, lat0, lat1],
                   aspect="auto", cmap=cmap)
    if context:
        _overlay_context(ax)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_title(title)
    ax.set_xlabel("долгота")
    ax.set_ylabel("широта")
    fig.colorbar(im, ax=ax, label=label, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_success_curve(area_frac, cum_pos, auc, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(area_frac, cum_pos, lw=2, label=f"модель (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", c="grey", label="случайная")
    ax.set_xlabel("доля территории (от высшего риска)")
    ax.set_ylabel("доля покрытых очагов")
    ax.set_title("Кривая успешности (success-rate)")
    ax.legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def plot_feature_importance(importances: pd.Series, path, title="Важность признаков"):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    importances[::-1].plot.barh(ax=ax, color="#3182bd")
    ax.set_title(title); ax.set_xlabel("важность (Gini)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def plot_roc_pr(y, proba, auc, ap, path):
    from sklearn.metrics import roc_curve, precision_recall_curve
    fpr, tpr, _ = roc_curve(y, proba)
    prec, rec, _ = precision_recall_curve(y, proba)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3))
    axes[0].plot(fpr, tpr, lw=2, label=f"AUC={auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", c="grey")
    axes[0].set_title("ROC (триггер, holdout)"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].legend()
    axes[1].plot(rec, prec, lw=2, color="#e6550d", label=f"AP={ap:.3f}")
    axes[1].set_title("Precision–Recall"); axes[1].set_xlabel("полнота"); axes[1].set_ylabel("точность")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def plot_id_threshold(events: pd.DataFrame, a: float, b: float, path):
    fig, ax = plt.subplots(figsize=(6, 4.8))
    ev0 = events[events["triggered"] == 0]
    ev1 = events[events["triggered"] == 1]
    ax.scatter(ev0["D_days"], ev0["I_mm_day"], s=18, c="#9ecae1", label="без схода", alpha=0.6)
    ax.scatter(ev1["D_days"], ev1["I_mm_day"], s=40, c="#de2d26", label="сход", zorder=5)
    D = np.linspace(events["D_days"].min(), events["D_days"].max(), 100)
    ax.plot(D, a * D ** (-b), "k-", lw=2, label=f"порог I={a:.1f}·D^(-{b:.2f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("длительность дождя D, сут")
    ax.set_ylabel("интенсивность I, мм/сут")
    ax.set_title("Порог интенсивность–длительность осадков")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def plot_event_timeline(meteo_pred: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    agg = meteo_pred.groupby("date")["trigger_prob"].max().reset_index()
    ax.plot(agg["date"], agg["trigger_prob"], lw=0.7, c="#2b8cbe", label="макс. P(триггер)")
    ev = meteo_pred[meteo_pred["event"] == 1]
    ax.scatter(ev["date"], np.full(len(ev), 1.02), marker="v", c="#de2d26", s=25,
               label="фактический сход", clip_on=False)
    ax.set_ylim(0, 1.05)
    ax.set_title("Суточная вероятность триггера и фактические сходы")
    ax.set_ylabel("P(триггер)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def plot_reliability(frac_pos, mean_pred, brier, path):
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], "--", c="grey", label="идеальная калибровка")
    ax.plot(mean_pred, frac_pos, "o-", lw=2, color="#2b8cbe",
            label=f"модель (Brier={brier:.3f})")
    ax.set_xlabel("средняя предсказанная вероятность")
    ax.set_ylabel("наблюдаемая доля событий")
    ax.set_title("Калибровка вероятности триггера")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def plot_metric_bars(groups: dict, title, path, metrics=("roc_auc", "ap")):
    labels = list(groups.keys())
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for i, m in enumerate(metrics):
        vals = [groups[l].get(m, np.nan) for l in labels]
        ax.bar(x + (i - 0.5) * w, vals, w, label=m.upper().replace("ROC_AUC", "ROC-AUC"))
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=9)
    ax.set_ylim(0, 1.02); ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def plot_seed_variance(var_df: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    cols = ["susc_auc", "susc_ap", "trig_auc", "trig_ap"]
    names = ["подвер.\nROC-AUC", "подвер.\nAP", "триггер\nROC-AUC", "триггер\nAP"]
    means = [var_df[c].mean() for c in cols]
    stds = [var_df[c].std() for c in cols]
    x = np.arange(len(cols))
    ax.bar(x, means, yerr=stds, capsize=6, color=["#3d5a4c", "#7fa99b", "#c0392b", "#e6a08f"])
    for xi, (m, s) in enumerate(zip(means, stds)):
        ax.text(xi, m + s + 0.02, f"{m:.2f}±{s:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Устойчивость метрик по {len(var_df)} seed (mean ± std)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def plot_risk_map(risk_df: pd.DataFrame, grid: pd.DataFrame, date, path):
    from .models.fusion import risk_color_map
    lon0, lat0, lon1, lat1 = aoi_bbox()
    arr = _grid_to_2d(grid, risk_df["risk"].to_numpy())
    fig, ax = plt.subplots(figsize=(8, 4.6))
    im = ax.imshow(arr, origin="lower", extent=[lon0, lon1, lat0, lat1],
                   aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
    _overlay_context(ax)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_title(f"Карта риска селей — {pd.Timestamp(date).date()}")
    ax.set_xlabel("долгота"); ax.set_ylabel("широта")
    fig.colorbar(im, ax=ax, label="риск (0..1)", shrink=0.85)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path
