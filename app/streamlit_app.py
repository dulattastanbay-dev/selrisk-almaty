"""Интерактивное демо: вероятность схода селя над Алматы на выбранную дату.

Запуск:  streamlit run app/streamlit_app.py
Перед запуском выполните пайплайн:  python scripts/run_all.py

Проценты — откалиброванная суточная вероятность схода в бассейне, та же модель,
что на сайте web/ (selrisk.models.triggering.train_calibrated_trigger).
Риск ячейки = подверженность склона × вероятность схода в её бассейне.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LogNorm

from selrisk.aoi import aoi_bbox
from selrisk.config import load_config
from selrisk.models.fusion import trigger_level
from selrisk.models.triggering import (
    predict_calibrated_trigger,
    train_calibrated_trigger,
    trigger_contributions,
)
from selrisk.viz import _grid_to_2d, _overlay_context

st.set_page_config(page_title="SelRisk-Almaty", page_icon="⛰️", layout="wide")
cfg = load_config()
ODIR = cfg.path("outputs")
BASINS = [b["name"] for b in cfg["basins"]]
BASIN_COLORS = dict(zip(BASINS, ["#2563eb", "#8b3fd9", "#0f8a7e"]))   # как на сайте
LEVELS = cfg["models"]["calibrated_trigger"]["levels"]
LEVEL_COLORS = {lv["name"]: lv["color"] for lv in LEVELS}
RISK_MIN, RISK_MAX = 0.001, 0.5          # шкала карты риска: 0,1 % … 50 % (логарифм.)
COND = {                                 # признак -> (подпись, единицы, знаков после запятой)
    "rain_1d": ("Осадки за сутки", "мм", 1),
    "rain_3d": ("Осадки за 3 дня", "мм", 1),
    "rain_7d": ("Осадки за 7 дней", "мм", 1),
    "api": ("Индекс увлажнённости грунта", "мм", 0),
    "pdd_3d": ("Тепло за 3 дня (сумма t > 0)", "°C·сут", 0),
    "swe_mm": ("Запас воды в снеге", "мм", 0),
    "snowmelt_mm": ("Таяние снега за сутки", "мм", 1),
    "eq_max_mag_3d": ("Макс. магнитуда за 3 дня", "M", 1),
}


# ------------------------------------------------------------------ формат
def comma(x: float, d: int) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def fmt_pct(p: float) -> str:
    v = p * 100
    if v < 0.1:
        return "<0,1 %"
    if v > 99.5:
        return ">99 %"
    return (comma(v, 1) if v < 10 else str(int(v + 0.5))) + " %"   # округление как на сайте


def times(x: float) -> str:
    """«1,5 раза», «3 раза», «25 раз»."""
    if x < 10 and round(x, 1) != round(x):
        return comma(x, 1) + " раза"
    n = round(x)
    few = 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14
    return f"{n} раза" if few else f"{n} раз"


def ratio_text(p: float, base: float) -> str:
    r = p / base
    if r >= 1.5:
        return f"в {times(r)} выше среднего дня"
    if r <= 0.67:
        return f"в {times(1 / r)} ниже среднего дня"
    return "как в средний день"


def fmt_mult(v: float) -> str:
    """Множитель шансов exp(v): «×12», «÷3,4»."""
    if abs(v) < 0.05:
        return "≈ 1"
    m = np.exp(abs(v))
    s = "1000+" if m >= 1000 else f"{m:.0f}" if m >= 10 else comma(m, 1)
    return ("×" if v >= 0 else "÷") + s


def show(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------------------------------------------ данные
@st.cache_data
def load_data():
    grid = pd.read_csv(ODIR / "grid_with_susceptibility.csv")
    meteo = pd.read_csv(ODIR / "meteo_with_trigger.csv", parse_dates=["date"])
    return grid, meteo


@st.cache_resource
def load_model():
    """Откалиброванная модель из train.py; если её ещё нет — обучаем здесь (секунды)."""
    path = cfg.path("models") / "trigger_calibrated.joblib"
    if path.exists():
        return joblib.load(path)
    return train_calibrated_trigger(load_data()[1])


# ------------------------------------------------------------------ графики
def risk_figure(grid, p_by_basin: pd.Series, date):
    lon0, lat0, lon1, lat1 = aoi_bbox()
    risk = grid["susceptibility"].to_numpy() * grid["basin"].map(p_by_basin).to_numpy()
    arr = _grid_to_2d(grid, np.clip(risk, RISK_MIN, RISK_MAX))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    im = ax.imshow(arr, origin="lower", extent=[lon0, lon1, lat0, lat1], aspect="auto",
                   cmap="YlOrRd", norm=LogNorm(RISK_MIN, RISK_MAX))
    _overlay_context(ax)
    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ax.set_title(f"Риск в ячейке за сутки — {date:%d.%m.%Y}")
    ax.set_xlabel("долгота"); ax.set_ylabel("широта")
    cb = fig.colorbar(im, ax=ax, shrink=0.85, label="подверженность × вероятность")
    cb.ax.minorticks_off()
    cb.set_ticks([0.001, 0.01, 0.05, 0.2, 0.5], labels=["0,1 %", "1 %", "5 %", "20 %", "50 %"])
    fig.tight_layout()
    return fig


def factor_figure(contrib: pd.Series):
    names = list(contrib.index)[::-1]
    vals = contrib.to_numpy()[::-1]
    lim = max(2.0, float(np.abs(vals).max()) * 1.4)
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.barh(names, vals, color=["#e0592a" if v >= 0 else "#3a8fd9" for v in vals], height=0.6)
    ax.axvline(0, color="0.3", lw=0.8)
    for y, v in enumerate(vals):
        ax.text(v + (0.04 if v >= 0 else -0.04) * lim, y, fmt_mult(v), va="center",
                ha="left" if v >= 0 else "right", fontsize=9, fontweight="bold")
    ax.set_xlim(-lim, lim)
    ax.set_xticks([])
    ax.grid(False)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def timeline_figure(meteo, date):
    w = meteo[(meteo["date"] - date).abs() <= pd.Timedelta(days=45)]
    ymax = max(float(w["p"].max()) * 1.12, 0.02) * 100
    fig, ax = plt.subplots(figsize=(10, 3.0))
    for b in BASINS:
        s = w[w["basin"] == b]
        ax.plot(s["date"], s["p"] * 100, color=BASIN_COLORS[b], lw=1.8, label=b)
        e = s[s["event"] == 1]
        ax.scatter(e["date"], e["p"] * 100, s=36, facecolors="white", edgecolors="k", zorder=5)
    ax.scatter([], [], s=36, facecolors="white", edgecolors="k", label="событие в данных")
    for lv in LEVELS[:-1]:
        if lv["max"] * 100 < ymax:
            ax.axhline(lv["max"] * 100, color=lv["color"], ls="--", lw=1)
    ax.axvline(date, color="0.2", lw=1.2, alpha=0.6)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("вероятность, %")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.legend(fontsize=8, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ страница
try:
    grid, meteo = load_data()
except FileNotFoundError:
    st.error("Нет результатов. Сначала выполните: `python scripts/run_all.py`")
    st.stop()
cal = load_model()
meteo = meteo.assign(p=predict_calibrated_trigger(cal, meteo))

st.title("⛰️ SelRisk-Almaty — вероятность схода селя над Алматы")
st.caption("Откалиброванная суточная вероятность схода в бассейне (та же модель, что на сайте) "
           "× подверженность склонов. Реальные данные: рельеф Copernicus GLO-30, погода "
           "ERA5-Land 2016–2023, сейсмика USGS. Метки сходов — демонстрационные.")

# --- боковая панель: выбор даты ---
dmin, dmax = meteo["date"].min().date(), meteo["date"].max().date()
top = meteo.sort_values("p", ascending=False).drop_duplicates("date").head(6)
events = meteo[meteo["event"] == 1].sort_values("date")
presets = {"— вручную —": None}
presets.update({f"Макс. вероятность: {r.date:%d.%m.%Y} · {r.basin} ({fmt_pct(r.p)})": r.date.date()
                for r in top.itertuples()})
presets.update({f"Событие в данных: {r.date:%d.%m.%Y} · {r.basin}": r.date.date()
                for r in events.itertuples()})

with st.sidebar:
    st.header("Дата оценки")
    choice = st.selectbox("Быстрый выбор", list(presets))
    default = presets[choice] or pd.Timestamp("2017-06-15").date()
    date = st.date_input("Дата", value=default, min_value=dmin, max_value=dmax, format="DD.MM.YYYY")
    st.caption("Погода — реальный реанализ ERA5-Land по каждому бассейну. Свои значения "
               "и живой прогноз — на [сайте-калькуляторе]"
               "(https://dulattastanbay-dev.github.io/selrisk-almaty/).")

date = pd.Timestamp(date)
day = meteo[meteo["date"] == date].set_index("basin").reindex(BASINS)
if day["p"].isna().any():
    st.warning("Нет данных на выбранную дату.")
    st.stop()
p = day["p"]

# --- карточки по бассейнам ---
st.subheader(f"Вероятность схода селя за сутки · {date:%d.%m.%Y}")
for col, b, lvl in zip(st.columns(len(BASINS)), BASINS, trigger_level(p.to_numpy())):
    col.metric(b, fmt_pct(p[b]), border=True)
    col.markdown(f"<span style='color:{LEVEL_COLORS[lvl]};font-weight:700'>■ {lvl}</span>"
                 f" · {ratio_text(p[b], cal['base_rate'])}", unsafe_allow_html=True)
ev_today = [b for b in BASINS if day.loc[b, "event"] == 1]
if ev_today:
    st.info("В обучающих данных в этот день — событие: " + ", ".join(ev_today)
            + " (демонстрационная метка).")

left, right = st.columns([3, 2])
with left:
    show(risk_figure(grid, p, date))
with right:
    sel = st.radio("Что влияет на вероятность — бассейн", BASINS,
                   index=int(np.argmax(p.to_numpy())), horizontal=True)
    contrib = trigger_contributions(cal, day.reset_index()).iloc[BASINS.index(sel)]
    st.caption("Во сколько раз каждый фактор меняет шансы схода по сравнению с днём "
               "со средними условиями 2016–2023.")
    show(factor_figure(contrib))

st.markdown("**Вероятность по дням — ±45 дней вокруг выбранной даты**")
show(timeline_figure(meteo, date))

st.markdown("**Условия выбранного дня**")
rows = {f"{label}, {unit}": [comma(day.loc[b, f], d) for b in BASINS]
        for f, (label, unit, d) in COND.items()}
rows["Вероятность схода за сутки"] = [fmt_pct(p[b]) for b in BASINS]
st.dataframe(pd.DataFrame(rows, index=BASINS).T, width="stretch")

with st.expander("Статическая карта подверженности склонов (модель «где»)"):
    lon0, lat0, lon1, lat1 = aoi_bbox()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    arr = _grid_to_2d(grid, grid["susceptibility"].to_numpy())
    im = ax.imshow(arr, origin="lower", extent=[lon0, lon1, lat0, lat1],
                   aspect="auto", cmap="YlOrRd")
    _overlay_context(ax)
    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
    ax.legend(loc="upper right", fontsize=7)
    fig.colorbar(im, ax=ax, label="P(очаг)", shrink=0.85)
    show(fig)

with st.expander("О модели и ограничения"):
    m = cal["metrics"]
    lo = 0.0
    lv_txt = []
    for lv in LEVELS:
        lv_txt.append(f"{lv['name']} — ≥ {lo * 100:.0f} %" if lv["max"] > 1
                      else f"{lv['name']} — {lo * 100:.0f}–{lv['max'] * 100:.0f} %")
        lo = lv["max"]
    st.markdown(
        f"- **Когда:** логистическая регрессия без взвешивания классов по осадкам за 1/3/7 дней, "
        f"увлажнённости, теплу, снегу, таянию, сейсмике и сезону. Проверка на 2020–2023: "
        f"ROC-AUC {comma(m['holdout_roc_auc'], 2)}, ожидала {comma(m['holdout_expected_events'], 1)} "
        f"событий при {m['holdout_events']} фактических.\n"
        f"- **Где:** Random Forest по рельефу, руслам и моренным озёрам.\n"
        f"- **Уровни за сутки:** {'; '.join(lv_txt)}. Средний день — "
        f"{comma(cal['base_rate'] * 100, 2)} %.\n"
        f"- **Метки сходов демонстрационные** (порождены физической моделью по реальной погоде), "
        f"поэтому проценты показывают работу метода, а не официальный прогноз. "
        f"Следуйте сообщениям МЧС РК и Казселезащиты.\n"
        f"- Значения вне диапазона 2016–2023 не экстраполируются."
    )
