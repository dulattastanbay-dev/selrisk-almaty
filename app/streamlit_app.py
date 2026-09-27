"""Интерактивное демо: оценка риска селей над Алматы на выбранную дату.

Запуск:  streamlit run app/streamlit_app.py
Перед запуском выполните пайплайн:  python scripts/run_all.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from selrisk.aoi import aoi_bbox
from selrisk.config import load_config
from selrisk.models.fusion import daily_risk, risk_color_map
from selrisk.viz import _grid_to_2d, _overlay_context

st.set_page_config(page_title="SelRisk-Almaty", page_icon="⛰️", layout="wide")
cfg = load_config()
ODIR = cfg.path("outputs")


@st.cache_data
def load_data():
    grid = pd.read_csv(ODIR / "grid_with_susceptibility.csv")
    meteo = pd.read_csv(ODIR / "meteo_with_trigger.csv", parse_dates=["date"])
    return grid, meteo


def risk_figure(risk, grid, date):
    lon0, lat0, lon1, lat1 = aoi_bbox()
    arr = _grid_to_2d(grid, risk["risk"].to_numpy())
    fig, ax = plt.subplots(figsize=(8, 4.6))
    im = ax.imshow(arr, origin="lower", extent=[lon0, lon1, lat0, lat1],
                   aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
    _overlay_context(ax)
    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ax.set_title(f"Карта риска селей — {pd.Timestamp(date).date()}")
    ax.set_xlabel("долгота"); ax.set_ylabel("широта")
    fig.colorbar(im, ax=ax, label="риск (0..1)", shrink=0.85)
    fig.tight_layout()
    return fig


try:
    grid, meteo = load_data()
except FileNotFoundError:
    st.error("Нет результатов. Сначала выполните: `python scripts/run_all.py`")
    st.stop()

st.title("⛰️ SelRisk-Almaty — риск селей и оползней над Алматы")
st.caption("Двухуровневая модель: подверженность склонов (где) × вероятность "
           "триггера (когда). Реальные данные: рельеф Copernicus GLO-30, осадки "
           "ERA5-Land 2016–2023, сейсмика USGS. Метки сходов — демонстрационные.")

# --- боковая панель: выбор даты ---
dmin, dmax = meteo["date"].min().date(), meteo["date"].max().date()
hist = pd.read_csv(cfg.path("events") / "historical_events.csv", parse_dates=["date"])
hist_in = hist[(hist["date"].dt.date >= dmin) & (hist["date"].dt.date <= dmax)]

with st.sidebar:
    st.header("Дата оценки")
    preset = st.selectbox(
        "Быстрый выбор",
        ["— вручную —", "День максимального риска"]
        + [f"Ист. сход: {r.date.date()} ({r.basin})" for r in hist_in.itertuples()],
    )
    if preset == "День максимального риска":
        default = meteo.loc[meteo["trigger_prob"].idxmax(), "date"].date()
    elif preset.startswith("Ист. сход"):
        default = pd.Timestamp(preset.split(": ")[1].split(" ")[0]).date()
    else:
        default = pd.Timestamp("2017-06-15").date()
    date = st.date_input("Дата", value=default, min_value=dmin, max_value=dmax)

date = pd.Timestamp(date)
if date not in set(meteo["date"]):
    st.warning("Нет данных на выбранную дату.")
    st.stop()

risk = daily_risk(grid, meteo, date)
by_basin = (
    risk.groupby("basin")
    .agg(trigger=("trigger_prob", "first"), max_risk=("risk", "max"),
         susc=("susceptibility", "mean"))
    .sort_values("max_risk", ascending=False)
)

# --- карточки по бассейнам ---
st.subheader(f"Оценка на {date.date()}")
cols = st.columns(len(by_basin))
colors = risk_color_map()
for col, (basin, r) in zip(cols, by_basin.iterrows()):
    lvl = ("экстремальный" if r.max_risk > 0.7 else "высокий" if r.max_risk > 0.4
           else "умеренный" if r.max_risk > 0.15 else "низкий")
    col.metric(basin, f"{r.max_risk:.2f}", help="максимальный риск по бассейну")
    col.markdown(f"<span style='color:{colors[lvl]};font-weight:600'>{lvl.upper()}</span>"
                 f" · P(триггер)={r.trigger:.2f}", unsafe_allow_html=True)

left, right = st.columns([3, 2])
with left:
    st.pyplot(risk_figure(risk, grid, date))
with right:
    st.markdown("**Факторы триггера в этот день (по бассейнам)**")
    drivers = meteo[meteo["date"] == date][
        ["basin", "rain_3d", "api", "snowmelt_mm", "eq_max_mag_3d", "trigger_prob"]
    ].round(2).set_index("basin")
    drivers.columns = ["дождь 3д, мм", "увлажн. API", "снеготаяние, мм",
                       "макс. M (3д)", "P(триггер)"]
    st.dataframe(drivers, use_container_width=True)
    st.markdown(
        "- **Осадки за 3 дня** — главный триггерный фактор (см. важности).\n"
        "- **Сейсмика** — вторична: на реальных данных USGS её вклад близок к нулю.\n"
        "- Риск ячейки = подверженность × P(триггер) её бассейна."
    )

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
    st.pyplot(fig)
