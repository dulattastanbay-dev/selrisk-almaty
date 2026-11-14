"""Real-data research UI. Never trains or synthesizes on page open."""
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from selrisk.realdata import load_regional
from selrisk.research import lookup_date

st.set_page_config(page_title="СельРиск Алматы · v0.2",layout="wide")
st.title("СельРиск Алматы")
st.caption("Реальный датасет v0.2 · историческое presence-background ранжирование")
st.warning("Не система оповещения. Балл не является вероятностью селя и не определяет безопасность маршрута.")
kind=st.selectbox("Источник признаков",["power","stations"],
                  format_func=lambda k:"NASA POWER · 1981–2020" if k=="power" else "NOAA: станции · 1937–1991")
try:
    frame,features,root=load_regional(kind)
    artifact=json.loads((ROOT/f"web/data/model_{kind}.json").read_text(encoding="utf-8"))
    report=json.loads((ROOT/f"web/data/report_{kind}.json").read_text(encoding="utf-8"))
except (OSError,ValueError) as e:
    st.error(f"Не удалось загрузить исследовательские артефакты: {e}. Запустите python scripts/run_all.py. Синтетическая подмена запрещена.")
    st.stop()

a,b,c=st.columns(3)
a.metric("Зарегистрированные даты",int(frame.target.eq(1).sum()))
b.metric("Неизвестные исходы",int(frame.target.isna().sum()))
c.metric("Подтверждённые отрицательные",0)
st.subheader("Архивный день")
selected=st.date_input("Дата",value=date(2019,6,5) if kind=="power" else date(1983,7,1),
                       min_value=date(1900,1,1),max_value=date(2100,12,31),key=f"date_{kind}")
result=lookup_date(frame,artifact,selected)
if result["score"] is None:
    st.info("Нет данных для расчёта" if result["status"]=="no_data" else
            "Расчёт недоступен: неполные признаки или выход за обучающий диапазон")
else:
    st.metric("Исследовательский балл",f'{result["score"]:.3f}')
st.write("Разметка:", "Есть запись в каталоге" if result["ground_truth"]=="catalogue_positive" else "Исход неизвестен")
st.caption("Ноль и отрицательный балл не означают отсутствие опасности. На будущие даты прогноз не выполняется.")
row=frame[frame.date.eq(selected.isoformat())]
if not row.empty:
    st.dataframe(row[["date","split","label_status"]+features],hide_index=True)
st.subheader("Проверка на отложенных годах")
records=[]
for name,values in report["metrics"].items():
    m=values["test"]
    records.append({"Модель":name,"Proxy AP":m["proxy_average_precision"],"Proxy AUC":m["proxy_roc_auc"],
                    "Захват в верхней десятой части":m["catalogue_positive_capture_top_decile"]})
st.dataframe(pd.DataFrame(records),hide_index=True)
st.caption("Это различение зарегистрированных дат и выбранного фона, не точность прогноза. Истинные отрицательные неизвестны.")
with st.expander("Протокол и ограничения"):
    st.json(report)
st.subheader("Каталог событий")
catalogue=pd.read_csv(root/"data/event_catalogue.csv")
country=st.selectbox("Страна",["Все"]+sorted(catalogue.country.dropna().unique().tolist()))
view=catalogue if country=="Все" else catalogue[catalogue.country.eq(country)]
st.dataframe(view[["event_id","source_date","country","basin","event_type","date_precision","source_url"]],hide_index=True)
st.download_button("Скачать каталог CSV",catalogue.to_csv(index=False),file_name="selrisk-events-v02.csv",mime="text/csv")
st.download_button("Скачать протокол JSON",json.dumps(report,ensure_ascii=False,indent=2),file_name=f"report-{kind}.json",mime="application/json")
st.markdown("[Данные Reading](https://www.nature.com/articles/s44304-024-00050-7) · "
            "[NASA POWER](https://power.larc.nasa.gov/docs/referencing/) · "
            "[NOAA GHCN-Daily](https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt)")
