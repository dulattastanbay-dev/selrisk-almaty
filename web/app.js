"use strict";
const $=id=>document.getElementById(id), ns="http://www.w3.org/2000/svg";
let kind="power",model,report,archive=[],events=[],centers,filtered=[],pageNo=0,mode="archive",request=0;
const names={precipitation_lag2utc_mm:"Осадки D−2 UTC, мм",temperature_mean_lag2utc_c:"Средняя температура D−2 UTC, °C",temperature_min_lag2utc_c:"Минимальная температура, °C",temperature_max_lag2utc_c:"Максимальная температура, °C",precipitation_prev3d_mm:"Осадки за 3 дня, мм",precipitation_prev7d_mm:"Осадки за 7 дней, мм",precipitation_prev14d_mm:"Осадки за 14 дней, мм",temperature_mean_prev7d_c:"Средняя температура за 7 дней, °C",positive_degree_days_prev7d:"Положительные градусо-дни за 7 дней",day_of_year_sin:"Сезон: sin",day_of_year_cos:"Сезон: cos"};
function label(k){if(names[k])return names[k];const m=k.match(/(.+)_precipitation_prev(\d+)d_lag2_mm/);if(m)return `${({mynzhylki:"Мынжылки",bao:"БАО",gorelnik:"Горельник",almaty:"Алматы"})[m[1]]||m[1]}: осадки за ${m[2]} дн., мм`;return k;}
async function get(path){const r=await fetch(path);if(!r.ok)throw Error(`Не удалось загрузить ${path}: HTTP ${r.status}`);return r.json();}
function el(tag,text){const x=document.createElement(tag);if(text!==undefined)x.textContent=text;return x;}
function sourceLink(url,text="Открыть ↗"){const a=el("a",text);if(typeof url==="string"&&/^https:\/\//.test(url)){a.href=url;a.target="_blank";a.rel="noopener noreferrer";}return a;}
function setResult(result,row,customNote){
  $("score").textContent=result.score===null?"—":result.score.toFixed(3);
  $("score-state").textContent=({available:mode==="manual"?"Сценарий внутри диапазона":"Исторический исследовательский балл",no_data:"Нет данных на эту дату",missing_features:"Недостаточно погодных признаков",out_of_training_domain:"Вне обучающего диапазона",unavailable:"Оперативный расчёт отключён",invalid:"Некорректный сценарий"})[result.status]||result.status;
  $("score-note").textContent=customNote||"Чем выше балл, тем сильнее сходство с зарегистрированными датами внутри данного эксперимента. Число не является вероятностью и несопоставимо между двумя моделями.";
  $("day-meta").replaceChildren();$("feature-values").replaceChildren();
  if(row){let meta=[["Дата",row.date||$("scenario-date").value],["Разметка",mode==="manual"?"Учебный сценарий":row.target===1?"Есть запись в каталоге":"Исход неизвестен"],["Часть",row.split||"Не участвует в оценке"],["Окно данных",row.feature_window_end_utc?`строго до ${row.feature_window_end_utc}`:row.feature_window_last_observation_date?`наблюдения до ${row.feature_window_last_observation_date}`:"Задано пользователем"]];
    for(const [k,v]of meta){$("day-meta").append(el("dt",k),el("dd",v));}
    for(const k of model.features)$("feature-values").append(el("div",`${label(k)}: ${row[k]===null||row[k]===undefined?"нет данных":Number(row[k]).toFixed(3)}`));
  }
}
function inspect(){const date=$("date").value,row=archive.find(x=>x.date===date);if(!date){setResult({score:null,status:"invalid"},null,"Укажите дату.");return;}setResult(row?SelRiskModel.score(model,row):{score:null,status:"no_data"},row);}
function switchMode(next){
  mode=next;
  for(const x of ["archive","manual","live"]){$(x+"-panel").hidden=x!==mode;$("tab-"+x).setAttribute("aria-selected",String(x===mode));}
  if(!model)return;
  if(mode==="archive")inspect();
  else if(mode==="live")setResult({score:null,status:"unavailable"},null,"Нет проверенной оперативной модели и входного потока. Отсутствие расчёта не означает низкий риск.");
  else setResult({score:null,status:"invalid"},null,"Заполните сценарий и нажмите «Рассчитать балл». По умолчанию подставлены наблюдения выбранной архивной даты, если они полны.");
}
async function loadKind(){
  const mine=++request;kind=$("dataset").value;
  model=null;
  $("score").textContent="…";$("score-state").textContent="Загрузка";$("inspect").disabled=true;
  $("day-meta").replaceChildren();$("feature-values").replaceChildren();
  try{
    const [m,r,a]=await Promise.all([get(`data/model_${kind}.json`),get(`data/report_${kind}.json`),get(`data/archive_${kind}.json`)]);
    if(mine!==request)return;
    const rows=a.data.map(values=>Object.fromEntries(a.columns.map((key,i)=>[key,values[i]])));
    model=m;report=r;archive=rows;$("load-error").hidden=true;
    $("positive-count").textContent=r.contract.positive_days.toLocaleString("ru");
    $("unknown-count").textContent=r.contract.unknown_days.toLocaleString("ru");
    $("period-note").textContent=`Доступны ${rows[0].date} — ${rows.at(-1).date}. Другие даты останутся без расчёта.`;
    $("date").value=kind==="power"?"2019-06-05":"1983-07-01";$("scenario-date").value=$("date").value;
    $("manual-fields").replaceChildren();
    const initial=rows.find(x=>x.date===$("date").value&&x.feature_complete)||rows.find(x=>x.feature_complete);
    for(const k of m.features.filter(k=>!k.startsWith("day_of_year"))){const wrap=el("label",label(k)),input=el("input");input.type="number";input.step="any";input.required=true;input.dataset.feature=k;input.value=initial[k];wrap.append(input);$("manual-fields").append(wrap);}
    $("metrics").replaceChildren();
    const display={presence_background:"Погода + сезон",rainfall_3day:"Только осадки за 3 дня",season_only:"Только сезон"};
    for(const [k,v]of Object.entries(r.metrics)){const tr=el("tr"),t=v.test;for(const x of [display[k],t.proxy_average_precision.toFixed(4),t.proxy_roc_auc.toFixed(4),t.catalogue_positive_capture_top_decile.toFixed(3),t.actual_selected_fraction.toFixed(3)])tr.append(el("td",x));$("metrics").append(tr);}
    const t=r.metrics.presence_background.test;
    $("test-size").textContent=`Test: ${t.n_days} допустимых дней, ${t.catalogue_positive_days} зарегистрированных дат. Доли указаны от 0 до 1. Все модели сравниваются на одинаковых днях. Пропущено положительных дней из-за неполных признаков: ${r.excluded_incomplete_positive_days}.`;
    $("split-description").textContent=kind==="power"?"Train: 1981–2005 · validation: 2006–2012 · test: 2013–2020. Признаки используют D−2 UTC и ранее.":"Train: 1937–1973 · validation: 1974–1981 · test: 1982–1991. Последнее наблюдение в признаках: D−2.";
    for(const key of ["model","report"])$(key+"-download").href=`data/${key}_${kind}.json`;
    $("pred-download").href=`data/predictions_${kind}.csv`;
    $("inspect").disabled=false;switchMode(mode);
  }catch(e){if(mine!==request)return;model=null;$("load-error").hidden=false;$("load-error").textContent=e.message;setResult({score:null,status:"no_data"},null,"Расчёт недоступен: не удалось загрузить данные. Старые значения не используются.");}
}
function svgEl(tag,attrs,text){const x=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))x.setAttribute(k,v);if(text)x.textContent=text;return x;}
function showMapDetail(p){$("map-detail").replaceChildren(el("span",`${p.event_id||p.center_id}: ${p.basin||"бассейн не указан"} · ${p.source_date||p.name||""} `),sourceLink(p.source_url));}
function drawMap(){
  const svg=$("map");svg.replaceChildren();
  const bounds=[76.65,42.65,77.8,43.5],x=lon=>45+(lon-bounds[0])/(bounds[2]-bounds[0])*920,y=lat=>410-(lat-bounds[1])/(bounds[3]-bounds[1])*380;
  for(let lon=76.7;lon<=77.81;lon+=.2){svg.append(svgEl("line",{x1:x(lon),x2:x(lon),y1:20,y2:415,stroke:"#d3ddcf"}),svgEl("text",{x:x(lon),y:440},`${lon.toFixed(1)}° E`));}
  for(let lat=42.8;lat<43.51;lat+=.2){svg.append(svgEl("line",{x1:45,x2:965,y1:y(lat),y2:y(lat),stroke:"#d3ddcf"}),svgEl("text",{x:2,y:y(lat)-5},`${lat.toFixed(1)}°`));}
  for(const f of centers.features){const g=f.geometry;if(!g)continue;const polygons=g.type==="MultiPolygon"?g.coordinates:[g.coordinates];let d="";for(const rings of polygons)for(const ring of rings)d+=ring.map(([lon,lat],i)=>`${i?"L":"M"}${x(lon)},${y(lat)}`).join("")+"Z";const path=svgEl("path",{d,"fill-rule":"evenodd",tabindex:0,role:"button","aria-label":`Очаг ${f.properties.name}`});path.append(svgEl("title",{},`Очаг ${f.properties.name}; не граница водосбора`));path.onclick=()=>showMapDetail(f.properties);path.onkeydown=e=>{if(e.key==="Enter")showMapDetail(f.properties);};svg.append(path);}
  const local=filtered.filter(e=>e.longitude>=bounds[0]&&e.longitude<=bounds[2]&&e.latitude>=bounds[1]&&e.latitude<=bounds[3]);
  for(const p of local){const c=svgEl("circle",{cx:x(p.longitude),cy:y(p.latitude),r:3.4,tabindex:0,role:"button","aria-label":p.event_id});c.append(svgEl("title",{},`${p.event_id} · ${p.source_date||"дата неизвестна"}`));c.onclick=()=>showMapDetail(p);c.onkeydown=e=>{if(e.key==="Enter")showMapDetail(p);};svg.append(c);}
  svg.append(svgEl("text",{x:65,y:35},`${local.length} записей внутри показанного региона`));
}
function filterEvents(){
  const search=$("search").value.toLowerCase().trim();
  filtered=events.filter(e=>(!$("country").value||e.country===$("country").value)&&(!$("basin").value||e.basin===$("basin").value)&&(!$("event-type").value||e.event_type===$("event-type").value)&&(!search||JSON.stringify(e).toLowerCase().includes(search)));pageNo=0;renderTable();drawMap();
}
function renderTable(){
  $("events").replaceChildren();const n=Math.max(1,Math.ceil(filtered.length/10));
  for(const e of filtered.slice(pageNo*10,pageNo*10+10)){const tr=el("tr"),first=el("td",e.source_date||"Дата неизвестна");first.append(el("small",e.event_id));tr.append(first,el("td",e.basin||"Не указан"),el("td",e.event_type||"Не указан"),el("td",e.date_year_conflict?"Конфликт года":e.date_precision));const td=el("td");td.append(sourceLink(e.source_url));tr.append(td);$("events").append(tr);}
  if(!filtered.length){const td=el("td","Ничего не найдено");td.colSpan=5;const tr=el("tr");tr.append(td);$("events").append(tr);}
  $("catalog-count").textContent=`Найдено записей: ${filtered.length} из ${events.length}`;
  $("page-count").textContent=`${pageNo+1} / ${n}`;$("prev").disabled=pageNo===0;$("next").disabled=pageNo>=n-1;
}
$("dataset").addEventListener("change",loadKind);$("inspect").onclick=inspect;$("date").onchange=()=>{if(model)inspect();};
document.querySelectorAll("[data-mode]").forEach(b=>b.onclick=()=>switchMode(b.dataset.mode));
$("manual-panel").onsubmit=e=>{e.preventDefault();try{if(!model)throw Error("Модель недоступна");const row={date:$("scenario-date").value,...SelRiskModel.season($("scenario-date").value)};document.querySelectorAll("[data-feature]").forEach(x=>{row[x.dataset.feature]=x.value.trim()===""?NaN:Number(x.value);});setResult(SelRiskModel.validateScenario(model,row),row);}catch(err){setResult({score:null,status:"invalid"},null,err.message);}};
$("manual-panel").addEventListener("input",()=>setResult({score:null,status:"invalid"},null,"Сценарий изменён. Выполните расчёт заново; предыдущий балл не используется."));
for(const id of ["country","basin","event-type","search"])$(id).addEventListener(id==="search"?"input":"change",filterEvents);
$("prev").onclick=()=>{pageNo--;renderTable();};$("next").onclick=()=>{pageNo++;renderTable();};
(async()=>{try{[events,centers]=await Promise.all([get("data/events.json"),get("data/centers.geojson")]);$("event-count").textContent=events.length.toLocaleString("ru");for(const [id,key]of [["country","country"],["basin","basin"],["event-type","event_type"]])for(const s of [...new Set(events.map(e=>e[key]).filter(Boolean))].sort()){$(id).append(new Option(s,s));}filterEvents();await loadKind();}catch(e){$("load-error").hidden=false;$("load-error").textContent=e.message;setResult({score:null,status:"no_data"},null,"Ошибка загрузки. Расчёт не выполнен.");}})();
