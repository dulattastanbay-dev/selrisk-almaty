(function(root){
  "use strict";
  const base=root.SelRiskModel||(typeof require==="function"?require("./model.js"):null);
  const trees=root.SelRiskTrees||(typeof require==="function"?require("./tree-model.js"):null);
  const shift=(iso,n)=>new Date(Date.parse(iso+"T00:00:00Z")+n*86400000).toISOString().slice(0,10);
  function featuresFromDaily(daily,date){
    base.season(date);
    const last=shift(date,-2),end=daily.time.indexOf(last);
    if(end<13)throw Error("Недостаточно погоды для 14-дневного окна");
    const window=daily.time.slice(end-13,end+1);
    if(window.some((v,i)=>v!==shift(last,i-13)))throw Error("В погодном ряду пропущена дата");
    const take=(key,n)=>{const a=daily[key]?.slice(end-n+1,end+1);if(!a||a.length!==n||a.some(x=>typeof x!=="number"||!Number.isFinite(x)))throw Error("Погодный ряд неполон");return a;};
    const sum=a=>a.reduce((s,v)=>s+v,0);
    const rain=n=>sum(take("precipitation_sum",n));
    const mean=take("temperature_2m_mean",7);
    return {date,...base.season(date),
      precipitation_lag2utc_mm:rain(1),precipitation_prev3d_mm:rain(3),
      precipitation_prev7d_mm:rain(7),precipitation_prev14d_mm:rain(14),
      temperature_mean_lag2utc_c:mean.at(-1),
      temperature_min_lag2utc_c:take("temperature_2m_min",1)[0],
      temperature_max_lag2utc_c:take("temperature_2m_max",1)[0],
      temperature_mean_prev7d_c:sum(mean)/7,
      positive_degree_days_prev7d:sum(mean.map(x=>Math.max(0,x)))};
  }
  function probability(model,row){
    base.validateFeatureValues(model,row);
    return trees.predict(model,row);
  }
  function color(p){
    if(p===null||!Number.isFinite(p))return "#9aaba5";
    const t=Math.max(0,Math.min(1,(Math.log10(Math.max(p,1e-12))+6)/5));
    const stops=[[38,132,91],[125,186,101],[225,205,87],[232,135,63],[186,63,56]];
    const a=Math.min(3,Math.floor(t*4)),f=t*4-a;
    return `rgb(${stops[a].map((v,i)=>Math.round(v+(stops[a+1][i]-v)*f)).join(",")})`;
  }
  const api={featuresFromDaily,probability,localRisk:trees.fuse,color,shift};
  root.SelRiskMap=api;if(typeof module!=="undefined")module.exports=api;
  if(typeof document==="undefined")return;
  const $=id=>document.getElementById(id);
  const today=()=>new Intl.DateTimeFormat("en-CA",{timeZone:"Asia/Almaty",year:"numeric",month:"2-digit",day:"2-digit"}).format(new Date());
  const fmt=p=>{const v=p*100;return v>0&&v<.000001?"<0,000001%":new Intl.NumberFormat("ru-RU",{maximumFractionDigits:v<.01?6:v<1?4:2}).format(v)+"%";};
  const city=[43.18,77.02],bounds=[[42.86,76.60],[43.43,77.65]];
  let model,forest,areas,map,archive=null,version=0,current=null,ready;
  function resetView(){
    if(root.innerWidth<=650)map.setView([43.17,77.02],10);
    else map.fitBounds(bounds);
  }
  async function get(url,signal){
    let r;
    try{r=await fetch(url,{signal});}
    catch(e){if(e.name==="AbortError")throw e;throw Error("Не удалось загрузить данные. Повторите расчёт.");}
    if(!r.ok)throw Error(`Данные недоступны (HTTP ${r.status})`);
    return r.json();
  }
  function clear(message){
    current=null;
    $("region-probability").textContent="—";$("status").textContent=message;
    $("weather-source").textContent="Погода: нет расчёта";
    if(map)map.closePopup();
    if(areas)areas.setStyle({fillColor:color(null),fillOpacity:.22});
  }
  async function weather(date,signal){
    if(date>="1981-01-01"&&date<="2020-12-31"){
      if(!archive){const a=await get("data/archive_power.json",signal);archive=a.data.map(v=>Object.fromEntries(a.columns.map((k,i)=>[k,v[i]])));}
      const row=archive.find(x=>x.date===date);
      if(!row||!row.feature_complete)throw Error("На выбранную дату нет полного погодного окна");
      return {row,source:"NASA POWER · исторический архив"};
    }
    const last=shift(date,-2),start=shift(last,-13);
    const recent=date>=shift(today(),-70);
    const endpoint=recent?"https://api.open-meteo.com/v1/forecast":"https://archive-api.open-meteo.com/v1/archive";
    const query=new URLSearchParams({latitude:"43.1",longitude:"77.0",start_date:start,end_date:last,
      daily:"precipitation_sum,temperature_2m_mean,temperature_2m_min,temperature_2m_max",timezone:"GMT"});
    const data=await get(`${endpoint}?${query}`,signal);
    if(!data.daily)throw Error("Провайдер не вернул суточную погоду");
    return {row:featuresFromDaily(data.daily,date),source:recent?"Open-Meteo · актуальная погода":"Open-Meteo · реанализ"};
  }
  let controller;
  async function calculate(date){
    const mine=++version;
    controller?.abort();controller=new AbortController();const signal=controller.signal;
    clear("Расчёт по выбранной дате…");
    $("day-label").textContent=date||"Выберите дату";
    for(const id of ["today","tomorrow"])$(id).setAttribute("aria-pressed",String(date===shift(today(),id==="today"?0:1)));
    try{
      base.season(date);
      if(date<"1981-01-16"||date>shift(today(),1))throw Error("Выберите дату с 16.01.1981 по завтра");
      await ready;
      if(mine!==version)return;
      if(!model||!forest||!areas)throw Error("Модели или география не загружены");
      const {row,source}=await weather(date,signal);
      if(mine!==version)return;
      const result=probability(model,row);
      if(result.value===null)throw Error(result.status==="out_of_training_domain"?"Погода вне обучающего диапазона":"Недостаточно погодных данных");
      current={date,p:result.value};
      $("region-probability").textContent=fmt(current.p);
      $("day-label").textContent=new Intl.DateTimeFormat("ru-RU",{dateStyle:"long",timeZone:"UTC"}).format(new Date(date+"T00:00:00Z"));
      $("status").textContent="Нажмите на цветной участок: лес × бустинг.";
      $("weather-source").textContent=`${source} · данные по ${shift(date,-2)} UTC`;
      areas.eachLayer(layer=>{
        const p=trees.fuse(layer.spatialScore,current.p);
        layer.setStyle({fillColor:color(p),fillOpacity:.62});
      });
      map.closePopup();
    }catch(e){if(mine!==version||e.name==="AbortError")return;clear(e.message);}
  }
  ready=(async()=>{
    if(typeof L==="undefined")throw Error("Не удалось загрузить библиотеку карты");
    map=L.map("map",{zoomControl:false,minZoom:8,maxZoom:16,zoomSnap:.25,preferCanvas:true});
    resetView();
    L.control.zoom({position:"topright"}).addTo(map);
    L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",{
      attribution:'<a href="https://www.esri.com/en-us/legal/terms/data-attributions" target="_blank" rel="noopener">&copy; Esri</a> &middot; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OSM contributors</a>',
      maxZoom:17,crossOrigin:true}).addTo(map);
    map.attributionControl.addAttribution('<a href="https://research.reading.ac.uk/glof-predicting-debris-flow/" target="_blank" rel="noopener">Очаги: Reading</a> &middot; <a href="https://open-meteo.com/" target="_blank" rel="noopener">Погода: Open-Meteo / POWER</a>');
    const [m,f,g,centers]=await Promise.all([get("data/model_boosting.json"),get("data/model_rf.json"),
      get("data/terrain_grid.json"),get("data/centers.geojson")]);
    model=trees.validate(m);forest=trees.validate(f);
    if(model.model_type!=="hist_gradient_boosting_assumed_negative_trigger"||
       forest.model_type!=="random_forest_inventory_background"||
       g.source_sha256!==forest.source_sha256||g.sample_spacing_m!==450||
       JSON.stringify(g.columns)!==JSON.stringify(["sample_id","latitude","longitude",...forest.features])||
       !Array.isArray(g.data)||!g.data.length)throw Error("Несовместимые модели и рельеф");
    areas=L.featureGroup().addTo(map);
    for(const values of g.data){
      const row=Object.fromEntries(g.columns.map((k,i)=>[k,values[i]]));
      if(!Number.isFinite(row.latitude)||!Number.isFinite(row.longitude))throw Error("Некорректная география");
      const spatial=trees.predict(forest,row);
      if(spatial.value===null)throw Error("Нет расчёта случайного леса для рельефа");
      // Approximate 450m sample footprints, not natural catchment boundaries.
      const dy=225/111320,dx=dy/Math.cos(row.latitude*Math.PI/180);
      const layer=L.rectangle([[row.latitude-dy,row.longitude-dx],[row.latitude+dy,row.longitude+dx]],
        {stroke:false,fillColor:color(null),fillOpacity:.22}).addTo(areas);
      layer.spatialScore=spatial.value;
      layer.on("click",()=>{
        const box=document.createElement("div"),title=document.createElement("b"),value=document.createElement("strong"),note=document.createElement("small");
        title.textContent=`Участок ${row.latitude.toFixed(4)}, ${row.longitude.toFixed(4)}`;
        box.append(title,document.createElement("br"));
        value.textContent=current?fmt(trees.fuse(layer.spatialScore,current.p)):"Нет расчёта";
        note.textContent=current?`${current.date} · условный индекс лес × бустинг`:"Выберите доступную дату";
        box.append(value,note);
        const detail=document.createElement("small");
        detail.textContent=current?`Лес (сходство рельефа): ${fmt(layer.spatialScore)}. Бустинг (погодный триггер): ${fmt(current.p)}.`:"";
        const terrain=document.createElement("small");
        terrain.textContent=`Высота ${Math.round(row.elevation_m)} м · склон ${row.slope_deg.toFixed(1)}° · сетка 450 м`;
        box.append(detail,terrain);layer.bindPopup(box).openPopup();
      });
    }
    L.geoJSON(centers,{interactive:false,style:{color:"#294e3d",weight:.75,fill:false,opacity:.65}}).addTo(map);
    $("reset-map").onclick=resetView;
  })();
  // Attach a rejection handler immediately; calculate also awaits the same promise.
  ready.catch(e=>clear(e.message));
  $("date").min="1981-01-16";$("date").max=shift(today(),1);
  $("date").value=new URLSearchParams(location.search).get("date")||today();
  $("date-form").onsubmit=e=>{e.preventDefault();calculate($("date").value);};
  $("date").onchange=()=>calculate($("date").value);
  for(const [id,offset]of [["today",0],["tomorrow",1]])$(id).onclick=()=>{$("date").value=shift(today(),offset);calculate($("date").value);};
  calculate($("date").value);
})(typeof globalThis!=="undefined"?globalThis:this);
