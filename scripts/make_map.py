"""Интерактивная веб-карта (Leaflet) + экспорт слоёв риска в GeoJSON и ESRI ASCII.

    python scripts/make_map.py

Создаёт:
  docs/map/risk_map.html                 — интерактивная карта (открыть в браузере)
  data/outputs/susceptibility.geojson/.asc
  data/outputs/risk_<дата>.geojson/.asc
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

import pandas as pd

from selrisk.config import PROJECT_ROOT, load_config
from selrisk.export import build_map_payload, to_esri_ascii, to_geojson
from selrisk.models.fusion import daily_risk

TEMPLATE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Карта риска селей · Алматы</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
  html,body{margin:0;height:100%;font-family:system-ui,"Segoe UI",sans-serif}
  #map{position:absolute;inset:0}
  .panel{position:absolute;z-index:1000;top:12px;left:12px;background:#fff;
    padding:12px 14px;border-radius:10px;box-shadow:0 2px 12px rgba(0,0,0,.25);max-width:290px}
  .panel h1{font-size:16px;margin:0 0 6px}
  .panel p{font-size:12px;color:#555;margin:4px 0}
  select,button{font:inherit;padding:5px 8px;border-radius:6px;border:1px solid #bbb;margin:3px 0}
  button{cursor:pointer;background:#3d5a4c;color:#fff;border:none}
  .legend{position:absolute;z-index:1000;bottom:18px;right:12px;background:#fff;
    padding:10px 12px;border-radius:10px;box-shadow:0 2px 12px rgba(0,0,0,.25);font-size:12px}
  .legend i{display:inline-block;width:14px;height:14px;margin-right:6px;border-radius:3px;vertical-align:-2px}
</style>
</head>
<body>
<div id="map"></div>
<div class="panel">
  <h1>⛰️ Риск селей — Алматы</h1>
  <p>Слой:</p>
  <select id="mode"></select>
  <p id="info"></p>
  <p style="color:#888">Подверженность × вероятность триггера. Демо-данные + реальный рельеф.</p>
</div>
<div class="legend" id="legend"></div>
<script>
const DATA = __PAYLOAD__;
const [lon0,lat0,lon1,lat1] = DATA.bbox, res = DATA.res;
const BOUNDS=[[lat0,lon0],[lat1,lon1]];
const map = L.map('map',{preferCanvas:true}).fitBounds(BOUNDS);
function fit(){ map.invalidateSize(); map.fitBounds(BOUNDS); }  // после фиксации размера
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom:16, attribution:'© OpenStreetMap'}).addTo(map);

function ylorrd(v){                       // рамп для подверженности
  const stops=[[255,255,178],[254,204,92],[253,141,60],[240,59,32],[189,0,38]];
  v=Math.max(0,Math.min(1,v)); const t=v*(stops.length-1); const i=Math.floor(t), f=t-i;
  const a=stops[i], b=stops[Math.min(i+1,stops.length-1)];
  return `rgb(${a[0]+(b[0]-a[0])*f|0},${a[1]+(b[1]-a[1])*f|0},${a[2]+(b[2]-a[2])*f|0})`;
}
function riskClassColor(v){
  for(const c of DATA.classes){ if(v < c.max) return c.color; }
  return DATA.classes[DATA.classes.length-1].color;
}

let layer=null;
function draw(mode){
  if(layer) layer.remove();
  const g=L.layerGroup(); const isSusc = (mode==='susc');
  const tp = isSusc ? null : DATA.triggers[mode];
  for(const c of DATA.cells){
    const [lon,lat,su,bi]=c;
    const val = isSusc ? su : su*tp[bi];
    if(!isSusc && val<0.03) continue;      // не рисуем почти нулевой риск
    const col = isSusc ? ylorrd(su) : riskClassColor(val);
    L.rectangle([[lat-res/2,lon-res/2],[lat+res/2,lon+res/2]],
      {stroke:false,fillColor:col,fillOpacity:isSusc?0.6:0.72}).addTo(g);
  }
  DATA.basins.forEach((b,i)=>{});
  g.addTo(map); layer=g;
  document.getElementById('info').textContent =
    isSusc ? 'Статическая подверженность склонов (0..1).'
           : 'Риск на '+mode+' (0..1), по классам.';
  legend(isSusc);
}
function legend(isSusc){
  const el=document.getElementById('legend');
  if(isSusc){ el.innerHTML='<b>Подверженность</b><br>'+
    [0.1,0.3,0.5,0.7,0.9].map(v=>`<i style="background:${ylorrd(v)}"></i>${v}`).join('<br>'); }
  else { el.innerHTML='<b>Класс риска</b><br>'+
    DATA.classes.map(c=>`<i style="background:${c.color}"></i>${c.name}`).join('<br>'); }
}

const sel=document.getElementById('mode');
sel.add(new Option('Подверженность (где)','susc'));
Object.keys(DATA.triggers).forEach(d=>sel.add(new Option('Риск на '+d,d)));
sel.onchange=()=>draw(sel.value);
draw('susc');
window.addEventListener('load', fit);
setTimeout(fit, 200);
</script>
</body>
</html>
"""


def main() -> None:
    cfg = load_config()
    odir = cfg.path("outputs")
    grid = pd.read_csv(odir / "grid_with_susceptibility.csv")
    meteo = pd.read_csv(odir / "meteo_with_trigger.csv", parse_dates=["date"])

    # выбор дат: день максимального риска и спокойный день
    peak = meteo.loc[meteo["trigger_prob"].idxmax(), "date"]
    calm = meteo.loc[meteo["trigger_prob"].idxmin(), "date"]
    dates = list(dict.fromkeys(d for d in (peak, calm) if d in set(meteo["date"])))

    # экспорт слоёв в ГИС-форматы
    to_geojson(grid, "susceptibility", odir / "susceptibility.geojson")
    to_esri_ascii(grid, grid["susceptibility"], odir / "susceptibility.asc")
    for d in dates[:2]:
        r = daily_risk(grid, meteo, d)
        tag = pd.Timestamp(d).date()
        to_geojson(r.assign(risk=r["risk"]), "risk", odir / f"risk_{tag}.geojson")
        to_esri_ascii(grid, r["risk"], odir / f"risk_{tag}.asc")

    payload = build_map_payload(grid, meteo, dates)
    html = TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    out = PROJECT_ROOT / "docs" / "map"
    out.mkdir(parents=True, exist_ok=True)
    (out / "risk_map.html").write_text(html, encoding="utf-8")

    print(f"[make_map] карта -> docs/map/risk_map.html ({len(payload['cells'])} ячеек, "
          f"{len(dates)} дат)")
    print(f"[make_map] ГИС-слои (GeoJSON + ESRI ASCII) -> {odir}")


if __name__ == "__main__":
    main()
