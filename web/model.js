/* Exact linear decision-function parity with selrisk.research.score_rows.
   No sigmoid, event probability, clipping or missing-value substitution. */
(function(root){
  function score(model,row){
    const values=model.features.map(k=>row[k]);
    if(values.some(x=>x===null||x===""||x===undefined||typeof x!=="number"||!Number.isFinite(x)))
      return {score:null,status:"missing_features"};
    if(values.some((v,i)=>v<model.feature_min[i]-1e-7||v>model.feature_max[i]+1e-7))
      return {score:null,status:"out_of_training_domain"};
    return {score:values.reduce((s,v,i)=>s+(v-model.mean[i])/model.scale[i]*model.coefficients[i],model.intercept),status:"available"};
  }
  function season(iso){
    if(!/^\d{4}-\d{2}-\d{2}$/.test(iso))throw Error("Укажите корректную дату");
    const dt=new Date(iso+"T00:00:00Z");
    if(!Number.isFinite(+dt)||dt.toISOString().slice(0,10)!==iso)throw Error("Укажите корректную дату");
    const doy=(dt-Date.UTC(dt.getUTCFullYear(),0,1))/86400000;
    return {day_of_year_sin:Math.sin(2*Math.PI*doy/365.2425),day_of_year_cos:Math.cos(2*Math.PI*doy/365.2425)};
  }
  function validateFeatureValues(model,row){
    for(const k of model.features){
      if(typeof row[k]!=="number"||!Number.isFinite(row[k]))throw Error("Заполните все признаки конечными числами");
      if((k.includes("precipitation")||k.includes("degree_days"))&&row[k]<0)throw Error("Осадки и градусо-дни не могут быть отрицательными");
    }
    const groups={};
    model.features.filter(k=>k.includes("precipitation")).forEach(k=>{
      const station=k.split("precipitation")[0];(groups[station]??=[]).push(k);
    });
    for(const keys of Object.values(groups)){
      const rank=k=>k.includes("prev14d")?14:k.includes("prev7d")?7:k.includes("prev3d")?3:1;
      keys.sort((a,b)=>rank(a)-rank(b));
      for(let i=1;i<keys.length;i++)if(row[keys[i-1]]>row[keys[i]]+1e-5)throw Error("Накопленные осадки должны возрастать: 1 ≤ 3 ≤ 7 ≤ 14 дней");
    }
    if("temperature_min_lag2utc_c" in row && !(row.temperature_min_lag2utc_c<=row.temperature_mean_lag2utc_c&&row.temperature_mean_lag2utc_c<=row.temperature_max_lag2utc_c))
      throw Error("Температуры должны удовлетворять min ≤ mean ≤ max");
  }
  function validateScenario(model,row){
    validateFeatureValues(model,row);
    return score(model,row);
  }
  const api={score,season,validateScenario,validateFeatureValues};root.SelRiskModel=api;
  if(typeof module!=="undefined")module.exports=api;
})(typeof globalThis!=="undefined"?globalThis:this);
