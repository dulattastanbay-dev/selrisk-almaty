(async function(){
  const targets=document.querySelectorAll("[data-results]");
  try{
    const response=await fetch("../../web/data/report_power.json");
    if(!response.ok)throw Error("HTTP "+response.status);
    const r=await response.json();
    const table=document.createElement("table"),head=document.createElement("thead"),hr=document.createElement("tr");
    for(const name of ["Модель","Proxy AP","Proxy AUC","Захват"]){const th=document.createElement("th");th.textContent=name;hr.append(th);}head.append(hr);table.append(head);
    const body=document.createElement("tbody"),names={presence_background:"Погода + сезон",rainfall_3day:"Осадки 3 дня",season_only:"Только сезон"};
    for(const [key,vals]of Object.entries(r.metrics)){const t=vals.test,tr=document.createElement("tr");for(const x of [names[key],t.proxy_average_precision.toFixed(4),t.proxy_roc_auc.toFixed(4),t.catalogue_positive_capture_top_decile.toFixed(3)]){const td=document.createElement("td");td.textContent=x;tr.append(td);}body.append(tr);}table.append(body);
    for(const x of targets){x.replaceChildren(table.cloneNode(true));const note=document.createElement("p");note.className="result-note";const t=r.metrics.presence_background.test;note.textContent=`Test: ${t.n_days} допустимых дней, ${t.catalogue_positive_days} зарегистрированные даты. Доли от 0 до 1; фон не является проверенным отрицательным исходом.`;x.append(note);}
  }catch(e){for(const x of targets)x.textContent="Результаты недоступны: "+e.message+". Запустите сайт через HTTP, не file://.";}
})();
(function(){
  const slides=[...document.querySelectorAll(".slide")];if(!slides.length)return;
  let i=Math.max(0,Math.min(slides.length-1,(Number(location.hash.slice(1))||1)-1));
  function show(){slides.forEach((s,j)=>s.hidden=i!==j);document.getElementById("slide-count").textContent=`${i+1} / ${slides.length}`;document.getElementById("slide-prev").disabled=i===0;document.getElementById("slide-next").disabled=i===slides.length-1;history.replaceState(null,"","#"+(i+1));window.scrollTo(0,0);}
  document.getElementById("slide-prev").onclick=()=>{if(i>0){i--;show();}};
  document.getElementById("slide-next").onclick=()=>{if(i<slides.length-1){i++;show();}};
  document.addEventListener("keydown",e=>{if(e.key==="ArrowRight"&&i<slides.length-1){i++;show();}if(e.key==="ArrowLeft"&&i>0){i--;show();}});show();
})();
