/* СельРиск Алматы — интерфейс: три режима ввода, карточки, факторы, график, карта. */
(function () {
  "use strict";

  const D = window.SELRISK_DATA;
  const M = D.model;
  const SR = window.SelRiskModel;
  const NB = M.basins.length;
  const BCOL = ["var(--b0)", "var(--b1)", "var(--b2)"];
  const OPEN_METEO = "https://api.open-meteo.com/v1/forecast";
  const USGS = "https://earthquake.usgs.gov/fdsnws/event/1/query";
  const HOUR_MS = 3600000;

  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));

  // ------------------------------------------------------------------ формат
  const num = (x, d) => x.toFixed(d).replace(".", ",").replace("-", "−");
  function fmtPct(p) {
    const v = p * 100;
    if (v < 0.1) return "<0,1";
    if (v < 10) return num(v, 1);
    if (v > 99.5) return ">99";
    return String(Math.round(v));
  }
  const fmtDay = (d) => d.toISOString().slice(8, 10) + "." + d.toISOString().slice(5, 7);
  const fmtDateRu = (d) => fmtDay(d) + "." + d.getUTCFullYear();
  function fmtMult(v) {
    const m = Math.exp(Math.abs(v));
    const s = m >= 1000 ? "1000+" : m >= 10 ? String(Math.round(m)) : num(m, 1);
    return (v >= 0 ? "×" : "÷") + s;
  }
  function todayUTC() {
    const n = new Date();
    return new Date(Date.UTC(n.getFullYear(), n.getMonth(), n.getDate()));
  }

  const FEATURE_LABEL = {
    rain_1d: ["Осадки за сутки", "мм", 1],
    rain_3d: ["Осадки за 3 дня", "мм", 1],
    rain_7d: ["Осадки за 7 дней", "мм", 1],
    api: ["Индекс увлажнённости грунта", "мм", 0],
    pdd_3d: ["Тепло за 3 дня (сумма t > 0)", "°C·сут", 0],
    swe_mm: ["Запас воды в снеге", "мм", 0],
    snowmelt_mm: ["Таяние снега за сутки", "мм", 1],
    eq_max_mag_3d: ["Макс. магнитуда за 3 дня", "M", 1],
  };

  // ------------------------------------------------------------------ состояние
  const state = { mode: "manual", sel: null, results: null, date: null, layer: "risk" };

  // ================================================================== РЕЗУЛЬТАТЫ
  function show(results, date, sub, feats) {
    state.results = results;
    state.date = date;
    if (state.sel === null) state.sel = argmax(results.map((r) => r.p));
    $("#res-title").textContent = "Вероятность схода селя за сутки · " + fmtDateRu(date);
    $("#res-sub").textContent = sub;
    renderCards();
    renderFactors();
    renderClipWarn();
    renderCond(feats);
    updateMap();
  }

  function argmax(a) {
    return a.reduce((best, v, i) => (v > a[best] ? i : best), 0);
  }

  function times(x) {                           // «1,5 раза», «3 раза», «25 раз»
    if (x < 10 && !Number.isInteger(+x.toFixed(1))) return num(x, 1) + " раза";
    const n = Math.round(x);
    const few = n % 10 >= 2 && n % 10 <= 4 && !(n % 100 >= 12 && n % 100 <= 14);
    return n + (few ? " раза" : " раз");
  }

  function meterPos(p) {                        // логарифмическая шкала 0,1 % … 100 %
    return clamp((Math.log10(Math.max(p, 1e-4)) + 3) / 3, 0, 1);
  }

  function renderCards() {
    const host = $("#cards");
    const segs = [];
    let prev = 0;
    M.levels.forEach((l) => {
      const to = meterPos(Math.min(l.max, 1));
      segs.push(`<b style="width:${(to - prev) * 100}%;background:${l.color}"></b>`);
      prev = to;
    });
    host.innerHTML = state.results.map((r, i) => {
      const lv = SR.level(M, r.p);
      const ratio = r.p / M.base_rate;
      const ratioTxt = ratio >= 1.5 ? `в ${times(ratio)} выше среднего дня`
        : ratio <= 0.67 ? `в ${times(1 / ratio)} ниже среднего дня`
          : "как в средний день";
      return `<button type="button" class="card" data-b="${i}" aria-pressed="${i === state.sel}" style="--bc:${BCOL[i]}">
        <div class="bn"><i></i>${esc(M.basins[i])}</div>
        <div class="pct">${fmtPct(r.p)}<span>%</span></div>
        <div class="lvl" style="--lc:${lv.color}">${lv.name}</div>
        <div class="ratio">${ratioTxt}</div>
        <div class="meter" aria-hidden="true">${segs.join("")}<em style="left:${meterPos(r.p) * 100}%"></em></div>
      </button>`;
    }).join("");
    $$(".card", host).forEach((c) => c.addEventListener("click", () => {
      state.sel = +c.dataset.b;
      renderCards();
      renderFactors();
    }));
  }

  function renderFactors() {
    const r = state.results[state.sel];
    $("#factor-basin").textContent = M.basins[state.sel];
    const groups = SR.groupContrib(M, r.contrib);
    const maxAbs = 6;                              // ±6 в логите = край шкалы (~×400)
    $("#factors").innerHTML = groups.map((g) => {
      const w = clamp(Math.abs(g.value) / maxAbs, 0, 1) * 50;
      const cls = g.value >= 0 ? "up" : "down";
      return `<div class="frow"><span>${g.label}</span>
        <div class="fbar"><i class="${cls}" style="width:${w}%"></i></div>
        <span class="fval">${Math.abs(g.value) < 0.05 ? "≈ 1" : fmtMult(g.value)}</span></div>`;
    }).join("");
  }

  function renderClipWarn() {
    const names = new Set();
    state.results.forEach((r) => r.clipped.forEach((f) => names.add(f)));
    const box = $("#clip-warn");
    if (!names.size) { box.hidden = true; return; }
    const list = Array.from(names).map((f) => {
      const [label, unit, d] = FEATURE_LABEL[f] || [f, "", 1];
      return `${label.toLowerCase()} (максимум в данных — ${num(M.clip_max[M.raw_features.indexOf(f)], d)} ${unit})`;
    });
    box.innerHTML = `<b>Выше, чем встречалось в 2016–2023:</b> ${list.join("; ")}. Модель не экстраполирует и считает эти
      значения равными максимуму, поэтому реальный риск может быть ещё выше.`;
    box.hidden = false;
  }

  function renderCond(feats) {
    const panel = $("#cond-panel");
    if (!feats) { panel.hidden = true; return; }
    panel.hidden = false;
    const head = `<thead><tr><th>Показатель</th>${M.basins.map((b, i) =>
      `<th><span style="color:${BCOL[i]}">●</span> ${esc(b)}</th>`).join("")}</tr></thead>`;
    const rows = Object.entries(FEATURE_LABEL).map(([f, [label, unit, d]]) => {
      const cells = feats.map((fv, i) => {
        const clipped = state.results[i].clipped.includes(f);
        return `<td class="${clipped ? "clip" : ""}">${num(fv[f], d)} <small>${unit}</small></td>`;
      }).join("");
      return `<tr><td>${label}</td>${cells}</tr>`;
    }).join("");
    $("#cond").innerHTML = head + "<tbody>" + rows + "</tbody>";
  }

  // ================================================================== ГРАФИК
  function drawChart(opts) {
    // opts: {start, from, to, preds[bi][i], events[bi]:Set, mark, forecastFrom, onPick}
    const host = $("#chart");
    lastChartW = host.clientWidth;
    const W = Math.max(host.clientWidth, 280);
    const H = 230;
    const P = { l: 44, r: 12, t: 10, b: 26 };
    const iw = W - P.l - P.r;
    const ih = H - P.t - P.b;
    const n = opts.to - opts.from + 1;
    let vmax = 0;
    for (let b = 0; b < NB; b++) for (let i = opts.from; i <= opts.to; i++) vmax = Math.max(vmax, opts.preds[b][i].p);
    vmax = Math.max(vmax * 1.12, 0.02);
    const steps = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.25];
    const step = steps.find((s) => vmax / s <= 5) || 0.25;
    vmax = Math.min(Math.ceil(vmax / step) * step, 1);
    const x = (i) => P.l + ((i - opts.from) / Math.max(n - 1, 1)) * iw;
    const y = (p) => P.t + ih - (p / vmax) * ih;

    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="График вероятности по дням">`;
    if (opts.forecastFrom != null && opts.forecastFrom <= opts.to) {
      const fx = x(Math.max(opts.forecastFrom, opts.from)) ;
      svg += `<rect x="${fx}" y="${P.t}" width="${P.l + iw - fx}" height="${ih}" fill="var(--surface-2)"/>`;
      svg += `<text x="${P.l + iw - 4}" y="${P.t + 13}" text-anchor="end" font-size="11" fill="var(--muted)">прогноз</text>`;
    }
    svg += `<g class="grid axis">`;
    for (let v = 0; v <= vmax + 1e-9; v += step) {
      svg += `<line x1="${P.l}" x2="${P.l + iw}" y1="${y(v)}" y2="${y(v)}"/>`;
      svg += `<text x="${P.l - 6}" y="${y(v) + 4}" text-anchor="end">${num(v * 100, step < 0.01 ? 1 : 0)}%</text>`;
    }
    const every = Math.max(1, Math.ceil(n / Math.max(3, Math.floor(iw / 70))));
    for (let i = opts.from; i <= opts.to; i += every) {
      svg += `<text x="${x(i)}" y="${H - 6}" text-anchor="middle">${fmtDay(SR.addDays(opts.start, i))}</text>`;
    }
    svg += `</g>`;
    M.levels.slice(0, -1).forEach((l) => {
      if (l.max < vmax) {
        svg += `<line x1="${P.l}" x2="${P.l + iw}" y1="${y(l.max)}" y2="${y(l.max)}" stroke="${l.color}" stroke-dasharray="4 4" opacity=".8"/>`;
      }
    });
    if (opts.mark != null) {
      svg += `<line x1="${x(opts.mark)}" x2="${x(opts.mark)}" y1="${P.t}" y2="${P.t + ih}" stroke="var(--text)" stroke-width="1.5" opacity=".55"/>`;
    }
    for (let b = 0; b < NB; b++) {
      let d = "";
      for (let i = opts.from; i <= opts.to; i++) d += (i === opts.from ? "M" : "L") + x(i).toFixed(1) + "," + y(opts.preds[b][i].p).toFixed(1);
      svg += `<path d="${d}" fill="none" stroke="${BCOL[b]}" stroke-width="2" stroke-linejoin="round"/>`;
    }
    for (let b = 0; b < NB; b++) {
      if (!opts.events) break;
      opts.events[b].forEach((i) => {
        if (i >= opts.from && i <= opts.to) {
          svg += `<circle cx="${x(i)}" cy="${y(opts.preds[b][i].p)}" r="4.5" fill="var(--surface)" stroke="var(--text)" stroke-width="2"/>`;
        }
      });
    }
    svg += `<line class="hover" x1="0" x2="0" y1="${P.t}" y2="${P.t + ih}" stroke="var(--muted)" stroke-dasharray="2 3" visibility="hidden"/>`;
    svg += `<rect class="hit" x="${P.l}" y="${P.t}" width="${iw}" height="${ih}" fill="transparent" style="cursor:pointer"/>`;
    svg += `</svg><div class="tip" hidden></div>`;
    host.innerHTML = svg;

    const hit = $(".hit", host);
    const tip = $(".tip", host);
    const hover = $(".hover", host);
    const idxAt = (ev) => {
      const rect = host.querySelector("svg").getBoundingClientRect();
      const px = ((ev.clientX - rect.left) / rect.width) * W;
      return clamp(Math.round(opts.from + ((px - P.l) / iw) * (n - 1)), opts.from, opts.to);
    };
    hit.addEventListener("mousemove", (ev) => {
      const i = idxAt(ev);
      hover.setAttribute("x1", x(i));
      hover.setAttribute("x2", x(i));
      hover.setAttribute("visibility", "visible");
      const ev_ = opts.events ? M.basins.map((_, b) => opts.events[b].has(i)) : [];
      tip.innerHTML = `<b>${fmtDateRu(SR.addDays(opts.start, i))}</b><br>` + M.basins.map((bn, b) =>
        `<span style="color:${BCOL[b]}">●</span> ${esc(bn)}: <b>${fmtPct(opts.preds[b][i].p)}%</b>${ev_[b] ? " · событие" : ""}`).join("<br>");
      tip.hidden = false;
      const left = (x(i) / W) * host.clientWidth;
      tip.style.left = Math.min(left + 12, host.clientWidth - tip.offsetWidth - 4) + "px";
      tip.style.top = "8px";
    });
    hit.addEventListener("mouseleave", () => { tip.hidden = true; hover.setAttribute("visibility", "hidden"); });
    hit.addEventListener("click", (ev) => opts.onPick && opts.onPick(idxAt(ev)));

    $("#chart-legend").innerHTML = M.basins.map((b, i) => `<span><i style="background:${BCOL[i]}"></i>${esc(b)}</span>`).join("")
      + (opts.events ? `<span><i class="dot"></i>событие в данных</span>` : "")
      + `<span><i style="background:${M.levels[2].color}"></i>пороги уровней 1 / 5 / 20 %</span>`;
    lastChart = opts;
  }
  let lastChart = null;
  let lastChartW = 0;
  let resizeTimer = null;
  if (window.ResizeObserver) {
    new ResizeObserver(() => {
      const w = $("#chart").clientWidth;
      if (!lastChart || $("#chart-panel").hidden || Math.abs(w - lastChartW) < 2) return;
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => drawChart(lastChart), 100);
    }).observe($("#chart"));
  }

  // ================================================================== РУЧНОЙ РЕЖИМ
  const manual = { date: todayUTC(), rain: [0, 0, 0, 0, 0, 0, 0], prev: null, t: 8, swe: 0, melt: 0, mag: 0 };
  const PRESETS = [
    { name: "Сухой летний день", md: [7, 20], rain: [0, 0, 0, 1, 0, 0, 0], prev: 40, t: 12, swe: 0, melt: 0, mag: 0 },
    { name: "Затяжные дожди", md: [5, 15], rain: [6, 9, 11, 8, 10, 9, 12], prev: 90, t: 5, swe: 60, melt: 3, mag: 0 },
    { name: "Сильный ливень", md: [6, 20], rain: [2, 0, 4, 6, 14, 24, 20], prev: 70, t: 8, swe: 10, melt: 1, mag: 0 },
    { name: "Снеготаяние + дождь", md: [5, 20], rain: [3, 4, 6, 5, 9, 14, 12], prev: 80, t: 6, swe: 300, melt: 14, mag: 0 },
    { name: "Землетрясение M4,5 в сухую погоду", md: [8, 10], rain: [0, 0, 0, 0, 0, 0, 0], prev: 30, t: 11, swe: 0, melt: 0, mag: 4.5 },
  ];

  function climPrev(date) {
    return Math.round(D.climatology_mm_day[date.getUTCMonth() + 1] * 21);
  }

  function buildManual() {
    $("#presets").innerHTML = PRESETS.map((p, i) => `<button type="button" class="chip" data-i="${i}" aria-pressed="false">${p.name}</button>`).join("");
    $$("#presets .chip").forEach((c) => c.addEventListener("click", () => applyPreset(PRESETS[+c.dataset.i], c)));

    const rain = $("#rain7");
    rain.innerHTML = manual.rain.map((_, k) => `<label><span data-k="${k}"></span>
      <input type="number" min="0" step="0.5" inputmode="decimal" data-k="${k}" aria-label="Осадки, день ${k + 1} из 7"></label>`).join("");
    $$("input", rain).forEach((inp) => inp.addEventListener("input", () => {
      manual.rain[+inp.dataset.k] = Math.max(parseFloat(inp.value) || 0, 0);
      unpress(); computeManual();
    }));

    $("#in-date").addEventListener("change", (e) => {
      if (!e.target.value) return;
      manual.date = SR.parseDate(e.target.value);
      unpress(); syncManual(); computeManual();
    });
    $("#in-prev").addEventListener("input", (e) => {
      manual.prev = Math.max(parseFloat(e.target.value) || 0, 0);
      unpress(); computeManual();
    });
    $("#climBtn").addEventListener("click", () => { manual.prev = null; syncManual(); computeManual(); });

    [["t", "in-t"], ["swe", "in-swe"], ["melt", "in-melt"], ["mag", "in-mag"]].forEach(([key, id]) => {
      const numEl = $("#" + id);
      const rngEl = $("#" + id + "-r");
      const set = (v) => { manual[key] = v; unpress(); computeManual(); };
      rngEl.addEventListener("input", () => { numEl.value = rngEl.value; set(parseFloat(rngEl.value)); });
      numEl.addEventListener("input", () => {
        const v = parseFloat(numEl.value);
        if (Number.isNaN(v)) return;
        rngEl.value = v;
        set(key === "t" ? v : Math.max(v, 0));
      });
    });
  }

  function unpress() {
    $$("#presets .chip").forEach((c) => c.setAttribute("aria-pressed", "false"));
  }

  function applyPreset(p, chip) {
    manual.date = new Date(Date.UTC(todayUTC().getUTCFullYear(), p.md[0] - 1, p.md[1]));
    Object.assign(manual, { rain: p.rain.slice(), prev: p.prev, t: p.t, swe: p.swe, melt: p.melt, mag: p.mag });
    syncManual();
    unpress();
    chip.setAttribute("aria-pressed", "true");
    computeManual();
  }

  function syncManual() {
    $("#in-date").value = SR.fmtDate(manual.date);
    const today = todayUTC().getTime();
    $$("#rain7 span").forEach((s) => {
      const d = SR.addDays(manual.date, +s.dataset.k - 6);
      const isToday = d.getTime() === today;
      s.textContent = isToday ? "сегодня" : fmtDay(d);
      s.className = +s.dataset.k === 6 ? "today" : "";
    });
    $$("#rain7 input").forEach((inp) => { inp.value = manual.rain[+inp.dataset.k]; });
    $("#in-prev").value = manual.prev === null ? climPrev(manual.date) : manual.prev;
    [["t", "in-t"], ["swe", "in-swe"], ["melt", "in-melt"], ["mag", "in-mag"]].forEach(([key, id]) => {
      $("#" + id).value = manual[key];
      $("#" + id + "-r").value = manual[key];
    });
  }

  function computeManual() {
    if (state.mode !== "manual") return;
    const prev = manual.prev === null ? climPrev(manual.date) : manual.prev;
    const f = SR.manualFeatures({
      date: manual.date, rain: manual.rain, rainPrev21: prev, tmean: manual.t,
      swe: manual.swe, melt: manual.melt, quakeMag: manual.mag,
    }, M);
    const res = M.basins.map((_, bi) => SR.predict(M, f, bi));
    $("#chart-panel").hidden = true;
    show(res, manual.date, "По введённым условиям; одинаковая погода для всех трёх бассейнов.", null);
  }

  // ================================================================== АРХИВ
  let arc = null;
  function buildArchive() {
    const A = D.archive;
    const start = SR.parseDate(A.start);
    const eqMax = new Array(A.n).fill(0);
    const eqEnergy = new Array(A.n).fill(0);
    Object.entries(A.eq).forEach(([k, [mag, e]]) => { eqMax[+k] = mag; eqEnergy[+k] = e; });
    const feats = [];
    const preds = [];
    const events = [];
    M.basins.forEach((b, bi) => {
      const s = A.basins[b];
      const f = SR.seriesFeatures({ start, precip: s.precip, tmean: s.tmean, swe: s.swe, eqMax, eqEnergy }, M);
      feats.push(f);
      // процент — эталонный из Python (как в Streamlit); расчёт в JS по округлённому
      // архиву совпадает с ним до 0,05 п.п. (tests/web_parity.js) и даёт вклад факторов
      preds.push(f.map((fi, i) => Object.assign(SR.predict(M, fi, bi), { p: s.p_py[i] })));
      events.push(new Set(s.events));
    });
    arc = { start, n: A.n, feats, preds, events, idx: null };

    const dateEl = $("#arc-date");
    dateEl.min = A.start;
    dateEl.max = SR.fmtDate(SR.addDays(start, A.n - 1));
    dateEl.addEventListener("change", () => {
      if (!dateEl.value) return;
      const i = Math.round((SR.parseDate(dateEl.value) - start) / SR.DAY_MS);
      if (i >= 0 && i < A.n) showArchive(i);
    });
    $("#arc-prev").addEventListener("click", () => showArchive(Math.max(arc.idx - 1, 0)));
    $("#arc-next").addEventListener("click", () => showArchive(Math.min(arc.idx + 1, A.n - 1)));

    const evList = [];
    M.basins.forEach((b, bi) => A.basins[b].events.forEach((i) => evList.push([i, bi])));
    evList.sort((a, b) => a[0] - b[0]);
    const sel = $("#arc-events");
    sel.innerHTML = `<option value="">— выберите (${evList.length}) —</option>` + evList.map(([i, bi]) =>
      `<option value="${i}">${fmtDateRu(SR.addDays(start, i))} · ${esc(M.basins[bi])}</option>`).join("");
    sel.addEventListener("change", () => { if (sel.value !== "") showArchive(+sel.value); });

    $("#arc-top").innerHTML = D.top_days.map((t) =>
      `<button type="button" class="chip" data-d="${t.date}">${fmtDateRu(SR.parseDate(t.date))} <small>${esc(t.basin)}</small></button>`).join("");
    $$("#arc-top .chip").forEach((c) => c.addEventListener("click", () =>
      showArchive(Math.round((SR.parseDate(c.dataset.d) - start) / SR.DAY_MS))));
  }

  function showArchive(i) {
    if (state.mode !== "archive") return;
    arc.idx = i;
    const date = SR.addDays(arc.start, i);
    $("#arc-date").value = SR.fmtDate(date);
    const res = arc.preds.map((p) => p[i]);
    const evHere = M.basins.filter((_, b) => arc.events[b].has(i));
    $("#chart-panel").hidden = false;
    $("#chart-title").textContent = "Вероятность по дням — ±45 дней вокруг выбранной даты";
    drawChart({
      start: arc.start, from: Math.max(0, i - 45), to: Math.min(arc.n - 1, i + 45),
      preds: arc.preds, events: arc.events, mark: i, forecastFrom: null, onPick: showArchive,
    });
    show(res, date, "Реальная погода ERA5-Land" + (evHere.length ? ` · в данных в этот день событие: ${evHere.join(", ")}` : ""),
      arc.feats.map((f) => f[i]));
  }

  // ================================================================== ЖИВАЯ ПОГОДА
  let live = null;
  let liveBusy = false;

  async function fetchJSON(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${new URL(url).host}: HTTP ${r.status}`);
    return r.json();
  }

  async function loadLive() {
    if (liveBusy) return;
    liveBusy = true;
    const btn = $("#live-load");
    const status = $("#live-status");
    btn.disabled = true;
    status.className = "status";
    status.textContent = "Загружаю прогноз погоды и каталог землетрясений…";
    try {
      const pts = [];
      const owner = [];
      D.basins.forEach((b, bi) => b.points.forEach((p) => { pts.push(p); owner.push(bi); }));
      const qs = new URLSearchParams({
        latitude: pts.map((p) => p[0]).join(","),
        longitude: pts.map((p) => p[1]).join(","),
        elevation: pts.map(() => "nan").join(","),            // средняя высота ячейки, как в ERA5
        hourly: "precipitation,temperature_2m,snow_depth_water_equivalent",
        past_days: "61", forecast_days: "8", timezone: "GMT",
      });
      const wx = await fetchJSON(OPEN_METEO + "?" + qs);
      const series = aggregateWeather(Array.isArray(wx) ? wx : [wx], owner);

      let quakes = [];
      let eqNote = "";
      try {
        const q = new URLSearchParams({
          format: "geojson", starttime: SR.fmtDate(SR.addDays(series.start, -7)),
          latitude: D.aoi.center[0], longitude: D.aoi.center[1],
          maxradiuskm: D.aoi.eq_radius_km, minmagnitude: D.aoi.eq_min_mag, orderby: "time-asc",
        });
        const js = await fetchJSON(USGS + "?" + q);
        quakes = js.features.map((f) => ({ t: f.properties.time, mag: f.properties.mag, place: f.properties.place }));
      } catch (e) {
        eqNote = " Каталог USGS недоступен — сейсмика не учтена.";
      }
      const n = series.precip[0].length;
      const eqMax = new Array(n).fill(0);
      const eqEnergy = new Array(n).fill(0);
      quakes.forEach((q) => {
        const i = Math.floor((q.t - series.start.getTime()) / SR.DAY_MS);
        if (i >= 0 && i < n && q.mag != null) {
          eqMax[i] = Math.max(eqMax[i], q.mag);
          eqEnergy[i] += SR.quakeEnergy(q.mag);
        }
      });

      const feats = [];
      const preds = [];
      M.basins.forEach((_, bi) => {
        const f = SR.seriesFeatures({
          start: series.start, precip: series.precip[bi], tmean: series.tmean[bi], swe: series.swe[bi], eqMax, eqEnergy,
        }, M);
        feats.push(f);
        preds.push(f.map((fi) => SR.predict(M, fi, bi)));
      });
      const todayIdx = clamp(Math.round((todayUTC() - series.start) / SR.DAY_MS), 0, n - 1);
      live = { start: series.start, n, feats, preds, todayIdx, idx: todayIdx };

      const recent = quakes.filter((q) => q.t >= Date.now() - 10 * SR.DAY_MS);
      const qTxt = recent.length
        ? `землетрясений M ≥ ${D.aoi.eq_min_mag} за 10 дней: ${recent.length} (макс. M${num(Math.max(...recent.map((q) => q.mag)), 1)})`
        : `землетрясений M ≥ ${D.aoi.eq_min_mag} за 10 дней не было`;
      status.textContent = `Обновлено ${new Date().toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" })}; ${qTxt}.${eqNote}`;

      $("#live-days").innerHTML = Array.from({ length: n - todayIdx }, (_, k) => {
        const i = todayIdx + k;
        const lbl = k === 0 ? "Сегодня" : k === 1 ? "Завтра" : fmtDay(SR.addDays(live.start, i));
        return `<button type="button" class="chip" data-i="${i}" aria-pressed="false">${lbl}</button>`;
      }).join("");
      $$("#live-days .chip").forEach((c) => c.addEventListener("click", () => showLive(+c.dataset.i)));
      showLive(todayIdx);
    } catch (e) {
      status.className = "status err";
      status.textContent = "Не удалось загрузить погоду: " + e.message + ". Проверьте подключение к интернету.";
    } finally {
      btn.disabled = false;
      liveBusy = false;
    }
  }

  /* Почасовые данные 21 узла → суточные ряды по бассейнам, так же как era5.py:
     осадки — сумма за часы 01…24 UTC, температура и SWE — среднее за сутки. */
  function aggregateWeather(locs, owner) {
    const times = locs[0].hourly.time.map((t) => Date.parse(t + ":00Z"));
    const t0 = new Date(Math.floor(times[0] / SR.DAY_MS) * SR.DAY_MS);
    const nd = Math.ceil((times[times.length - 1] - t0) / SR.DAY_MS) + 1;
    const acc = () => ({ s: new Array(nd).fill(0), c: new Array(nd).fill(0) });
    const A = M.basins.map(() => ({ p: acc(), t: acc(), w: acc() }));
    const hourMean = (key, bi, h) => {
      let s = 0;
      let c = 0;
      locs.forEach((l, li) => {
        const v = l.hourly[key][h];
        if (owner[li] === bi && v != null) { s += v; c++; }
      });
      return c ? s / c : null;
    };
    times.forEach((tm, h) => {
      const dT = Math.floor((tm - t0) / SR.DAY_MS);
      const dP = Math.floor((tm - HOUR_MS - t0) / SR.DAY_MS);
      M.basins.forEach((_, bi) => {
        const p = hourMean("precipitation", bi, h);
        const t = hourMean("temperature_2m", bi, h);
        const w = hourMean("snow_depth_water_equivalent", bi, h);
        if (p != null && dP >= 0) { A[bi].p.s[dP] += p; A[bi].p.c[dP]++; }
        if (t != null) { A[bi].t.s[dT] += t; A[bi].t.c[dT]++; }
        if (w != null) { A[bi].w.s[dT] += w; A[bi].w.c[dT]++; }
      });
    });
    // полные сутки: 24 часа осадков и температуры во всех бассейнах
    let first = -1;
    let last = -1;
    for (let d = 0; d < nd; d++) {
      const ok = A.every((a) => a.p.c[d] === 24 && a.t.c[d] === 24);
      if (ok && first < 0) first = d;
      if (ok) last = d;
      else if (first >= 0) break;
    }
    if (first < 0) throw new Error("в ответе нет полных суток данных");
    const out = { start: SR.addDays(t0, first), precip: [], tmean: [], swe: [] };
    A.forEach((a) => {
      const P = [];
      const T = [];
      const W = [];
      let lastW = 0;
      for (let d = first; d <= last; d++) {
        P.push(a.p.s[d]);
        T.push(a.t.s[d] / 24);
        lastW = a.w.c[d] ? a.w.s[d] / a.w.c[d] : lastW;
        W.push(lastW);
      }
      out.precip.push(P);
      out.tmean.push(T);
      out.swe.push(W);
    });
    return out;
  }

  function showLive(i) {
    if (state.mode !== "live" || !live) return;
    live.idx = i;
    $$("#live-days .chip").forEach((c) => c.setAttribute("aria-pressed", String(+c.dataset.i === i)));
    const date = SR.addDays(live.start, i);
    $("#chart-panel").hidden = false;
    $("#chart-title").textContent = "Последние 2 недели и прогноз";
    drawChart({
      start: live.start, from: Math.max(0, live.todayIdx - 14), to: live.n - 1, preds: live.preds, events: null,
      mark: i, forecastFrom: live.todayIdx + 1, onPick: (k) => { if (k >= live.todayIdx) showLive(k); },
    });
    const kind = i < live.todayIdx ? "Прошедший день" : i === live.todayIdx ? "Сегодня" : "Прогноз";
    show(live.preds.map((p) => p[i]), date, `${kind} · погода Open-Meteo, сейсмика USGS`, live.feats.map((f) => f[i]));
  }

  // ================================================================== КАРТА
  let map = null;
  let overlay = null;
  let legend = null;
  let basinLabels = [];
  const G = D.grid;
  const canvas = document.createElement("canvas");
  canvas.width = G.nlon;
  canvas.height = G.nlat;
  const RAMP = [[255, 255, 204], [255, 237, 160], [254, 217, 118], [254, 178, 76], [253, 141, 60], [252, 78, 42], [227, 26, 28], [177, 0, 38]];
  function ramp(t) {
    const x = clamp(t, 0, 1) * (RAMP.length - 1);
    const i = Math.min(Math.floor(x), RAMP.length - 2);
    const f = x - i;
    return RAMP[i].map((c, k) => Math.round(c + (RAMP[i + 1][k] - c) * f));
  }
  const RISK_LO = -3;                                 // 0,1 %
  const RISK_HI = Math.log10(0.5);                    // 50 %
  const riskT = (r) => (Math.log10(Math.max(r, 1e-6)) - RISK_LO) / (RISK_HI - RISK_LO);

  function initMap() {
    const b = [[G.lat0, G.lon0], [G.lat0 + G.nlat * G.res, G.lon0 + G.nlon * G.res]];
    map = L.map("map", { scrollWheelZoom: false, zoomSnap: 0.25 }).fitBounds(b);
    // если страница открылась в скрытой вкладке, у карты нулевой размер и fitBounds
    // даёт масштаб всего мира — подгоняем заново, когда контейнер получит размер
    let fitted = map.getSize().x > 0;
    if (window.ResizeObserver) {
      new ResizeObserver(() => {
        map.invalidateSize();
        if (!fitted && map.getSize().x > 0) { fitted = true; map.fitBounds(b); }
      }).observe($("#map"));
    }
    const osm = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 17, attribution: "© участники OpenStreetMap",
    }).addTo(map);
    const topo = L.tileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", {
      maxZoom: 17, attribution: "© OpenStreetMap, SRTM · стиль © OpenTopoMap (CC-BY-SA)",
    });
    L.control.layers({ "Карта": osm, "Рельеф": topo }, null, { position: "topright" }).addTo(map);
    overlay = L.imageOverlay(canvas.toDataURL(), b, { className: "grid-overlay", interactive: false }).addTo(map);

    D.lakes.forEach((l) => {
      L.circleMarker([l.lat, l.lon], { radius: 5, color: "#fff", weight: 1.5, fillColor: "#1e88e5", fillOpacity: 1 })
        .bindTooltip(`${esc(l.name)} · ${num(l.area_km2, 3)} км²`).addTo(map);
    });
    D.history.forEach((h) => {
      L.marker([h.source_lat, h.source_lon], { icon: L.divIcon({ className: "hist-mark", html: "<div></div>", iconSize: [12, 12] }) })
        .bindPopup(`<b>${esc(h.date)} · ${esc(h.basin)}</b><br>${esc(h.notes)}`).addTo(map);
    });
    basinLabels = D.basins.map((bs, i) => L.marker([bs.lat, bs.lon], {
      icon: L.divIcon({ className: "basin-label", html: "", iconSize: [0, 0] }), interactive: false,
    }).addTo(map));

    legend = L.control({ position: "bottomright" });
    legend.onAdd = () => L.DomUtil.create("div", "map-legend");
    legend.addTo(map);

    map.on("click", (e) => {
      const col = Math.floor((e.latlng.lng - G.lon0) / G.res);
      const row = Math.floor((e.latlng.lat - G.lat0) / G.res);
      if (col < 0 || row < 0 || col >= G.nlon || row >= G.nlat || !state.results) return;
      const k = row * G.nlon + col;
      const bi = +G.basin[k];
      const s = G.susc[k] / 1000;
      const p = state.results[bi].p;
      L.popup().setLatLng(e.latlng).setContent(
        `<b>${esc(M.basins[bi])}</b><br>Подверженность склона: <b>${num(s, 2)}</b><br>` +
        `Вероятность в бассейне: <b>${fmtPct(p)}%</b><br>Риск ячейки: <b>${fmtPct(s * p)}%</b>`).openOn(map);
    });

    $$(".seg button").forEach((btn) => btn.addEventListener("click", () => {
      state.layer = btn.dataset.layer;
      $$(".seg button").forEach((x) => x.setAttribute("aria-pressed", String(x === btn)));
      updateMap();
    }));
  }

  function updateMap() {
    if (!map || !state.results) return;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(G.nlon, G.nlat);
    const ps = state.results.map((r) => r.p);
    for (let row = 0; row < G.nlat; row++) {
      for (let col = 0; col < G.nlon; col++) {
        const k = row * G.nlon + col;
        const s = G.susc[k] / 1000;
        let rgb;
        let a;
        if (state.layer === "susc") {
          rgb = ramp(s);
          a = 0.2 + 0.55 * s;
        } else {
          const r = s * ps[+G.basin[k]];
          const t = riskT(r);
          if (t <= 0) { a = 0; rgb = [0, 0, 0]; } else { rgb = ramp(t); a = 0.12 + 0.55 * clamp(t, 0, 1); }
        }
        const o = ((G.nlat - 1 - row) * G.nlon + col) * 4;          // строка 0 — юг
        img.data[o] = rgb[0]; img.data[o + 1] = rgb[1]; img.data[o + 2] = rgb[2]; img.data[o + 3] = Math.round(a * 255);
      }
    }
    ctx.putImageData(img, 0, 0);
    overlay.setUrl(canvas.toDataURL());

    basinLabels.forEach((m, i) => m.setIcon(L.divIcon({
      className: "basin-label", iconSize: [0, 0],
      html: `<div style="--bc:${BCOL[i]}">${esc(M.basins[i])} · ${fmtPct(ps[i])}%</div>`,
    })));

    const stops = Array.from({ length: 9 }, (_, i) => `rgb(${ramp(i / 8).join(",")})`).join(",");
    legend.getContainer().innerHTML = state.layer === "susc"
      ? `<b>Подверженность склона</b><div class="bar" style="background:linear-gradient(90deg,${stops})"></div>
         <div class="ticks"><span>0</span><span>0,5</span><span>1</span></div>`
      : `<b>Риск в ячейке за сутки</b><div class="bar" style="background:linear-gradient(90deg,${stops})"></div>
         <div class="ticks">${[0.001, 0.01, 0.05, 0.2, 0.5].map((v) => `<span>${num(v * 100, v < 0.01 ? 1 : 0)}%</span>`).join("")}</div>`;
  }

  // ================================================================== ВКЛАДКИ
  function setMode(mode) {
    state.mode = mode;
    $$(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === mode)));
    $$("[data-pane]").forEach((p) => { p.hidden = p.dataset.pane !== mode; });
    state.sel = null;
    if (mode === "manual") computeManual();
    if (mode === "archive") showArchive(arc.idx != null ? arc.idx : topIdx());
    if (mode === "live") {
      if (live) showLive(live.idx);
      else {
        $("#chart-panel").hidden = true;
        loadLive();
      }
    }
  }

  function topIdx() {
    return Math.round((SR.parseDate(D.top_days[0].date) - arc.start) / SR.DAY_MS);
  }

  // ================================================================== О МОДЕЛИ
  function renderAbout() {
    const m = M.metrics;
    $("#m-susc").textContent = num(D.susc_metrics.cv_roc_auc, 2);
    $("#m-auc").textContent = num(m.holdout_roc_auc, 2);
    $("#m-exp").textContent = num(m.holdout_expected_events, 1);
    $("#m-obs").textContent = m.holdout_events;
    $("#m-eq").textContent = Object.keys(D.archive.eq).length;
    $("#m-gen").textContent = fmtDateRu(SR.parseDate(D.generated));
    let lo = 0;
    $("#levels").innerHTML = `<thead><tr><th>Уровень</th><th>Вероятность за сутки</th><th>Доля дней 2016–2023</th></tr></thead><tbody>` +
      M.levels.map((l, i) => {
        const range = l.max > 1 ? `≥ ${num(lo * 100, 0)} %` : `${num(lo * 100, 0)}–${num(l.max * 100, 0)} %`;
        lo = l.max;
        return `<tr><td><span class="lvl" style="--lc:${l.color}">${l.name}</span></td><td>${range}</td><td>${num(M.level_share[i] * 100, 1)} %</td></tr>`;
      }).join("") +
      `<tr><td>средний день</td><td>${num(M.base_rate * 100, 2)} %</td><td>${M.n_events} событий за ${M.n_days} дней × ${NB} бассейна</td></tr></tbody>`;
  }

  // ================================================================== СТАРТ
  $$(".tabs button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  $("#live-load").addEventListener("click", loadLive);
  buildManual();
  buildArchive();
  renderAbout();
  syncManual();
  try { initMap(); } catch (e) { $("#map").textContent = "Карта не загрузилась (нет доступа к интернету?)."; }
  applyPreset(PRESETS[2], $$("#presets .chip")[2]);
})();
