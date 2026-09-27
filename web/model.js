/* СельРиск — расчёт признаков и суточной вероятности схода селя.
 *
 * Точная копия Python-версии: признаки — selrisk/features.py + dataset.py
 * (снеготаяние по убыли SWE, сейсмика по окнам 3/7 дней), модель —
 * логистическая регрессия из scripts/export_web.py. Работает в браузере и в
 * Node (для проверки паритета с Python: tests/test_web_parity.py).
 */
(function (root) {
  "use strict";

  const DAY_MS = 86400000;

  const GROUPS = [
    { key: "rain", label: "Осадки за 1–7 дней", feats: ["rain_1d", "rain_3d", "rain_7d"] },
    { key: "api", label: "Увлажнённость грунта", feats: ["api"] },
    { key: "snow", label: "Снег и его таяние", feats: ["pdd_3d", "snowmelt_mm", "swe_mm"] },
    { key: "eq", label: "Землетрясения", feats: ["eq_max_mag_3d", "eq_energy_7d"] },
    { key: "season", label: "Время года", feats: ["doy_sin", "doy_cos"] },
    { key: "basin", label: "Особенности бассейна", feats: null },   // все is_* признаки
  ];

  function parseDate(s) {
    const [y, m, d] = s.split("-").map(Number);
    return new Date(Date.UTC(y, m - 1, d));
  }

  function fmtDate(d) {
    return d.toISOString().slice(0, 10);
  }

  function addDays(d, n) {
    return new Date(d.getTime() + n * DAY_MS);
  }

  function dayOfYear(d) {
    return Math.round((d - Date.UTC(d.getUTCFullYear(), 0, 1)) / DAY_MS) + 1;
  }

  function seasonFeatures(d) {
    const a = (2 * Math.PI * dayOfYear(d)) / 365.25;
    return { doy_sin: Math.sin(a), doy_cos: Math.cos(a) };
  }

  function quakeEnergy(mag) {
    return Math.pow(10, 1.5 * mag);          // как energy_from_magnitude() в Python
  }

  /* Признаки для каждого дня суточного ряда одного бассейна.
   * s = {start: Date, precip[], tmean[], swe[], eqMax[], eqEnergy[]}
   * eqMax/eqEnergy — суточный максимум магнитуды и сумма энергии (0, если толчков не было). */
  function seriesFeatures(s, model) {
    const n = s.precip.length;
    const decay = model.api_decay;
    const base = model.melt_base_temp_c;
    const out = new Array(n);
    const winSum = (arr, i, w) => {
      let t = 0;
      for (let k = Math.max(0, i - w + 1); k <= i; k++) t += arr[k] || 0;
      return t;
    };
    const pdd = s.tmean.map((t) => Math.max(t - base, 0));
    let api = 0;
    for (let i = 0; i < n; i++) {
      api = decay * api + s.precip[i];
      let eqMax = 0;
      for (let k = Math.max(0, i - 2); k <= i; k++) eqMax = Math.max(eqMax, (s.eqMax && s.eqMax[k]) || 0);
      const melt = i === 0 ? 0 : Math.max(s.swe[i - 1] - s.swe[i], 0);
      out[i] = Object.assign({
        rain_1d: s.precip[i],
        rain_3d: winSum(s.precip, i, 3),
        rain_7d: winSum(s.precip, i, 7),
        api: api,
        pdd_3d: winSum(pdd, i, 3),
        snowmelt_mm: melt,
        swe_mm: s.swe[i],
        eq_max_mag_3d: eqMax,
        eq_energy_7d: Math.log1p(winSum(s.eqEnergy || [], i, 7)),
      }, seasonFeatures(addDays(s.start, i)));
    }
    return out;
  }

  /* Признаки из ручного ввода.
   * inp = {date: Date, rain: [7 значений, последнее — сегодня], rainPrev21, tmean, swe, melt, quakeMag} */
  function manualFeatures(inp, model) {
    const q = model.api_decay;
    const r = inp.rain.slice().reverse();       // r[0] — сегодня, r[6] — 6 дней назад
    let api = 0;
    r.forEach((p, k) => { api += Math.pow(q, k) * p; });
    // предыстория: до этой недели осадки считаем равномерными, rainPrev21/21 мм/сут
    // (сумма геометрического ряда по всем более ранним дням)
    const perDay = Math.max(inp.rainPrev21, 0) / 21;
    api += (Math.pow(q, 7) * perDay) / (1 - q);
    const m = Math.max(inp.quakeMag || 0, 0);
    return Object.assign({
      rain_1d: r[0],
      rain_3d: r[0] + r[1] + r[2],
      rain_7d: r.reduce((a, b) => a + b, 0),
      api: api,
      pdd_3d: 3 * Math.max(inp.tmean - model.melt_base_temp_c, 0),
      snowmelt_mm: Math.max(inp.melt, 0),
      swe_mm: Math.max(inp.swe, 0),
      eq_max_mag_3d: m,
      eq_energy_7d: m > 0 ? Math.log1p(quakeEnergy(m)) : 0,
    }, seasonFeatures(inp.date));
  }

  /* Вероятность схода за сутки в бассейне basinIdx.
   * Значения за пределами диапазона обучающих данных обрезаются до границ
   * (модель не экстраполирует) и перечисляются в `clipped`. */
  function predict(model, raw, basinIdx) {
    const x = [];
    const clipped = [];
    model.raw_features.forEach((f, j) => {
      let v = raw[f];
      if (v > model.clip_max[j]) { v = model.clip_max[j]; clipped.push(f); }
      if (v < model.clip_min[j]) v = model.clip_min[j];
      x.push(v);
    });
    model.basins.slice(1).forEach((_, j) => x.push(basinIdx === j + 1 ? 1 : 0));
    let logit = model.intercept;
    const contrib = {};
    x.forEach((v, j) => {
      const c = (model.coef[j] * (v - model.mean[j])) / model.scale[j];
      contrib[model.features[j]] = c;
      logit += c;
    });
    return { p: 1 / (1 + Math.exp(-logit)), logit, contrib, clipped, raw };
  }

  /* Вклад групп факторов в логит (относительно дня со средними условиями). */
  function groupContrib(model, contrib) {
    return GROUPS.map((g) => {
      const feats = g.feats || model.features.filter((f) => f.startsWith("is_"));
      return { key: g.key, label: g.label, value: feats.reduce((a, f) => a + (contrib[f] || 0), 0) };
    });
  }

  function level(model, p) {
    return model.levels.find((l) => p < l.max) || model.levels[model.levels.length - 1];
  }

  const api = {
    DAY_MS, GROUPS, parseDate, fmtDate, addDays, dayOfYear, quakeEnergy,
    seriesFeatures, manualFeatures, predict, groupContrib, level,
  };
  root.SelRiskModel = api;
  if (typeof module === "object" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
