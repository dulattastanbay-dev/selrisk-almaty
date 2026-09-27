// Паритет JS ↔ Python: вероятности, посчитанные web/model.js по архивным рядам,
// должны совпадать с p_py, посчитанными в scripts/export_web.py.
// Запуск: node tests/web_parity.js   (печатает JSON с максимальным расхождением)
const path = require("path");
global.window = {};
require(path.join(__dirname, "..", "web", "data", "selrisk_data.js"));
const SR = require(path.join(__dirname, "..", "web", "model.js"));

const D = window.SELRISK_DATA;
const M = D.model;
const A = D.archive;
const start = SR.parseDate(A.start);
const eqMax = new Array(A.n).fill(0);
const eqEnergy = new Array(A.n).fill(0);
for (const [k, [mag, energy]] of Object.entries(A.eq)) {
  eqMax[+k] = mag;
  eqEnergy[+k] = energy;
}

let maxAbs = 0;
let maxRel = 0;
let n = 0;
M.basins.forEach((b, bi) => {
  const s = A.basins[b];
  const feats = SR.seriesFeatures({ start, precip: s.precip, tmean: s.tmean, swe: s.swe, eqMax, eqEnergy }, M);
  feats.forEach((f, i) => {
    const p = SR.predict(M, f, bi).p;
    const ref = s.p_py[i];
    maxAbs = Math.max(maxAbs, Math.abs(p - ref));
    if (ref > 1e-3) maxRel = Math.max(maxRel, Math.abs(p - ref) / ref);
    n++;
  });
});
console.log(JSON.stringify({ n, maxAbs, maxRel }));
