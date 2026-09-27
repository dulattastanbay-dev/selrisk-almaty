"""Сайт (web/model.js) должен считать те же вероятности, что и Python-модель."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "web" / "data" / "selrisk_data.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="нужен Node.js")
@pytest.mark.skipif(not DATA.exists(), reason="сначала python scripts/export_web.py")
def test_js_matches_python():
    out = subprocess.run(["node", str(ROOT / "tests" / "web_parity.js")],
                         capture_output=True, text=True, check=True, encoding="utf-8")
    res = json.loads(out.stdout)
    assert res["n"] > 1000
    assert res["maxAbs"] < 2e-3     # архив хранится с округлением до 0.01 мм / 0.01 °C
    assert res["maxRel"] < 0.05
