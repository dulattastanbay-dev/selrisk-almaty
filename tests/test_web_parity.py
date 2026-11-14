import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from selrisk.realdata import load_regional
from selrisk.research import score_rows
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("kind",["power","stations"])
@pytest.mark.skipif(shutil.which("node") is None,reason="Node required")
def test_js_matches_python_all_days(kind):
    out=subprocess.run(["node",str(ROOT/"tests/web_parity.js"),kind],capture_output=True,text=True,check=True)
    js=json.loads(out.stdout)
    frame,_,_=load_regional(kind)
    artifact=json.loads((ROOT/f"web/data/model_{kind}.json").read_text())
    scores,status=score_rows(artifact,frame)
    assert len(js)==len(frame)
    assert [r["status"] for r in js]==status.tolist()
    vals=np.array([np.nan if r["score"] is None else r["score"] for r in js])
    np.testing.assert_allclose(vals,scores,rtol=0,atol=2e-8,equal_nan=True)
