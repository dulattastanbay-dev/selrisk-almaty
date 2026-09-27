"""Streamlit-демо запускается без ошибок и показывает проценты по всем бассейнам."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.mark.skipif(not (ROOT / "data" / "outputs" / "meteo_with_trigger.csv").exists(),
                    reason="сначала python scripts/run_all.py")
def test_streamlit_app_runs():
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception
    assert len(at.metric) == 3
    assert all(m.value.endswith("%") for m in at.metric)
