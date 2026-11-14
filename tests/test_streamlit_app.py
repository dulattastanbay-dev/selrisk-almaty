from pathlib import Path
import pytest
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]

def test_streamlit_real_data():
    at=AppTest.from_file(str(ROOT/"app/streamlit_app.py"),default_timeout=120).run()
    assert not at.exception
    assert at.metric[0].value=="137"
    assert at.metric[2].value=="0"
    assert all(not m.value.endswith("%") for m in at.metric)
    at.selectbox[0].select("stations").run()
    assert not at.exception
    assert at.metric[0].value=="102"
    from datetime import date
    at.date_input[0].set_value(date(2030,1,1)).run()
    assert not at.exception
    assert any("Нет данных" in x.value for x in at.info)
