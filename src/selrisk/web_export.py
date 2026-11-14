"""Reproducible static export of the real research workflow."""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from .realdata import ROOT, ensure_dataset, verify_file, load_regional
from .research import OUTPUT, write_json


def records(frame):
    return json.loads(frame.to_json(orient="records",force_ascii=False,double_precision=10))


def compact(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n",encoding="utf-8")


def export_web():
    root=ensure_dataset()
    out=ROOT/"web/data"
    out.mkdir(parents=True,exist_ok=True)
    for kind in ("power","stations"):
        frame,features,_=load_regional(kind)
        for name in ("model","report"):
            path=OUTPUT/f"{name}_{kind}.json"
            if not path.exists():
                raise ValueError("Run python scripts/train.py first; export never silently fits a model")
            shutil.copy2(path,out/path.name)
        pred=pd.read_csv(OUTPUT/f"predictions_{kind}.csv")
        cols=["date","target","label_status","split","feature_complete",
              "date_uncertainty_overlap"]+features
        if kind=="power":
            cols+=["feature_window_end_utc"]
        else:
            cols+=["feature_window_last_observation_date"]
        compact(out/f"archive_{kind}.json",
                json.loads(frame[cols].to_json(orient="split",index=False,force_ascii=False,double_precision=10)))
        shutil.copy2(OUTPUT/f"predictions_{kind}.csv",out/f"predictions_{kind}.csv")
    events=verify_file(root,"data/event_catalogue.csv")
    shutil.copy2(events,out/"event_catalogue.csv")
    compact(out/"events.json",records(pd.read_csv(events)))
    from .presentation import presentation_examples
    examples = presentation_examples(
        pd.read_csv(OUTPUT/"predictions_power.csv"),
        json.loads((OUTPUT/"model_power.json").read_text(encoding="utf-8")),
        pd.read_csv(events))
    write_json(out/"presentation_examples.json", examples)
    shutil.copy2(verify_file(root,"data/mudflow_centers.geojson"),out/"centers.geojson")
    shutil.copy2(verify_file(root,"metadata/quality_report.json"),out/"quality_report.json")
    shutil.copy2(verify_file(root,"metadata/source_register.csv"),out/"source_register.csv")
    print(f"Static real-data export: {out}")
