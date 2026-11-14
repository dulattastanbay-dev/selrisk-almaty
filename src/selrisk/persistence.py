"""Persist fitted pipelines; portable inference never unpickles external files."""
from __future__ import annotations
import hashlib
import json
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from .research import write_json


def save_pipeline(model, output, kind):
    output = Path(output)
    path = output / f"model_{kind}.joblib"
    joblib.dump(model, path, compress=3)
    files = [path, output / f"model_{kind}.json"]
    write_json(output / f"training_{kind}.json", {
        "dataset_kind": kind, "python": platform.python_version(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__, "joblib": joblib.__version__},
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "joblib_warning": "Load trusted files only, with matching training dependencies.",
        "portable_format": f"model_{kind}.json",
        "contains_fitted_preprocessing": True,
        "final_refit_on_all_data": False,
    })


def load_artifact(path, kind=None):
    """Read the declarative JSON format; never execute pickle/joblib code."""
    from .realdata import load_regional
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    actual_kind = artifact.get("dataset_kind")
    if actual_kind not in {"power", "stations"} or (kind and actual_kind != kind):
        raise ValueError("Model dataset kind mismatch")
    _, allowed, _ = load_regional(actual_kind)
    if (artifact.get("features") != allowed or artifact.get("version") != "0.2"
            or artifact.get("model_type") != "presence_background_logistic_ranking"):
        raise ValueError("Unsupported model schema or feature order")
    for key in ("mean", "scale", "coefficients", "feature_min", "feature_max"):
        a = np.asarray(artifact[key], dtype=float)
        if a.shape != (len(allowed),) or not np.isfinite(a).all():
            raise ValueError(f"Invalid model field: {key}")
    if (np.any(np.asarray(artifact["scale"]) <= 0)
            or np.any(np.asarray(artifact["feature_min"]) > artifact["feature_max"])
            or not np.isfinite(float(artifact["intercept"]))):
        raise ValueError("Invalid model bounds, scale or intercept")
    return artifact
