# Scientific panel integration: scope and limitations

## Safe default

`python scripts/run_all.py` and `python scripts/train.py` no longer generate or select synthetic training data by default. They validate the real v0.1 panel and write a separate scientific report. There is no fallback to `data/sample`.

Place the previously generated `model_dataset.csv` at `data/real/model_dataset.csv`, or pass its path:

```sh
python scripts/run_all.py --dataset /path/to/model_dataset.csv
python scripts/train.py --dataset /path/to/model_dataset.parquet
```

Parquet additionally requires `pip install pyarrow`; CSV works with the existing pandas dependency. No uploaded dataset is bundled by this patch. The previous dataset repository was empty at inspection, so neither its presence nor the integrity of the earlier downloadable snapshot is claimed here.

The schema contains exactly 29 columns: 20 explicit features and 9 metadata/target columns. Validation does not mutate labels, drop unlabeled rows, or map point coordinates into fabricated catchment polygons. It checks keys, ranges, dates, nested accumulation windows, event metadata, seasonal encoding and chronological split. Earthquake magnitude/distance remain null when count is zero. A SHA-256 binds the report to the provided file; it is not proof of upstream provenance authenticity.

## Expected blocked result, not false success

The report is `data/outputs/scientific/validation.json`. Exit status 2 means training remains blocked. A valid v0.1 dataset can have `schema_status=valid` and `training_status=blocked_insufficient_ground_truth` simultaneously. The historical panel has positives and unknowns, not reviewed negatives. Its original training split has no positive labels. No calibrated model, AUC, risk percentage or GIS hazard layer is produced by the real-data route. A future supervised contract must review negatives, independent storm counts, all splits, availability times and minimum evidence before fitting. Do not fill null targets with zero just to make training pass.

An invalid or missing input overwrites the validation report with an explicit blocked status rather than leaving a previous success report. Existing production models and static website assets are not removed or overwritten; their presence must not be interpreted as a validated result of this patch.

## Explicit legacy demo

```sh
python scripts/run_all.py --demo
```

This explicitly runs the existing synthetic/mixed-data demonstration. Model and tabular output paths are isolated under `demo/`. Direct `train.py --demo` only uses `data/sample`, never silently selects processed files. The demo can still use the legacy generated terrain/labels and is not a scientific benchmark. The legacy map generator may write demo map files under its configured docs location: this patch does not certify those maps or a website deployment.

## Tests and remaining work

`python -m pytest tests/test_scientific_panel.py` checks artificial contract fixtures. These fixtures are not source observations and are not an added dataset. No full historical panel, full legacy pipeline, browser deployment or remote CI run has been exercised as part of these local unit tests. This is a safety/integration patch, not a claim that every project defect is fixed.

Remaining: supply the exact real dataset snapshot and source manifests, expand reviewed event/negative coverage, replace synthetic spatial layers, audit probability calibration and realtime data availability, verify all other direct CLI/API/UI entrypoints, and run the complete existing test suite and deployment checks. Calling low-level legacy model functions or old evaluation/export scripts directly is outside the new entrypoint guard. Do not publish their outputs as real validation.
