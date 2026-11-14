# Dataset v0.2: provenance and terms

The MIT license covers original program code, not third-party datasets. The ZIP
is the frozen research snapshot requested by the repository owner; publication
does not create or resolve upstream redistribution rights.

- Reading event and center records: [research platform](https://research.reading.ac.uk/glof-predicting-debris-flow/)
  and [ArcGIS service](https://services1.arcgis.com/PhkL97KbkzUBf4PQ/arcgis/rest/services/Platform/FeatureServer/0).
  Preserve attribution to the authors of [the 2024 paper](https://www.nature.com/articles/s44304-024-00050-7).
  The service metadata snapshot supplies no clear `licenseInfo`. Obtain written
  clarification before a third-party republication or commercial deployment.
- NASA POWER: preserve [POWER acknowledgement](https://power.larc.nasa.gov/docs/referencing/)
  and [API provenance](https://power.larc.nasa.gov/docs/services/api/temporal/daily/).
- NOAA GHCN-Daily: preserve station IDs, element flags and
  [format and quality documentation](https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt).
- Terrain Tiles: retain [dataset attribution](https://registry.opendata.aws/terrain-tiles/)
  and [upstream mosaic attribution](https://github.com/tilezen/joerd/blob/master/docs/attribution.md).
  This is a resampled regional terrain product, not a new Copernicus GLO-30 survey.
- HMA GLOF supplement: [upstream repository](https://github.com/fidelsteiner/HMAGLOFDB),
  pinned commit `18d9d336e8e152053ee78246781cc37d510194c7`.
  Upstream LICENSE and README disagree (CC0 / CC-BY in the recorded snapshot).
  Keep both notices and citations; no blanket relicensing is asserted.
- Official event confirmations: attribution and URLs accompany individual records.
  Metadata and derived factual summaries do not imply permission to republish
  complete news stories or official bulletins.

Inside the archive, `metadata/source_register.csv`, `downloads.json`,
`event_evidence.md`, `checksums.sha256` and `sources/` retain provenance.
No credentials or private records are needed to use the frozen snapshot.
