# C13 - GEO Fallback / T11 Metadata Reconciliation

This folder demonstrates the fallback path for BioClaw benchmark task T11:
recovering GSE68849 metadata when the primary NCBI GEO call fails.

## Verified GSE68849 Metadata

Source files:

- `gse68849_ncbi_esearch.json`
- `gse68849_ncbi_esummary_6063.json`
- `geo_pubchem_fallback_demo.log`

Verified answer from NCBI GEO DataSets / ESummary:

| Field | Value |
|-------|-------|
| Series | GSE68849 |
| DataSet | GDS6063 |
| Title | Influenza A effect on plasmacytoid dendritic cells |
| Series title | Impact of influenza A on human plasmacytoid dendritic cells (pDC) gene expression |
| Organism | Homo sapiens |
| Samples | 10 |
| Type | Expression profiling by array |
| Platform | GPL10558 / Illumina HumanHT-12 V4.0 expression beadchip |
| PubMed ID | 26826244 |

NCBI page:

https://www.ncbi.nlm.nih.gov/gds/6063

## Reconciliation

Older draft rubrics incorrectly expected pancreatic islets / human + mouse /
26 samples for GSE68849. That is not the current NCBI GEO DataSets metadata for
GSE68849. The benchmark T11 expected answer has been updated to match NCBI.

## What C13 Proves

Supported:

- The fallback mechanism can route from GEO failure to a secondary literature
  source and back to GEO metadata.
- The verified T11 metadata is now recorded from NCBI GEO.

Not supported by this artifact alone:

- A full rerun of all model answers after the T11 correction.
