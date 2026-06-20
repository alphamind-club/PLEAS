---
description: BioPLEASE data lake catalog and usage guidance
when_to_use: Use when the task could benefit from searching or reusing the bundled BioPLEASE data lake instead of downloading a new dataset from scratch.
---

# BioPLEASE Data Lake

The bundled BioPLEASE data lake lives at:
`G:/BioClaw/BioPlease_tools/data_lake`

Use it as a read-only catalog unless the user explicitly asks to modify those assets.

Representative files:
- affinity_capture-ms.parquet
- affinity_capture-rna.parquet
- BindingDB_All_202409.tsv
- bioplease_package.zip
- broad_repurposing_hub_molecule_with_smiles.parquet
- broad_repurposing_hub_phase_moa_target_info.parquet
- co-fractionation.parquet
- czi_census_datasets_v4.parquet
- ddinter_alimentary_tract_metabolism.csv
- ddinter_antineoplastic.csv
- ddinter_antiparasitic.csv
- ddinter_blood_organs.csv
- ddinter_dermatological.csv
- ddinter_hormonal.csv
- ddinter_respiratory.csv
- ddinter_various.csv
- DepMap_CRISPRGeneDependency.csv
- DepMap_CRISPRGeneEffect.csv

Guidance:
- Search this catalog before downloading redundant biomedical datasets.
- Copy only the specific files needed into the opened working folder when a run needs local writable outputs.
- Record provenance in reports or summaries when using a bundled dataset.
