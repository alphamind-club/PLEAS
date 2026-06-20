# C4 Case Study - BioClaw 24-Task Benchmark

This folder contains the C4 benchmark evidence for the paper. It compares three
systems on the same 24 biomedical tool-use tasks:

- BioClaw
- Biomni
- ChatGPT (`gpt-5.5-thinking-extended`)

The benchmark is scored with the BioClaw 0/1/2 rubric for each task:

- `2` = complete answer, correct database/tool, requested fields present
- `1` = partial answer, minor factual error, wrong/missing field, or incomplete tool use
- `0` = failed, wrong answer, hallucinated answer, no answer, or required tool not used

There are 24 tasks, so the maximum score is 48 points.

Scoring provenance: the score sheets in this folder were graded by a fresh
instance of `gpt-5.5-thinking-extended` using the BioClaw rubric. They are not
author-scored results.

## GPT-5.5 Thinking Extended Grading Summary

| System | Score | Percent | Complete | Partial | Failed |
|--------|------:|--------:|---------:|--------:|-------:|
| BioClaw | 42/48 | 87.5% | 20 | 2 | 2 |
| Biomni | 38/48 | 79.2% | 15 | 8 | 1 |
| ChatGPT (`gpt-5.5-thinking-extended`) | 28/48 | 58.3% | 11 | 6 | 7 |

This table preserves the GPT-5.5 Thinking Extended grading rows. Because T11 has
since been corrected against NCBI GEO, this table should be rechecked against
raw system outputs before being used as final paper wording.

## What This Case Study Tests

The benchmark covers six biomedical task families:

| Category | Tasks | Examples |
|----------|-------|----------|
| Sequence and variant analysis | T01-T04 | UniProt, BLAST, ClinVar, Ensembl |
| Drug-target discovery | T05-T08 | ChEMBL, PubChem, Open Targets, dual inhibitors |
| Literature and clinical mining | T09-T12 | PubMed, ClinicalTrials.gov, GEO, Europe PMC |
| Transcriptomics and single-cell | T13-T16 | Cell markers, GEO DEGs, Enrichr/KEGG, HCA/scRNA-seq |
| Structural and protein biology | T17-T20 | AlphaFold DB, STRING, RCSB PDB, UniProt phosphorylation sites |
| Pathway and systems biology | T21-T24 | KEGG, Reactome, STRING degree, QuickGO |

Each task requires a concrete biomedical database or API, not just a plausible
free-text biological answer.

## Source Of Truth

Use these two files for tables, plots, and paper claims:

| File | Use |
|------|-----|
| `c4_results_summary.csv` | System-level comparison: one row each for BioClaw, Biomni, and ChatGPT. |
| `c4_task_scores_by_system.csv` | Task-level comparison with one row per task and one score column per system. |

The task definitions and official rubric are in:

| File | Use |
|------|-----|
| `../bioclaw_24tasks/bioclaw_24_tasks.txt` | Full 24-task benchmark prompt set and scoring rubric. |

## Grading Notes

The detailed GPT-5.5 Thinking Extended grading rationales are preserved
separately:

| File | System |
|------|--------|
| `bioclaw_uploaded_json_harsh_grading.csv` | BioClaw task-level grading |
| `bioclaw_uploaded_json_harsh_grading_notes.md` | BioClaw human-readable grading notes |
| `biomni_baseline_consolidated_answer.csv` | Biomni task-level grading |
| `biomni_baseline_notes.md` | Biomni human-readable grading notes |
| `chatgpt_baseline_gpt-5.5-thinking-extended.csv` | ChatGPT task-level grading |
| `chatgpt_baseline_notes.md` | ChatGPT human-readable grading notes |

## Interpretation

Under the GPT-5.5 Thinking Extended grading, BioClaw has the highest total score
and the highest number of fully completed tasks. Its remaining failures were
recorded as:

- `T10` - ClinicalTrials.gov recruiting U.S. TNBC trial count
- `T11` - GEO GSE68849 metadata, which now requires regrading after the C13
  correction

BioClaw receives partial credit on:

- `T03` - ClinVar BRCA1 variant ID mismatch
- `T09` - PubMed most-recent PMID issue

Biomni is competitive but loses more points through partial answers, especially
where required fields or exact task constraints are missing. ChatGPT is weakest
under this rubric because several answers either omit required tool/API use or
fail exact dataset/query constraints.

## Important Consistency Rule

For C4, use only the three-system 48-point comparison after T11 is regraded:

- BioClaw: `42/48 = 87.5%`
- Biomni: `38/48 = 79.2%`
- ChatGPT: `28/48 = 58.3%`

Do not use older coverage-style numbers for the C4 comparison. The final
comparison must stay on the same 24-task, 48-point rubric for all three systems.

## T11 Correction Notice

After C13 reconciliation, the benchmark T11 expected answer was corrected
against NCBI GEO. GSE68849 maps to GDS6063, "Influenza A effect on
plasmacytoid dendritic cells", Homo sapiens, 10 samples. Older grading notes
penalized pDC/influenza answers because an earlier draft rubric incorrectly
expected pancreatic islets / human + mouse / 26 samples.

The score files in this folder preserve the GPT-5.5 Thinking Extended grading
rows. Before C4 is used as a fully final result, T11 should be regraded from the
raw system outputs against the corrected T11 rubric.

## Limitations

- Raw API logs are not included for all systems.
- Biomni grading accepts stated database/API usage even where raw logs are absent.
- The ChatGPT score is computed from the GPT-5.5 Thinking Extended task-level
  rubric rows.
- T11 requires regrading from raw outputs after the official GEO correction.
- This folder is an evidence artifact for the C4 benchmark result, not a full
  reproducibility package with rerunnable tool traces.
