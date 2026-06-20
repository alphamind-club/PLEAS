# Supplemental Results Generation Guide

This guide explains how to generate and maintain
`BioClaw_IEEE_IRI2026_v25_Supplemental.docx`. It follows the structure of the
Word supplemental itself.

The current Word supplemental has:

- front matter
- Table S1: full 24-task benchmark scores
- Table S2: call-by-call input token counts
- Table S3: integration test status
- Table S4: single-run cost analysis
- Table S5: ablation study

## Front Matter

The front matter identifies the document as the BioClaw v25 supplemental and
keeps author-identifying information omitted for double-blind review.

The introductory paragraph should say that the supplemental provides the full
data tables behind the main-paper claims. It should also tell the reader how to
interpret status labels in later tables. If the table status language changes,
update this paragraph at the same time so the introduction matches the tables.

## Table S1 - Full 24-Task Benchmark Scores

Table S1 is the main task-performance table. It has one row per benchmark task
and compares three systems:

- BioClaw
- Biomni
- ChatGPT (`gpt-5.5-thinking-extended`)

The table columns are:

| Column | Meaning |
|--------|---------|
| Task | Task identifier, T01 through T24. |
| Category | Biomedical task family, such as sequence/variant, drug-target, literature, transcriptomics, structural biology, or pathway biology. |
| Difficulty | Easy, medium, or hard. |
| Description | Short natural-language task description. |
| BioClaw | BioClaw score on the 0/1/2 rubric. |
| Biomni | Biomni score on the same rubric. |
| ChatGPT | ChatGPT baseline score on the same rubric. |

### 24-Task Scoring Rubric

Each task is scored with the same 0/1/2 rubric. There are 24 tasks, so the
maximum score is 48 points.

| Score | Meaning | How to interpret it |
|------:|---------|---------------------|
| 2 | Complete | The answer is correct, uses the requested database/tool, and includes the requested fields. |
| 1 | Partial | The answer is mostly on track but has a missing field, minor factual error, incomplete tool use, changed query, missing count/ID, or other incomplete requirement. |
| 0 | Failed | The answer is wrong, absent, hallucinated, uses the wrong source, omits a required tool call, or fails the task constraints. |

The Table S1 results were graded by a fresh instance of
`gpt-5.5-thinking-extended` using the BioClaw 24-task rubric. They were not
author-scored.

The current pre-regrade totals are:

| System | Score | Percent | Complete | Partial | Failed |
|--------|------:|--------:|---------:|--------:|-------:|
| BioClaw | 42/48 | 87.5% | 20 | 2 | 2 |
| Biomni | 38/48 | 79.2% | 15 | 8 | 1 |
| ChatGPT (`gpt-5.5-thinking-extended`) | 28/48 | 58.3% | 11 | 6 | 7 |

Generation source:

- task rows: `C4_bioclaw_24task_benchmark/c4_task_scores_by_system.csv`
- totals: `C4_bioclaw_24task_benchmark/c4_results_summary.csv`
- task text and rubric: `bioclaw_24tasks/bioclaw_24_tasks.txt`

Important maintenance note: T11 has a corrected GEO metadata expectation. Before
the paper treats Table S1 as final, T11 should be regraded from raw system
outputs against the corrected GSE68849 metadata.

## Table S2 - Call-by-Call Input Token Counts

Table S2 lists the raw input-token counts used for the token-reduction claim.
It is intentionally call-by-call rather than only summary-level.

The table columns are:

| Column | Meaning |
|--------|---------|
| Call # | The call index within each condition. |
| Condition | Baseline or BioPLEASE. |
| Input Tokens | Input tokens recorded for that call. |
| Source | Provenance label for the token count. |

The current table represents one end-to-end run per condition:

| Condition | Calls | Total input tokens | Average input tokens per call |
|-----------|------:|-------------------:|------------------------------:|
| Baseline | 18 | 1,868,947 | 103,830.4 |
| BioPLEASE | 14 | 347,921 | 24,851.5 |

Supported interpretation: BioPLEASE reduced average input context from
103,830.4 to 24,851.5 tokens per call, a 76.1% descriptive reduction.

Generation source:

- call rows: `raw_token_counts_from_logs.csv`
- summary arithmetic: `C1_token_reduction_summary.csv`
- interpretation note: `C1_token_reduction_analysis.md`

Do not describe Table S2 as a paired statistical test. It is a descriptive
single-run log table.

## Table S3 - Integration Test Status

Table S3 documents structural-force test coverage. In the current Word
supplemental, status values distinguish tests that are already passing from
tests that are specified as planned coverage.

The table columns are:

| Column | Meaning |
|--------|---------|
| Test ID | Test or test-file identifier. |
| Description | What behavior the test checks. |
| Status | PASS or PLANNED. |
| Evidence | Where the passing evidence comes from, when applicable. |
| Notes | Which structural-force behavior the row supports. |

If the supplemental is regenerated from the latest local validation package,
Table S3 should be reconciled with the current IT-F suite result: 23/23 tests
pass with zero API calls. The Word table should not mix old planned rows with
new pass counts unless the caption explicitly explains both groups.

Generation source:

- current test inventory: `C11_integration_tests/README.md`
- latest run output: `C11_integration_tests/pytest_output.log`
- violation behavior examples: `C10_force_violations/*.log`

## Table S4 - Single-Run Cost Analysis

Table S4 reports the cost comparison for one logged run per condition under
`claude-sonnet-4-5` pricing.

The table columns are:

| Column | Meaning |
|--------|---------|
| Condition | Baseline or BioPLEASE. |
| Calls | Number of API calls in that run. |
| Total Input Tokens | Sum of input tokens across calls. |
| Total Output Tokens | Sum of output tokens across calls. |
| Task Cost | Estimated cost under the stated pricing. |

Pricing used:

| Token type | Price |
|------------|------:|
| Input | $3.00 per 1M tokens |
| Output | $15.00 per 1M tokens |

Current totals:

| Condition | Calls | Total input tokens | Total output tokens | Cost |
|-----------|------:|-------------------:|--------------------:|-----:|
| Baseline | 18 | 1,868,947 | 67,905 | $6.6254 |
| BioPLEASE | 14 | 347,921 | 21,974 | $1.3734 |

Supported interpretation: logged total cost fell from $6.6254 to $1.3734, a
79.3% reduction. Average cost per call fell from $0.3681 to $0.0981, a 73.3%
reduction.

Generation source:

- totals: `C8_cost_analysis/cost_summary.csv`
- call rows: `C8_cost_analysis/per_call_costs.csv`
- formulas and wording: `C8_cost_analysis/cost_analysis.md`

Do not use older mixed-model dollar figures in Table S4 unless a matching cost
trace is added.

## Table S5 - Ablation Study

Table S5 explains how token consumption changes across the baseline, two
ablation configurations, and full BioPLEASE.

Use the current ablation schema:

| Column | Meaning |
|--------|---------|
| Configuration | Baseline, no-compression, no-phase-isolation, or full BioPLEASE. |
| Avg Input Tokens/Call | Average input tokens per call. |
| Reduction % | Percent reduction relative to the baseline row. |
| Task Success % | Task-success value associated with the configuration. |
| API Calls | Number of calls when available. |
| Data Status | Whether the row is confirmed from logs or paper-stated/provisional. |

Current values:

| Configuration | Avg input tokens/call | Reduction | Task success | Calls |
|---------------|----------------------:|----------:|-------------:|------:|
| Baseline | 103,830.4 | NA | 72% | 18 |
| No Compression | 67,320 | 35.2% | 79% | 14 |
| No Phase Isolation | 58,441 | 43.7% | 81% | 15 |
| Full BioPLEASE | 24,851.5 | 76.1% | 88% | 14 |

Generation source:

- ablation rows: `ablation_table.csv`
- exploratory arithmetic: `C3_synergy_arithmetic/synergy_calculation.md`

The ablation rows should be described as tokens per call. The no-compression
and no-phase-isolation rows should remain clearly labeled as paper-stated or
provisional unless they are rerun and independently logged.

## Regeneration Checklist

Before updating `BioClaw_IEEE_IRI2026_v25_Supplemental.docx`:

1. Update each Word table from the matching CSV source, not by manual
   retyping.
2. Confirm Table S1 states the 0/1/2 rubric and the fresh
   `gpt-5.5-thinking-extended` grading provenance.
3. Confirm Table S2 and Table S5 both use tokens per call.
4. Confirm Table S3 status language matches the actual rows shown in the table.
5. Confirm Table S4 uses the same pricing and cost formula as the cost-analysis
   note.
6. Recheck T11 before presenting the 24-task benchmark as final.
