# Case Studies & Benchmark Data

This directory consolidates all benchmark data, scoring results, and supplemental materials for the IEEE IRI 2026 paper.

## 24-Task Biomedical Benchmark (Section V-B)

Scores determined by **blinded multi-judge evaluation** — 3 independent LLM judges (Claude Opus 4.8, Claude Sonnet 4.6, Gemini 2.5 Pro), 216 total API calls, majority-vote aggregation.

### Per-Category Breakdown

| Category | BioClaw (PLEAS) | BioPLEAS (baseline) | GPT-5.5 |
|---|---|---|---|
| Sequence & Variant Analysis | 7/8 (88%) | 6/8 (75%) | 7/8 (88%) |
| Drug-Target Discovery | 6/8 (75%) | 6/8 (75%) | 4/8 (50%) |
| Literature & Clinical Mining | 6/8 (75%) | 8/8 (100%) | 6/8 (75%) |
| Transcriptomics & Single-Cell | 7/8 (88%) | 6/8 (75%) | 2/8 (25%) |
| Structural & Protein Biology | 8/8 (100%) | 7/8 (88%) | 7/8 (88%) |
| Pathway & Systems Biology | 8/8 (100%) | 6/8 (75%) | 5/8 (62%) |
| **Total** | **42/48 (87.5%)** | **39/48 (81.2%)** | **31/48 (64.6%)** |

### Inter-Rater Reliability

| Metric | Value |
|---|---|
| Fleiss' kappa | 0.5319 |
| Krippendorff's alpha | 0.1128 |
| Judge agreement | Moderate (kappa 0.41–0.60) |

## Rubric

- **Score 2:** Correct factual answer, correct database/tool, explicit reasoning
- **Score 1:** Correct approach with incomplete output
- **Score 0:** Failure

## Difficulty Distribution

- Easy: 6 tasks
- Medium: 12 tasks
- Hard: 6 tasks

## Directory Structure

```
case_studies/
├── tasks/                  # 24-task definitions with expected outputs
├── results/                # Raw outputs from all three systems
├── scoring/                # Blinded multi-judge scoring pipeline
│   ├── rubric.py           # 24-task rubric definitions
│   ├── blinding.py         # HMAC-SHA256 cryptographic blinding
│   ├── judges.py           # Multi-model judge interface
│   ├── aggregator.py       # Score aggregation + inter-rater reliability
│   ├── run_blinded_scoring.py  # CLI entry point
│   ├── config.py           # Judge panel configuration
│   ├── model_answers/      # System answer files (BioClaw, BioPLEAS, GPT)
│   └── results/            # Pre-computed blinded scoring results
│       ├── scoring_report.txt          # Human-readable summary
│       ├── blinded_task_scores.csv     # Per-task, per-judge scores (72 rows)
│       ├── blinded_system_summary.csv  # System-level aggregates
│       ├── inter_rater_reliability.json
│       ├── judge_rationales.json       # Full rationale text from each judge
│       └── blinding_manifest.json      # Sealed de-anonymization mapping
├── token_analysis/         # Call-by-call token counts, ablation table
├── cost_analysis/          # Cost tracking data
├── figures/                # Publication figures
├── framework_code/         # Reference framework snapshots
├── migration_artifacts/    # Migration diffs and analysis
├── phase_prompt_templates/ # Phase prompt templates
├── state_schema/           # State schema analysis
├── langgraph_probe/        # LangGraph portability middleware
├── test_it_forces.py       # 23 force-aligned integration tests (F1–F5)
└── trigger_force_violations.py  # Force violation demonstrations
```

## Running the Scoring Pipeline

### Dry Run (no API keys needed)

```bash
cd scoring
pip install -r requirements.txt
python run_blinded_scoring.py --dry-run
```

### Full Scoring Run

Requires API keys for the three judge models:

```bash
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...

python run_blinded_scoring.py
```

### Run Blinding Tests (offline)

```bash
pytest test_blinding.py -v
```

## Scoring Methodology

1. **Blinding:** Responses are anonymized using HMAC-SHA256 with a session-seeded key. System-identifying names are scrubbed via regex.
2. **Independent Judging:** Three frontier LLMs (Claude Opus 4.8, Claude Sonnet 4.6, Gemini 2.5 Pro) independently score each blinded response against the rubric.
3. **Aggregation:** Majority-vote and mean scores are computed per task and per system. Inter-rater reliability is measured via Fleiss' kappa and Krippendorff's alpha.
4. **Transparency:** Full judge rationales and the sealed blinding manifest are included for reproducibility.
