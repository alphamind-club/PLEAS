# 24-Task Biomedical Scientific Workflow Benchmark

This benchmark (Section V-B of the paper) evaluates scientific AI workflow agents across six biomedical categories using a 0/1/2 rubric.

## Categories (4 tasks each, max 8 points per category)

| Category | BioClaw (PLEAS) | Biomni (baseline) | ChatGPT |
|---|---|---|---|
| Sequence & Variant Analysis | 7/8 (88%) | 7/8 (88%) | 5/8 (62%) |
| Drug-Target Discovery | 8/8 (100%) | 7/8 (88%) | 5/8 (62%) |
| Literature & Clinical Mining | 3/8 (38%) | 6/8 (75%) | 3/8 (38%) |
| Transcriptomics & Single-Cell | 8/8 (100%) | 6/8 (75%) | 3/8 (38%) |
| Structural & Protein Biology | 8/8 (100%) | 7/8 (88%) | 7/8 (88%) |
| Pathway & Systems Biology | 8/8 (100%) | 6/8 (75%) | 4/8 (50%) |
| **Total** | **42/48 (87.5%)** | **38/48 (79.2%)** | **28/48 (58.3%)** |

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
benchmark/
├── tasks/          # Task definitions with expected outputs
├── results/        # Raw outputs from all three systems + scoring CSVs
└── scoring/        # Blinded multi-judge scoring pipeline
    ├── rubric.py               # 24-task rubric definitions
    ├── blinding.py             # HMAC-SHA256 cryptographic blinding
    ├── judges.py               # Multi-model judge interface
    ├── aggregator.py           # Score aggregation + inter-rater reliability
    ├── run_blinded_scoring.py  # CLI entry point
    ├── test_blinding.py        # Offline test suite
    └── config.py               # Judge panel configuration
```

## Running the Scoring Pipeline

### Dry Run (no API keys needed)

```bash
cd scoring
pip install -r requirements.txt
python run_blinded_scoring.py --dry-run
```

### Full Scoring Run

Requires API keys for three judge models (OpenAI, Anthropic, Google):

```bash
export OPENAI_API_KEY=...
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
2. **Independent Judging:** Three frontier LLMs (GPT-5.5-thinking-extended, Claude Opus 4.8, Gemini 3.1 Pro) independently score each blinded response.
3. **Aggregation:** Majority-vote and mean scores are computed per task and per system. Inter-rater reliability is measured via Fleiss' kappa and Krippendorff's alpha.

## Limitations

- Scoring was non-blinded in the original paper (GPT-5.5-thinking-extended only)
- The blinded multi-judge pipeline in `scoring/` is the improved version
- ChatGPT comparator uses the same model family as the original judge (disclosed confound)
