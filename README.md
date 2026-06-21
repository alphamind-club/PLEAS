# PLEAS: Plan, Learn, Execute, Assess, Share

**Scientific AI Agents Must Learn and Share**

Thomas Pan<sup>1</sup>, Jake Y. Chen<sup>2</sup>

<sup>1</sup>Alpha Mind Club, LLC, Birmingham, AL, USA
<sup>2</sup>Department of Biomedical Informatics and Data Science, University of Alabama at Birmingham, Birmingham, AL, USA

> **IEEE IRI 2026** | [Citation](#citation)

---

## Overview

PLEAS is a **substrate-independent five-phase specification** for scientific AI workflow agents. It extends the prevailing Plan-Execute-Evaluate loop with two architecturally novel phases:

- **Learn** — structured knowledge retrieval *before* any tool is invoked
- **Share** — artifact externalization *after* assessment

Five structural forces (F1-F5) produce a context bound that grows with task complexity rather than session length.

```
Research Task → [Plan] → [Learn] → [Execute] → [Assess] → [Share] → Reusable Output
                  ↕          ↕          ↕           ↕          ↕
              Persistent State + Compressed Trace + Reusable Artifacts
```

### Key Results

Benchmark scores determined by **blinded multi-judge evaluation** (3 independent LLM judges, 216 API calls, Fleiss' kappa = 0.53):

| Metric | Value |
|---|---|
| Force-aligned integration tests | **23/23 pass** (0.95s, zero API calls) |
| Benchmark score (BioClaw/PLEAS) | **42/48** (87.5%) |
| Benchmark score (non-PLEAS baseline) | 39/48 (81.2%) |
| Benchmark score (GPT-5.5-thinking-extended) | 31/48 (64.6%) |
| Token reduction (single-run) | **65.4% fewer** input tokens (avg 35,935 vs 103,830/call) |
| Cost reduction (single-run) | **62.3% lower** cost |

---

## Repository Structure

```
PLEAS/
├── engines/
│   ├── biopleas/         # BioPLEAS on Biomni (Python) — Section IV-A
│   └── bioclaw/          # BioClaw on OpenClaude (TypeScript + Python) — Section IV-B
│
└── case_studies/         # All benchmark data, results, and supplemental materials
    ├── tasks/            # 24-task benchmark definitions — Section V-B
    ├── results/          # Raw outputs from all three systems
    ├── scoring/          # Blinded multi-judge scoring pipeline and results
    ├── token_analysis/   # Call-by-call token counts, ablation table — Section V-C
    ├── cost_analysis/    # Cost tracking data — Section V-E
    ├── figures/          # Publication figures
    ├── framework_code/   # Reference framework snapshots
    ├── migration_artifacts/  # Migration diffs and analysis
    ├── phase_prompt_templates/  # Phase prompt templates
    ├── state_schema/     # State schema analysis
    ├── langgraph_probe/  # LangGraph portability middleware — Section IV-C
    ├── test_it_forces.py           # Force-aligned integration tests (IT-F suite) — Section V-A
    └── trigger_force_violations.py # Force violation demonstrations
```

---

## Quick Start

### BioPLEAS Engine (Python)

The BioPLEAS engine is a Python-native PLEAS implementation with Pydantic-typed tool schemas and a rich biomedical tool registry.

```bash
# Clone and install
git clone https://github.com/alphamind-club/PLEAS.git
cd PLEAS/engines/biopleas

# Create environment (Python >= 3.11)
python -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install -e .

# Install scientific stack (for full replication)
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys (Anthropic, OpenAI, and/or Google)

# Run the force-aligned integration tests (zero API calls needed)
cd ../../case_studies
pytest test_it_forces.py -v
```

### BioClaw Engine (TypeScript + Python)

BioClaw re-instantiates PLEAS on OpenClaude using SKILL.md phase files and MCP tool schemas.

```bash
cd PLEAS/engines/bioclaw

# Install Node.js dependencies (requires Bun or npm)
bun install
# or: npm install

# Set up the Python biomedical tool environment
cd BioPlease_tools/bioplease_env
bash setup.sh

# Configure API keys
cp .env.example .env
# Edit .env with your provider keys

# Run BioClaw
bun run bioplease
```

See `engines/bioclaw/docs/` for platform-specific setup guides:
- [macOS / Linux](engines/bioclaw/docs/quick-start-mac-linux.md)
- [Windows](engines/bioclaw/docs/quick-start-windows.md)
- [Advanced Setup](engines/bioclaw/docs/advanced-setup.md)

---

## Replication Guide

### 1. Run the Force-Aligned Integration Tests (Section V-A)

These 23 tests validate all five structural forces with zero API calls:

```bash
cd case_studies
pytest test_it_forces.py -v
```

Expected output: `23 passed in < 1 second`

### 2. Run the 24-Task Biomedical Benchmark (Section V-B)

The benchmark tasks, rubric, and raw system outputs are in `case_studies/`. To run the blinded scoring pipeline:

```bash
cd case_studies/scoring

# Install scoring dependencies
pip install -r requirements.txt

# Dry run (no API calls — generates blinding manifest only)
python run_blinded_scoring.py --dry-run

# Full scoring run (requires Anthropic and Google API keys)
python run_blinded_scoring.py
```

The scoring pipeline:
1. Loads system answers from three systems (BioClaw, BioPLEAS, GPT-5.5-thinking-extended)
2. Blinds responses using HMAC-SHA256 (scrubs system-identifying names)
3. Sends blinded responses to three independent LLM judges (Claude Opus 4.8, Claude Sonnet 4.6, Gemini 2.5 Pro)
4. Aggregates scores with inter-rater reliability statistics (Fleiss' kappa = 0.53, Krippendorff's alpha = 0.11)

Pre-computed results from the blinded evaluation (216 judge calls) are in `case_studies/scoring/results/`:
- `scoring_report.txt` — Human-readable summary
- `blinded_task_scores.csv` — Per-task, per-judge scores (72 rows)
- `blinded_system_summary.csv` — System-level aggregates with category breakdowns
- `inter_rater_reliability.json` — Fleiss' kappa and Krippendorff's alpha
- `judge_rationales.json` — Full rationale text from each judge
- `blinding_manifest.json` — Sealed session mapping for de-anonymization

### 3. Token Efficiency Analysis (Section V-C)

Raw token count data and ablation results are in `case_studies/token_analysis/`:
- `raw_token_counts_from_logs.csv` — Call-by-call token counts
- `ablation_table.csv` — Per-force ablation results
- `C1_token_reduction_summary.csv` — Summary statistics

### 4. Health-Economics Case Study (Section V-E)

The ICER case study trace and calculator template are in `case_studies/`.

---

## The Five Structural Forces

| Force | Name | Enforcement |
|-------|------|-------------|
| **F1** | Context Boundedness | Emerges from F2-F5; active phase context is bounded |
| **F2** | Phase-Scoped Execution | Each phase receives only required inputs; raw history is pruned |
| **F3** | Typed Contracts | Pydantic schemas validate inputs/outputs before state mutation |
| **F4** | Bounded Retry | Execute retries at most R_max=3; failures pass to Assess |
| **F5** | Trace Compression | Summaries capped at S_max tokens, stored in long-term memory |

**Context Bound (Corollary 1):** Under fixed S_max and R_max:

```
context(i) <= 2·S_max + K·B + T + A
```

where K = Execute tool calls, B = avg bounded tool-response size, T = plan object size, A = schema overhead. This bound is O(1) with respect to session length.

---

## Portability Invariants

PLEAS defines three substrate-compliance invariants:

- **P1 (Portability):** Five-phase semantics preserved under substrate migration
- **P2 (Composability):** Any phase may be substituted without altering adjacent contracts
- **P3 (Domain-Extensibility):** Domain-specific tools injected at Execute without modifying the orchestrator

This repo demonstrates P1 with two production substrates (BioPLEAS, BioClaw) and a preliminary LangGraph probe.

---

## Citation

If you use PLEAS in your research, please cite:

```bibtex
@inproceedings{pan2026pleas,
  title     = {Scientific {AI} Agents Must Learn and Share},
  author    = {Pan, Thomas and Chen, Jake Y.},
  booktitle = {Proceedings of the IEEE International Conference on
               Information Reuse and Integration (IRI)},
  year      = {2026},
  note      = {To appear}
}
```

---

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.

## Acknowledgments

This work was supported in part by UAB startup funding to JYC and NIH grants to JYC: U54-OD036472 and UM1TR004771.
