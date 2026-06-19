# PLEAS: Plan, Learn, Execute, Assess, Share

**Scientific AI Agents Must Learn and Share**

Thomas Pan<sup>1</sup>, Jake Y. Chen<sup>2</sup>

<sup>1</sup>Alpha Mind Club, LLC, Birmingham, AL, USA
<sup>2</sup>Department of Biomedical Informatics and Data Science, University of Alabama at Birmingham, Birmingham, AL, USA

> **IEEE IRI 2026** | [Paper PDF](#) | [Citation](#citation)

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

### Key Results (Preliminary)

| Metric | Value |
|---|---|
| Force-aligned integration tests | **23/23 pass** (0.95s, zero API calls) |
| Benchmark score (BioClaw/PLEAS) | **42/48** (87.5%) |
| Benchmark score (Biomni baseline) | 38/48 (79.2%) |
| Benchmark score (ChatGPT gpt-5.5-thinking-extended) | 28/48 (58.3%) |
| Token reduction (single-run) | **76.1% fewer** input tokens |
| Cost reduction (single-run) | **79.3% lower** cost |

---

## Repository Structure

```
PLEAS/
├── engines/
│   ├── biomni/           # PLEAS on Biomni (Python) — Section IV-A
│   └── bioclaw/          # BioClaw on OpenClaude (TypeScript + Python) — Section IV-B
│
├── benchmark/            # 24-task biomedical scientific workflow benchmark — Section V-B
│   ├── tasks/            # Task definitions and rubric
│   ├── results/          # Raw outputs from all three systems
│   └── scoring/          # Blinded multi-judge scoring pipeline
│
├── tests/                # Force-aligned integration tests (IT-F suite) — Section V-A
│
├── supplemental/         # Token analysis, cost data, case studies — Sections V-C, V-E
│   ├── token_analysis/   # Call-by-call token counts, ablation table
│   ├── cost_analysis/    # Cost tracking data
│   ├── case_studies/     # ICER health-economics case study
│   ├── langgraph_probe/  # LangGraph portability middleware — Section IV-C
│   └── figures/          # Publication figures
│
└── examples/             # Usage examples and demos
```

---

## Quick Start

### Biomni Engine (Python)

The Biomni engine is a Python-native PLEAS implementation with Pydantic-typed tool schemas and a rich biomedical tool registry.

```bash
# Clone and install
git clone https://github.com/alphamind-club/PLEAS.git
cd PLEAS/engines/biomni

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
cd ../../tests
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
cd tests
pytest test_it_forces.py -v
```

Expected output: `23 passed in < 1 second`

### 2. Run the 24-Task Biomedical Benchmark (Section V-B)

The benchmark tasks, rubric, and raw system outputs are in `benchmark/`. To run the blinded scoring pipeline:

```bash
cd benchmark/scoring

# Install scoring dependencies
pip install -r requirements.txt

# Dry run (no API calls — generates blinding manifest only)
python run_blinded_scoring.py --dry-run

# Full scoring run (requires OpenAI, Anthropic, and Google API keys)
python run_blinded_scoring.py
```

The scoring pipeline:
1. Loads system answers from three systems
2. Blinds responses using HMAC-SHA256 (scrubs system-identifying names)
3. Sends blinded responses to three independent LLM judges
4. Aggregates scores with inter-rater reliability statistics (Fleiss' kappa, Krippendorff's alpha)

### 3. Token Efficiency Analysis (Section V-C)

Raw token count data and ablation results are in `supplemental/token_analysis/`:
- `raw_token_counts_from_logs.csv` — Call-by-call token counts
- `ablation_table.csv` — Per-force ablation results
- `C1_token_reduction_summary.csv` — Summary statistics

### 4. Health-Economics Case Study (Section V-E)

The ICER case study trace and calculator template are in `supplemental/case_studies/`.

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

This repo demonstrates P1 with two production substrates (Biomni, BioClaw) and a preliminary LangGraph probe.

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
