# PLEAS on Biomni

The Biomni engine is the Python-native PLEAS implementation described in **Section IV-A** of the paper. It wraps the [Biomni](https://doi.org/10.1101/2025.05.30.656746) scientific agent platform with PLEAS five-phase orchestration, Pydantic-typed phase contracts, and a rich biomedical tool registry.

## Architecture

```
bioplease/
├── agent/
│   ├── pleas.py              # Core PLEAS five-phase orchestrator
│   ├── react.py              # ReAct agent (LangGraph-based)
│   ├── state_manager.py      # Phase-scoped state persistence (F2)
│   ├── memory.py             # Two-tier memory: short-term + long-term compressed (F5)
│   ├── cost_manager.py       # Token/cost/time budget tracking (F4)
│   ├── phase_logger.py       # Per-phase structured file logging
│   ├── a1.py                 # Primary agent implementation
│   ├── a1_langchain.py       # LangChain-integrated agent variant
│   └── claude_a1.py          # Claude-specific agent variant
├── model/
│   └── retriever.py          # LLM-based tool/resource retrieval
├── tool/                     # 19 domain-specific biomedical tool modules
│   ├── biochemistry.py
│   ├── cancer_biology.py
│   ├── cell_biology.py
│   ├── database.py           # Cross-domain database queries
│   ├── genetics.py
│   ├── genomics.py
│   ├── immunology.py
│   ├── literature.py         # PubMed, Europe PMC
│   ├── molecular_biology.py
│   ├── pharmacology.py
│   ├── systems_biology.py
│   ├── tool_registry.py      # Dynamic tool registration
│   └── ...
├── task/                     # Task/benchmark definitions
├── biorxiv_scripts/          # bioRxiv benchmark task generation
├── env_desc.py               # Data lake registry (COSMIC, DepMap, BioGRID, etc.)
├── llm.py                    # Multi-provider LLM factory
└── utils.py                  # Code execution helpers (R, Bash, Python, CLI)
```

## Installation

### Prerequisites

- Python >= 3.11
- API keys for at least one LLM provider (Anthropic, OpenAI, or Google)

### Core Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs the minimal dependencies: `pydantic`, `langchain`, `python-dotenv`.

### Full Scientific Stack

For full replication (single-cell analysis, embedding computation, etc.):

```bash
pip install -r requirements.txt
```

This adds: `scanpy`, `scvi-tools`, `torch`, `pandas`, `numpy`, `matplotlib`, `faiss-cpu`, and other scientific packages.

### Configuration

```bash
cp .env.example .env
```

Edit `.env` with your API keys. Supported providers:

| Provider | Environment Variable |
|---|---|
| Anthropic (Claude) | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Google (Gemini) | `GOOGLE_API_KEY` |
| Ollama (local) | `OLLAMA_BASE_URL` |

## Structural Force Mapping

| Force | Implementation |
|---|---|
| F1 (Context Boundedness) | Emerges from F2-F5 |
| F2 (Phase-Scoped Execution) | `state_manager.py` — per-phase subdirectories |
| F3 (Typed Contracts) | Pydantic schemas in `pleas.py` |
| F4 (Bounded Retry) | R_max=3 enforcement in `pleas.py` |
| F5 (Trace Compression) | `memory.py` — S_max=2000 token cap |

## Key Files for Replication

- **`agent/pleas.py`** — The core PLEAS runner. Start here to understand the five-phase loop.
- **`agent/state_manager.py`** — Phase-scoped state isolation (F2).
- **`agent/memory.py`** — Two-tier memory with trace compression (F5).
- **`tool/`** — All 19 biomedical tool modules registered via `tool_registry.py`.
