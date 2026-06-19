# BioClaw — PLEAS on OpenClaude

BioClaw is the TypeScript-based PLEAS instantiation described in **Section IV-B** of the paper. It re-implements the PLEAS five-phase specification on [OpenClaude](https://github.com/Gitlawb/openclaude), using SKILL.md phase files and Model Context Protocol (MCP) tool schemas instead of Biomni's Python registry.

## Architecture

```
src/bioplease/                 # Core BioPLEASE runtime (TypeScript)
├── types.ts                   # Phase types, provider profiles, runtime state
├── runner.ts                  # Session runner — phase transitions, journal, artifacts
├── prompts.ts                 # System prompt / CLAUDE.md generation, skill bundles
├── workspace.ts               # Project scaffolding (.bioplease/, reports/, data/)
├── biocontext.ts              # BioContextAI MCP server integration
├── journal.ts                 # Runtime state persistence (state.json, transcripts)
├── ledger.ts                  # Workspace file tracking (workspace-ledger.json)
├── artifacts.ts               # Artifact manifest builder
├── cli.ts                     # CLI entrypoint (doctor, init, run, status, web)
├── doctor.ts                  # Runtime dependency health checker
└── projects.ts                # Multi-project management

python/                        # Python provider bridge
├── smart_router.py            # Multi-provider auto-router (latency/cost scoring)
├── ollama_provider.py         # Local Ollama model support
└── atomic_chat_provider.py    # Apple Silicon local provider

BioPlease_tools/               # Biomedical tool registry
├── tool/                      # 15 domain-specific Python tool modules
│   ├── biochemistry.py ... systems_biology.py
│   ├── tool_registry.py       # Dynamic tool registration
│   ├── schema_db/             # 27 pre-extracted API schemas (UniProt, PDB, etc.)
│   └── example_mcp_tools/     # MCP tool server examples
├── bioplease_env/             # Reproducible environment setup
│   ├── environment.yml        # Conda environment definition
│   ├── setup.sh               # Automated setup script
│   └── install_r_packages.R   # R package installation
└── biorxiv_scripts/           # Benchmark task generation from bioRxiv
```

## Installation

### Prerequisites

- [Bun](https://bun.sh/) >= 1.0 (or Node.js >= 18)
- Python >= 3.11
- API keys for at least one LLM provider

### Install Dependencies

```bash
# Install Node.js/TypeScript dependencies
bun install
# or: npm install

# Set up the biomedical tool environment
cd BioPlease_tools/bioplease_env
bash setup.sh
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env` with your provider keys. BioClaw supports multiple providers:

| Provider | Environment Variable |
|---|---|
| Anthropic (Claude) | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Google (Gemini) | `GOOGLE_API_KEY` |
| Ollama (local) | `OLLAMA_BASE_URL` |

### Run BioClaw

```bash
# Check system dependencies
bun run bioplease:doctor

# Initialize a new project
bun run bioplease:init

# Run a task
bun run bioplease:run

# Launch the web console
bun run bioplease:web
```

## Portability Invariants

BioClaw demonstrates **P1** (Portability) and **P2** (Composability) from the PLEAS specification:

- The five-phase semantics are preserved from Biomni
- Substrate-specific differences are confined to:
  - **Execute adapter:** MCP tool schemas (vs. Biomni's Python registry)
  - **Learn retriever:** SKILL.md knowledge files (vs. Biomni's Pydantic schemas)

## Platform-Specific Setup

- [macOS / Linux](docs/quick-start-mac-linux.md)
- [Windows](docs/quick-start-windows.md)
- [Advanced Setup](docs/advanced-setup.md)
- [Non-Technical Setup](docs/non-technical-setup.md)
