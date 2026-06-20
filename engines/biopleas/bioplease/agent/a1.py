import glob
import inspect
import os
import re
import sys
import time
import json
import unicodedata
from pathlib import Path
from typing import Any, Literal, TypedDict, Union

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from bioplease.env_desc import data_lake_dict, library_content_dict
from bioplease.llm import SourceType, get_llm
from bioplease.model.retriever import ToolRetriever
from bioplease.tool.support_tools import run_python_repl
from bioplease.tool.tool_registry import ToolRegistry
from bioplease.utils import (
    check_and_download_s3_files,
    download_and_unzip,
    function_to_api_schema,
    pretty_print,
    read_module2api,
    run_bash_script,
    run_r_code,
    run_with_timeout,
    textify_api_dict,
)
from bioplease.agent.phase_logger import PhaseLogger
from bioplease.agent.communication_hub import CommunicationHub
from bioplease.agent.unified_phase_logger import UnifiedPhaseLogger

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import glob, json, subprocess, textwrap, shutil

print(os.path)

if os.path.exists(".env"):
    load_dotenv(".env", override=False)
    print("Loaded environment variables from .env")

LOG_FILE = "ai_interactive.txt"
open(LOG_FILE, "w", encoding="utf-8").close()

def _msgs_to_text(msgs) -> str:
    if isinstance(msgs, str):
        return msgs
    if isinstance(msgs, (list, tuple)):
        parts = []
        for m in msgs:
            # Try to take message.content; fall back to str(m)
            parts.append(getattr(m, "content", str(m)))
        return "\n".join(parts)
    return str(msgs)

def log_llm_event(tag: str, content) -> None:
    text = _msgs_to_text(content).strip()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{tag}:\n")
        f.write(text + "\n\n")

class AgentState(TypedDict):
    messages: list[BaseMessage]
    next_step: str | None
    phase: Literal["PLAN","LEARN","EXECUTE","MINI_SHARE","ASSESS","SHARE"]  # NEW
    artifacts: dict[str, Any]  # NEW (file paths, selected tools, notes, etc.)



class A1:
    def __init__(
        self,
        path="./data",
        llm="gpt-4o-mini",
        source: SourceType | None = None,
        use_tool_retriever=True,
        timeout_seconds=600,
        base_url: str | None = None,
        api_key: str = "sk-fb0Oh5seld3UPAjFwhIgodwojLqZSwAwm_txEWOn_fT3BlbkFJEHxEJmlZmEVPcKtDwB7MWLiG087nH9KgQG2ivVymYA",
        cost_budget: float = None,
        enable_pm: bool = True,
        fastmode: bool = False,
        auto_retry_on_error: bool = False,
        max_execute_retries: int = 3,
        use_llm_error_detection: bool = True,
        error_detection_model: str = "gpt-4o-mini",
        scientific_mindset_first_plan_only: bool = True,
    ):
        """Initialize the BioPLEASE agent.

        Args:
            path: Path to the data
            llm: LLM to use for the agent
            source (str): Source provider: "OpenAI", "AzureOpenAI", "Anthropic", "Ollama", "Gemini", "Bedrock", or "Custom"
            use_tool_retriever: If True, use a tool retriever
            timeout_seconds: Timeout for code execution in seconds
            base_url: Base URL for custom model serving (e.g., "http://localhost:8000/v1")
            api_key: API key for the custom LLM
            enable_pm: If True, automatically enable Product Manager agent for monitoring and reporting
            fastmode: If True, strictly prevent backward edges in workflow for faster execution
            auto_retry_on_error: If True, automatically retry EXECUTE phase on code errors
            max_execute_retries: Maximum number of auto-retry attempts for EXECUTE phase
            use_llm_error_detection: If True, use LLM to detect errors. If False, use pattern matching
            error_detection_model: Model to use for LLM-based error detection (default: gpt-4o-mini)
            scientific_mindset_first_plan_only: If True, show scientific reasoning mindset only on first plan.
                                                 If False, show on all plans. Set to False for maximum rigor.

        """
        
        self.max_overview_loops = 3  # max iterations for OVERVIEW phase
        self.cost_budget = cost_budget  # User-settable cost budget
        self.fastmode = fastmode  # Fast mode flag
        
        # Auto-retry configuration
        self.auto_retry_on_error = auto_retry_on_error
        self.max_execute_retries = max_execute_retries
        self.use_llm_error_detection = use_llm_error_detection
        self.error_detection_model = error_detection_model
        
        # Scientific mindset configuration
        # TOGGLE THIS: Set to False to show scientific mindset on ALL plans (more rigorous)
        #              Set to True to show only on FIRST plan (more efficient)
        self.scientific_mindset_first_plan_only = scientific_mindset_first_plan_only
        self._plan_count = 0  # Track number of times PLAN phase has been called

        self.path = path

        self.system_prompts: dict[str, str] = {
            "PLAN": "",
            "LEARN": "",
            "EXECUTE": "",
            "ASSESS": "",
        }

        self.phase_llms = {
            "PLAN": None,
            "LEARN": None,
            "EXECUTE": None,   # used by generate()
            "ASSESS": None,     # used by assess()
            "SHARE": None,
        }

        self.paper_format = "latex"      # "latex" | "markdown"
        self.compile_pdf = True          # build paper.pdf via latexmk
        self.max_code_bytes = 20000      # how much code to embed in paper
        self.code_globs = ["*.py"]       # which files to consider for embedding

        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")

        # --- Begin custom folder/file checks ---
        benchmark_dir = os.path.join(path, "bioplease_data", "benchmark")
        data_lake_dir = os.path.join(path, "bioplease_data", "data_lake")

        # Create the bioplease_data directory structure
        os.makedirs(benchmark_dir, exist_ok=True)
        os.makedirs(data_lake_dir, exist_ok=True)

        expected_data_lake_files = list(data_lake_dict.keys())

        # filter out COSMIC files from expected list
        expected_data_lake_files = [
            f for f in data_lake_dict.keys()
            if not f.lower().startswith("cosmic_")
        ]

        # Check and download missing data lake files
        print("Checking and downloading missing data lake files...")
        check_and_download_s3_files(
            s3_bucket_url="https://biomni-release.s3.amazonaws.com",
            local_data_lake_path=data_lake_dir,
            expected_files=expected_data_lake_files,
            folder="data_lake",
        )

        # Check if benchmark directory structure is complete
        benchmark_ok = False
        if os.path.isdir(benchmark_dir):
            patient_gene_detection_dir = os.path.join(benchmark_dir, "hle")
            if os.path.isdir(patient_gene_detection_dir):
                benchmark_ok = True

        if not benchmark_ok:
            print("Checking and downloading benchmark files...")
            check_and_download_s3_files(
                s3_bucket_url="https://biomni-release.s3.amazonaws.com",
                local_data_lake_path=benchmark_dir,
                expected_files=[],  # Empty list - will download entire folder
                folder="benchmark",
            )

        self.path = os.path.join(path, "bioplease_data")
        module2api = read_module2api()

        stop_sequences = None
        if source in ["Anthropic", "OpenAI"]:
            stop_sequences = ["</execute>", "</solution>"]
        
        self.llm = get_llm(
            llm, stop_sequences=stop_sequences, source=source, base_url=base_url, api_key=api_key
        )
        # instantiate cost/time manager (simple defaults)
        try:
            from .cost_manager import CostManager
            self.cost_manager = CostManager()
        except Exception:
            # fail-safe: if import fails, set to None so agent still loads
            self.cost_manager = None
        self.module2api = module2api
        self.use_tool_retriever = use_tool_retriever

        if self.use_tool_retriever:
            self.tool_registry = ToolRegistry(module2api)
            self.retriever = ToolRetriever()

        # Add timeout parameter
        self.timeout_seconds = timeout_seconds  # 10 minutes default timeout

        # --- Short/Long-term memory controls (tune as you like) ---
        # initialize before configure() so prompt-builder and memory code see them
        self.short_window = 6          # how many recent messages to keep verbatim
        self.summary_max_chars = 4000  # guardrail on summary size (increased from 2500)

        self.configure()

        # --- Auto-enable Product Manager if requested ---
        self.enable_pm = enable_pm
        self.product_manager = None
        if self.enable_pm:
            try:
                from .pm_integration import add_product_manager_to_agent
                add_product_manager_to_agent(self)
                print("✓ Product Manager agent enabled - monitoring progress, costs, and decisions")
            except Exception as e:
                print(f"Warning: Could not enable Product Manager: {e}")
                self.enable_pm = False

    @property
    def pm(self):
        """Shorthand for accessing the product_manager"""
        return self.product_manager







    def _record_llm_usage(self, messages, llm_obj=None):
        """Estimate input/output tokens separately and record via cost_manager.

        Behaviour:
        - If `messages` is a list and its last element appears to be a model response
        (an `AIMessage` or similar object), treat everything before it as input
        and the last element as output. Otherwise, treat `messages` entirely as
        input and assume no output was passed.
        - If `messages` is a single string, treat it as input.
        """
        if not getattr(self, "cost_manager", None):
            return
        try:
            input_msgs = []
            output_msgs = []

            # Determine if messages includes a response at the end
            if isinstance(messages, (list, tuple)):
                if messages:
                    last = messages[-1]
                    is_response = False
                    try:
                        # Prefer langchain's AIMessage type when available
                        if isinstance(last, AIMessage):
                            is_response = True
                    except Exception:
                        # fallback heuristics
                        role = getattr(last, "role", None)
                        if role and str(role).lower() in ("assistant", "ai"):
                            is_response = True

                    if is_response:
                        input_msgs = list(messages[:-1])
                        output_msgs = [last]
                    else:
                        input_msgs = list(messages)
                        output_msgs = []
                else:
                    input_msgs = []
                    output_msgs = []
            else:
                # single message/string
                input_msgs = [messages]
                output_msgs = []

            input_text = _msgs_to_text(input_msgs)
            output_text = _msgs_to_text(output_msgs)

            input_tokens = self.cost_manager.estimate_tokens(input_text)
            output_tokens = self.cost_manager.estimate_tokens(output_text)

            # derive model key (prefer the provided llm_obj)
            model_key = None
            if llm_obj is not None:
                model_key = getattr(llm_obj, "model_name", None) or getattr(llm_obj, "model", None)
            if not model_key:
                model_key = getattr(self.llm, "model_name", None) or getattr(self.llm, "model", None) or "unknown"

            self.cost_manager.record_usage(str(model_key), input_tokens=int(input_tokens), output_tokens=int(output_tokens))
            # Print estimated cost after each model call


            try:
                cost = self.cost_manager.estimate_cost(
                    tokens=int(input_tokens) + int(output_tokens),
                    model=str(model_key),
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens)
                )
                total_cost = getattr(self.cost_manager, "total_cost", None) if hasattr(self, "cost_manager") else None
                if total_cost is not None:
                    print(f"[COST] Model: {model_key} | Input tokens: {input_tokens} | Output tokens: {output_tokens} | Estimated cost: ${cost:.6f} | Total cost so far: ${total_cost:.4f}")
                else:
                    print(f"[COST] Model: {model_key} | Input tokens: {input_tokens} | Output tokens: {output_tokens} | Estimated cost: ${cost:.6f}")
            except Exception as e:
                print(f"[COST] Could not estimate cost: {e}")

            # Print total cost and budget after each LLM usage record
            total_cost = getattr(self.cost_manager, "total_cost", None) if hasattr(self, "cost_manager") else None
            budget = getattr(self, "cost_budget", None)
            cost_msg = None
            if total_cost is not None and budget is not None:
                cost_msg = f"[COST] Total cost so far: ${total_cost:.4f} / Budget: ${budget:.2f}"
            elif total_cost is not None:
                cost_msg = f"[COST] Total cost so far: ${total_cost:.4f}"
            elif budget is not None:
                cost_msg = f"[COST] Budget: ${budget:.2f}"
            if cost_msg:
                print(cost_msg)
                # Add as HumanMessage so the agent can reference cost
                try:
                    from langchain_core.messages import HumanMessage
                    if isinstance(messages, list):
                        messages.append(HumanMessage(content=cost_msg))
                except Exception:
                    pass

        except Exception:
            # don't crash the agent for telemetry issues
            pass

    def _ensure_pdf_from_share(self, paper_md_path: str) -> str | None:
        """
        Build a PDF for the current run.
        Returns the PDF path or None if failed.
        """
        # --- directories ---
        run_dir = getattr(self, "run_dir", self.path)
        papers_dir = os.path.join(run_dir, "papers")
        logs_dir   = os.path.join(run_dir, "logs")
        figures_dir = os.path.join(papers_dir, "figures")
        os.makedirs(papers_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)

        # --- gather artifacts from this run ---
        arts = self._collect_run_artifacts() if hasattr(self, "_collect_run_artifacts") else {}
        if not paper_md_path:
            paper_md_path = arts.get("paper_md")

        # --- collect figures + optional code snippets for the paper ---
        figs = self._gather_figures_for_latex(figures_dir) if hasattr(self, "_gather_figures_for_latex") else []
        code_items = self._select_code_snippets(work_dir=self.path) if hasattr(self, "_select_code_snippets") else []

        # --- try user template first ---
        tmpl_dir = os.path.join(self.path, "templates")
        user_tex = os.path.join(tmpl_dir, "agents4science_2025.tex")
        user_sty = os.path.join(tmpl_dir, "agents4science_2025.sty")

        if os.path.exists(user_tex):
            # 1) copy template into papers_dir as main.tex (+ .sty if present)
            main_tex = os.path.join(papers_dir, "main.tex")
            shutil.copy2(user_tex, main_tex)
            if os.path.exists(user_sty):
                shutil.copy2(user_sty, os.path.join(papers_dir, os.path.basename(user_sty)))

            # 2) generate the content body this template will include
            content_tex = os.path.join(papers_dir, "content_generated.tex")
            # uses internal renderer to produce a LaTeX body from the md, figs, code
            self._render_paper_latex(md_paper_path=paper_md_path,
                                    figures=figs,
                                    code_items=code_items,
                                    out_tex_path=content_tex)

            # 3) patch template to include content + frontmatter
            with open(main_tex, "r", encoding="utf-8") as f:
                templ = f.read()

            # basic front matter (customize as you like)
            paper_title = "Auto-Generated Research Paper"
            authors = "Thomas Pan"
            frontmatter = (
                f"\\def\\PaperTitle{{{paper_title}}}\n"
                f"\\def\\PaperAuthors{{{authors}}}\n"
            )

            if "%%CONTENT_HERE" in templ:
                templ = templ.replace("%%CONTENT_HERE", "\\input{content_generated.tex}")
            else:
                # insert before \end{document}
                m = re.search(r"\\end{document}", templ)
                if m:
                    i = m.start()
                    templ = templ[:i] + "\n\\input{content_generated.tex}\n" + templ[i:]
                else:
                    templ += "\n\\input{content_generated.tex}\n\\end{document}\n"

            templ = frontmatter + "\n" + templ
            with open(main_tex, "w", encoding="utf-8") as f:
                f.write(templ)

            # 4) compile and return pdf path (also writes main.build.log)
            pdf_path = self._compile_pdf_from_tex(main_tex)
            return pdf_path if pdf_path and os.path.exists(pdf_path) else None

        # --- fallback: no user template; render a single-file tex and compile ---
        fallback_tex = os.path.join(papers_dir, "paper_generated.tex")
        self._render_paper_latex(md_paper_path=paper_md_path,
                                figures=figs,
                                code_items=code_items,
                                out_tex_path=fallback_tex)
        pdf_path = self._compile_pdf_from_tex(fallback_tex)
        return pdf_path if pdf_path and os.path.exists(pdf_path) else None
        
    def _record_figure(self, abs_path: str, caption: str, label: str = "Figure") -> None:
        """Append a JSON line to the current run's figures manifest and emit a breadcrumb marker.
        Use only from inside EXECUTE code via builtins if you want (optional).
        """
        import json, os
        run_dir = getattr(self, "run_dir", None)
        if not run_dir:
            return
        fig_dir = os.path.join(run_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        manifest = os.path.join(fig_dir, "manifest.jsonl")
        rec = {"path": abs_path, "caption": caption, "label": label}
        try:
            with open(manifest, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass
        # Standardized breadcrumb the SHARE phase (or you) can parse
        print(f"<<FIGURE:{abs_path}|{label}. {caption}>>")
        # --- Phase-specific LLM plumbing ---------------------------------------------
        # Map phase name -> LangChain chat model (or None = fallback to self.llm)


    def _latest_run_dirs(self):
        # returns (run_dir, papers_dir, logs_dir, figures_dir) for most recent run
        runs_root = os.path.join(self.path, "runs")
        if not os.path.isdir(runs_root):
            return (None, None, None, None)
        runs = sorted(
            [d for d in os.listdir(runs_root) if os.path.isdir(os.path.join(runs_root, d))],
            reverse=True
        )
        if not runs:
            return (None, None, None, None)
        run_dir = os.path.join(runs_root, runs[0])
        return (
            run_dir,
            os.path.join(run_dir, "papers"),
            os.path.join(run_dir, "logs"),
            os.path.join(run_dir, "figures"),
        )

    def _collect_run_artifacts(self):
        run_dir, papers_dir, logs_dir, figures_dir = self._latest_run_dirs()
        paper = None
        if papers_dir and os.path.isdir(papers_dir):
            md = sorted([f for f in os.listdir(papers_dir) if f.endswith(".md")], reverse=True)
            if md:
                paper = os.path.join(papers_dir, md[0])

        log_file = None
        if logs_dir and os.path.isdir(logs_dir):
            logs = sorted([f for f in os.listdir(logs_dir) if f.endswith(".txt") or f.endswith(".log")], reverse=True)
            if logs:
                log_file = os.path.join(logs_dir, logs[0])

        fig_manifest = None
        if figures_dir:
            mf = os.path.join(figures_dir, "manifest.jsonl")
            if os.path.exists(mf):
                fig_manifest = mf

        return {
            "run_dir": run_dir,
            "paper_md": paper,
            "log_file": log_file,
            "fig_manifest": fig_manifest,
    }



    def _gather_figures_for_latex(self, figures_dir: str):
        """
        figures_dir: DESTINATION (usually papers/figures). We'll copy here.
        We will read a manifest from DEST first; if missing, we fall back to run_dir/figures.
        """
        import json, shutil, glob, os

        os.makedirs(figures_dir, exist_ok=True)
        items = []

        # 1) Try manifest in the destination (papers/figures)
        manifest = os.path.join(figures_dir, "manifest.jsonl")
        src_for_manifest = figures_dir

        if not os.path.exists(manifest):
            # 2) Fallback: look in the run-scoped figures dir
            src_fig_dir = getattr(self, "run_figures_dir",
                                os.path.join(getattr(self, "run_dir", self.path), "figures"))
            alt_manifest = os.path.join(src_fig_dir, "manifest.jsonl")
            if os.path.exists(alt_manifest):
                manifest = alt_manifest
                src_for_manifest = src_fig_dir

        # Read manifest if present
        if os.path.exists(manifest):
            with open(manifest, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    src = (rec.get("path") or rec.get("abs_path") or "").strip()
                    cap = rec.get("caption", "").strip()
                    lab = rec.get("label", "Figure")
                    if not src:
                        continue
                    # Copy into destination next to the TeX
                    if os.path.isfile(src):
                        dst = os.path.join(figures_dir, os.path.basename(src))
                        if os.path.abspath(src) != os.path.abspath(dst):
                            try: shutil.copy2(src, dst)
                            except Exception: pass
                        items.append({"path": dst, "caption": cap, "label": lab})
            if items:
                return items

        # 3) No manifest? Fallback: include whatever is already in DEST
        for ext in ("*.png","*.jpg","*.jpeg","*.pdf","*.svg"):
            for p in sorted(glob.glob(os.path.join(figures_dir, ext))):
                items.append({"path": p, "caption": os.path.basename(p), "label": "Figure"})
        return items

    def _select_code_snippets(self, work_dir: str):
        """Pick the most relevant code to embed (by size, recency).
        Returns a list of dicts: [{name, language, content}]"""
        if not work_dir or not os.path.isdir(work_dir):
            return []
        candidates = []
        for pattern in self.code_globs:
            for p in glob.glob(os.path.join(work_dir, "**", pattern), recursive=True):
                try:
                    st = os.stat(p)
                    candidates.append((p, st.st_mtime, st.st_size))
                except:
                    pass
        # prefer recent and non-huge
        candidates.sort(key=lambda t: (-t[1], t[2]))
        picked, total = [], 0
        for p, _, size in candidates:
            if total >= self.max_code_bytes:
                break
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            # tiny heuristic: prioritize files that mention "execute(" or "agent.go"
            score = 0
            if "def execute" in content or "<execute>" in content:
                score += 2
            if "agent.go" in content or "StateGraph" in content:
                score += 1
            picked.append({"name": os.path.relpath(p, work_dir), "language": "python", "content": content, "score": score})
            total += min(size, self.max_code_bytes - total)
        # re-sort by score then recency implied by order
        picked.sort(key=lambda d: -d["score"])
        return picked[:5]  # keep top few




    def _latex_template(self) -> str:
        # Uses minted for code (requires -shell-escape). Falls back to listings if minted is missing.
        return r"""
    \documentclass[11pt]{article}
    \usepackage[margin=1in]{geometry}
    \usepackage[T1]{fontenc}
    \usepackage{lmodern}
    \usepackage{graphicx}
    \usepackage{caption}
    \usepackage{float}
    \usepackage{hyperref}
    \usepackage{booktabs}
    \usepackage{amsmath, amssymb}
    \usepackage{siunitx}
    \usepackage{xcolor}

    % Try minted; if not available at compile time, switch to listings manually.
    \IfFileExists{minted.sty}{
    \usepackage[cache=false]{minted}
    \newcommand{\codeblock}[2]{\begin{minted}[fontsize=\small,breaklines]{#1}#2\end{minted}}
    }{
    \usepackage{listings}
    \lstset{basicstyle=\ttfamily\small,breaklines=true}
    \newcommand{\codeblock}[2]{\begin{lstlisting}[language=Python]#2\end{lstlisting}}
    }

    \title{\textbf{<<TITLE>>}}
    \author{<<AUTHORS>>}
    \date{\today}

    \begin{document}
    \maketitle

    \begin{abstract}
    <<ABSTRACT>>
    \end{abstract}

    \textbf{Keywords:} <<KEYWORDS>>

    \section{Introduction}
    <<INTRODUCTION>>

    \section{Methods}
    <<METHODS>>

    \section{Results}
    <<RESULTS>>

    \section{Discussion}
    <<DISCUSSION>>

    \section{Figures}
    <<FIGURES>>

    \section{Code Excerpts}
    <<CODE>>

    \section{References}
    <<REFERENCES>>

    \end{document}
    """.lstrip()

    def _escape_tex(self, txt: str) -> str:
        # light escape for raw text blocks that go in normal paragraphs
        if not txt:
            return ""
        repl = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        for k,v in repl.items():
            txt = txt.replace(k, v)
        return txt

    def _paper_sections_from_md(self, paper_md_path: str):
        """If your SHARE phase still writes a markdown paper, parse rough sections to fill LaTeX fields."""
        fields = {
            "title": "Untitled",
            "authors": "Anonymous",
            "abstract": "",
            "keywords": "",
            "introduction": "",
            "methods": "",
            "results": "",
            "discussion": "",
            "references": "",
        }
        if not paper_md_path or not os.path.exists(paper_md_path):
            return fields
        txt = Path(paper_md_path).read_text(encoding="utf-8", errors="ignore")

        # super light parsing by headings
        import re
        def grab(head):
            m = re.search(rf"^#+\s*{head}\b(.*?)(^#|\Z)", txt, flags=re.I|re.M|re.S)
            return m.group(1).strip() if m else ""
        # Try to scrape a title line
        m_title = re.search(r"^#\s+(.+)$", txt, flags=re.M)
        if m_title: fields["title"] = m_title.group(1).strip()

        # Simple keys
        fields["abstract"]     = grab("Abstract")
        fields["introduction"] = grab("Introduction")
        fields["methods"]      = grab("Methods")
        fields["results"]      = grab("Results")
        fields["discussion"]   = grab("Discussion")
        fields["references"]   = grab("References|Bibliography")

        # Try authors/keywords from front matter lines
        m_auth = re.search(r"(?i)^Authors?:\s*(.+)$", txt, flags=re.M)
        if m_auth: fields["authors"] = m_auth.group(1).strip()
        m_key = re.search(r"(?i)^Keywords?:\s*(.+)$", txt, flags=re.M)
        if m_key: fields["keywords"] = m_key.group(1).strip()

        return fields

    def _render_paper_latex(self, md_paper_path: str, figures: list, code_items: list, out_tex_path: str):
        fields = self._paper_sections_from_md(md_paper_path)
        tmpl = self._latex_template()
        # Build the figures block
        # We’ll copy figures into a local ./figures subdir next to the tex for portability
        figs_block = []
        figs_dir = os.path.join(os.path.dirname(out_tex_path), "figures")
        os.makedirs(figs_dir, exist_ok=True)
        for i, f in enumerate(figures, start=1):
            dst = os.path.join(figs_dir, f"fig_{i}" + os.path.splitext(f["path"])[1].lower())
            try:
                if not os.path.exists(dst):
                    shutil.copyfile(f["path"], dst)
                cap = self._escape_tex(f.get("caption",""))
                label = self._escape_tex(f.get("label","Figure"))
                figs_block.append(
                    textwrap.dedent(f"""
                    \\begin{{figure}}[H]
                    \\centering
                    \\includegraphics[width=0.95\\linewidth]{{figures/{os.path.basename(dst)}}}
                    \\caption{{{label}. {cap}}}
                    \\end{{figure}}
                    """).strip()
                )
            except Exception as e:
                # skip if copy fails
                continue
        if not figs_block:
            figs_block.append("No figures were produced.")

        # Build the code block
        code_block_parts = []
        for item in code_items:
            name = self._escape_tex(item["name"])
            # Do not escape code content; minted/listings handle verbatim.
            code_block_parts.append(
                textwrap.dedent(f"""
                \\subsection*{{{name}}}
                \\codeblock{{python}}{{
    {item["content"]}
                }}
                """).strip()
            )
        if not code_block_parts:
            code_block_parts.append("No code excerpts selected.")

        replacements = {
            "<<TITLE>>":       self._escape_tex(fields["title"]),
            "<<AUTHORS>>":     self._escape_tex(fields["authors"]),
            "<<ABSTRACT>>":    self._escape_tex(fields["abstract"]),
            "<<KEYWORDS>>":    self._escape_tex(fields["keywords"]),
            "<<INTRODUCTION>>":self._escape_tex(fields["introduction"]),
            "<<METHODS>>":     self._escape_tex(fields["methods"]),
            "<<RESULTS>>":     self._escape_tex(fields["results"]),
            "<<DISCUSSION>>":  self._escape_tex(fields["discussion"]),
            "<<REFERENCES>>":  self._escape_tex(fields["references"]),
            "<<FIGURES>>":     "\n\n".join(figs_block),
            "<<CODE>>":        "\n\n".join(code_block_parts),
        }
        for k,v in replacements.items():
            tmpl = tmpl.replace(k, v)

        Path(out_tex_path).write_text(tmpl, encoding="utf-8")
        return out_tex_path

    def _compile_pdf_from_tex(self, tex_path: str):
        """Compile with latexmk; log stdout/stderr to a build log; minted-first, listings-fallback."""
        workdir = os.path.dirname(tex_path)
        base = os.path.splitext(os.path.basename(tex_path))[0]
        log_file = os.path.join(workdir, f"{base}.build.log")

        def run_and_log(cmd):
            try:
                p = subprocess.run(
                    cmd, cwd=workdir, check=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                Path(log_file).write_text(
                    (p.stdout or b"").decode("utf-8","ignore")
                    + "\n--- STDERR ---\n"
                    + (p.stderr or b"").decode("utf-8","ignore"),
                    encoding="utf-8"
                )
                return True
            except FileNotFoundError as e:
                Path(log_file).write_text(f"[FileNotFoundError] {e}\nCommand: {cmd}\n", encoding="utf-8")
                return False
            except subprocess.CalledProcessError as e:
                Path(log_file).write_text(
                    (e.stdout or b"").decode("utf-8","ignore")
                    + "\n--- STDERR ---\n"
                    + (e.stderr or b"").decode("utf-8","ignore"),
                    encoding="utf-8"
                )
                return False

        # minted (needs -shell-escape) → listings fallback
        if run_and_log(["latexmk", "-pdf", "-interaction=nonstopmode", "-shell-escape", os.path.basename(tex_path)]):
            return os.path.join(workdir, base + ".pdf")
        if run_and_log(["latexmk", "-pdf", "-interaction=nonstopmode", os.path.basename(tex_path)]):
            return os.path.join(workdir, base + ".pdf")
        return None




    def configure_phase_llms(self, **phase_to_llm):
        """
        Accepts either ready model objects or (model, source, base_url, api_key) tuples.

        Examples:
        agent.v(
            PLAN=("gpt-4o-mini","OpenAI"),
            LEARN=("gemini-1.5-pro","Gemini"),
            EXECUTE=("claude-3-5-sonnet-20240620","Anthropic"),
            ASSESS=("gpt-4o-mini","OpenAI"),
            SHARE=("gpt-4o-mini","OpenAI"),
        )

        # or pass instantiated chat models
        from langchain_openai import ChatOpenAI
        agent.configure_phase_llms(
            PLAN=ChatOpenAI(model="gpt-4o-mini"),
            EXECUTE=ChatOpenAI(model="gpt-4o-mini"),
        )
        """
        for phase, model in phase_to_llm.items():
            key = str(phase).upper()
            if isinstance(model, tuple):
                llm_name, source, *rest = model
                base_url         = rest[0] if len(rest) > 0 else None
                api_key          = rest[1] if len(rest) > 1 else None
                extended_thinking = bool(rest[2]) if len(rest) > 2 else False
                # Fallback to environment variable if api_key is not provided
                if api_key is None:
                    if source == "OpenAI":
                        api_key = os.environ.get("OPENAI_API_KEY")
                    elif source == "Anthropic":
                        api_key = os.environ.get("ANTHROPIC_API_KEY")
                    elif source == "Perplexity":
                        # Try both possible env var names
                        api_key = os.environ.get("Perplexity_API_KEY") or os.environ.get("PPLX_API_KEY")
                    elif source == "Gemini":
                        api_key = os.environ.get("GEMINI_API_KEY")
                    elif source == "Google":
                        api_key = os.environ.get("GOOGLE_API_KEY")
                    elif source == "MiniMax":
                        api_key = os.environ.get("MINIMAX_API_KEY")
                    # Add more providers as needed
                self.phase_llms[key] = get_llm(
                    llm_name, source=source, base_url=base_url, api_key=api_key,
                    extended_thinking=extended_thinking,
                )
            else:
                self.phase_llms[key] = model

    def _llm_for(self, phase: str):
        llm = self.phase_llms.get(phase.upper()) or self.llm
        print(f"[LLM_FOR] phase={phase} → {type(llm).__name__} ({getattr(llm, 'model_name', getattr(llm, 'model', 'unknown'))})")
        return llm
    
    def _prompt_for(self, phase: str) -> str:
        """Return the phase-specific system prompt, with robust fallback."""
        p = (phase or "").upper()
        return (
            self.system_prompts.get(p)
            or self.system_prompts.get("EXECUTE")
            or getattr(self, "system_prompt", "")  # backwards-compat
            or ""
        )

# ----------------------------------------------------------------------------- 

        
        
    def add_tool(self, api):
        """Add a new tool to the agent's tool registry and make it available for retrieval.

        Args:
            api: A callable function to be added as a tool

        """
        try:
            # Get function information
            function_code = inspect.getsource(api)
            module_name = api.__module__ if hasattr(api, "__module__") else "custom_tools"
            function_name = api.__name__ if hasattr(api, "__name__") else str(api)

            # Generate API schema using the existing utility function
            schema = function_to_api_schema(function_code, self.llm)

            # Ensure the schema has all required fields for the tool registry
            if not isinstance(schema, dict):
                raise ValueError("Generated schema is not a dictionary")

            # Validate and enhance the schema

            # Set default values if missing
            if "name" not in schema:
                schema["name"] = function_name
            if "description" not in schema:
                schema["description"] = f"Custom tool: {function_name}"
            if "required_parameters" not in schema:
                # Try to extract from parameters if available
                if "parameters" in schema and isinstance(schema["parameters"], dict):
                    required_params = []
                    params = schema["parameters"]
                    if "properties" in params:
                        for param_name in params["properties"]:
                            if param_name in params.get("required", []):
                                required_params.append(param_name)
                    schema["required_parameters"] = required_params
                else:
                    schema["required_parameters"] = []

            # Add module information to the schema
            schema["module"] = module_name

            # Add the tool to the tool registry if it exists
            if hasattr(self, "tool_registry") and self.tool_registry is not None:
                try:
                    self.tool_registry.register_tool(schema)
                    print(f"Successfully registered tool '{schema['name']}' in tool registry")
                except Exception as e:
                    print(f"Warning: Failed to register tool in registry: {e}")
                    # Continue with adding to module2api even if registry fails

            # Add the tool to module2api structure for system prompt generation
            if not hasattr(self, "module2api") or self.module2api is None:
                self.module2api = {}

            if module_name not in self.module2api:
                self.module2api[module_name] = []

            # Check if tool already exists in module2api to avoid duplicates
            existing_tool = None
            for existing in self.module2api[module_name]:
                if existing.get("name") == schema["name"]:
                    existing_tool = existing
                    break

            if existing_tool:
                # Update existing tool
                existing_tool.update(schema)
                print(f"Updated existing tool '{schema['name']}' in module '{module_name}'")
            else:
                # Add new tool
                self.module2api[module_name].append(schema)
                print(f"Added new tool '{schema['name']}' to module '{module_name}'")

            # Update the tool registry's document dataframe if it exists
            if hasattr(self, "tool_registry") and self.tool_registry is not None:
                try:
                    # Rebuild the document dataframe
                    docs = []
                    for tool_id in range(len(self.tool_registry.tools)):
                        docs.append(
                            [
                                int(tool_id),
                                self.tool_registry.get_tool_by_id(int(tool_id)),
                            ]
                        )
                    self.tool_registry.document_df = pd.DataFrame(docs, columns=["docid", "document_content"])
                except Exception as e:
                    print(f"Warning: Failed to update tool registry document dataframe: {e}")

            # Store the original function for potential future use
            if not hasattr(self, "_custom_functions"):
                self._custom_functions = {}
            self._custom_functions[schema["name"]] = api

            # Also store in _custom_tools for highlighting
            if not hasattr(self, "_custom_tools"):
                self._custom_tools = {}
            self._custom_tools[schema["name"]] = {
                "name": schema["name"],
                "description": schema["description"],
                "module": module_name,
            }

            # Make the function available in the global namespace for execution
            import builtins

            if not hasattr(builtins, "_biomni_custom_functions"):
                builtins._biomni_custom_functions = {}
            builtins._biomni_custom_functions[schema["name"]] = api

            print(
                f"Tool '{schema['name']}' successfully added and ready for use in both direct execution and retrieval"
            )
            self.configure()
            return schema

        except Exception as e:
            print(f"Error adding tool: {e}")
            import traceback

            traceback.print_exc()
            raise

    def add_mcp(self, config_path: str | Path = "./tutorials/examples/mcp_config.yaml") -> None:
        """
        Add MCP (Model Context Protocol) tools from configuration file.

        This method dynamically registers MCP server tools as callable functions within
        the BioPLEASE agent system. Each MCP server is loaded as an independent module
        with its tools exposed as synchronous wrapper functions.

        Supports both manual tool definitions and automatic tool discovery from MCP servers.

        Args:
            config_path: Path to the MCP configuration YAML file containing server
                        definitions and tool specifications.

        Raises:
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If the config file is malformed
            RuntimeError: If MCP server initialization fails
        """
        import asyncio
        import os
        import sys
        import types
        from pathlib import Path

        import nest_asyncio
        import yaml
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        nest_asyncio.apply()

        def discover_mcp_tools_sync(server_params: StdioServerParameters) -> list[dict]:
            """Discover available tools from MCP server synchronously."""
            try:

                async def _discover_async():
                    async with stdio_client(server_params) as (reader, writer):
                        async with ClientSession(reader, writer) as session:
                            await session.initialize()

                            # Get available tools
                            tools_result = await session.list_tools()
                            tools = tools_result.tools if hasattr(tools_result, "tools") else tools_result

                            discovered_tools = []
                            for tool in tools:
                                if hasattr(tool, "name"):
                                    discovered_tools.append(
                                        {
                                            "name": tool.name,
                                            "description": tool.description,
                                            "inputSchema": tool.inputSchema,
                                        }
                                    )
                                else:
                                    print(f"Warning: Skipping tool with no name attribute: {tool}")

                            return discovered_tools

                return asyncio.run(_discover_async())
            except Exception as e:
                print(f"Failed to discover tools: {e}")
                return []

        def make_mcp_wrapper(cmd: str, args: list[str], tool_name: str, doc: str, env_vars: dict = None):
            """Create a synchronous wrapper for an async MCP tool call."""

            def sync_tool_wrapper(**kwargs):
                """Synchronous wrapper for MCP tool execution."""
                try:
                    server_params = StdioServerParameters(command=cmd, args=args, env=env_vars)

                    async def async_tool_call():
                        async with stdio_client(server_params) as (reader, writer):
                            async with ClientSession(reader, writer) as session:
                                await session.initialize()
                                result = await session.call_tool(tool_name, kwargs)
                                content = result.content[0]
                                if hasattr(content, "json"):
                                    return content.json()
                                return content.text

                    try:
                        loop = asyncio.get_running_loop()
                        return loop.create_task(async_tool_call())
                    except RuntimeError:
                        return asyncio.run(async_tool_call())

                except Exception as e:
                    raise RuntimeError(f"MCP tool execution failed for '{tool_name}': {e}") from e

            sync_tool_wrapper.__name__ = tool_name
            sync_tool_wrapper.__doc__ = doc
            return sync_tool_wrapper

        # Initialize registries if they don't exist
        self._custom_functions = getattr(self, "_custom_functions", {})
        self._custom_tools = getattr(self, "_custom_tools", {})

        # Load and validate configuration
        try:
            config_content = Path(config_path).read_text(encoding="utf-8")
            cfg: dict[str, Any] = yaml.safe_load(config_content) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"MCP config file not found: {config_path}") from None
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in MCP config: {e}") from e

        mcp_servers: dict[str, Any] = cfg.get("mcp_servers", {})
        if not mcp_servers:
            print("Warning: No MCP servers found in configuration")
            return

        # Process each MCP server configuration
        for server_name, server_meta in mcp_servers.items():
            if not server_meta.get("enabled", True):
                continue

            # Validate command configuration
            cmd_list = server_meta.get("command", [])
            if not cmd_list or not isinstance(cmd_list, list):
                print(f"Warning: Invalid command configuration for server '{server_name}'")
                continue

            cmd, *args = cmd_list

            # Process environment variables
            env_vars = server_meta.get("env", {})
            if env_vars:
                processed_env = {}
                for key, value in env_vars.items():
                    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                        var_name = value[2:-1]
                        processed_env[key] = os.getenv(var_name, "")
                    else:
                        processed_env[key] = value
                env_vars = processed_env

            # Create module namespace for this MCP server
            mcp_module_name = f"mcp_servers.{server_name}"
            if mcp_module_name not in sys.modules:
                sys.modules[mcp_module_name] = types.ModuleType(mcp_module_name)
            server_module = sys.modules[mcp_module_name]

            tools_config = server_meta.get("tools", [])

            if not tools_config:
                try:
                    server_params = StdioServerParameters(command=cmd, args=args, env=env_vars)
                    tools_config = discover_mcp_tools_sync(server_params)

                    if tools_config:
                        print(f"Discovered {len(tools_config)} tools from {server_name} MCP server")
                    else:
                        print(f"Warning: No tools discovered from {server_name} MCP server")
                        continue

                except Exception as e:
                    print(f"Failed to discover tools for {server_name}: {e}")
                    continue

            # Register each tool
            for tool_meta in tools_config:
                if isinstance(tool_meta, dict) and "biomni_name" in tool_meta:
                    # Manual tool definition
                    tool_name = tool_meta.get("biomni_name")
                    description = tool_meta.get("description", f"MCP tool: {tool_name}")
                    parameters = tool_meta.get("parameters", {})
                else:
                    # Auto-discovered tool
                    tool_name = tool_meta.get("name")
                    description = tool_meta.get("description", f"MCP tool: {tool_name}")
                    parameters = tool_meta.get("inputSchema", {}).get("properties", {})

                if not tool_name:
                    print(f"Warning: Skipping tool with no name in {server_name}")
                    continue

                # Create wrapper function
                wrapper_function = make_mcp_wrapper(cmd, args, tool_name, description, env_vars)

                # Add to module namespace
                setattr(server_module, tool_name, wrapper_function)

                # Build parameter lists
                required_params, optional_params = [], []
                for param_name, param_spec in parameters.items():
                    param_info = {
                        "name": param_name,
                        "type": str(param_spec.get("type", "string")),
                        "description": param_spec.get("description", ""),
                        "default": param_spec.get("default", None),
                    }

                    if param_spec.get("required", False):
                        required_params.append(param_info)
                    else:
                        optional_params.append(param_info)

                # Create tool schema
                tool_schema = {
                    "name": tool_name,
                    "description": description,
                    "parameters": parameters,
                    "required_parameters": required_params,
                    "optional_parameters": optional_params,
                    "module": mcp_module_name,
                    "fn": wrapper_function,
                }

                # Register in tool registry
                self.tool_registry.register_tool(tool_schema)

                # Add to module2api mapping
                if mcp_module_name not in self.module2api:
                    self.module2api[mcp_module_name] = []
                self.module2api[mcp_module_name].append(tool_schema)

                # Add to instance registries
                self._custom_functions[tool_name] = wrapper_function
                self._custom_tools[tool_name] = {
                    "name": tool_name,
                    "description": description,
                    "module": mcp_module_name,
                }

        # Update agent configuration
        self.configure()

    def get_custom_tool(self, name):
        """Get a custom tool by name.

        Args:
            name: The name of the custom tool

        Returns:
            The custom tool function if found, None otherwise

        """
        if hasattr(self, "_custom_functions") and name in self._custom_functions:
            return self._custom_functions[name]
        return None

    def list_custom_tools(self):
        """List all custom tools that have been added.

        Returns:
            A list of custom tool names

        """
        if hasattr(self, "_custom_functions"):
            return list(self._custom_functions.keys())
        return []

    def remove_custom_tool(self, name):
        """Remove a custom tool.

        Args:
            name: The name of the custom tool to remove

        Returns:
            True if the tool was removed, False if it wasn't found

        """
        removed = False

        # Remove from custom functions
        if hasattr(self, "_custom_functions") and name in self._custom_functions:
            del self._custom_functions[name]
            removed = True

        # Remove from custom tools (for highlighting)
        if hasattr(self, "_custom_tools") and name in self._custom_tools:
            del self._custom_tools[name]
            removed = True

        # Remove from global namespace
        import builtins

        if hasattr(builtins, "_biomni_custom_functions") and name in builtins._biomni_custom_functions:
            del builtins._biomni_custom_functions[name]

        # Remove from tool registry
        if hasattr(self, "tool_registry") and self.tool_registry is not None:
            if self.tool_registry.remove_tool_by_name(name):
                removed = True
                # Rebuild the document dataframe
                try:
                    docs = []
                    for tool_id in range(len(self.tool_registry.tools)):
                        docs.append(
                            [
                                int(tool_id),
                                self.tool_registry.get_tool_by_id(int(tool_id)),
                            ]
                        )
                    self.tool_registry.document_df = pd.DataFrame(docs, columns=["docid", "document_content"])
                except Exception as e:
                    print(f"Warning: Failed to update tool registry document dataframe: {e}")

        # Remove from module2api
        if hasattr(self, "module2api"):
            for tools in self.module2api.values():
                for i, tool in enumerate(tools):
                    if tool.get("name") == name:
                        del tools[i]
                        removed = True
                        break

        if removed:
            print(f"Custom tool '{name}' has been removed")
        else:
            print(f"Custom tool '{name}' was not found")

        return removed

    def add_data(self, data):
        """Add new data to the data lake.

        Args:
            data: Dictionary with file path as key and description as value
                  e.g., {'my_dataset.csv': 'A dataset containing gene expression data'}
                  or {'path/to/file.txt': 'Description of the file'}

        """
        try:
            if not isinstance(data, dict):
                raise ValueError("Data must be a dictionary with file path as key and description as value")

            # Initialize custom data storage if it doesn't exist
            if not hasattr(self, "_custom_data"):
                self._custom_data = {}

            # Add each data item
            for file_path, description in data.items():
                if not isinstance(file_path, str) or not isinstance(description, str):
                    print("Warning: Skipping invalid data entry - file_path and description must be strings")
                    continue

                # Extract filename from path for storage
                filename = os.path.basename(file_path) if "/" in file_path else file_path

                # Store the data with both the full path and description
                self._custom_data[filename] = {
                    "path": file_path,
                    "description": description,
                }

                # Also add to the data_lake_dict for consistency
                self.data_lake_dict[filename] = description

                print(f"Added data item '{filename}': {description}")
            self.configure()
            print(f"Successfully added {len(data)} data item(s) to the data lake")
            return True

        except Exception as e:
            print(f"Error adding data: {e}")
            import traceback

            traceback.print_exc()
            return False

    def get_custom_data(self, name):
        """Get a custom data item by name.

        Args:
            name: The name of the custom data item

        Returns:
            The custom data item info if found, None otherwise

        """
        if hasattr(self, "_custom_data") and name in self._custom_data:
            return self._custom_data[name]
        return None

    def list_custom_data(self):
        """List all custom data items that have been added.

        Returns:
            A list of custom data item names and descriptions

        """
        if hasattr(self, "_custom_data"):
            return [(name, info["description"]) for name, info in self._custom_data.items()]
        return []

    def remove_custom_data(self, name):
        """Remove a custom data item.

        Args:
            name: The name of the custom data item to remove

        Returns:
            True if the data item was removed, False if it wasn't found

        """
        removed = False

        # Remove from custom data
        if hasattr(self, "_custom_data") and name in self._custom_data:
            del self._custom_data[name]
            removed = True

        # Remove from data_lake_dict
        if hasattr(self, "data_lake_dict") and name in self.data_lake_dict:
            del self.data_lake_dict[name]
            removed = True

        if removed:
            print(f"Custom data item '{name}' has been removed")
        else:
            print(f"Custom data item '{name}' was not found")

        return removed

    def add_software(self, software):
        """Add new software to the software library.

        Args:
            software: Dictionary with software name as key and description as value
                     e.g., {'custom_tool': 'A custom analysis tool for processing data'}
                     or {'my_package': 'Description of the package functionality'}

        """
        try:
            if not isinstance(software, dict):
                raise ValueError("Software must be a dictionary with software name as key and description as value")

            # Initialize custom software storage if it doesn't exist
            if not hasattr(self, "_custom_software"):
                self._custom_software = {}

            # Add each software item
            for software_name, description in software.items():
                if not isinstance(software_name, str) or not isinstance(description, str):
                    print("Warning: Skipping invalid software entry - software_name and description must be strings")
                    continue

                # Store the software with description
                self._custom_software[software_name] = {
                    "name": software_name,
                    "description": description,
                }

                # Also add to the library_content_dict for consistency
                self.library_content_dict[software_name] = description

                print(f"Added software '{software_name}': {description}")

            print(f"Successfully added {len(software)} software item(s) to the library")
            self.configure()
            return True

        except Exception as e:
            print(f"Error adding software: {e}")
            import traceback

            traceback.print_exc()
            return False

    def get_custom_software(self, name):
        """Get a custom software item by name.

        Args:
            name: The name of the custom software item

        Returns:
            The custom software item info if found, None otherwise

        """
        if hasattr(self, "_custom_software") and name in self._custom_software:
            return self._custom_software[name]
        return None

    def list_custom_software(self):
        """List all custom software items that have been added.

        Returns:
            A list of custom software item names and descriptions

        """
        if hasattr(self, "_custom_software"):
            return [(name, info["description"]) for name, info in self._custom_software.items()]
        return []

    def remove_custom_software(self, name):
        """Remove a custom software item.

        Args:
            name: The name of the custom software item to remove

        Returns:
            True if the software item was removed, False if it wasn't found

        """
        removed = False

        # Remove from custom software
        if hasattr(self, "_custom_software") and name in self._custom_software:
            del self._custom_software[name]
            removed = True

        # Remove from library_content_dict
        if hasattr(self, "library_content_dict") and name in self.library_content_dict:
            del self.library_content_dict[name]
            removed = True

        if removed:
            print(f"Custom software item '{name}' has been removed")
        else:
            print(f"Custom software item '{name}' was not found")

        return removed

    def _generate_system_prompt(
        self,
        tool_desc,
        data_lake_content,
        library_content_list,
        self_critic=False,
        is_retrieval=False,
        custom_tools=None,
        custom_data=None,
        custom_software=None,
    ):
        """Generate the system prompt based on the provided resources.

        Args:
            tool_desc: Dictionary of tool descriptions
            data_lake_content: List of data lake items
            library_content_list: List of libraries
            self_critic: Whether to include self-critic instructions
            is_retrieval: Whether this is for retrieval (True) or initial configuration (False)
            custom_tools: List of custom tools to highlight
            custom_data: List of custom data items to highlight
            custom_software: List of custom software items to highlight

        Returns:
            The generated system prompt

        """

        def format_item_with_description(name, description):
            """Format an item with its description in a readable way."""
            # Handle None or empty descriptions
            if not description:
                description = f"Data lake item: {name}"

            # Check if the item is already formatted (contains a colon)
            if isinstance(name, str) and ": " in name:
                return name

            # Wrap long descriptions to make them more readable
            max_line_length = 80
            if len(description) > max_line_length:
                # Simple wrapping for long descriptions
                wrapped_desc = []
                words = description.split()
                current_line = ""

                for word in words:
                    if len(current_line) + len(word) + 1 <= max_line_length:
                        if current_line:
                            current_line += " " + word
                        else:
                            current_line = word
                    else:
                        wrapped_desc.append(current_line)
                        current_line = word

                if current_line:
                    wrapped_desc.append(current_line)

                # Join with newlines and proper indentation
                formatted_desc = f"{name}:\n  " + "\n  ".join(wrapped_desc)
                return formatted_desc
            else:
                return f"{name}: {description}"

        # Separate custom and default resources
        default_data_lake_content = []
        default_library_content_list = []

        # Filter out custom items from default lists
        custom_data_names = set()
        custom_software_names = set()

        if custom_data:
            custom_data_names = {item.get("name") if isinstance(item, dict) else item for item in custom_data}
        if custom_software:
            custom_software_names = {item.get("name") if isinstance(item, dict) else item for item in custom_software}

        # Separate default data lake items
        for item in data_lake_content:
            if isinstance(item, dict):
                name = item.get("name", "")
                if name not in custom_data_names:
                    default_data_lake_content.append(item)
            elif item not in custom_data_names:
                default_data_lake_content.append(item)

        # Separate default library items
        for lib in library_content_list:
            if isinstance(lib, dict):
                name = lib.get("name", "")
                if name not in custom_software_names:
                    default_library_content_list.append(lib)
            elif lib not in custom_software_names:
                default_library_content_list.append(lib)

        # Format the default data lake content
        if isinstance(default_data_lake_content, list) and all(
            isinstance(item, str) for item in default_data_lake_content
        ):
            # Simple list of strings - check if they already have descriptions
            data_lake_formatted = []
            for item in default_data_lake_content:
                # Check if the item already has a description (contains a colon)
                if ": " in item:
                    data_lake_formatted.append(item)
                else:
                    description = self.data_lake_dict.get(item, f"Data lake item: {item}")
                    data_lake_formatted.append(format_item_with_description(item, description))
        else:
            # List with descriptions
            data_lake_formatted = []
            for item in default_data_lake_content:
                if isinstance(item, dict):
                    name = item.get("name", "")
                    description = self.data_lake_dict.get(name, f"Data lake item: {name}")
                    data_lake_formatted.append(format_item_with_description(name, description))
                # Check if the item already has a description (contains a colon)
                elif isinstance(item, str) and ": " in item:
                    data_lake_formatted.append(item)
                else:
                    description = self.data_lake_dict.get(item, f"Data lake item: {item}")
                    data_lake_formatted.append(format_item_with_description(item, description))

        # Format the default library content
        if isinstance(default_library_content_list, list) and all(
            isinstance(item, str) for item in default_library_content_list
        ):
            if (
                len(default_library_content_list) > 0
                and isinstance(default_library_content_list[0], str)
                and "," not in default_library_content_list[0]
            ):
                # Simple list of strings
                libraries_formatted = []
                for lib in default_library_content_list:
                    description = self.library_content_dict.get(lib, f"Software library: {lib}")
                    libraries_formatted.append(format_item_with_description(lib, description))
            else:
                # Already formatted string
                libraries_formatted = default_library_content_list
        else:
            # List with descriptions
            libraries_formatted = []
            for lib in default_library_content_list:
                if isinstance(lib, dict):
                    name = lib.get("name", "")
                    description = self.library_content_dict.get(name, f"Software library: {name}")
                    libraries_formatted.append(format_item_with_description(name, description))
                else:
                    description = self.library_content_dict.get(lib, f"Software library: {lib}")
                    libraries_formatted.append(format_item_with_description(lib, description))

        # Format custom resources with highlighting
        custom_tools_formatted = []
        if custom_tools:
            for tool in custom_tools:
                if isinstance(tool, dict):
                    name = tool.get("name", "Unknown")
                    desc = tool.get("description", "")
                    module = tool.get("module", "custom_tools")
                    custom_tools_formatted.append(f"🔧 {name} (from {module}): {desc}")
                else:
                    custom_tools_formatted.append(f"🔧 {str(tool)}")

        custom_data_formatted = []
        if custom_data:
            for item in custom_data:
                if isinstance(item, dict):
                    name = item.get("name", "Unknown")
                    desc = item.get("description", "")
                    custom_data_formatted.append(f"📊 {format_item_with_description(name, desc)}")
                else:
                    desc = self.data_lake_dict.get(item, f"Custom data: {item}")
                    custom_data_formatted.append(f"📊 {format_item_with_description(item, desc)}")

        custom_software_formatted = []
        if custom_software:
            for item in custom_software:
                if isinstance(item, dict):
                    name = item.get("name", "Unknown")
                    desc = item.get("description", "")
                    custom_software_formatted.append(f"⚙️ {format_item_with_description(name, desc)}")
                else:
                    desc = self.library_content_dict.get(item, f"Custom software: {item}")
                    custom_software_formatted.append(f"⚙️ {format_item_with_description(item, desc)}")

        # Base prompt
        prompt_modifier = """
You are a helpful biomedical assistant assigned with the task of problem-solving.
To achieve this, you will be using an interactive coding environment equipped with a variety of tool functions, data, and softwares to assist you throughout the process.

Given a task, make a plan first. The plan should be a numbered list of steps that you will take to solve the task. Be specific and detailed.

CRITICAL FORMATTING RULE: You must maintain a "=== GLOBAL PLAN ===" that persists throughout the entire run. If execution fails, DO NOT replace the plan with a short debugging checklist; instead, add indented sub-tasks.

Format your plan as a checklist with empty checkboxes like this:

=== GLOBAL PLAN ===
1. [ ] Phase 1: High-level goal (e.g. Data Gathering)
   - [ ] Specific sub-task
2. [ ] Phase 2: High-level goal (e.g. Model Training)
3. [ ] Phase 3: High-level goal (e.g. Evaluation)
===================

Follow the plan step by step. After completing each step, update the checklist by replacing the empty checkbox:
=== GLOBAL PLAN ===
1. [✓] Phase 1: High-level goal (completed)
2. [ ] Phase 2: High-level goal (current)
   - [ ] New active sub-task
3. [ ] Phase 3: High-level goal
===================

If a step fails or needs modification, mark it with an X, explain why, AND KEEP ALL FUTURE PHASES IN THE LIST:
=== GLOBAL PLAN ===
1. [✗] Phase 1: High-level goal (failed due to missing file)
   - [ ] MICRO-STEP: Write script to download missing file
2. [ ] Phase 2: High-level goal
3. [ ] Phase 3: High-level goal
===================

Always show the updated plan after each step so the user can track progress.

At each turn, you should first provide your thinking and reasoning given the conversation history.
After that, you have two options:

1) Interact with a programming environment and receive the corresponding output within <observe></observe>. Your code should be enclosed using "<execute>" tag, for example: <execute> print("Hello World!") </execute>. IMPORTANT: You must end the code block with </execute> tag.
   - For Python code (default): <execute> print("Hello World!") </execute>
   - For R code: <execute> #!R\nlibrary(ggplot2)\nprint("Hello from R") </execute>
   - For Bash scripts and commands: <execute> #!BASH\necho "Hello from Bash"\nls -la </execute>
   - For CLI softwares, use Bash scripts.

2) When you think it is ready, directly provide a solution that adheres to the required format for the given task to the user. Your solution should be enclosed using "<solution>" tag, for example: The answer is <solution> A </solution>. IMPORTANT: You must end the solution block with </solution> tag.

You have many chances to interact with the environment to receive the observation. So you can decompose your code into multiple steps.
Don't overcomplicate the code. Keep it simple and easy to understand.
When writing the code, please print out the steps and results in a clear and concise manner, like a research log.
When calling the existing python functions in the function dictionary, YOU MUST SAVE THE OUTPUT and PRINT OUT the result.
For example, result = understand_scRNA(XXX) print(result)
Otherwise the system will not be able to know what has been done.

For R code, use the #!R marker at the beginning of your code block to indicate it's R code.
For Bash scripts and commands, use the #!BASH marker at the beginning of your code block. This allows for both simple commands and multi-line scripts with variables, loops, conditionals, loops, and other Bash features.

In each response, you must include EITHER <execute> or <solution> tag. Not both at the same time. Do not respond with messages without any tags. No empty messages.
"""

        # Add self-critic instructions if needed
        if self_critic:
            prompt_modifier += """
You may or may not receive feedbacks from human. If so, address the feedbacks by following the same procedure of multiple rounds of thinking, execution, and then coming up with a new solution.
"""

        # Add custom resources section first (highlighted)
        has_custom_resources = any([custom_tools_formatted, custom_data_formatted, custom_software_formatted])

        if has_custom_resources:
            prompt_modifier += """

PRIORITY CUSTOM RESOURCES
===============================
IMPORTANT: The following custom resources have been specifically added for your use.
    PRIORITIZE using these resources as they are directly relevant to your task.
    Always consider these FIRST and in the meantime using default resources.

"""

            if custom_tools_formatted:
                prompt_modifier += """
CUSTOM TOOLS (USE THESE FIRST):
{custom_tools}

"""

            if custom_data_formatted:
                prompt_modifier += """
CUSTOM DATA (PRIORITIZE THESE DATASETS):
{custom_data}

"""

            if custom_software_formatted:
                prompt_modifier += """
⚙️ CUSTOM SOFTWARE (USE THESE LIBRARIES):
{custom_software}

"""

            prompt_modifier += """===============================
"""

        # Add environment resources
        prompt_modifier += """

Environment Resources:

- Function Dictionary:
{function_intro}
---
{tool_desc}
---

{import_instruction}

- Biological data lake
You can access a biological data lake at the following path: {data_lake_path}.
{data_lake_intro}
Each item is listed with its description to help you understand its contents.
----
{data_lake_content}
----

- Software Library:
{library_intro}
Each library is listed with its description to help you understand its functionality.
----
{library_content_formatted}
----

- Note on using R packages and Bash scripts:
  - R packages: Use subprocess.run(['Rscript', '-e', 'your R code here']) in Python, or use the #!R marker in your execute block.
  - Bash scripts and commands: Use the #!BASH marker in your execute block for both simple commands and complex shell scripts with variables, loops, conditionals, etc.
        """

        # Set appropriate text based on whether this is initial configuration or after retrieval
        if is_retrieval:
            function_intro = "Based on your query, I've identified the following most relevant functions that you can use in your code:"
            data_lake_intro = "Based on your query, I've identified the following most relevant datasets:"
            library_intro = (
                "Based on your query, I've identified the following most relevant libraries that you can use:"
            )
            import_instruction = "IMPORTANT: When using any function, you MUST first import it from its module. For example:\nfrom [module_name] import [function_name]"
        else:
            function_intro = "In your code, you will need to import the function location using the following dictionary of functions:"
            data_lake_intro = "You can write code to understand the data, process and utilize it for the task. Here is the list of datasets:"
            library_intro = "The environment supports a list of libraries that can be directly used. Do not forget the import statement:"
            import_instruction = ""

        # Format the content consistently for both initial and retrieval cases
        library_content_formatted = "\n".join(libraries_formatted)
        data_lake_content_formatted = "\n".join(data_lake_formatted)

        # Format the prompt with the appropriate values
        format_dict = {
            "function_intro": function_intro,
            "tool_desc": textify_api_dict(tool_desc) if isinstance(tool_desc, dict) else tool_desc,
            "import_instruction": import_instruction,
            "data_lake_path": self.path + "/data_lake",
            "data_lake_intro": data_lake_intro,
            "data_lake_content": data_lake_content_formatted,
            "library_intro": library_intro,
            "library_content_formatted": library_content_formatted,
        }

        # Add custom resources to format dict (always add them, even if empty)
        format_dict["custom_tools"] = "\n".join(custom_tools_formatted) if custom_tools_formatted else ""
        format_dict["custom_data"] = "\n".join(custom_data_formatted) if custom_data_formatted else ""
        format_dict["custom_software"] = "\n".join(custom_software_formatted) if custom_software_formatted else ""

        run_dir = getattr(self, "run_dir", self.path)
        format_dict["run_dir"] = run_dir
        format_dict["fig_dir"] = getattr(self, "run_figures_dir", os.path.join(run_dir, "figures"))
        format_dict["arts_dir"] = getattr(self, "run_artifacts_dir", os.path.join(run_dir, "artifacts"))

        formatted_prompt = prompt_modifier.format(**format_dict)

        return formatted_prompt
    
    def _plan_system_prompt(
        self,
        tool_desc,
        data_lake_content,
        library_content_list,
        self_critic=False,
        is_retrieval=False,
        custom_tools=None,
        custom_data=None,
        custom_software=None,
        current_cost=None,
        cost_budget=None,
        show_scientific_mindset=True,
    ):
        """Generate the system prompt based on the provided resources.

        Args:
            tool_desc: Dictionary of tool descriptions
            data_lake_content: List of data lake items
            library_content_list: List of libraries
            self_critic: Whether to include self-critic instructions
            is_retrieval: Whether this is for retrieval (True) or initial configuration (False)
            custom_tools: List of custom tools to highlight
            custom_data: List of custom data items to highlight
            custom_software: List of custom software items to highlight

        Returns:
            The generated system prompt

        """

        def format_item_with_description(name, description):
            """Format an item with its description in a readable way."""
            # Handle None or empty descriptions
            if not description:
                description = f"Data lake item: {name}"

            # Check if the item is already formatted (contains a colon)
            if isinstance(name, str) and ": " in name:
                return name

            # Wrap long descriptions to make them more readable
            max_line_length = 80
            if len(description) > max_line_length:
                # Simple wrapping for long descriptions
                wrapped_desc = []
                words = description.split()
                current_line = ""

                for word in words:
                    if len(current_line) + len(word) + 1 <= max_line_length:
                        if current_line:
                            current_line += " " + word
                        else:
                            current_line = word
                    else:
                        wrapped_desc.append(current_line)
                        current_line = word

                if current_line:
                    wrapped_desc.append(current_line)

                # Join with newlines and proper indentation
                formatted_desc = f"{name}:\n  " + "\n  ".join(wrapped_desc)
                return formatted_desc
            else:
                return f"{name}: {description}"

        # Separate custom and default resources
        default_data_lake_content = []
        default_library_content_list = []

        # Filter out custom items from default lists
        custom_data_names = set()
        custom_software_names = set()

        if custom_data:
            custom_data_names = {item.get("name") if isinstance(item, dict) else item for item in custom_data}
        if custom_software:
            custom_software_names = {item.get("name") if isinstance(item, dict) else item for item in custom_software}

        # Separate default data lake items
        for item in data_lake_content:
            if isinstance(item, dict):
                name = item.get("name", "")
                if name not in custom_data_names:
                    default_data_lake_content.append(item)
            elif item not in custom_data_names:
                default_data_lake_content.append(item)

        # Separate default library items
        for lib in library_content_list:
            if isinstance(lib, dict):
                name = lib.get("name", "")
                if name not in custom_software_names:
                    default_library_content_list.append(lib)
            elif lib not in custom_software_names:
                default_library_content_list.append(lib)

        # Format the default data lake content
        if isinstance(default_data_lake_content, list) and all(
            isinstance(item, str) for item in default_data_lake_content
        ):
            # Simple list of strings - check if they already have descriptions
            data_lake_formatted = []
            for item in default_data_lake_content:
                # Check if the item already has a description (contains a colon)
                if ": " in item:
                    data_lake_formatted.append(item)
                else:
                    description = self.data_lake_dict.get(item, f"Data lake item: {item}")
                    data_lake_formatted.append(format_item_with_description(item, description))
        else:
            # List with descriptions
            data_lake_formatted = []
            for item in default_data_lake_content:
                if isinstance(item, dict):
                    name = item.get("name", "")
                    description = self.data_lake_dict.get(name, f"Data lake item: {name}")
                    data_lake_formatted.append(format_item_with_description(name, description))
                # Check if the item already has a description (contains a colon)
                elif isinstance(item, str) and ": " in item:
                    data_lake_formatted.append(item)
                else:
                    description = self.data_lake_dict.get(item, f"Data lake item: {item}")
                    data_lake_formatted.append(format_item_with_description(item, description))

        # Format the default library content
        if isinstance(default_library_content_list, list) and all(
            isinstance(item, str) for item in default_library_content_list
        ):
            if (
                len(default_library_content_list) > 0
                and isinstance(default_library_content_list[0], str)
                and "," not in default_library_content_list[0]
            ):
                # Simple list of strings
                libraries_formatted = []
                for lib in default_library_content_list:
                    description = self.library_content_dict.get(lib, f"Software library: {lib}")
                    libraries_formatted.append(format_item_with_description(lib, description))
            else:
                # Already formatted string
                libraries_formatted = default_library_content_list
        else:
            # List with descriptions
            libraries_formatted = []
            for lib in default_library_content_list:
                if isinstance(lib, dict):
                    name = lib.get("name", "")
                    description = self.library_content_dict.get(name, f"Software library: {name}")
                    libraries_formatted.append(format_item_with_description(name, description))
                else:
                    description = self.library_content_dict.get(lib, f"Software library: {lib}")
                    libraries_formatted.append(format_item_with_description(lib, description))

        # Format custom resources with highlighting
        custom_tools_formatted = []
        if custom_tools:
            for tool in custom_tools:
                if isinstance(tool, dict):
                    name = tool.get("name", "Unknown")
                    desc = tool.get("description", "")
                    module = tool.get("module", "custom_tools")
                    custom_tools_formatted.append(f"🔧 {name} (from {module}): {desc}")
                else:
                    custom_tools_formatted.append(f"🔧 {str(tool)}")

        custom_data_formatted = []
        if custom_data:
            for item in custom_data:
                if isinstance(item, dict):
                    name = item.get("name", "Unknown")
                    desc = item.get("description", "")
                    custom_data_formatted.append(f"📊 {format_item_with_description(name, desc)}")
                else:
                    desc = self.data_lake_dict.get(item, f"Custom data: {item}")
                    custom_data_formatted.append(f"📊 {format_item_with_description(item, desc)}")

        custom_software_formatted = []
        if custom_software:
            for item in custom_software:
                if isinstance(item, dict):
                    name = item.get("name", "Unknown")
                    desc = item.get("description", "")
                    custom_software_formatted.append(f"⚙️ {format_item_with_description(name, desc)}")
                else:
                    desc = self.library_content_dict.get(item, f"Custom software: {item}")
                    custom_software_formatted.append(f"⚙️ {format_item_with_description(item, desc)}")
        # removed In each response, you must include <nextStep> tag. YOU DO NOT NEED TO CLOSE THE <nextStep> TAG. Do not respond with messages without any tags. No empty messages.
        # Cost-awareness section
        cost_section = ""
        if cost_budget is not None and current_cost is not None:
            cost_section = f"You have a cost budget of ${cost_budget:.2f} for this task. The current cost spent is ${current_cost:.2f}.\n"
        elif cost_budget is not None:
            cost_section = f"You have a cost budget of ${cost_budget:.2f} for this task.\n"
        elif current_cost is not None:
            cost_section = f"The current cost spent so far is ${current_cost:.2f}.\n"

        # Base prompt
        prompt_modifier = f"""
You are a helpful, fully autonomous biomedical assistant assigned with the task of providing a plan towards problem-solving.
You are also the Plan agent of the PLAN-LEARN-EXECUTE-ASSESS-SHARE (or PLEAS) agentic framework.

{cost_section}

Try to keep the total cost within the budget if one is provided.

When you wish to update the plan after ASSESS phase feedback, you MUST extract the previous plan from that phase output and revise it accordingly. CRITICAL: You must absolutely force yourself to remember and include the HIGH-LEVEL, full-scope original plan. NEVER delete future high-level steps. Only insert new micro-steps as sub-tasks for the current step, keeping the rest of the overarching global plan intact.

PLEAS Framework for Agents:

P — Plan
- Break down the user query into sub-tasks
- Outline solution path and required tools/data

L — Learn
- Gather info from sources (APIs, data, papers)
- Summarize and refine plan with context

E — Execute
- Carry out the plan
- Run code, call APIs, perform computations

A — Assess
- Evaluate outputs for correctness and relevance
- Decide whether to loop back or move forward

S — Share
- Deliver final, user-facing response
- Present results clearly and in usable form


[PHASE: PLAN]
"""
        
        # Conditionally add scientific mindset section
        if show_scientific_mindset:
            prompt_modifier += """
═══════════════════════════════════════════════════════════════════════════════
🔬 SCIENTIFIC REASONING MINDSET 🔬
═══════════════════════════════════════════════════════════════════════════════

**YOU ARE A COMPUTATIONAL SCIENTIST, NOT JUST AN INFORMATION RETRIEVER.**

For any question that is not immediately and trivially answerable, you must adopt an EXPERIMENTAL, EVIDENCE-BASED approach:

**The Scientific Method for Planning:**

1. **QUESTION ANALYSIS** → Identify what claims need to be tested or validated
2. **HYPOTHESIS FORMATION** → Generate testable predictions (including for multiple-choice options)
3. **EXPERIMENTAL DESIGN** → Design computational experiments using available data and tools
4. **EMPIRICAL VALIDATION** → Plan to collect quantitative evidence through code execution
5. **EVIDENCE-BASED CONCLUSION** → Ensure answers are grounded in computational results

**PRIORITIZE (Experimental Approaches):**
✓ Analyzing real datasets from the data lake (GTEx, COSMIC, STRING, DisGeNET, MSigDB, etc.)
✓ Running simulations to test biological mechanisms
✓ Computing quantitative metrics (correlations, p-values, enrichment scores, network statistics)
✓ Performing statistical analyses to discriminate between hypotheses
✓ Generating empirical evidence through code execution
✓ Validating predictions against observed data
✓ Running parameter sweeps or sensitivity analyses

**DEPRIORITIZE (Pure Information Retrieval):**
✗ Relying solely on literature search without computational follow-up
✗ Making claims based only on textbook definitions
✗ Selecting answers through logical reasoning alone without empirical testing
✗ Retrieving information without validating it computationally
✗ Accepting assertions without generating supporting evidence

**When to Use Each Approach:**
- **Trivial/Definitional**: "What does CRISPR stand for?" → Direct answer OK
- **Factual Lookup**: "What is the function of TP53?" → Literature search + database validation
- **Mechanistic/Quantitative**: "Which mechanism explains X?" → DESIGN EXPERIMENTS to test each mechanism
- **Comparative**: "Which gene is most associated with Y?" → ANALYZE DATA to rank candidates
- **Hypothesis Testing**: "Does A correlate with B?" → COMPUTE correlations from real data
- **Multiple Choice**: ANY MCQ → Treat each choice as a hypothesis and TEST empirically

**Example Transformation:**

❌ BAD PLAN (information retrieval):
1. [ ] Search PubMed for papers about gene regulation mechanisms
2. [ ] Summarize the most cited mechanism
3. [ ] Select the answer that matches the literature

✓ GOOD PLAN (scientific experimentation):
1. [ ] Experimental Design - For each proposed mechanism, identify testable predictions
2. [ ] Data Analysis - Load GTEx expression data + STRING network to test predictions
3. [ ] Hypothesis Testing - Compute metrics for each mechanism (e.g., co-expression patterns, network topology, regulatory motifs)
4. [ ] Evidence Synthesis - Rank mechanisms by quantitative support from data analysis
5. [ ] Conclusion - Select answer with strongest empirical evidence

**Your Planning Directive:**
Unless the question is completely straightforward, your plan MUST include steps that:
- Load and analyze relevant datasets
- Compute quantitative metrics
- Generate empirical evidence through code
- Test hypotheses with real data
- Provide numerical justification for conclusions

Think: "How would I design an experiment to answer this if I were in a lab?" Then translate that to computational/data analysis steps.

═══════════════════════════════════════════════════════════════════════════════

"""
        
        prompt_modifier += """
To problem solve, you will need to create a high-level and well orchestrated plan that incorporates the use of a variety of tool functions, data, and softwares to assist you throughout the process.
You are the planning agent. Your responsibility is to carefully analyze the user’s query, break it down into structured goals, and design a step-by-step plan for how to solve it. You do not execute or code yet — only plan.  

Given a task, make a plan first. The plan should be a numbered list of steps that you will take to solve the task. Be specific and detailed.

CRITICAL FORMATTING RULE: You must maintain a "=== GLOBAL PLAN ===" that persists throughout the entire run. If execution fails, DO NOT replace the plan with a short debugging checklist; instead, add indented sub-tasks.

Format your plan as a checklist with empty checkboxes like this:

=== GLOBAL PLAN ===
1. [ ] Phase 1: High-level goal (e.g. Data Gathering)
   - [ ] Specific sub-task
2. [ ] Phase 2: High-level goal (e.g. Model Training)
3. [ ] Phase 3: High-level goal (e.g. Evaluation)
===================

Follow the plan step by step. After completing each step, update the checklist by replacing the empty checkbox:
=== GLOBAL PLAN ===
1. [✓] Phase 1: High-level goal (completed)
2. [ ] Phase 2: High-level goal (current)
   - [ ] New active sub-task
3. [ ] Phase 3: High-level goal
===================

If a step fails or needs modification, mark it with an X, explain why, AND KEEP ALL FUTURE PHASES IN THE LIST:
=== GLOBAL PLAN ===
1. [✗] Phase 1: High-level goal (failed due to missing file)
   - [ ] MICRO-STEP: Write script to download missing file
2. [ ] Phase 2: High-level goal
3. [ ] Phase 3: High-level goal
===================

Rules:
- Do not retrieve, run code, or output results.  
- Do not include any `<execute>` blocks.  
- Think like a project manager or strategist: what needs to be done, in what order, and with what resources.
- YOU CANNOT DIRECTLY INTERACT WITH THE USER AND ASK FOR FEEDBACK OR FUTURE DIRECTIONS.

EVIDENCE AND RIGOR RULES — STRICTLY ENFORCED:
- DO NOT INVENT data, results, file paths, package versions, dataset properties, or any facts. Only state things that are confirmed by actual execution observations or explicitly provided context.
- DO NOT TRUST noisy or unverified intermediate artifacts. If a prior EXECUTE step produced ambiguous, partial, or suspicious output (e.g. mock data passed off as real, empty files, error-wrapped results), flag this in the plan and require explicit verification before treating the artifact as a valid input to the next step.
- DO NOT make unsupported assumptions. Every assumption about the environment, data, or tools must be labeled as an assumption and must include a concrete verification step in the plan. If an assumption cannot be verified, route to EXECUTE to check it before proceeding.
- If prior results are unreliable (e.g. mock data used as stand-in, file not confirmed to exist, metric computed on dummy data), the plan MUST explicitly note this and add a step to re-run on real data.

Always show the updated plan after assess passes back to you so the user can track progress.

Provide the current step that you are working on in the plan so that learn and execute agents know what to do next.

"If you see a human message starting with [OVERVIEW→PLAN], revise the plan to address each critique, mark failed steps with ✗, and add the new steps.

"""

        # Add self-critic instructions if needed
        if self_critic:
            prompt_modifier += """
You may or may not receive feedbacks from human. If so, address the feedbacks by following the same procedure of multiple rounds of thinking, execution, and then coming up with a new solution.
"""

        # Add custom resources section first (highlighted)
        has_custom_resources = any([custom_tools_formatted, custom_data_formatted, custom_software_formatted])

        if has_custom_resources:
            prompt_modifier += """

PRIORITY CUSTOM RESOURCES
===============================
IMPORTANT: The following custom resources have been specifically added for your use.
    PRIORITIZE using these resources as they are directly relevant to your task.
    Always consider these FIRST and in the meantime using default resources.

"""

            if custom_tools_formatted:
                prompt_modifier += """
CUSTOM TOOLS (USE THESE FIRST):
{custom_tools}

"""

            if custom_data_formatted:
                prompt_modifier += """
CUSTOM DATA (PRIORITIZE THESE DATASETS):
{custom_data}

"""

            if custom_software_formatted:
                prompt_modifier += """
⚙️ CUSTOM SOFTWARE (USE THESE LIBRARIES):
{custom_software}

"""

            prompt_modifier += """===============================
"""

        # Add environment resources
        prompt_modifier += """

Environment Resources:

- Function Dictionary:
{function_intro}
---
{tool_desc}
---

{import_instruction}

- Biological data lake
You can access a biological data lake at the following path: {data_lake_path}.
{data_lake_intro}
Each item is listed with its description to help you understand its contents.
----
{data_lake_content}
----

- Software Library:
{library_intro}
Each library is listed with its description to help you understand its functionality.
----
{library_content_formatted}
----

**CRITICAL: Create 3-5 HIGH-LEVEL BATCHED STEPS, NOT 15-20 MICRO-STEPS**

GOOD Example (3 batched steps):
1. [ ] Data Gathering Phase - Query all databases in parallel (PubMed, KEGG, STRING, DisGeNET), load MSigDB, analyze GTEx
2. [ ] Analysis Phase - Score candidates, rank by evidence, select top 32 genes across functional categories
3. [ ] Deliverables Phase - Design sgRNAs, create visualizations, generate final report

BAD Example (too granular, causes 15+ iterations):
1. [ ] Query PubMed for T cell exhaustion
2. [ ] Save PubMed results to CSV
3. [ ] Query KEGG database
4. [ ] Process KEGG results
... (continues for 20 steps)

Guidelines:
- Batch related operations into single steps (all database queries together)
- Group by logical phase (gather → analyze → report)
- Each step should take 1-2 EXECUTE iterations max
- Aim for completion in 3-5 total iterations
        """

        # Set appropriate text based on whether this is initial configuration or after retrieval
        if is_retrieval:
            function_intro = "Based on your query, I've identified the following most relevant functions that you can use in your code:"
            data_lake_intro = "Based on your query, I've identified the following most relevant datasets:"
            library_intro = (
                "Based on your query, I've identified the following most relevant libraries that you can use:"
            )
            import_instruction = "IMPORTANT: When using any function, you MUST first import it from its module. For example:\nfrom [module_name] import [function_name]"
        else:
            function_intro = "In your code, you will need to import the function location using the following dictionary of functions:"
            data_lake_intro = "You can write code to understand the data, process and utilize it for the task. Here is the list of datasets:"
            library_intro = "The environment supports a list of libraries that can be directly used. Do not forget the import statement:"
            import_instruction = ""

        # Format the content consistently for both initial and retrieval cases
        library_content_formatted = "\n".join(libraries_formatted)
        data_lake_content_formatted = "\n".join(data_lake_formatted)

        # Format the prompt with the appropriate values
        format_dict = {
            "function_intro": function_intro,
            "tool_desc": textify_api_dict(tool_desc) if isinstance(tool_desc, dict) else tool_desc,
            "import_instruction": import_instruction,
            "data_lake_path": self.path + "/data_lake",
            "data_lake_intro": data_lake_intro,
            "data_lake_content": data_lake_content_formatted,
            "library_intro": library_intro,
            "library_content_formatted": library_content_formatted,
        }

        # Add custom resources to format dict (always add them, even if empty)
        format_dict["custom_tools"] = "\n".join(custom_tools_formatted) if custom_tools_formatted else ""
        format_dict["custom_data"] = "\n".join(custom_data_formatted) if custom_data_formatted else ""
        format_dict["custom_software"] = "\n".join(custom_software_formatted) if custom_software_formatted else ""

        run_dir = getattr(self, "run_dir", self.path)
        format_dict["run_dir"] = run_dir
        format_dict["fig_dir"] = getattr(self, "run_figures_dir", os.path.join(run_dir, "figures"))
        format_dict["arts_dir"] = getattr(self, "run_artifacts_dir", os.path.join(run_dir, "artifacts"))

        formatted_prompt = prompt_modifier.format(**format_dict)

        return formatted_prompt
    
    
    def _learn_system_prompt(
        self,
        tool_desc,
        data_lake_content,
        library_content_list,
        self_critic=False,
        is_retrieval=False,
        custom_tools=None,
        custom_data=None,
        custom_software=None,
    ):
        """Generate the system prompt based on the provided resources.

        Args:
            tool_desc: Dictionary of tool descriptions
            data_lake_content: List of data lake items
            library_content_list: List of libraries
            self_critic: Whether to include self-critic instructions
            is_retrieval: Whether this is for retrieval (True) or initial configuration (False)
            custom_tools: List of custom tools to highlight
            custom_data: List of custom data items to highlight
            custom_software: List of custom software items to highlight

        Returns:
            The generated system prompt

        """

        def format_item_with_description(name, description):
            """Format an item with its description in a readable way."""
            # Handle None or empty descriptions
            if not description:
                description = f"Data lake item: {name}"

            # Check if the item is already formatted (contains a colon)
            if isinstance(name, str) and ": " in name:
                return name

            # Wrap long descriptions to make them more readable
            max_line_length = 80
            if len(description) > max_line_length:
                # Simple wrapping for long descriptions
                wrapped_desc = []
                words = description.split()
                current_line = ""

                for word in words:
                    if len(current_line) + len(word) + 1 <= max_line_length:
                        if current_line:
                            current_line += " " + word
                        else:
                            current_line = word
                    else:
                        wrapped_desc.append(current_line)
                        current_line = word

                if current_line:
                    wrapped_desc.append(current_line)

                # Join with newlines and proper indentation
                formatted_desc = f"{name}:\n  " + "\n  ".join(wrapped_desc)
                return formatted_desc
            else:
                return f"{name}: {description}"

        # Separate custom and default resources
        default_data_lake_content = []
        default_library_content_list = []

        # Filter out custom items from default lists
        custom_data_names = set()
        custom_software_names = set()

        if custom_data:
            custom_data_names = {item.get("name") if isinstance(item, dict) else item for item in custom_data}
        if custom_software:
            custom_software_names = {item.get("name") if isinstance(item, dict) else item for item in custom_software}

        # Separate default data lake items
        for item in data_lake_content:
            if isinstance(item, dict):
                name = item.get("name", "")
                if name not in custom_data_names:
                    default_data_lake_content.append(item)
            elif item not in custom_data_names:
                default_data_lake_content.append(item)

        # Separate default library items
        for lib in library_content_list:
            if isinstance(lib, dict):
                name = lib.get("name", "")
                if name not in custom_software_names:
                    default_library_content_list.append(lib)
            elif lib not in custom_software_names:
                default_library_content_list.append(lib)

        # Format the default data lake content
        if isinstance(default_data_lake_content, list) and all(
            isinstance(item, str) for item in default_data_lake_content
        ):
            # Simple list of strings - check if they already have descriptions
            data_lake_formatted = []
            for item in default_data_lake_content:
                # Check if the item already has a description (contains a colon)
                if ": " in item:
                    data_lake_formatted.append(item)
                else:
                    description = self.data_lake_dict.get(item, f"Data lake item: {item}")
                    data_lake_formatted.append(format_item_with_description(item, description))
        else:
            # List with descriptions
            data_lake_formatted = []
            for item in default_data_lake_content:
                if isinstance(item, dict):
                    name = item.get("name", "")
                    description = self.data_lake_dict.get(name, f"Data lake item: {name}")
                    data_lake_formatted.append(format_item_with_description(name, description))
                # Check if the item already has a description (contains a colon)
                elif isinstance(item, str) and ": " in item:
                    data_lake_formatted.append(item)
                else:
                    description = self.data_lake_dict.get(item, f"Data lake item: {item}")
                    data_lake_formatted.append(format_item_with_description(item, description))

        # Format the default library content
        if isinstance(default_library_content_list, list) and all(
            isinstance(item, str) for item in default_library_content_list
        ):
            if (
                len(default_library_content_list) > 0
                and isinstance(default_library_content_list[0], str)
                and "," not in default_library_content_list[0]
            ):
                # Simple list of strings
                libraries_formatted = []
                for lib in default_library_content_list:
                    description = self.library_content_dict.get(lib, f"Software library: {lib}")
                    libraries_formatted.append(format_item_with_description(lib, description))
            else:
                # Already formatted string
                libraries_formatted = default_library_content_list
        else:
            # List with descriptions
            libraries_formatted = []
            for lib in default_library_content_list:
                if isinstance(lib, dict):
                    name = lib.get("name", "")
                    description = self.library_content_dict.get(name, f"Software library: {name}")
                    libraries_formatted.append(format_item_with_description(name, description))
                else:
                    description = self.library_content_dict.get(lib, f"Software library: {lib}")
                    libraries_formatted.append(format_item_with_description(lib, description))

        # Format custom resources with highlighting
        custom_tools_formatted = []
        if custom_tools:
            for tool in custom_tools:
                if isinstance(tool, dict):
                    name = tool.get("name", "Unknown")
                    desc = tool.get("description", "")
                    module = tool.get("module", "custom_tools")
                    custom_tools_formatted.append(f"🔧 {name} (from {module}): {desc}")
                else:
                    custom_tools_formatted.append(f"🔧 {str(tool)}")

        custom_data_formatted = []
        if custom_data:
            for item in custom_data:
                if isinstance(item, dict):
                    name = item.get("name", "Unknown")
                    desc = item.get("description", "")
                    custom_data_formatted.append(f"📊 {format_item_with_description(name, desc)}")
                else:
                    desc = self.data_lake_dict.get(item, f"Custom data: {item}")
                    custom_data_formatted.append(f"📊 {format_item_with_description(item, desc)}")

        custom_software_formatted = []
        if custom_software:
            for item in custom_software:
                if isinstance(item, dict):
                    name = item.get("name", "Unknown")
                    desc = item.get("description", "")
                    custom_software_formatted.append(f"⚙️ {format_item_with_description(name, desc)}")
                else:
                    desc = self.library_content_dict.get(item, f"Custom software: {item}")
                    custom_software_formatted.append(f"⚙️ {format_item_with_description(item, desc)}")

        # Base prompt
        prompt_modifier = """
You are a helpful, fully autonomous biomedical assistant assigned with the task of identifying the necessary resources for effective problem-solving.
You are the LEARN agent of the PLAN-LEARN-EXECUTE-ASSESS-SHARE (or PLEAS) agentic framework.

[PHASE: LEARN]
Your job is to identify and select the resources needed to execute the plan. You DO NOT execute code or solve the problem yet — that happens in the EXECUTE phase.

Your Responsibilities:
1. Review the current step in the plan and identify what resources are needed
2. Select relevant tools/functions from the available function dictionary
3. Identify relevant datasets from the data lake
4. Identify relevant software libraries needed
5. Document why each resource is needed and how it will be used
6. Provide a clear summary of selected resources for the EXECUTE phase

Output Format:
Provide your response as a structured resource selection summary:

**Current Step Analysis:**
[Describe which step of the plan you're addressing]

**Knowledge Gaps to Fill:**
[List what information or capabilities are needed]

**Selected Resources:**

1. **Functions/Tools:**
   - [Function name from tool dictionary]: [Why this is needed]
   - [Function name]: [Why this is needed]
   
2. **Datasets:**
   - [Dataset name from data lake]: [What information it provides]
   - [Dataset name]: [What information it provides]

3. **Software Libraries:**
   - [Library name]: [What capabilities it provides]
   - [Library name]: [What capabilities it provides]

**Resource Usage Plan:**
[Brief description of how these resources will be used in the EXECUTE phase]

Rules:
- DO NOT write code or use <execute> tags — that is the EXECUTE phase's job
- DO NOT call functions or run analyses — only identify which ones are needed  
- DO NOT invent resources — only select from the provided lists below
- DO NOT try to solve the entire problem — focus on one step at a time
- Focus on selecting the RIGHT resources, not all possible resources
- Be specific about WHY each resource is needed
- YOU CANNOT DIRECTLY INTERACT WITH THE USER AND ASK FOR FEEDBACK OR FUTURE DIRECTIONS

EVIDENCE AND RIGOR RULES — STRICTLY ENFORCED:
- DO NOT INVENT facts about datasets, packages, tools, or results. Only reference things that appear in the provided resource lists, execution observations, or explicitly confirmed prior state.
- DO NOT ASSUME a dataset has certain columns, a package is installed, or a file exists unless it was confirmed in a prior EXECUTE observation. All such assumptions must be stated explicitly as unverified and must include a check step.
- DO NOT fabricate expected outputs, metric ranges, or tool behaviors. If you are uncertain whether a tool or library is available or behaves a certain way, note it as a knowledge gap to be verified in EXECUTE — do not guess.
- If prior EXECUTE outputs were produced on mock/synthetic data rather than real data, explicitly flag this and list it as a gap that must be resolved before proceeding.

"""

        # Add self-critic instructions if needed
        if self_critic:
            prompt_modifier += """
You may or may not receive feedbacks from human. If so, address the feedbacks by following the same procedure of multiple rounds of thinking, execution, and then coming up with a new solution.
"""

        # Add custom resources section first (highlighted)
        has_custom_resources = any([custom_tools_formatted, custom_data_formatted, custom_software_formatted])

        if has_custom_resources:
            prompt_modifier += """

PRIORITY CUSTOM RESOURCES
===============================
IMPORTANT: The following custom resources have been specifically added for your use.
    PRIORITIZE using these resources as they are directly relevant to your task.
    Always consider these FIRST and in the meantime using default resources.

"""

            if custom_tools_formatted:
                prompt_modifier += """
CUSTOM TOOLS (USE THESE FIRST):
{custom_tools}

"""

            if custom_data_formatted:
                prompt_modifier += """
CUSTOM DATA (PRIORITIZE THESE DATASETS):
{custom_data}

"""

            if custom_software_formatted:
                prompt_modifier += """
⚙️ CUSTOM SOFTWARE (USE THESE LIBRARIES):
{custom_software}

"""

            prompt_modifier += """===============================
"""

        # Add environment resources
        prompt_modifier += """

Environment Resources:

- Function Dictionary:
{function_intro}
---
{tool_desc}
---

{import_instruction}

- Biological data lake
You can access a biological data lake at the following path: {data_lake_path}.
{data_lake_intro}
Each item is listed with its description to help you understand its contents.
----
{data_lake_content}
----

- Software Library:
{library_intro}
Each library is listed with its description to help you understand its functionality.
----
{library_content_formatted}
----

- Note on using R packages and Bash scripts:
  - R packages: Use subprocess.run(['Rscript', '-e', 'your R code here']) in Python, or use the #!R marker in your execute block.
  - Bash scripts and commands: Use the #!BASH marker in your execute block for both simple commands and complex shell scripts with variables, loops, conditionals, etc.
        """

        # Set appropriate text based on whether this is initial configuration or after retrieval
        if is_retrieval:
            function_intro = "Based on your query, I've identified the following most relevant functions that you can use in your code:"
            data_lake_intro = "Based on your query, I've identified the following most relevant datasets:"
            library_intro = (
                "Based on your query, I've identified the following most relevant libraries that you can use:"
            )
            import_instruction = "IMPORTANT: When using any function, you MUST first import it from its module. For example:\nfrom [module_name] import [function_name]"
        else:
            function_intro = "In your code, you will need to import the function location using the following dictionary of functions:"
            data_lake_intro = "You can write code to understand the data, process and utilize it for the task. Here is the list of datasets:"
            library_intro = "The environment supports a list of libraries that can be directly used. Do not forget the import statement:"
            import_instruction = ""

        # Format the content consistently for both initial and retrieval cases
        library_content_formatted = "\n".join(libraries_formatted)
        data_lake_content_formatted = "\n".join(data_lake_formatted)

        # Format the prompt with the appropriate values
        format_dict = {
            "function_intro": function_intro,
            "tool_desc": textify_api_dict(tool_desc) if isinstance(tool_desc, dict) else tool_desc,
            "import_instruction": import_instruction,
            "data_lake_path": self.path + "/data_lake",
            "data_lake_intro": data_lake_intro,
            "data_lake_content": data_lake_content_formatted,
            "library_intro": library_intro,
            "library_content_formatted": library_content_formatted,
        }

        # Add custom resources to format dict (always add them, even if empty)
        format_dict["custom_tools"] = "\n".join(custom_tools_formatted) if custom_tools_formatted else ""
        format_dict["custom_data"] = "\n".join(custom_data_formatted) if custom_data_formatted else ""
        format_dict["custom_software"] = "\n".join(custom_software_formatted) if custom_software_formatted else ""

        run_dir = getattr(self, "run_dir", self.path)
        format_dict["run_dir"] = run_dir
        format_dict["fig_dir"] = getattr(self, "run_figures_dir", os.path.join(run_dir, "figures"))
        format_dict["arts_dir"] = getattr(self, "run_artifacts_dir", os.path.join(run_dir, "artifacts"))

        formatted_prompt = prompt_modifier.format(**format_dict)

        return formatted_prompt
    
    
    
    
    
    def _execute_system_prompt(
        self,
        tool_desc,
        data_lake_content,
        library_content_list,
        self_critic=False,
        is_retrieval=False,
        custom_tools=None,
        custom_data=None,
        custom_software=None,
    ):
        """Generate the system prompt based on the provided resources.

        Args:
            tool_desc: Dictionary of tool descriptions
            data_lake_content: List of data lake items
            library_content_list: List of libraries
            self_critic: Whether to include self-critic instructions
            is_retrieval: Whether this is for retrieval (True) or initial configuration (False)
            custom_tools: List of custom tools to highlight
            custom_data: List of custom data items to highlight
            custom_software: List of custom software items to highlight

        Returns:
            The generated system prompt

        """

        def format_item_with_description(name, description):
            """Format an item with its description in a readable way."""
            # Handle None or empty descriptions
            if not description:
                description = f"Data lake item: {name}"

            # Check if the item is already formatted (contains a colon)
            if isinstance(name, str) and ": " in name:
                return name

            # Wrap long descriptions to make them more readable
            max_line_length = 80
            if len(description) > max_line_length:
                # Simple wrapping for long descriptions
                wrapped_desc = []
                words = description.split()
                current_line = ""

                for word in words:
                    if len(current_line) + len(word) + 1 <= max_line_length:
                        if current_line:
                            current_line += " " + word
                        else:
                            current_line = word
                    else:
                        wrapped_desc.append(current_line)
                        current_line = word

                if current_line:
                    wrapped_desc.append(current_line)

                # Join with newlines and proper indentation
                formatted_desc = f"{name}:\n  " + "\n  ".join(wrapped_desc)
                return formatted_desc
            else:
                return f"{name}: {description}"

        # Separate custom and default resources
        default_data_lake_content = []
        default_library_content_list = []

        # Filter out custom items from default lists
        custom_data_names = set()
        custom_software_names = set()

        if custom_data:
            custom_data_names = {item.get("name") if isinstance(item, dict) else item for item in custom_data}
        if custom_software:
            custom_software_names = {item.get("name") if isinstance(item, dict) else item for item in custom_software}

        # Separate default data lake items
        for item in data_lake_content:
            if isinstance(item, dict):
                name = item.get("name", "")
                if name not in custom_data_names:
                    default_data_lake_content.append(item)
            elif item not in custom_data_names:
                default_data_lake_content.append(item)

        # Separate default library items
        for lib in library_content_list:
            if isinstance(lib, dict):
                name = lib.get("name", "")
                if name not in custom_software_names:
                    default_library_content_list.append(lib)
            elif lib not in custom_software_names:
                default_library_content_list.append(lib)

        # Format the default data lake content
        if isinstance(default_data_lake_content, list) and all(
            isinstance(item, str) for item in default_data_lake_content
        ):
            # Simple list of strings - check if they already have descriptions
            data_lake_formatted = []
            for item in default_data_lake_content:
                # Check if the item already has a description (contains a colon)
                if ": " in item:
                    data_lake_formatted.append(item)
                else:
                    description = self.data_lake_dict.get(item, f"Data lake item: {item}")
                    data_lake_formatted.append(format_item_with_description(item, description))
        else:
            # List with descriptions
            data_lake_formatted = []
            for item in default_data_lake_content:
                if isinstance(item, dict):
                    name = item.get("name", "")
                    description = self.data_lake_dict.get(name, f"Data lake item: {name}")
                    data_lake_formatted.append(format_item_with_description(name, description))
                # Check if the item already has a description (contains a colon)
                elif isinstance(item, str) and ": " in item:
                    data_lake_formatted.append(item)
                else:
                    description = self.data_lake_dict.get(item, f"Data lake item: {item}")
                    data_lake_formatted.append(format_item_with_description(item, description))

        # Format the default library content
        if isinstance(default_library_content_list, list) and all(
            isinstance(item, str) for item in default_library_content_list
        ):
            if (
                len(default_library_content_list) > 0
                and isinstance(default_library_content_list[0], str)
                and "," not in default_library_content_list[0]
            ):
                # Simple list of strings
                libraries_formatted = []
                for lib in default_library_content_list:
                    description = self.library_content_dict.get(lib, f"Software library: {lib}")
                    libraries_formatted.append(format_item_with_description(lib, description))
            else:
                # Already formatted string
                libraries_formatted = default_library_content_list
        else:
            # List with descriptions
            libraries_formatted = []
            for lib in default_library_content_list:
                if isinstance(lib, dict):
                    name = lib.get("name", "")
                    description = self.library_content_dict.get(name, f"Software library: {name}")
                    libraries_formatted.append(format_item_with_description(name, description))
                else:
                    description = self.library_content_dict.get(lib, f"Software library: {lib}")
                    libraries_formatted.append(format_item_with_description(lib, description))

        # Format custom resources with highlighting
        custom_tools_formatted = []
        if custom_tools:
            for tool in custom_tools:
                if isinstance(tool, dict):
                    name = tool.get("name", "Unknown")
                    desc = tool.get("description", "")
                    module = tool.get("module", "custom_tools")
                    custom_tools_formatted.append(f"🔧 {name} (from {module}): {desc}")
                else:
                    custom_tools_formatted.append(f"🔧 {str(tool)}")

        custom_data_formatted = []
        if custom_data:
            for item in custom_data:
                if isinstance(item, dict):
                    name = item.get("name", "Unknown")
                    desc = item.get("description", "")
                    custom_data_formatted.append(f"📊 {format_item_with_description(name, desc)}")
                else:
                    desc = self.data_lake_dict.get(item, f"Custom data: {item}")
                    custom_data_formatted.append(f"📊 {format_item_with_description(item, desc)}")

        custom_software_formatted = []
        if custom_software:
            for item in custom_software:
                if isinstance(item, dict):
                    name = item.get("name", "Unknown")
                    desc = item.get("description", "")
                    custom_software_formatted.append(f"⚙️ {format_item_with_description(name, desc)}")
                else:
                    desc = self.library_content_dict.get(item, f"Custom software: {item}")
                    custom_software_formatted.append(f"⚙️ {format_item_with_description(item, desc)}")

        # Base prompt
        prompt_modifier = """
You are a helpful, fully autonomous biomedical assistant assigned with the task of writing the proper code and performing effective problem-solving.
You are also the Execute agent of the PLAN-LEARN-EXECUTE-ASSESS-SHARE (or PLEAS) agentic framework.

[PHASE: EXECUTE]
To problem solve, you will need to create an execute code that incorporates the use of a variety of tool functions, data, and softwares.

You are the executing agent. Your job is to implement the plan and apply the knowledge gathered in the LEARN phase. This is where you write and run code, calculate, or directly generate outputs.

If you were routed here after the ASSESS phase, carefully review the feedback and adjust your execution accordingly. Look for the previous observation of the code execution and search for the previous code written in EXECUTE phase and properly debug the code or modify it to address the issues raised. This should be located in the previous state.

Instructions:
1. Re-check the PLAN and LEARN summaries to ensure alignment.  
2. Write clear, efficient code or step-by-step calculations.  
3. Use the appropriate tools and libraries.  
4. Only produce `<execute>...</execute>` blocks for code execution — no explanatory text inside them.  
5. After execution, output structured results that can be assessed.  
6. If an error occurs, explain it briefly and suggest corrections.
7. DO NOT ASK THE USER FOR FEEDBACK. You must solve the task autonomously in the best interest of rigor.
8. If debugging existing code, do NOT rewrite the entire script from scratch. Write small script patches, or print out only the problematic sections, leaving your previously generated code files intact.

❌ WRONG - Contains narrative inside execute block:
<execute>
**Error Analysis:**
1. The error was X
2. I will fix it by Y

import os
print("hello")
</execute>

❌ WRONG - Has markdown and explanations:
<execute>
Here's the corrected code:
```python
import os
print("hello")
```
</execute>

✅ CORRECT - Explanation outside, pure code inside:
I'll fix the import error by adding the os module.
<execute>
import os
print("hello")
</execute>

✅ CORRECT - Just code, nothing else:
<execute>
import os
import json
data = {{"key": "value"}}
print(json.dumps(data))
</execute>

Rules:
- Only code and computation happen here — no extra commentary outside results.  
- Do not re-plan or re-learn; only act on what is prepared.  
- Precision and correctness are the priority.
- YOU CANNOT DIRECTLY INTERACT WITH THE USER AND ASK FOR FEEDBACK OR FUTURE DIRECTIONS.
- Try to solve the plan one task at a time. Do not try to solve everything at once.

**OUTPUT DIRECTORY RULES — ALWAYS FOLLOW THESE:**

Every run has exactly THREE authorized output subdirectories. Access them via env vars:
  import os
  RUN_DIR   = os.environ.get("BIOPLEASE_RUN_DIR", ".")
  ARTIFACTS = os.environ.get("BIOPLEASE_ARTIFACTS_DIR", os.path.join(RUN_DIR, "artifacts"))
  FIGURES   = os.environ.get("BIOPLEASE_FIGURES_DIR",   os.path.join(RUN_DIR, "figures"))
  CODE      = os.environ.get("BIOPLEASE_CODE_DIR",      os.path.join(RUN_DIR, "code"))

The ONLY valid destinations for any output file are: artifacts/, figures/, code/

Rules:
1. ALL data outputs (CSV, TSV, JSON, TXT, Parquet, PKL, HDF5, etc.) MUST be saved to ARTIFACTS.
   - Always: os.makedirs(ARTIFACTS, exist_ok=True)
   - Always print the full path: print(f"Saved: {{path}}")
2. ALL figures/plots/images MUST be saved to FIGURES.
   - Always: os.makedirs(FIGURES, exist_ok=True)
3. Code files are harvested automatically into CODE — do NOT manually save .py/.R files elsewhere.
4. NEVER save any file directly into RUN_DIR or its root. NEVER use cwd, /tmp, or ad-hoc paths.
   The main run directory MUST stay clean — no stray files.
5. To find a file YOU previously generated, search ARTIFACTS or FIGURES:
   import glob
   matches = glob.glob(os.path.join(ARTIFACTS, "**", "*keyword*"), recursive=True)
   Do NOT assume a file is in RUN_DIR or cwd — it will not be there.

Within the execute step, you should first provide your thinking and reasoning given the conversation history.
After that, you have two options:

1) Interact with a programming environment and receive the corresponding output within <observe></observe>. Your code should be enclosed using "<execute>" tag, for example: <execute> print("Hello World!") </execute>. IMPORTANT: You must end the code block with </execute> tag.
   - For Python code (default): <execute> print("Hello World!") </execute>
   - For R code: <execute> #!R\nlibrary(ggplot2)\nprint("Hello from R") </execute>
   - For Bash scripts and commands: <execute> #!BASH\necho "Hello from Bash"\nls -la </execute>
   - For CLI softwares, use Bash scripts.

2) When you think it is ready, directly provide a solution that adheres to the required format for the given task to the user. Your solution should be enclosed using "<solution>" tag, for example: The answer is <solution> A </solution>. IMPORTANT: You must end the solution block with </solution> tag.

You have many chances to interact with the environment to receive the observation. So you can decompose your code into multiple steps.
Don't overcomplicate the code. Keep it simple and easy to understand.
When writing the code, please print out the steps and results in a clear and concise manner, like a research log.
When calling the existing python functions in the function dictionary, YOU MUST SAVE THE OUTPUT and PRINT OUT the result.
For example, result = understand_scRNA(XXX) print(result)
Otherwise the system will not be able to know what has been done.

For R code, use the #!R marker at the beginning of your code block to indicate it's R code.
For Bash scripts and commands, use the #!BASH marker at the beginning of your code block. This allows for both simple commands and multi-line scripts with variables, loops, conditionals, loops, and other Bash features.

If you see a human message starting with [OVERVIEW→EXECUTE], you MUST implement those code/logging changes now. 
Emit exactly one <execute>...</execute> block and then stop.

IMPORTANT FILE SYSTEM RULES:
1) EXPLORE BEFORE READING: Before you try to directly read or get a file, you MUST first list the contents of the directory containing that file to see what files are in that directory (e.g. using `ls -la` in Bash or `os.listdir()` in Python).
2) SAVE LOCATIONS: You MUST save any output files to your active run directory! Do NOT save them in the current working directory unless it is the run directory.
   - Run Directory Base: {run_dir}
   - Figures should be saved to: {fig_dir}
   - Code and Artifacts should be saved to: {arts_dir}
3) TRACKING: You MUST print out a tracking list of all newly created files (artifacts, figures, code) and their absolute file locations so they are recorded in the state logs.

2. Look at the TOOLS TO USE (from LEARN phase)
3. Write ONE code block to start that task
4. Don't overthink - just write simple, working code

"""

        # Add self-critic instructions if needed
        if self_critic:
            prompt_modifier += """
You may or may not receive feedbacks from human. If so, address the feedbacks by following the same procedure of multiple rounds of thinking, execution, and then coming up with a new solution.
"""

        # Add custom resources section first (highlighted)
        has_custom_resources = any([custom_tools_formatted, custom_data_formatted, custom_software_formatted])

        if has_custom_resources:
            prompt_modifier += """

PRIORITY CUSTOM RESOURCES
===============================
IMPORTANT: The following custom resources have been specifically added for your use.
    PRIORITIZE using these resources as they are directly relevant to your task.
    Always consider these FIRST and in the meantime using default resources.

"""

            if custom_tools_formatted:
                prompt_modifier += """
CUSTOM TOOLS (USE THESE FIRST):
{custom_tools}

"""

            if custom_data_formatted:
                prompt_modifier += """
CUSTOM DATA (PRIORITIZE THESE DATASETS):
{custom_data}

"""

            if custom_software_formatted:
                prompt_modifier += """
⚙️ CUSTOM SOFTWARE (USE THESE LIBRARIES):
{custom_software}

"""

            prompt_modifier += """===============================
"""

        # Add environment resources
        prompt_modifier += """

Environment Resources:

- Function Dictionary:
{function_intro}
---
{tool_desc}
---

{import_instruction}

- Biological data lake
You can access a biological data lake at the following path: {data_lake_path}.
{data_lake_intro}
Each item is listed with its description to help you understand its contents.
----
{data_lake_content}
----

- Software Library:
{library_intro}
Each library is listed with its description to help you understand its functionality.
----
{library_content_formatted}
----

- Note on using R packages and Bash scripts:
  - R packages: Use subprocess.run(['Rscript', '-e', 'your R code here']) in Python, or use the #!R marker in your execute block.
  - Bash scripts and commands: Use the #!BASH marker in your execute block for both simple commands and complex shell scripts with variables, loops, conditionals, etc.
        """

        # Set appropriate text based on whether this is initial configuration or after retrieval
        if is_retrieval:
            function_intro = "Based on your query, I've identified the following most relevant functions that you can use in your code:"
            data_lake_intro = "Based on your query, I've identified the following most relevant datasets:"
            library_intro = (
                "Based on your query, I've identified the following most relevant libraries that you can use:"
            )
            import_instruction = "IMPORTANT: When using any function, you MUST first import it from its module. For example:\nfrom [module_name] import [function_name]"
        else:
            function_intro = "In your code, you will need to import the function location using the following dictionary of functions:"
            data_lake_intro = "You can write code to understand the data, process and utilize it for the task. Here is the list of datasets:"
            library_intro = "The environment supports a list of libraries that can be directly used. Do not forget the import statement:"
            import_instruction = ""

        # Format the content consistently for both initial and retrieval cases
        library_content_formatted = "\n".join(libraries_formatted)
        data_lake_content_formatted = "\n".join(data_lake_formatted)

        # Format the prompt with the appropriate values
        format_dict = {
            "function_intro": function_intro,
            "tool_desc": textify_api_dict(tool_desc) if isinstance(tool_desc, dict) else tool_desc,
            "import_instruction": import_instruction,
            "data_lake_path": self.path + "/data_lake",
            "data_lake_intro": data_lake_intro,
            "data_lake_content": data_lake_content_formatted,
            "library_intro": library_intro,
            "library_content_formatted": library_content_formatted,
        }

        # Add custom resources to format dict (always add them, even if empty)
        format_dict["custom_tools"] = "\n".join(custom_tools_formatted) if custom_tools_formatted else ""
        format_dict["custom_data"] = "\n".join(custom_data_formatted) if custom_data_formatted else ""
        format_dict["custom_software"] = "\n".join(custom_software_formatted) if custom_software_formatted else ""

        run_dir = getattr(self, "run_dir", self.path)
        format_dict["run_dir"] = run_dir
        format_dict["fig_dir"] = getattr(self, "run_figures_dir", os.path.join(run_dir, "figures"))
        format_dict["arts_dir"] = getattr(self, "run_artifacts_dir", os.path.join(run_dir, "artifacts"))

        formatted_prompt = prompt_modifier.format(**format_dict)

        return formatted_prompt
    
    
    def _assess_system_prompt(
        self,
        tool_desc,
        data_lake_content,
        library_content_list,
        self_critic=False,
        is_retrieval=False,
        custom_tools=None,
        custom_data=None,
        custom_software=None,
    ):
        """Generate the system prompt based on the provided resources.

        Args:
            tool_desc: Dictionary of tool descriptions
            data_lake_content: List of data lake items
            library_content_list: List of libraries
            self_critic: Whether to include self-critic instructions
            is_retrieval: Whether this is for retrieval (True) or initial configuration (False)
            custom_tools: List of custom tools to highlight
            custom_data: List of custom data items to highlight
            custom_software: List of custom software items to highlight

        Returns:
            The generated system prompt

        """

        def format_item_with_description(name, description):
            """Format an item with its description in a readable way."""
            # Handle None or empty descriptions
            if not description:
                description = f"Data lake item: {name}"

            # Check if the item is already formatted (contains a colon)
            if isinstance(name, str) and ": " in name:
                return name

            # Wrap long descriptions to make them more readable
            max_line_length = 80
            if len(description) > max_line_length:
                # Simple wrapping for long descriptions
                wrapped_desc = []
                words = description.split()
                current_line = ""

                for word in words:
                    if len(current_line) + len(word) + 1 <= max_line_length:
                        if current_line:
                            current_line += " " + word
                        else:
                            current_line = word
                    else:
                        wrapped_desc.append(current_line)
                        current_line = word

                if current_line:
                    wrapped_desc.append(current_line)

                # Join with newlines and proper indentation
                formatted_desc = f"{name}:\n  " + "\n  ".join(wrapped_desc)
                return formatted_desc
            else:
                return f"{name}: {description}"

        # Separate custom and default resources
        default_data_lake_content = []
        default_library_content_list = []

        # Filter out custom items from default lists
        custom_data_names = set()
        custom_software_names = set()

        if custom_data:
            custom_data_names = {item.get("name") if isinstance(item, dict) else item for item in custom_data}
        if custom_software:
            custom_software_names = {item.get("name") if isinstance(item, dict) else item for item in custom_software}

        # Separate default data lake items
        for item in data_lake_content:
            if isinstance(item, dict):
                name = item.get("name", "")
                if name not in custom_data_names:
                    default_data_lake_content.append(item)
            elif item not in custom_data_names:
                default_data_lake_content.append(item)

        # Separate default library items
        for lib in library_content_list:
            if isinstance(lib, dict):
                name = lib.get("name", "")
                if name not in custom_software_names:
                    default_library_content_list.append(lib)
            elif lib not in custom_software_names:
                default_library_content_list.append(lib)

        # Format the default data lake content
        if isinstance(default_data_lake_content, list) and all(
            isinstance(item, str) for item in default_data_lake_content
        ):
            # Simple list of strings - check if they already have descriptions
            data_lake_formatted = []
            for item in default_data_lake_content:
                # Check if the item already has a description (contains a colon)
                if ": " in item:
                    data_lake_formatted.append(item)
                else:
                    description = self.data_lake_dict.get(item, f"Data lake item: {item}")
                    data_lake_formatted.append(format_item_with_description(item, description))
        else:
            # List with descriptions
            data_lake_formatted = []
            for item in default_data_lake_content:
                if isinstance(item, dict):
                    name = item.get("name", "")
                    description = self.data_lake_dict.get(name, f"Data lake item: {name}")
                    data_lake_formatted.append(format_item_with_description(name, description))
                # Check if the item already has a description (contains a colon)
                elif isinstance(item, str) and ": " in item:
                    data_lake_formatted.append(item)
                else:
                    description = self.data_lake_dict.get(item, f"Data lake item: {item}")
                    data_lake_formatted.append(format_item_with_description(item, description))

        # Format the default library content
        if isinstance(default_library_content_list, list) and all(
            isinstance(item, str) for item in default_library_content_list
        ):
            if (
                len(default_library_content_list) > 0
                and isinstance(default_library_content_list[0], str)
                and "," not in default_library_content_list[0]
            ):
                # Simple list of strings
                libraries_formatted = []
                for lib in default_library_content_list:
                    description = self.library_content_dict.get(lib, f"Software library: {lib}")
                    libraries_formatted.append(format_item_with_description(lib, description))
            else:
                # Already formatted string
                libraries_formatted = default_library_content_list
        else:
            # List with descriptions
            libraries_formatted = []
            for lib in default_library_content_list:
                if isinstance(lib, dict):
                    name = lib.get("name", "")
                    description = self.library_content_dict.get(name, f"Software library: {name}")
                    libraries_formatted.append(format_item_with_description(name, description))
                else:
                    description = self.library_content_dict.get(lib, f"Software library: {lib}")
                    libraries_formatted.append(format_item_with_description(lib, description))

        # Format custom resources with highlighting
        custom_tools_formatted = []
        if custom_tools:
            for tool in custom_tools:
                if isinstance(tool, dict):
                    name = tool.get("name", "Unknown")
                    desc = tool.get("description", "")
                    module = tool.get("module", "custom_tools")
                    custom_tools_formatted.append(f"🔧 {name} (from {module}): {desc}")
                else:
                    custom_tools_formatted.append(f"🔧 {str(tool)}")

        custom_data_formatted = []
        if custom_data:
            for item in custom_data:
                if isinstance(item, dict):
                    name = item.get("name", "Unknown")
                    desc = item.get("description", "")
                    custom_data_formatted.append(f"📊 {format_item_with_description(name, desc)}")
                else:
                    desc = self.data_lake_dict.get(item, f"Custom data: {item}")
                    custom_data_formatted.append(f"📊 {format_item_with_description(item, desc)}")

        custom_software_formatted = []
        if custom_software:
            for item in custom_software:
                if isinstance(item, dict):
                    name = item.get("name", "Unknown")
                    desc = item.get("description", "")
                    custom_software_formatted.append(f"⚙️ {format_item_with_description(name, desc)}")
                else:
                    desc = self.library_content_dict.get(item, f"Custom software: {item}")
                    custom_software_formatted.append(f"⚙️ {format_item_with_description(item, desc)}")

        # Base prompt
        prompt_modifier = """
You are a helpful, fully autonomous biomedical assistant assigned with the task of assessing the outputs from PLAN, LEARN, and EXECUTE for effective problem-solving.
You are also the Assess agent of the PLAN-LEARN-EXECUTE-ASSESS-SHARE (or PLEAS) agentic framework.

[PHASE: ASSESS]
You are the assessor. Your job is to critically evaluate the outputs from EXECUTE and decide the next step. You act like a reviewer.
To problem solve, you will need to decide whether to go back to the previous steps of Plan, Learn, or Execute based on the outputs you have so far.

look at the code and observations from the EXECUTE phase and assess whether they meet the objectives set out in the PLAN and LEARN phases. Search for that information in the previous state.

Instructions:
1. Compare the execution output with the plan objectives.  
2. Check for correctness, completeness, and consistency.  
3. Evaluate scientific reasoning and validity — assess whether the chosen methods, statistical approaches, biological assumptions, and drawn conclusions make scientific sense. Flag any methodological flaws or unsupported interpretations.
4. Identify errors, missing pieces, or improvements.  
5. Decide whether the workflow should:
   - Go back to PLAN (if the approach itself is flawed, to map out the next high-level step, or to update progress. When routing to plan, explicitly instruct the PLAN agent to KEEP THE FULL GLOBAL PLAN INTACT and just add sub-tasks.),  
   - Or move to SHARE (if the solution is scientifically valid, correct, and complete).  
6. Justify your routing decision clearly.
7. DO NOT ASK THE USER FOR FEEDBACK. Make the decision independently in the best interest of rigor.
8. Try to route to plan after each successful task completion within plan. Try to update the plan so the user can see the progress. Only route to learn and execute if there are problems or bugs within those sections.

Rules:
- DO NOT WRITE ANY NEW CODE AND DO NOT CHANGE OR CONTINUE ANY CODE.
- Be critical but constructive.
- DO NOT WRITE FOLLOW UP QUESTIONS
- Route decisively: <goplan> or <goshare>. THESE ARE THE ONLY OPTIONS.

EVIDENCE AND RIGOR RULES — STRICTLY ENFORCED:
- DO NOT TRUST noisy intermediate artifacts. If execution output is based on mock/synthetic data, is partial, throws warnings, or produces suspiciously clean results, flag it explicitly and route back to EXECUTE or PLAN to re-run on real data.
- DO NOT accept unsupported assumptions as facts. If the EXECUTE output assumes a file exists, a package is installed, or a column is present without confirming it in the observation, treat this as an unverified assumption and penalize it in your assessment.
- DO NOT credit steps as complete if the observed output does not directly confirm the deliverable. A print statement saying 'done' without showing the actual result does NOT count as confirmation.
- DO NOT INVENT conclusions. If the output is ambiguous or absent, say so clearly and route to re-execute. Never fill gaps with plausible-sounding invented results.
- PENALIZE any EXECUTE output that: (a) uses mock/dummy data without explicitly labeling it, (b) asserts a result without printing evidence, (c) silently swallows errors, or (d) completes a step using hardcoded values where dynamic computation was required.
- YOU MUST PICK ONE OF THE ROUTES! THIS IS ABOSLUTELY MANDATORY.
- YOU CANNOT DIRECTLY END. IF YOU WANT TO END OR FINISH, YOU MUST ROUTE TO SHARE. IF YOU DON'T ROUTE TO SHARE, YOU CANNOT END OR FINISH SINCE YOU WILL LOOP INDEFINITELY.
- YOU CANNOT CONTINUE TO THE NEXT STEP WITHOUT ASSESSING.
- YOU CANNOT DIRECTLY INTERACT WITH THE USER AND ASK FOR FEEDBACK OR FUTURE DIRECTIONS.

In each response, you must include EITHER <goplan> or <goshare> tag. Not multiple at the same time. YOU DO NOT NEED TO CLOSE THE <goplan> OR <goshare> TAGS. Do not respond with messages without any tags. No empty messages.
If you provide more than one tag, even if you mentioned them in passing, you will misroute yourself. So be careful and only provide one tag.


MANDATORY RESPONSE STRUCTURE — you MUST use exactly these section headers in this order:
1) Comparison with PLAN objectives
   - State which plan objectives were attempted and whether each was met.

2) Correctness, completeness, and consistency
   - Evaluate whether the outputs are correct and complete relative to the plan step.

3) Scientific reasoning and validity
   - Evaluate whether the methods, tools, statistical approaches, and interpretations used follow sound scientific reasoning.
   - Flag any methodological flaws, inappropriate assumptions, unsupported conclusions, or steps that do not make biological/scientific sense.
   - If the science is sound, explicitly state why the approach is valid.

4) Key errors / missing pieces
   - List specific bugs, missing artifacts, wrong values, or incomplete steps. If none, write "None."

5) Decision / routing
   - State which phase to route to and WHY in 1-3 sentences.
   - Include the exact routing tag (e.g. <goexecute>).

6) Plan progress update (what's done, what's next)
   - Bullet list: what has been completed so far and what the next step should be.

### MISSING STEPS
   - If (AND ONLY IF) you chose <goexecute> to correct issues or finish steps, provide a markdown checklist here of the exact micro-steps needed next. (Do not include this section for other routes).

<goplan|golearn|goexecute|goshare>  ← replace with exactly ONE tag

DO NOT skip any section. DO NOT collapse sections together. Each section must have at least 2 sentences or bullet points.

PLEAS LOOK FOR THE COST MANAGEMENT INSTRUCTIONS BELOW. IF IT SAYS THE BUGGET IS EXCEEDED, YOU MUST TAKE IMMEDIATELY ROUTE TO SHARE PHASE.
THIS IS MANDATORY AND KEY, DO NOT IGNORE.
"""

        # Add self-critic instructions if needed
        if self_critic:
            prompt_modifier += """
You may or may not receive feedbacks from human. If so, address the feedbacks by following the same procedure of multiple rounds of thinking, execution, and then coming up with a new solution.
"""

        # Add custom resources section first (highlighted)
        has_custom_resources = any([custom_tools_formatted, custom_data_formatted, custom_software_formatted])

        if has_custom_resources:
            prompt_modifier += """

PRIORITY CUSTOM RESOURCES
===============================
IMPORTANT: The following custom resources have been specifically added for your use.
    PRIORITIZE using these resources as they are directly relevant to your task.
    Always consider these FIRST and in the meantime using default resources.

"""

            if custom_tools_formatted:
                prompt_modifier += """
CUSTOM TOOLS (USE THESE FIRST):
{custom_tools}

"""

            if custom_data_formatted:
                prompt_modifier += """
CUSTOM DATA (PRIORITIZE THESE DATASETS):
{custom_data}

"""

            if custom_software_formatted:
                prompt_modifier += """
⚙️ CUSTOM SOFTWARE (USE THESE LIBRARIES):
{custom_software}

"""

            prompt_modifier += """===============================
"""

        # Add environment resources
        prompt_modifier += """

Environment Resources:

- Function Dictionary:
{function_intro}
---
{tool_desc}
---

{import_instruction}

- Biological data lake
You can access a biological data lake at the following path: {data_lake_path}.
{data_lake_intro}
Each item is listed with its description to help you understand its contents.
----
{data_lake_content}
----

- Software Library:
{library_intro}
Each library is listed with its description to help you understand its functionality.
----
{library_content_formatted}
----

- Note on using R packages and Bash scripts:
  - R packages: Use subprocess.run(['Rscript', '-e', 'your R code here']) in Python, or use the #!R marker in your execute block.
  - Bash scripts and commands: Use the #!BASH marker in your execute block for both simple commands and complex shell scripts with variables, loops, conditionals, etc.
        """

        # Set appropriate text based on whether this is initial configuration or after retrieval
        if is_retrieval:
            function_intro = "Based on your query, I've identified the following most relevant functions that you can use in your code:"
            data_lake_intro = "Based on your query, I've identified the following most relevant datasets:"
            library_intro = (
                "Based on your query, I've identified the following most relevant libraries that you can use:"
            )
            import_instruction = "IMPORTANT: When using any function, you MUST first import it from its module. For example:\nfrom [module_name] import [function_name]"
        else:
            function_intro = "In your code, you will need to import the function location using the following dictionary of functions:"
            data_lake_intro = "You can write code to understand the data, process and utilize it for the task. Here is the list of datasets:"
            library_intro = "The environment supports a list of libraries that can be directly used. Do not forget the import statement:"
            import_instruction = ""

        # Format the content consistently for both initial and retrieval cases
        library_content_formatted = "\n".join(libraries_formatted)
        data_lake_content_formatted = "\n".join(data_lake_formatted)

        # Format the prompt with the appropriate values
        format_dict = {
            "function_intro": function_intro,
            "tool_desc": textify_api_dict(tool_desc) if isinstance(tool_desc, dict) else tool_desc,
            "import_instruction": import_instruction,
            "data_lake_path": self.path + "/data_lake",
            "data_lake_intro": data_lake_intro,
            "data_lake_content": data_lake_content_formatted,
            "library_intro": library_intro,
            "library_content_formatted": library_content_formatted,
        }

        # Add custom resources to format dict (always add them, even if empty)
        format_dict["custom_tools"] = "\n".join(custom_tools_formatted) if custom_tools_formatted else ""
        format_dict["custom_data"] = "\n".join(custom_data_formatted) if custom_data_formatted else ""
        format_dict["custom_software"] = "\n".join(custom_software_formatted) if custom_software_formatted else ""

        run_dir = getattr(self, "run_dir", self.path)
        format_dict["run_dir"] = run_dir
        format_dict["fig_dir"] = getattr(self, "run_figures_dir", os.path.join(run_dir, "figures"))
        format_dict["arts_dir"] = getattr(self, "run_artifacts_dir", os.path.join(run_dir, "artifacts"))

        formatted_prompt = prompt_modifier.format(**format_dict)

        return formatted_prompt
    

    
    
    
    
    def _build_phase_prompts(
        self,
        tool_desc,
        data_lake_with_desc,
        library_content_list,
        *,
        is_retrieval: bool,
        self_critic: bool,
        custom_tools: list[dict] | None,
        custom_data: list[dict] | None,
        custom_software: list[dict] | None,
    ) -> None:
        """(Re)compute PLAN/LEARN/EXECUTE prompts from the same inputs."""
        # Pass cost_budget and current_cost to PLAN prompt
        current_cost = None
        if hasattr(self, "cost_manager") and self.cost_manager is not None:
            current_cost = getattr(self.cost_manager, "total_cost", None)
        
        # Determine whether to show scientific mindset
        # If first_plan_only=True, only show on first plan (when count=0)
        # If first_plan_only=False, always show it
        show_mindset = (not self.scientific_mindset_first_plan_only) or (self._plan_count == 0)
        self._plan_count += 1  # Increment after check
        
        self.system_prompts["PLAN"] = self._plan_system_prompt(
            tool_desc=tool_desc,
            data_lake_content=data_lake_with_desc,
            library_content_list=library_content_list,
            self_critic=self_critic,
            is_retrieval=is_retrieval,
            custom_tools=custom_tools,
            custom_data=custom_data,
            custom_software=custom_software,
            current_cost=current_cost,
            cost_budget=self.cost_budget,
            show_scientific_mindset=show_mindset,
        )

        self.system_prompts["LEARN"] = self._learn_system_prompt(
            tool_desc=tool_desc,
            data_lake_content=data_lake_with_desc,
            library_content_list=library_content_list,
            self_critic=self_critic,
            is_retrieval=is_retrieval,
            custom_tools=custom_tools,
            custom_data=custom_data,
            custom_software=custom_software,
        )

        self.system_prompts["EXECUTE"] = self._execute_system_prompt(
            tool_desc=tool_desc,
            data_lake_content=data_lake_with_desc,
            library_content_list=library_content_list,
            self_critic=self_critic,
            is_retrieval=is_retrieval,
            custom_tools=custom_tools,
            custom_data=custom_data,
            custom_software=custom_software,
        )
        
        self.system_prompts["ASSESS"] = self._assess_system_prompt(
            tool_desc=tool_desc,
            data_lake_content=data_lake_with_desc,
            library_content_list=library_content_list,
            self_critic=self_critic,
            is_retrieval=is_retrieval,
            custom_tools=custom_tools,
            custom_data=custom_data,
            custom_software=custom_software,
        )

        # keep legacy single prompt for anything that still reads self.system_prompt
        self.system_prompt = self.system_prompts["PLAN"]

    
    def overview_assess(self, state: "AgentState") -> "AgentState":
        state["phase"] = "ASSESS"  # reuse assess LLM; this is an overview pass
        llm = self._llm_for("ASSESS")

        # read artifacts
        arts = self._collect_run_artifacts()
        paper_txt = ""
        if arts["paper_md"] and os.path.exists(arts["paper_md"]):
            try:
                paper_txt = Path(arts["paper_md"]).read_text(encoding="utf-8")
            except Exception:
                paper_txt = ""
        log_txt = ""
        if arts["log_file"] and os.path.exists(arts["log_file"]):
            try:
                log_txt = Path(arts["log_file"]).read_text(encoding="utf-8")[-100_000:]  # tail
            except Exception:
                log_txt = ""

        # cheap heuristics
        has_traceback = "Traceback (most recent call last)" in log_txt or "ERROR" in log_txt or "Exception" in log_txt
        imrad_ok = all(h in paper_txt for h in ["Introduction", "Methods", "Results", "Discussion"])
        has_abstract = "Abstract" in paper_txt
        has_refs = "References" in paper_txt or "Bibliography" in paper_txt

        # figures
        fig_count = 0
        if arts["fig_manifest"] and os.path.exists(arts["fig_manifest"]):
            try:
                with open(arts["fig_manifest"], "r", encoding="utf-8") as f:
                    for _ in f:
                        fig_count += 1
            except Exception:
                pass

        # build a short, structured summary for the model to refine into concrete fixes
        checklist = {
            "errors_in_logs": has_traceback,
            "imrad_complete": imrad_ok,
            "has_abstract": has_abstract,
            "has_references": has_refs,
            "figure_count": fig_count,
            "paper_path": arts["paper_md"],
            "log_path": arts["log_file"],
        }

        # minimal prompt; model turns this into actionable patch instructions
        prompt = f"""
You are an **area-chair–caliber overview reviewer** performing a rapid but rigorous pass on the latest run. 
Act like a NeurIPS reviewer who *must* enforce the **NeurIPS Paper Checklist** (reproducibility, transparency, ethics, societal impact). 
Your job is to (a) diagnose, (b) prescribe precise code and writing changes, and (c) route the graph to the correct next step.

## Inputs
CHECKS: {json.dumps(checklist, ensure_ascii=False)}
--- PAPER (md/LaTeX) ---
{paper_txt}
--- LOG (tail) ---
{log_txt}

## Review Focus (NeurIPS Checklist–driven)
For each item below, answer **Yes/No/N/A** with a one-sentence justification that points to evidence (section heading, figure/table label, file path, or log line pattern). 
Then propose *concrete* fixes if "No" or if the item is weak.

1) **Claims** — Do abstract & intro accurately reflect contributions/scope and match theoretical/experimental results? Are assumptions/limitations clearly stated?
2) **Limitations** — Is there a dedicated Limitations section that honestly scopes assumptions, robustness, datasets/runs coverage, and factors influencing performance?
3) **Theory/Assumptions/Proofs** — If theory exists: are assumptions complete and clearly stated? Are full proofs provided (main vs appendix) with proper references?
4) **Reproducibility Path** — Is there a viable path for others to verify results (code, data, hosted model, or step-by-step replication instructions)?
5) **Code/Data/Instructions** — Are code/data + exact commands/env (incl. seed control) present (or justified if closed/proprietary/anonymized)?
6) **Experimental Details** — Are training details (splits, hparams, selection procedure) specified in paper or appendix? Are baselines/datasets sufficiently described?
7) **Statistical Rigor** — Are error bars/CI/tests appropriate and defined (what variability, how computed, assumptions)? Are asymmetric distributions handled correctly?
8) **Compute Disclosure** — Are compute types (CPU/GPU), memory/storage, per-run & total compute (incl. failed/preliminary) disclosed?
9) **Code of Ethics Compliance** — Any deviations disclosed (anonymized) and justified? 
10) **Broader Impacts / Negative Uses** — Are potential harms, fairness/privacy/security issues, and mitigations discussed (if applicable)?
11) **Safeguards (high-risk release)** — For dual-use models/assets, are access controls or usage restrictions described (if applicable)?
12) **Licenses & Attribution** — Are third-party assets cited with versions, URLs, and licenses? Is repackaging license treatment correct?
13) **New Assets Documentation** — If releasing assets, are training, consent, license, and limitations documented via a structured template?
14) **Human Subjects / Crowdwork** — If applicable, are instructions, screenshots, and compensation details included (main or appendix)?
15) **IRB / Ethics Board** — If applicable, is approval stated in an anonymized way?
16) **Declaration of LLM Usage** — If LLMs are integral/novel in the method, is usage described per policy?

## PDF Build & Artifacts Sanity (fast checklists)
- **LaTeX/PDF build**: Confirm presence/paths for main.tex, main.log (or equivalent), generated PDF, NeurIPS checklist page after references.
- **Figures**: Confirm figure count = {fig_count} from fig_manifest; ensure each figure is referenced and has a caption; note broken/missing paths.
- **Bibs**: Check that references compile (no “??” or citation warnings visible in logs).
- **Tracebacks/Errors**: From logs, does a traceback or runtime error exist? `errors_in_logs={checklist.get('errors_in_logs')}`

## Output Schema (STRICT)
Return **all** sections below, in order. Keep each part concise, surgical, and actionable.

A) **PASS/FAIL (1 sentence)** — Gate on scientific clarity & checklist adequacy (a weak but fixable paper can be FAIL with concrete path to PASS).
B) **Top 5 Blocking Issues** — Bullet list; each issue = (Title · Evidence · Why it matters · Minimal fix).
C) **Code Fixes (precise)** — Provide **mini unified diffs** or exact line-level edits (file:line → replacement). 
   - Include build fixes for LaTeX/PDF (missing files, wrong paths, package installs, bibliography, figure includes).
   - Include instrumentation/logging code (see next section) directly in diffs when possible.
D) **Logging & Build Improvements** — Concrete changes to:
   - Ensure LaTeX build emits `main.log`/`build.log` and fails loudly on errors (non-zero exit).
   - Add runtime guards for figure existence; print a **figure manifest** and unresolved references.
   - Emit a **conda/pip freeze**, **git commit hash**, **seed settings**, and **hardware/compute summary** to the log.
E) **Paper Edits (surgical)** — Section-scoped edits with *draft text*. 
   - Add/trim sentences; insert missing Limitations, Broader Impacts, Compute Disclosure, Reproducibility Instructions; 
   - For stats: specify variability source, CI or σ vs s.e.m., and method (e.g., bootstrap N=10k).
F) **NeurIPS Checklist Table (Yes/No/N/A + 1-line justification)** — 16 rows corresponding to the items above. 
   - If “No”, include **mini-fix** (≤1 sentence) and pointer to where it should go (e.g., “Appendix A.2”).
G) **Exact Next Step** — Choose **ONE**:
   - <goexecute> if build is broken, assets missing, or diffs/logging need implementation **now**.
   - <goplan> if experimental design, dataset choice, or evaluation plan must be revised before execution.
   - <goshare> if PDF, checklist, and artifacts are sound and it’s ready to package/publish.
   **Emit exactly one tag and do not close it. No other tags.**

## Style Constraints
- Be **decisive** and **specific**. Avoid vague advice like “improve clarity.” Show the exact sentence or code replacement.
- Keep total under ~700–900 words. Prioritize the **Top 5 Blocking Issues** and the **exact diffs** that unblock PDF + checklist compliance.
- Do **not** emit multiple routing tags; emit only one of <goplan>, <goexecute>, or <goshare>. 
- Do **not** leave any output section empty; if N/A, state “N/A” and 1-line rationale.

(Notes for you, the reviewer agent)
- If `errors_in_logs=True` or the PDF artifacts/checklist page are missing, default the routing to **<goexecute>** unless there’s an upstream design flaw that needs planning.
- Prefer fixes that minimize scope and risk for the next iteration; ask for broader refactors only if essential for correctness or compliance.

In each response, you must include EITHER <goplan>, <goexecute>, or <goshare> tag. Not multiple at the same time. Do not respond with messages without any tags. No empty messages. DO NOT GO TO <goshare> IF THERE ARE ERRORS ELSEWHERE.
"""
        #{paper_txt[:4000]}
        #{log_txt[-4000:]}

        # Log with phase logger
        self.phase_logger.set_phase("OVERVIEW_ASSESS")
        self.phase_logger.log_prompt(prompt)
        log_llm_event("overview_assess_PROMPT", prompt)

        overview_msgs = [SystemMessage(content="You refine outputs into concrete, actionable fixes."),
                         HumanMessage(content=prompt)]
        ai = llm.invoke(overview_msgs)

        # Record usage for overview_assess (phase-specific LLM)
        try:
            self._record_llm_usage(overview_msgs + [ai], llm_obj=llm)
        except Exception:
            pass

        feedback = str(getattr(ai, "content", ai))



        # Log with phase logger
        self.phase_logger.log_response(feedback)
        log_llm_event("overview_assess_RESPONSE", feedback)
        
        if "<goplan>" in feedback.lower():
            next_step = "<goplan>"
        elif "<goexecute>" in feedback.lower():
            next_step = "<goexecute>"
        elif "<goshare>" in feedback.lower():
            next_step = "<goshare>"
        
        # update loop counter
        loop_count = int(state.get("artifacts", {}).get("loop_count", 0))
        loop_count += 1
        if "artifacts" not in state:
            state["artifacts"] = {}
        state["artifacts"]["loop_count"] = loop_count
        state["artifacts"]["last_overview"] = {
            "summary": feedback,
            "checks": checklist,
            "artifacts": arts,
            "decided_next": next_step,
        }

        # guard against infinite loops
        if loop_count >= self.max_overview_loops:
            state["next_step"] = "end"
            state["messages"].append(
                HumanMessage(content="[OVERVIEW] Reached maximum overview loops. Summarize best results and limitations.")
            )
            return state

        # push feedback to the next phase as a HumanMessage
        if next_step == "<goplan>":
            state["messages"].append(HumanMessage(content="[OVERVIEW→PLAN] Apply these critiques to revise the plan:\n" + feedback))
            state["next_step"] = "plan"
        elif next_step == "<goexecute>":
            state["messages"].append(HumanMessage(content="[OVERVIEW→EXECUTE] Implement the following code/logging fixes, then re-run:\n" + feedback))
            state["next_step"] = "execute"
        elif next_step == "<goshare>":
            state["messages"].append(HumanMessage(content="[OVERVIEW→SHARE] Apply these paper/logging fixes, then proceed to SHARE:\n" + feedback))
            state["next_step"] = "share"
        elif next_step == "end":
            state["messages"].append(HumanMessage(content="[OVERVIEW] Reached maximum overview loops. Summarize best results and limitations."))
            state["next_step"] = "end"
        else:
            print("parsing error...")
            # Robust retry counter stored in state
            tries = state["artifacts"].get("parse_retries", 0) if "artifacts" in state else 0
            tries += 1
            if "artifacts" not in state:
                state["artifacts"] = {}
            state["artifacts"]["parse_retries"] = tries

            if tries >= 3:
                print("Detected repeated parsing errors, ending conversation")
                state["messages"].append(AIMessage(content="Terminating: model failed to emit <goplan>, <goexecute>, or <goshare> after 3 retries."))
                state["next_step"] = "end"
            else:
                # Nudge once more with a precise instruction
                state["messages"].append(HumanMessage(
                    content="[FORMAT ERROR] Emit EITHER <goplan>, <goexecute>, or <goshare> tag. Not multiple at the same time. Do not respond with messages without any tags. No empty messages."
                ))
                state["next_step"] = "overview_assess"  # retry assess
        return state
    
    
    
    def _ensure_mem(self, state):
        """Initialize memory fields once."""
        arts = state.setdefault("artifacts", {})
        mem  = arts.setdefault("memory", {})
        mem.setdefault("long_term_summary", "")   # rolling summary
        mem.setdefault("events", [])              # optional structured breadcrumbs

    def _summarize_messages_for_long_term(self, prev_summary: str, new_msgs: list[BaseMessage]) -> str:
        """Condense recent dialog into a comprehensive, detailed summary we can carry across phases."""
        if not new_msgs:
            return prev_summary or ""
        
        # Use gpt-4o-mini for memory summarization
        from bioplease.llm import get_llm
        llm = get_llm(
            model="gpt-4o-mini",
            source="OpenAI",
            temperature=0.3
        )
        
        # Build comprehensive context from recent messages
        recent = []
        for m in new_msgs[-self.short_window:]:
            role = m.__class__.__name__.replace("Message","").lower()
            content = getattr(m, 'content', str(m))
            recent.append(f"[{role}]: {content}")
        
        # Enhanced prompt for detailed, in-depth summarization
        prompt = (
            "You are maintaining a comprehensive long-term memory of a scientific research project.\n"
            "Your role is to create a DETAILED, STRUCTURED summary that captures ALL important information.\n\n"
            
            "=== REQUIREMENTS ===\n"
            "1. Be COMPREHENSIVE - include all key details, not just highlights\n"
            "2. Maintain STRUCTURE - organize by research stages/phases\n"
            "3. Preserve TECHNICAL DETAILS - specific values, parameters, methods, file names\n"
            "4. Track DECISIONS - why choices were made, alternatives considered\n"
            "5. Document PROGRESS - what worked, what failed, lessons learned\n"
            "6. Note DATA/FILES - specific datasets, outputs, paths referenced\n"
            "7. Record ERRORS/FIXES - problems encountered and how they were resolved\n"
            "8. Capture NEXT STEPS - planned actions and open questions\n\n"
            
            "=== FORMAT ===\n"
            "Use markdown with clear sections:\n"
            "## Research Goal\n"
            "## Approach & Methods\n"
            "## Key Decisions\n"
            "## Progress Summary\n"
            "## Data & Resources\n"
            "## Results & Findings\n"
            "## Issues & Solutions\n"
            "## Next Steps\n\n"
            
            "=== STYLE ===\n"
            "- Use bullet points for clarity\n"
            "- Include specific numbers, file names, parameters\n"
            "- Be precise and technical\n"
            "- Avoid vague summaries\n"
            "- Build upon and enhance the previous summary rather than replacing it\n\n"
            
            f"=== PREVIOUS SUMMARY ===\n{prev_summary if prev_summary else '(Starting new research session)'}\n\n"
            f"=== RECENT CONVERSATION ===\n" + "\n\n".join(recent) + "\n\n"
            
            "Now update the summary by integrating the recent conversation into the existing structure. "
            "Expand on what was there before and add new information from the recent messages."
        )
        
        # Add timeout protection for LLM invocation
        try:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Memory summarization timed out")
            
            # Set 60 second timeout for memory summarization
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)
            
            try:
                out = llm.invoke([HumanMessage(content=prompt)])
                new_summary = (getattr(out, "content", str(out)) or "").strip()
            finally:
                signal.alarm(0)  # Cancel the alarm
                
        except TimeoutError as e:
            print(f"[WARNING] Memory summarization timeout: {e}. Using previous summary.")
            return prev_summary or ""
        except Exception as e:
            print(f"[WARNING] Memory summarization error: {e}. Using previous summary.")
            return prev_summary or ""
        
        return new_summary

    def _detect_execution_error(self, result: str) -> bool:
        """Detect if code execution resulted in an error.
        
        Args:
            result: The execution result string from code execution
            
        Returns:
            True if an error was detected, False otherwise
        """
        if self.use_llm_error_detection:
            return self._llm_detect_error(result)
        else:
            return self._pattern_detect_error(result)
    
    def _pattern_detect_error(self, result: str) -> bool:
        """Pattern-based error detection using regex.
        
        Args:
            result: The execution result string
            
        Returns:
            True if error patterns detected, False otherwise
        """
        if not result:
            return False
            
        error_patterns = [
            r'Traceback \(most recent call last\)',
            r'Error:',
            r'Exception:',
            r'SyntaxError:',
            r'NameError:',
            r'TypeError:',
            r'ValueError:',
            r'AttributeError:',
            r'KeyError:',
            r'IndexError:',
            r'ImportError:',
            r'ModuleNotFoundError:',
            r'FileNotFoundError:',
            r'PermissionError:',
            r'RuntimeError:',
            r'AssertionError:',
        ]
        
        import re
        for pattern in error_patterns:
            if re.search(pattern, result, re.IGNORECASE):
                return True
        return False
    
    def _llm_detect_error(self, result: str) -> bool:
        """LLM-based error detection for more nuanced analysis.
        
        Args:
            result: The execution result string
            
        Returns:
            True if LLM detects an error, False otherwise
        """
        if not result or len(result.strip()) == 0:
            return False
        
        # Truncate very long results to avoid token overflow
        max_chars = 4000
        truncated_result = result[:max_chars]
        if len(result) > max_chars:
            truncated_result += "\n[...truncated...]"
        
        prompt = f"""Analyze this code execution output and determine if it contains an error.

<observation>
{truncated_result}
</observation>

Respond with ONLY "YES" if there is an error (exception, traceback, failure), or "NO" if the code executed successfully.
Do not provide any explanation, just YES or NO."""

        try:
            # Use the configured error detection model (default: gpt-4o-mini)
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=self.error_detection_model, temperature=0)
            
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
            answer = response.content.strip().upper()
            
            return answer == "YES"
        except Exception as e:
            print(f"[ERROR] LLM error detection failed: {e}")
            # Fall back to pattern-based detection
            return self._pattern_detect_error(result)

    def _validate_code_format(self, code: str) -> tuple[bool, str]:
        """Validate code block format before execution to catch common mistakes.
        
        Args:
            code: Raw code extracted from <execute> tags
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        import re
        
        # Check for narrative text markers (common mistake)
        narrative_patterns = [
            r'^\*\*.*?\*\*',  # Bold markdown
            r'^#{2,6}\s',      # Markdown headers (but not single # which is valid Python comment)
            r'^\d+\.\s+\*\*', # Numbered lists with bold
            r'Error Analysis:',
            r'Root [Cc]ause:',
            r'Corrected Implementation',
            r'I will now',
            r'Let me',
            r'The error',
        ]
        
        for pattern in narrative_patterns:
            if re.search(pattern, code, re.MULTILINE | re.IGNORECASE):
                return (False, f"Contains narrative text (matched pattern: {pattern}). Code blocks must contain ONLY executable code.")
        
        # Check for excessive non-code content at start
        lines = code.strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            # First line should look like code, not prose
            if first_line and not any([
                first_line.startswith('#'),           # Comment
                first_line.startswith('import '),     # Import
                first_line.startswith('from '),       # Import
                first_line.startswith('def '),        # Function def
                first_line.startswith('class '),      # Class def
                first_line.startswith(('"', "'")),    # String/docstring
                '=' in first_line,                    # Assignment
                first_line.startswith('if '),         # Control flow
                first_line.startswith('for '),
                first_line.startswith('while '),
                first_line.startswith('with '),
                first_line.startswith('try:'),
                first_line.startswith('#!'),          # Shebang for R/Bash
            ]):
                # Check if it looks like prose (multiple words, no code patterns)
                if len(first_line.split()) > 5 and not any(c in first_line for c in ['(', ')', '{', '}', '[', ']', '=']):
                    return (False, f"First line appears to be prose, not code: '{first_line[:60]}'")
        
        return (True, "")
    
    def _extract_current_plan_step(self, state: dict) -> str:
        """Extract the full current plan step (including all its numbered sub-steps).

        Strategy (in priority order):
        1. PRIMARY: Pull plan_output and learn_output from
           self.product_manager.current_state.previous_states — these are the
           authoritative persisted PM state records and survive message pruning.
        2. FALLBACK: Scan LangGraph messages for plan/learn content.

        Returns the complete block for the active plan step, framed so the
        micro-plan LLM knows it must cover every sub-step in one EXECUTE call.
        """
        import re

        plan_text = ""
        learn_text = ""

        # ── 1. PRIMARY: product manager persisted state ──────────────────────
        if self.product_manager and self.product_manager.current_state:
            prev_states = self.product_manager.current_state.previous_states or []

            # Walk in reverse to prefer the most recent entries for each phase
            for entry in reversed(prev_states):
                phase = (entry.get("phase") or "").upper()
                output = str(entry.get("output") or "")
                if not plan_text and phase == "PLAN" and output:
                    plan_text = output
                if not learn_text and phase == "LEARN" and output:
                    learn_text = output
                if plan_text and learn_text:
                    break

        # ── 2. FALLBACK: scan LangGraph messages ─────────────────────────────
        if not plan_text or not learn_text:
            all_messages = state.get("messages", [])
            for msg in reversed(all_messages):
                content = str(getattr(msg, "content", ""))
                if not plan_text and re.search(r'(?i)(step\s*\d|\d+\.\s+\w)', content) and len(content) > 200:
                    plan_text = content
                if not learn_text and re.search(r'(?i)(?:plan\s+)?step\s+\d', content) and len(content) > 100:
                    learn_text = content
                if plan_text and learn_text:
                    break

        if not plan_text:
            return "Complete the current task from the PLAN"

        # ── Detect active step number from learn_text ─────────────────────────
        current_step_num = None
        if learn_text:
            # LEARN outputs say e.g. "Plan Step 2.1", "addressing Step 2", "Step 2.1:"
            m = re.search(r'(?i)(?:plan\s+)?step\s+(\d+(?:\.\d+)?)', learn_text)
            if m:
                current_step_num = m.group(1)

        if current_step_num is None:
            return f"FULL PLAN (could not determine active step):\n\n{plan_text[:3000]}"

        top_num = current_step_num.split(".")[0]

        # ── Extract the block for the active top-level step ───────────────────
        # Patterns tried in order; each captures from "Step N" / "N." header up
        # to (but not including) the next top-level step header or document end.
        patterns = [
            # "Step 2: ..." or "**Step 2**" (case-insensitive), multi-line block
            rf'(?is)((?:\*{{0,2}}step\s+{re.escape(top_num)}[^\d][^\n]*\n)(?:(?!\*{{0,2}}step\s+\d).*\n?)*)',
            # "2. ..." numbered list item, multi-line block
            rf'(?m)(^{re.escape(top_num)}\.\s.*(?:\n(?!\d+\.\s).*)*)',
        ]
        step_block = ""
        for pat in patterns:
            match = re.search(pat, plan_text)
            if match:
                step_block = match.group(0).strip()
                break

        if step_block:
            return (
                f"CURRENT PLAN STEP (Step {top_num}) — "
                f"ALL sub-steps below must be fully completed in this single EXECUTE call:\n\n"
                f"{step_block}"
            )
        # Fallback: full plan
        return f"FULL PLAN (active step ~{current_step_num}):\n\n{plan_text[:3000]}"

    def _build_micro_plan_state(
        self,
        state: dict,
        micro_plan: list,
        step_idx: int,
        bailout_reason: str,
        failed_step: dict | None = None,
        remaining_from_idx: int | None = None,
    ) -> None:
        """Build and store rich micro-plan state for ASSESS to consume.

        Populates state["artifacts"]["micro_plan_state"] with completed steps
        (harvested from accumulators), the failed step, and the remaining steps
        still to be run.  Also writes the remaining steps back into
        state["artifacts"]["micro_plan"] so the resume logic fires on the next
        EXECUTE entry.

        Args:
            state: Agent state (mutated in-place)
            micro_plan: The current micro-plan list (remaining steps in this call)
            step_idx: Index within micro_plan of the step that caused the bailout
                      (or the last completed step for a limit-bailout)
            bailout_reason: Human-readable reason string
            failed_step: The step dict that failed (None for limit-bailout)
            remaining_from_idx: Override start index for remaining slice.
                                 Defaults to step_idx (include failed step).
        """
        arts = state.setdefault("artifacts", {})

        # ── Build completed_steps from accumulators ──────────────────────────
        raw_obs  = arts.get("micro_accumulated_observations") or []
        raw_code = arts.get("micro_accumulated_code") or []

        completed_steps = []
        for i, obs_str in enumerate(raw_obs):
            # obs_str format: "[Step N] <observation text>"
            import re as _re
            m = _re.match(r'\[Step\s+(\d+)\]\s*(.*)', obs_str, _re.DOTALL)
            obs_text = m.group(2) if m else obs_str
            # Try to get description from the code accumulator header
            desc = ""
            if i < len(raw_code):
                hdr = raw_code[i].split('\n', 1)[0]  # "# Micro-step N: <desc>"
                dm = _re.match(r'#\s*Micro-step\s+\d+:\s*(.*)', hdr)
                desc = dm.group(1).strip() if dm else hdr.lstrip('# ').strip()
            completed_steps.append({
                "step": desc or f"Step {i+1}",
                "observation": obs_text[:400],
                "success": True,
            })

        # ── Remaining steps (from the failing/limit-hit index onward) ────────
        if remaining_from_idx is None:
            remaining_from_idx = step_idx  # include failed step for retry
        remaining_steps = micro_plan[remaining_from_idx:]

        # ── Write micro_plan_state ────────────────────────────────────────────
        arts["micro_plan_state"] = {
            "completed_steps": completed_steps,
            "remaining_steps": [
                {"description": s["description"], "expected_outputs": s.get("expected_outputs", "")}
                if isinstance(s, dict) else {"description": str(s)}
                for s in remaining_steps
            ],
            "failed_step": (
                {"description": failed_step["description"],
                 "expected_outputs": failed_step.get("expected_outputs", "")}
                if failed_step else None
            ),
            "bailout_reason": bailout_reason,
            "accumulated_observations": [s["observation"] for s in completed_steps],
        }

        # ── Preserve remaining steps for resume ──────────────────────────────
        if remaining_steps:
            arts["micro_plan"]          = remaining_steps
            arts["micro_step_index"]    = 0
            arts["micro_step_retries"]  = {}
            arts["bailout_reason"]      = bailout_reason
            arts["bailout_step"]        = step_idx
        else:
            arts["micro_plan"] = None

    def _generate_micro_plan(self, state: dict, current_plan_step: str) -> list[dict]:
        """Generate micro-plan: break down current PLAN step (all its sub-steps) into executable tasks.
        
        Args:
            state: Current agent state
            current_plan_step: The high-level PLAN step (including all sub-steps) to decompose
            
        Returns:
            List of micro-step dicts with 'id', 'description', 'expected_outputs'
        """
        import os, glob
        from langchain_core.messages import SystemMessage, HumanMessage

        artifacts = state.get("artifacts", {})

        # ── 1. Prior micro-execution state (completed/failed steps) ──────────
        prior_execution_ctx = ""
        micro_plan_state = artifacts.get("micro_plan_state")
        if micro_plan_state:
            completed = micro_plan_state.get("completed_steps") or []
            failed = micro_plan_state.get("failed_step")
            bailout = micro_plan_state.get("bailout_reason", "")
            remaining = micro_plan_state.get("remaining_steps") or []

            completed_lines = "\n".join(
                f"  [{'✓' if s.get('success') else '✗'}] {s.get('step','?')}\n"
                f"      Observation: {str(s.get('observation',''))[:300]}"
                for s in completed
            ) or "  (none)"

            remaining_lines = "\n".join(
                f"  [ ] {s.get('description', s) if isinstance(s, dict) else s}"
                for s in remaining
            ) or "  (none)"

            prior_execution_ctx = f"""
=== PRIOR MICRO-EXECUTION STATE (from previous EXECUTE call) ===
BAILOUT REASON: {bailout}

ALREADY COMPLETED — do NOT regenerate these steps:
{completed_lines}

STEPS THAT STILL NEED TO BE DONE (start from these):
{remaining_lines}

FAILED STEP (caused bailout):
{failed if failed else 'None — hit step-count limit'}

IMPORTANT: Your micro-plan must cover ONLY the remaining/failed steps above.
Do not repeat any already-completed step.
"""

        # ── 2. ASSESS-provided revised micro-plan ────────────────────────────
        revised_ctx = ""
        revised_plan = artifacts.get("revised_micro_plan")
        if revised_plan:
            revised_lines = "\n".join(
                f"  {i+1}. {s.get('step', s) if isinstance(s, dict) else s}"
                + (f"\n     Instructions: {s['instructions']}" if isinstance(s, dict) and s.get('instructions') else "")
                + (f"\n     Expected: {s['expected']}" if isinstance(s, dict) and s.get('expected') else "")
                for i, s in enumerate(revised_plan)
            )
            revised_ctx = f"""
=== ASSESS-PROVIDED REVISED PLAN ===
The ASSESS phase reviewed the prior execution and recommends these steps:
{revised_lines}

Use this revised plan as the basis for your micro-plan. Adjust only if the
concrete instructions need to be more specific; do not add or remove steps
unless clearly necessary.
"""

        # ── 3. Available artifacts on disk ───────────────────────────────────
        artifacts_ctx = ""
        try:
            run_dir = getattr(self, "run_dir", None)
            if not run_dir:
                run_dir_tuple = self._latest_run_dirs()
                run_dir = run_dir_tuple[0] if run_dir_tuple else None
            if run_dir:
                arts_dir = os.path.join(run_dir, "artifacts")
                if os.path.isdir(arts_dir):
                    files = sorted(glob.glob(os.path.join(arts_dir, "**", "*"), recursive=True))
                    files = [f for f in files if os.path.isfile(f)]
                    if files:
                        file_lines = "\n".join(
                            f"  {os.path.relpath(f, run_dir)}  ({os.path.getsize(f):,} bytes)"
                            for f in files[:60]  # cap at 60 to avoid bloat
                        )
                        artifacts_ctx = f"""
=== ARTIFACTS ALREADY ON DISK (run/artifacts/) ===
These files have already been saved — avoid recomputing them unless they need fixing:
{file_lines}
"""
        except Exception:
            pass  # non-fatal — proceed without artifact listing

        micro_plan_prompt = f"""You are a senior research software engineer converting a high-level plan step into a CONCRETE, fully-specified execution plan that a coding agent can follow step by step without ambiguity.

RULE: ONE full EXECUTE call must complete EVERY sub-step listed below. Do not skip any sub-step.
{prior_execution_ctx}{revised_ctx}{artifacts_ctx}
{current_plan_step}

INSTRUCTIONS:
- Generate one micro-task per sub-step. If the plan has explicit sub-steps (2.1, 2.2, 2.3 …), produce at least one micro-task per sub-step.
- Maximum 12 micro-tasks; combine trivial actions (e.g. save + print) into one.
- Each micro-task MUST include:
  • A one-line TITLE (what it does + which sub-step it fulfils)
  • INSTRUCTIONS: exact file paths, column names, variable names, pandas/numpy calls, expected shapes/counts, output paths — concrete enough to write code without guessing
  • EXPECTED: a specific observable output (e.g. "prints 18639 unique genes, dtype object")

FORMAT (reproduce exactly — no extra prose before or after MICRO-PLAN:):
MICRO-PLAN:
2a. [sub-step ref] One-line title
   Instructions: <step-by-step concrete instructions>
   Expected: <specific observable confirming success>
2b. [sub-step ref] One-line title
   Instructions: <concrete instructions>
   Expected: <success observable>
...

EXAMPLE (plan step with sub-steps 2.1 / 2.2 / 2.3):
MICRO-PLAN:
2a. [2.1] Load GTEx + gene_info and inspect identifier columns
   Instructions: pd.read_parquet("./data/.../gtex_tissue_gene_tpm.parquet"), print shape, dtypes, first 5 rows; pd.read_parquet("./data/.../gene_info.parquet"), print shape, columns, 10 sample gene_id and gene_name values.
   Expected: GTEx shape (1007910, 4) with columns [Description, Tissue, Expression, Gene]; gene_info ~(63086, 13) with gene_id = ENSG strings
2b. [2.1] Classify GTEx Gene column identifier format
   Instructions: Extract df["Gene"].unique(); regex classify each as ENSG/ENSG.version/symbol/other; print value_counts and 10 examples per class.
   Expected: Prints classification counts; all/most should be "symbol"
2c. [2.1] Map GTEx gene symbols to gene_info.gene_id; collect unmapped
   Instructions: merged = gtex_genes.merge(gene_info[["gene_id","gene_name"]], left_on="Gene", right_on="gene_name", how="left"); unmapped = merged[merged.gene_id.isna()]["Gene"]; print mapped_count, unmapped_count, first 10 unmapped.
   Expected: Prints e.g. "Mapped: 16800  Unmapped: 1839"
2d. [2.1] Resolve unmapped via txgnn_name_mapping; save final mapping CSV
   Instructions: Load pickle("./data/.../txgnn_name_mapping.pkl"); resolve unmapped symbols; build final df with cols [GTEx_Gene, gene_id, gene_name, source]; save to "./data/.../gtex_id_mapping.csv"; print coverage %.
   Expected: File saved; coverage e.g. "Coverage: 97.3 %"
2e. [2.2] Confirm genome assembly match between GTEx and gene_info
   Instructions: Check gene_info chr values vs GRCh38/hg38 conventions; print unique chr values; cross-check TP53 coordinates vs Ensembl GRCh38 range (chr17:7,668,402-7,687,550).
   Expected: Prints "Assembly: GRCh38 confirmed" or flags mismatch
2f. [2.3] Locate Ensembl GTF file; write provenance entry
   Instructions: Check "./data/.../gencode.v38.annotation.gtf.gz"; if missing log expected URL; write provenance entry to "./code/provenance_schema.json" with keys [file, source, version, date].
   Expected: Prints file path + size or "not found — URL logged"; provenance updated

NOW GENERATE THE MICRO-PLAN:"""

        messages = [
            SystemMessage(content="You are a senior research software engineer. Produce concrete, specific, actionable execution plans."),
            HumanMessage(content=micro_plan_prompt)
        ]
        
        llm = self._llm_for("EXECUTE")
        response = llm.invoke(messages)
        
        # Parse the micro-plan from response
        micro_steps = self._parse_micro_plan(response.content)
        
        return micro_steps
    
    def _parse_micro_plan(self, response: str) -> list[dict]:
        """Parse micro-plan from LLM response.
        
        Args:
            response: LLM response containing MICRO-PLAN
            
        Returns:
            List of parsed micro-step dictionaries
        """
        import re
        
        micro_steps = []
        
        # Find MICRO-PLAN section
        if "MICRO-PLAN:" in response:
            plan_section = response.split("MICRO-PLAN:")[1]
        else:
            plan_section = response
        
        # Split into individual step blocks on "Na. " boundaries (e.g. "2a. ", "2b. ")
        step_block_pattern = r'(\d+[a-z])\. '
        parts = re.split(step_block_pattern, plan_section)
        # parts = ['preamble', '2a', 'Title\n   Instructions:...\n   Expected:...', '2b', ...]
        i = 1
        while i < len(parts) - 1:
            step_id = parts[i].strip()
            body = parts[i + 1] if i + 1 < len(parts) else ""
            i += 2

            lines = body.split("\n")
            description = lines[0].strip()

            instr_lines = []
            exp_lines = []
            mode = None
            for line in lines[1:]:
                stripped = line.strip()
                if re.match(r'(?i)^instructions?\s*:', stripped):
                    mode = 'instructions'
                    instr_lines.append(re.sub(r'(?i)^instructions?\s*:\s*', '', stripped))
                elif re.match(r'(?i)^expected\s*:', stripped):
                    mode = 'expected'
                    exp_lines.append(re.sub(r'(?i)^expected\s*:\s*', '', stripped))
                elif mode == 'instructions' and stripped:
                    instr_lines.append(stripped)
                elif mode == 'expected' and stripped:
                    exp_lines.append(stripped)

            instructions = ' '.join(filter(None, instr_lines)).strip()
            expected = ' '.join(filter(None, exp_lines)).strip()

            if step_id and description:
                micro_steps.append({
                    'id': step_id,
                    'description': description,
                    'concrete_instructions': instructions or description,
                    'expected_outputs': expected or "Code executes without errors"
                })
        
        # Fallback: if parsing fails, create simple sequential steps
        if not micro_steps:
            print("[WARNING] Failed to parse micro-plan, using fallback")
            micro_steps = [
                {'id': '1', 'description': 'Complete current task',
                 'concrete_instructions': 'Complete the current task from the plan.',
                 'expected_outputs': 'Task completed successfully'}
            ]
        
        return micro_steps
    
    def _micro_assess_result(self, result: str, micro_step: dict, retry_count: int) -> dict:
        """Validate result of a micro-step execution using LLM assessment.
        
        Args:
            result: Execution output
            micro_step: Current micro-step dict
            retry_count: How many retries already attempted
            
        Returns:
            Dict with 'is_success', 'is_fixable', 'should_retry', 'error', 'stuck_reason', 'feedback'
        """
        # Use LLM to intelligently assess the execution result
        assessment_prompt = f"""[MICRO-STEP ASSESSMENT]
You are a rigorous code execution assessor enforcing EXACT micro-step completion.

MICRO-STEP DETAILS:
ID: {micro_step.get('id', 'unknown')}
Description: {micro_step.get('description', 'No description')}
Expected Output: {micro_step.get('expected_outputs', 'No specific expectation')}
Retry Count: {retry_count}

EXECUTION RESULT:
{result[:2000]}

TASK:
Assess if this EXACT micro-step was successfully completed.
Do NOT accept it if the agent performed a different or future step (e.g., ran UMAP when asked to initialize PCA params).

Respond in JSON format with these fields:
{{
  "is_success": true/false,  // Did the step achieve its exact goal?
  "should_retry": true/false,  // Should we retry if it failed?
  "is_fixable": true/false,  // Can the error be fixed by regenerating code?
  "error_summary": "concise error description or null",
  "retry_guidance": "specific guidance for retry or null",
  "reasoning": "brief explanation of your assessment"
}}

ASSESSMENT CRITERIA:
1. is_success = true ONLY if:
   - No errors occurred
   - The EXACT expected output was requested and produced
   - No off-step or future work was performed (if it jumped ahead, fail it)

2. should_retry = true if:
   - Error occurred or output is incomplete
   - The agent performed off-step work (feedback MUST tell it to only do the requested step)
   - Error is likely fixable (NameError, TypeError, AttributeError, etc.)
   - Retry count is still low (< 20)

3. should_retry = false if:
   - Missing dependencies (ModuleNotFoundError, ImportError)
   - Fundamental data/file missing
   - Same error repeated multiple times
   - Retry count too high

Respond ONLY with valid JSON."""

        try:
            # Get LLM for assessment (use EXECUTE phase LLM for consistency)
            llm = self._llm_for("EXECUTE")
            
            messages = [
                SystemMessage(content="You are an expert code execution assessor. Analyze execution results and provide JSON assessments."),
                HumanMessage(content=assessment_prompt)
            ]
            
            # Log the assessment
            self.phase_logger.set_phase("EXECUTE")
            self.phase_logger.log_prompt(messages)
            
            # Invoke LLM
            response = llm.invoke(messages)
            
            # Record usage
            try:
                self._record_llm_usage(messages + [response], llm_obj=llm)
            except Exception:
                pass
            
            msg = str(response.content).strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', msg, re.DOTALL)
            if json_match:
                assessment = json.loads(json_match.group())
                
                # Build return dict
                return {
                    'is_success': assessment.get('is_success', False),
                    'should_retry': assessment.get('should_retry', False),
                    'is_fixable': assessment.get('is_fixable', False),
                    'error': assessment.get('error_summary', 'Unknown error'),
                    'stuck_reason': None if assessment.get('should_retry') else assessment.get('error_summary'),
                    'feedback': assessment.get('retry_guidance', assessment.get('reasoning', 'No feedback provided')),
                    'reasoning': assessment.get('reasoning', '')
                }
            else:
                # Fallback if JSON parsing fails
                print("[WARNING] Failed to parse LLM assessment, using fallback logic")
                return self._fallback_assess_result(result, micro_step, retry_count)
                
        except Exception as e:
            print(f"[WARNING] LLM assessment failed: {e}, using fallback logic")
            return self._fallback_assess_result(result, micro_step, retry_count)
    
    def _fallback_assess_result(self, result: str, micro_step: dict, retry_count: int) -> dict:
        """Fallback pattern-based assessment when LLM assessment fails."""
        # Check for execution errors
        has_error = self._detect_execution_error(result)
        
        if has_error:
            # Categorize error type
            error_text = self._extract_error_summary(result)
            
            # Check if it's a fixable error
            fixable_patterns = [
                r"NameError",
                r"AttributeError", 
                r"KeyError",
                r"IndexError",
                r"TypeError.*argument",
                r"SyntaxError",
                r"IndentationError",
            ]
            
            is_fixable = any(re.search(pattern, error_text, re.IGNORECASE) for pattern in fixable_patterns)
            
            # Check for stuck condition (same error after retries)
            should_retry = is_fixable and retry_count < 20
            
            if "ModuleNotFoundError" in error_text or "ImportError" in error_text:
                should_retry = False
                stuck_reason = f"Missing dependency: {error_text[:100]}"
            else:
                stuck_reason = None if should_retry else "Unfixable error"
            
            return {
                'is_success': False,
                'should_retry': should_retry,
                'is_fixable': is_fixable,
                'error': error_text[:500],
                'stuck_reason': stuck_reason,
                'feedback': f"Error in step {micro_step.get('id', 'unknown')}: {error_text[:200]}"
            }
        
        # Check for expected outputs
        expected = micro_step.get('expected_outputs', '')
        has_expected = self._check_expected_patterns(result, expected)
        
        # Check for progress signals
        progress_signals = [
            r'Shape:.*\(',
            r'Loaded \d+',
            r'Saved to',
            r'Processing \d+',
            r'Complete',
            r'Successfully',
            r'\d+ records',
            r'File.*created',
        ]
        has_progress = any(re.search(pattern, result, re.IGNORECASE) for pattern in progress_signals)
        
        if not has_progress and len(result.strip()) < 20:
            return {
                'is_success': False,
                'should_retry': True,
                'is_fixable': True,
                'error': "No meaningful output produced",
                'stuck_reason': None,
                'feedback': f"Step {micro_step.get('id', 'unknown')} produced minimal output. Add print statements to verify execution."
            }
        
        # Success!
        return {
            'is_success': True,
            'should_retry': False,
            'is_fixable': None,
            'error': None,
            'stuck_reason': None,
            'feedback': f"Step {micro_step.get('id', 'unknown')} completed successfully"
        }
    
    def _consolidate_micro_execution_state(self, state: AgentState):
        """Consolidate all micro-step code and observations into final state and save.
        
        This is called only after all micro-steps complete or bailout occurs.
        """
        if not self.product_manager or not self.product_manager.current_state:
            return
        
        # Get accumulated code and observations
        accumulated_code = state["artifacts"].get("micro_accumulated_code", [])
        accumulated_observations = state["artifacts"].get("micro_accumulated_observations", [])
        
        # Merge all code into one block
        consolidated_code = "\n\n".join(accumulated_code) if accumulated_code else ""
        
        # Merge all observations into one
        consolidated_observations = "\n\n".join(accumulated_observations) if accumulated_observations else ""
        
        # Format the final output
        final_output = ""
        if consolidated_code:
            final_output += f"<execute>\n{consolidated_code}\n</execute>\n\n"
        if consolidated_observations:
            final_output += f"<observation>\n{consolidated_observations}\n</observation>"
        
        # Update the state with consolidated output
        self.product_manager.current_state.execute_output = final_output
        
        # Save the state (only once after all micro-steps complete or bailout)
        self.product_manager.state_manager.save_state(self.product_manager.current_state)
        
        print(f"  💾 Saved consolidated state with {len(accumulated_code)} code blocks and {len(accumulated_observations)} observations")

        # Persist consolidated working code to run_code_dir
        if consolidated_code and hasattr(self, "run_code_dir"):
            import time as _time
            os.makedirs(self.run_code_dir, exist_ok=True)
            ts_label = _time.strftime("%Y%m%dT%H%M%SZ", _time.gmtime())
            code_path = os.path.join(self.run_code_dir, f"execute_{ts_label}.py")
            with open(code_path, "w", encoding="utf-8") as _cf:
                _cf.write(f"# Auto-saved consolidated execute code — {ts_label}\n")
                _cf.write(f"# ARTIFACTS = {getattr(self, 'run_artifacts_dir', 'N/A')}\n")
                _cf.write(f"# FIGURES   = {getattr(self, 'run_figures_dir', 'N/A')}\n\n")
                _cf.write(consolidated_code + "\n")
            print(f"  📄 Code saved: {code_path}")
    
    def _check_expected_patterns(self, result: str, expected: str) -> bool:
        """Check if result contains expected patterns."""
        if not expected or expected == "Code executes without errors":
            return True
        
        # Simple keyword matching for now
        keywords = expected.lower().split()
        result_lower = result.lower()
        
        # At least 50% of keywords should be present
        matches = sum(1 for kw in keywords if kw in result_lower)
        return matches >= len(keywords) * 0.5
    
    def _extract_error_summary(self, result: str) -> str:
        """Extract concise error summary from execution output."""
        lines = result.split('\n')
        error_lines = []
        
        for i, line in enumerate(lines):
            if any(err in line for err in ['Error:', 'Exception:', 'Traceback']):
                # Get this line and next few lines
                error_lines.extend(lines[i:min(i+5, len(lines))])
                break
        
        if error_lines:
            return '\n'.join(error_lines)
        return result[:300]  # Fallback: first 300 chars
    
    def _validate_response_structure(self, msg: str) -> tuple[bool, str]:
        """Validate LLM response structure before parsing.
        
        Args:
            msg: The LLM response content
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        import re
        
        # Check for execute tags
        if "<execute>" not in msg:
            return (False, "No <execute> tag found in response. You must emit a code block.")
        
        # Check for mismatched or multiple execute blocks
        execute_open_count = msg.count("<execute>")
        execute_close_count = msg.count("</execute>")
        
        if execute_open_count > 1:
            return (False, f"Found {execute_open_count} <execute> tags. Emit exactly ONE code block.")
        
        if execute_close_count > 1:
            return (False, f"Found {execute_close_count} </execute> tags. Emit exactly ONE code block.")
        
        if execute_open_count != execute_close_count:
            return (False, f"Mismatched tags: {execute_open_count} opening, {execute_close_count} closing.")
        
        # Check if execute block is empty or just whitespace
        match = re.search(r"<execute>(.*?)</execute>", msg, re.DOTALL)
        if match:
            content = match.group(1).strip()
            if not content:
                return (False, "Execute block is empty. Include actual code.")
            
            # Check if it starts with common narrative patterns
            if re.match(r'^(\*\*|#{1,6}\s|\d+\.\s)', content):
                return (False, "Execute block starts with markdown/narrative formatting. Include ONLY code inside <execute> tags.")
        
        return (True, "")
    
    def _clean_extracted_code(self, code: str) -> str:
        """Clean extracted code by removing markdown artifacts and normalizing whitespace.
        
        Args:
            code: Raw code extracted from <execute> tags
            
        Returns:
            Cleaned code ready for execution
        """
        import re
        
        # Remove markdown code fence markers (```python, ```, etc.)
        code = re.sub(r'^```[a-zA-Z]*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n```$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```$', '', code, flags=re.MULTILINE)
        
        # Remove any stray backticks at start/end
        code = code.strip('`')
        
        # Strip leading/trailing whitespace but preserve internal structure
        code = code.strip()
        
        return code

    def _validate_python_syntax(self, code: str) -> tuple[bool, str]:
        """Validate Python code syntax before execution.
        
        Args:
            code: Python code to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        import ast
        
        try:
            ast.parse(code)
            return (True, "")
        except SyntaxError as e:
            error_msg = f"Line {e.lineno}: {e.msg}"
            if e.text:
                error_msg += f"\n  {e.text.strip()}"
                if e.offset:
                    error_msg += f"\n  {' ' * (e.offset - 1)}^"
            return (False, error_msg)
        except Exception as e:
            return (False, str(e))

    def _roll_memory(self, state, phase_tag: str):
        """Update long-term summary with the newest short-context, and add a breadcrumb event."""
        self._ensure_mem(state)
        mem = state["artifacts"]["memory"]
        prev = mem.get("long_term_summary", "")
        # Only summarize *non-system* messages the agent just appended
        msgs = [m for m in state.get("messages", []) if not isinstance(m, SystemMessage)]
        try:
            mem["long_term_summary"] = self._summarize_messages_for_long_term(prev, msgs)
        except Exception as e:
            print(f"[WARNING] Memory summarization failed: {e}. Keeping previous summary.")
            # Keep the previous summary if summarization fails
        mem["events"].append({
            "t": time.time(),
            "phase": phase_tag,
            "note": f"Rolled memory at {phase_tag}",
        })

    def _prune_history(self, state):
        """Keep just the last K conversational messages in RAM; the rest live in the long-term summary."""
        K = self.short_window
        msgs = state.get("messages", [])
        # Keep: last K non-system messages, but allow phases to add fresh System prompts as needed
        sys_msgs = [m for m in msgs if isinstance(m, SystemMessage)]
        non_sys  = [m for m in msgs if not isinstance(m, SystemMessage)]
        state["messages"] = sys_msgs + non_sys[-K:]

    
    
    
    
    
    
    
    
    
    

    def configure(self, self_critic=False, test_time_scale_round=0):
        """Configure the agent with the initial system prompt and workflow.

        Args:
            self_critic: Whether to enable self-critic mode
            test_time_scale_round: Number of rounds for test time scaling

        """
        # Store self_critic for later use
        self.self_critic = self_critic

        # Get data lake content
        data_lake_path = self.path + "/data_lake"
        data_lake_content = glob.glob(data_lake_path + "/*")
        data_lake_items = [x.split("/")[-1] for x in data_lake_content]

        # Store data_lake_dict as instance variable for use in retrieval
        self.data_lake_dict = data_lake_dict
        # Store library_content_dict directly without library_content
        self.library_content_dict = library_content_dict

        # Prepare tool descriptions
        tool_desc = {i: [x for x in j if x["name"] != "run_python_repl"] for i, j in self.module2api.items()}

        # Prepare data lake items with descriptions
        data_lake_with_desc = []
        for item in data_lake_items:
            description = self.data_lake_dict.get(item, f"Data lake item: {item}")
            data_lake_with_desc.append({"name": item, "description": description})

        # Add custom data items if they exist
        if hasattr(self, "_custom_data") and self._custom_data:
            for name, info in self._custom_data.items():
                data_lake_with_desc.append({"name": name, "description": info["description"]})

        # Prepare library content list including custom software
        library_content_list = list(self.library_content_dict.keys())
        if hasattr(self, "_custom_software") and self._custom_software:
            for name in self._custom_software:
                if name not in library_content_list:  # Avoid duplicates
                    library_content_list.append(name)

        # Generate the system prompt for initial configuration (is_retrieval=False)
        # Prepare custom resources for highlighting
        custom_tools = []
        if hasattr(self, "_custom_tools") and self._custom_tools:
            for name, info in self._custom_tools.items():
                custom_tools.append(
                    {
                        "name": name,
                        "description": info["description"],
                        "module": info["module"],
                    }
                )

        custom_data = []
        if hasattr(self, "_custom_data") and self._custom_data:
            for name, info in self._custom_data.items():
                custom_data.append({"name": name, "description": info["description"]})

        custom_software = []
        if hasattr(self, "_custom_software") and self._custom_software:
            for name, info in self._custom_software.items():
                custom_software.append({"name": name, "description": info["description"]})

        self.system_prompt = self._generate_system_prompt(
            tool_desc=tool_desc,
            data_lake_content=data_lake_with_desc,
            library_content_list=library_content_list,
            self_critic=self_critic,
            is_retrieval=False,
            custom_tools=custom_tools if custom_tools else None,
            custom_data=custom_data if custom_data else None,
            custom_software=custom_software if custom_software else None,
        )
        
        # Announces phases
        def _announce_phase(state: AgentState) -> None:
            phase = state.get("phase", "?")
            next_step = state.get("next_step", "NULL")
            tag = f"[PHASE] {phase} {next_step}"
            print(tag)                                # visible in stdout
            state["messages"].append(AIMessage(content=tag))  # visible in stream/log
        
        # Define the PLEAS nodes
        def plan(state: AgentState) -> AgentState:
            state["phase"] = "PLAN"
            _announce_phase(state)
            
            self._ensure_mem(state)
            lts = state["artifacts"]["memory"].get("long_term_summary","")
            
            # Start State
            if self.product_manager:
                self.product_manager.start_phase_state("PLAN", long_term_memory=lts)

            # Inject State Context if available (includes long-term memory)
            state_context_msg = ""
            if self.product_manager:
                ctx = self.product_manager.get_state_context()
                if ctx.get("formatted_state"):
                    state_context_msg = f"\n[PREVIOUS STATE]\n{ctx['formatted_state']}\n"

            messages = [SystemMessage(content=self._prompt_for("PLAN") + state_context_msg)] + state["messages"]
            llm = self._llm_for("PLAN")

            # Log with phase logger
            self.phase_logger.set_phase("PLAN")
            self.phase_logger.log_prompt(messages)
            log_llm_event("plan_prompt", messages)

            response = llm.invoke(messages)

            try:
                self._record_llm_usage(messages + [response], llm_obj=llm)
            except Exception:
                pass


            msg = str(response.content).strip()

            # Update State
            if self.product_manager:
                lts = state["artifacts"]["memory"].get("long_term_summary","")
                self.product_manager.end_phase_state("PLAN", msg, long_term_memory=lts)

            # Log with phase logger
            self.phase_logger.log_response(msg)
            log_llm_event("plan_response", msg)
            

            state["messages"].append(AIMessage(content=msg))

            # <<< NEW: roll long-term + prune short-term >>>
            self._roll_memory(state, "PLAN")
            self._prune_history(state)

            state["next_step"] = "learn"
            return state


        def learn(state: AgentState) -> AgentState:
            state["phase"] = "LEARN"
            _announce_phase(state)
            
            self._ensure_mem(state)
            lts = state["artifacts"]["memory"].get("long_term_summary","")
            
            # Start State
            if self.product_manager:
                self.product_manager.start_phase_state("LEARN", long_term_memory=lts)

            # Inject State Context if available (includes long-term memory)
            state_context_msg = ""
            if self.product_manager:
                ctx = self.product_manager.get_state_context()
                if ctx.get("formatted_state"):
                    state_context_msg = f"\n[PREVIOUS STATE]\n{ctx['formatted_state']}\n"

            messages = [SystemMessage(content=self._prompt_for("LEARN") + state_context_msg)] + state["messages"]

            # Log with phase logger
            self.phase_logger.set_phase("LEARN")
            self.phase_logger.log_prompt(messages)
            log_llm_event("learn_prompt", messages)

            llm = self._llm_for("LEARN")
            response = llm.invoke(messages)

            try:
                self._record_llm_usage(messages + [response], llm_obj=llm)
            except Exception:
                pass


            msg = str(response.content).strip()

            # Update State
            if self.product_manager:
                lts = state["artifacts"]["memory"].get("long_term_summary","")
                self.product_manager.end_phase_state("LEARN", msg, long_term_memory=lts)

            # Log with phase logger
            self.phase_logger.log_response(msg)
            log_llm_event("learn_response", msg)
            
            
            state["messages"].append(AIMessage(content=msg))

            # <<< NEW >>>
            self._roll_memory(state, "LEARN")
            self._prune_history(state)

            state["next_step"] = "execute"
            return state

        
            '''
            # reuse existing descriptors you already build elsewhere
            data_lake_path = self.path + "/data_lake"
            data_lake_items = [x.split("/")[-1] for x in glob.glob(data_lake_path + "/*")]
            data_lake_descs = [{"name": i, "description": self.data_lake_dict.get(i, f"Data lake item: {i}")} for i in data_lake_items]
            library_descs = [{"name": k, "description": v} for k, v in self.library_content_dict.items()]

            selected_tool_names = []
            if self.use_tool_retriever:
                all_tools = self.tool_registry.tools if hasattr(self, "tool_registry") else []
                
                # New
                query = state["messages"][-1].content
                selected = self.retriever.prompt_based_retrieval(
                    query,
                    {
                        "tools": [
                            {"name": getattr(t, "name", None), "description": getattr(t, "description", "")}
                            for t in all_tools
                        ],
                        "data_lake": data_lake_descs,
                        "libraries": library_descs,
                    },
                    self.llm,   # keep this if retriever signature allows llm
                )
                # new
                selected_tool_names = [t["name"] for t in selected.get("tools", [])]

            state["artifacts"]["selected_tool_names"] = selected_tool_names
            state["artifacts"]["data_lake_descs"] = data_lake_descs
            state["artifacts"]["library_descs"] = library_descs

            summary = (
                "Selected tools: " + (", ".join(selected_tool_names) or "(none)") + "\n"
                "Data lake: " + ", ".join(i["name"] for i in data_lake_descs) + "\n"
                "Libraries: " + ", ".join(i["name"] for i in library_descs)
            )
            state["messages"].append(HumanMessage(content=f"[LEARN] Grounding resources:\n{summary}"))
            state["next_step"] = "generate"
            '''
        def execute(state: AgentState) -> AgentState:
            
            state["phase"] = "EXECUTE"
            _announce_phase(state)
            
            # Start State
            if self.product_manager:
                self._ensure_mem(state)
                lts = state["artifacts"]["memory"].get("long_term_summary","")
                self.product_manager.start_phase_state("EXECUTE", long_term_memory=lts)
            
            # Initialize micro-plan state if not present
            if "artifacts" not in state:
                state["artifacts"] = {}
            
            # If ASSESS provided a revised micro-plan, clear the preserved bailout
            # plan so _generate_micro_plan() is called fresh with that context.
            # The revised plan + micro_plan_state will both be injected into the
            # generation prompt so the new plan is fully informed.
            if state["artifacts"].get("revised_micro_plan"):
                print(f"\n[MICRO-PLANNING] ASSESS provided a revised plan — regenerating with updated context.")
                state["artifacts"]["micro_plan"] = None  # force re-generation

            # If a micro-plan is already present (carried over from a bailout),
            # skip all planning/re-generation and go straight to execution.
            elif state["artifacts"].get("micro_plan"):
                remaining = state["artifacts"]["micro_plan"]
                print(f"\n[RESUMING MICRO-EXECUTION] Continuing with {len(remaining)} remaining step(s) — skipping re-planning.")
                return _execute_with_microplan(self, state)

            # Check if we need to generate a micro-plan (first time in EXECUTE phase, or after previous micro-plan completed/was cleared)
            if "micro_plan" not in state["artifacts"] or state["artifacts"]["micro_plan"] is None:

                # ── Print the full plan so the user can see where we are ──────
                full_plan_text = ""
                if self.product_manager and self.product_manager.current_state:
                    for entry in reversed(self.product_manager.current_state.previous_states or []):
                        if (entry.get("phase") or "").upper() == "PLAN" and entry.get("output"):
                            full_plan_text = str(entry["output"])
                            break
                if not full_plan_text:
                    for msg in reversed(state.get("messages", [])):
                        content = str(getattr(msg, "content", ""))
                        import re as _re
                        if _re.search(r'(?i)(step\s*\d|\d+\.\s+\w)', content) and len(content) > 200:
                            full_plan_text = content
                            break

                print("\n" + "="*80)
                print("FULL PLAN")
                print("="*80)
                print(full_plan_text if full_plan_text else "(plan not found in state)")
                print("="*80 + "\n")

                # ── Extract the current step and generate micro-plan ──────────
                print("="*80)
                print("MICRO-PLANNING: Breaking down current step into sub-tasks")
                print("="*80)

                # Extract current PLAN step (full step + all sub-steps) from messages
                plan_context = self._extract_current_plan_step(state)

                print("\nCURRENT STEP (to be executed now):")
                print("-"*60)
                print(plan_context)
                print("-"*60 + "\n")

                # Generate concrete micro-plan
                try:
                    micro_plan = self._generate_micro_plan(state, plan_context)
                    state["artifacts"]["micro_plan"] = micro_plan
                    state["artifacts"]["micro_step_index"] = 0
                    state["artifacts"]["micro_step_retries"] = {}
                    # Consumed — clear so they don't pollute the next fresh planning call
                    state["artifacts"].pop("revised_micro_plan", None)
                    state["artifacts"].pop("micro_plan_state", None)

                    # ── Print full concrete plan for the user ─────────────────
                    print("\n" + "="*80)
                    print(f"CONCRETE EXECUTION PLAN  ({len(micro_plan)} steps)")
                    print("="*80)
                    for ms in micro_plan:
                        print(f"\n  [{ms['id']}]  {ms['description']}")
                        instr = ms.get('concrete_instructions', '')
                        if instr and instr != ms['description']:
                            # Word-wrap at 100 chars
                            words, line_buf, wrapped = instr.split(), [], []
                            for w in words:
                                line_buf.append(w)
                                if len(' '.join(line_buf)) > 100:
                                    wrapped.append('         ' + ' '.join(line_buf))
                                    line_buf = []
                            if line_buf:
                                wrapped.append('         ' + ' '.join(line_buf))
                            print('         Instructions: ' + '\n'.join(wrapped).lstrip())
                        print(f"         Expected:     {ms.get('expected_outputs', '')}")
                    print("\n" + "="*80 + "\n")
                except Exception as e:
                    print(f"[WARNING] Micro-plan generation failed: {e}")
                    print("Falling back to single-step execution")
                    state["artifacts"]["micro_plan"] = None
            
            # Check if using micro-planning
            use_micro_planning = (
                state["artifacts"].get("micro_plan") is not None 
                and len(state["artifacts"]["micro_plan"]) > 0
            )
            
            if use_micro_planning:
                return _execute_with_microplan(self, state)
            else:
                return _execute_single_step(self, state)
        
        def _execute_single_step(self, state: AgentState) -> AgentState:
            """Original single-step execution (fallback when micro-planning not used)."""
            
            phase_hint = f"""
[PHASE: EXECUTE]
OUTPUT DIRECTORY RULES — THE RUN DIRECTORY HAS 3 SUBDIRS AND NO FILES LIVE ANYWHERE ELSE:

  AUTHORIZED DESTINATIONS (the ONLY 3 valid places to write files):
    artifacts/  →  {self.run_artifacts_dir}
    figures/    →  {self.run_figures_dir}
    code/       →  {self.run_code_dir}

In every code block, resolve paths EXACTLY like this:
  import os, json, glob
  ARTIFACTS = os.environ.get("BIOPLEASE_ARTIFACTS_DIR", "{self.run_artifacts_dir}")
  FIGURES   = os.environ.get("BIOPLEASE_FIGURES_DIR",   "{self.run_figures_dir}")
  os.makedirs(ARTIFACTS, exist_ok=True)
  os.makedirs(FIGURES,   exist_ok=True)

• ALL data outputs (CSV, TSV, JSON, Parquet, PKL, TXT, HDF5 …) → ARTIFACTS
• ALL plots/images → FIGURES
• NEVER save any file to RUN_DIR itself, cwd, /tmp, or any other path
• The main run directory MUST remain clean — no stray files
• Always print after saving: print(f"Saved: {{path}}")
• To locate a previously generated file: glob.glob(os.path.join(ARTIFACTS, "**", "*keyword*"), recursive=True)

Figure saving — after saving a plot at `img_path`, also do BOTH:
  1) with open(os.path.join(FIGURES, "manifest.jsonl"), "a", encoding="utf-8") as _f:
         _f.write(json.dumps({{"path": img_path, "caption": caption, "label": "Figure"}}) + "\\n")
  2) print(f"<<FIGURE:{{img_path}}|{{caption}}>>"
"""

            # Inject State Context if available (summarized for EXECUTE)
            state_context_msg = ""
            if self.product_manager:
                ctx = self.product_manager.get_state_context(for_phase="EXECUTE")
                if ctx.get("formatted_state"):
                    state_context_msg = f"\n[PREVIOUS STATE]\n{ctx['formatted_state']}\n"
            
            messages = [SystemMessage(content=phase_hint)] + [SystemMessage(content=self._prompt_for("EXECUTE") + state_context_msg)] + state["messages"]
            llm = self._llm_for("EXECUTE")

            # Log with phase logger
            self.phase_logger.set_phase("EXECUTE")
            self.phase_logger.log_prompt(messages)
            log_llm_event("execute_prompt", messages)

            response = llm.invoke(messages)

            try:
                self._record_llm_usage(messages + [response], llm_obj=llm)
            except Exception:
                pass

            msg = str(response.content)
            
            return _process_execute_response(self, state, msg)
        
        def _execute_with_microplan(self, state: AgentState) -> AgentState:
            """Execute using micro-planning: loop through sub-steps with validation."""
            
            MAX_MICRO_STEPS = 20
            RETRY_LIMIT_PER_STEP = 20
            
            micro_plan = state["artifacts"]["micro_plan"]
            start_idx = state["artifacts"].get("micro_step_index", 0)
            
            # Initialize accumulators for code and observations
            if "micro_accumulated_code" not in state["artifacts"]:
                state["artifacts"]["micro_accumulated_code"] = []
            if "micro_accumulated_observations" not in state["artifacts"]:
                state["artifacts"]["micro_accumulated_observations"] = []
            
            print(f"\n{'='*80}")
            print(f"MICRO-EXECUTION: Step {start_idx + 1}/{len(micro_plan)}")
            print(f"{'='*80}\n")
            
            # Loop through micro-steps
            for step_idx in range(start_idx, min(len(micro_plan), MAX_MICRO_STEPS)):
                micro_step = micro_plan[step_idx]
                print(f"\n[MICRO-STEP {step_idx + 1}] {micro_step['description']}")
                if micro_step.get('expected_outputs'):
                    print(f"  Expected: {micro_step['expected_outputs']}")
                
                # Get retry count for this step — always reset to 0 on entry
                retry_key = f"step_{step_idx}"
                state["artifacts"]["micro_step_retries"][retry_key] = 0
                retry_count = 0
                
                # Try executing this micro-step (with retries)
                while retry_count < RETRY_LIMIT_PER_STEP:
                    print(f"  🔄 Attempt {retry_count + 1}/{RETRY_LIMIT_PER_STEP}")
                    
                    if retry_count == 0:
                        # Append the initial prompt for this micro-step to the message history
                        # so the LLM gets proper conversation context and we avoid back-to-back AI messages.
                        concrete = micro_step.get('concrete_instructions') or micro_step['description']
                        initial_prompt = f"[MICRO-STEP {micro_step['id']}]\nTask: {micro_step['description']}\\n\\nConcrete instructions:\\n{concrete}\\n\\nSuccess criteria: {micro_step.get('expected_outputs', 'Code executes without errors')}\\n\\nANTI-DRIFT RULES:\\n- DO NOT perform future steps.\\n- DO NOT rerun previous steps unless explicitly asked.\\n- DO NOT substitute \"equivalent progress\"; perform EXACTLY this step and stop.\\n- FAIL rather than jump ahead.\\n\\nWrite ONLY the Python code needed for THIS step in <execute>...</execute> tags.\\nFollow the concrete instructions above exactly — do not skip ahead or do other steps.\\nKeep it focused."
                        state["messages"].append(HumanMessage(content=initial_prompt))

                    # Generate and execute code for this specific micro-step
                    code, observation, success = _execute_micro_step(self, state, micro_step, retry_count)
                    
                    # Check if this is a structure/syntax/format error (skip LLM assessment for these)
                    is_structure_error = any(err in observation for err in [
                        "STRUCTURE ERROR:", "SYNTAX ERROR:", "FORMAT ERROR:", "PARSING ERROR:"
                    ])
                    
                    if is_structure_error:
                        # Skip LLM assessment for obvious structural errors - retry immediately
                        print(f"  ⚠ Structure/syntax error detected - retrying without assessment")
                        retry_count += 1
                        state["artifacts"]["micro_step_retries"][retry_key] = retry_count
                        
                        # Add brief feedback for retry
                        feedback_msg = f"""[MICRO-STEP {step_idx + 1} RETRY {retry_count}/{RETRY_LIMIT_PER_STEP}]
Task: {micro_step['description']}

{observation[:300]}

Fix the structural issue and emit corrected code in <execute>...</execute> tags."""
                        state["messages"].append(HumanMessage(content=feedback_msg))
                        continue
                    
                    # Log the observation for transparency (only for actual execution)
                    print(f"  📝 Observation: {observation[:200]}...")  # Show first 200 chars
                    
                    # Validate the result using LLM assessment (ONLY for actual execution results)
                    validation = self._micro_assess_result(observation, micro_step, retry_count)
                    
                    # Log the assessment reasoning
                    if validation.get('reasoning'):
                        print(f"  📊 Assessment: {validation['reasoning']}")
                    
                    if validation['is_success']:
                        print(f"  ✓ Step {step_idx + 1} completed successfully")
                        # Accumulate successful code and observation
                        state["artifacts"]["micro_accumulated_code"].append(f"# Micro-step {step_idx + 1}: {micro_step['description']}\n{code}")
                        state["artifacts"]["micro_accumulated_observations"].append(f"[Step {step_idx + 1}] {observation}")
                        
                        # Add success observation to the conversation history so the LLM sees the result
                        success_msg = f"[Step {step_idx + 1} SUCCESS]\nObservation:\n{observation[:1000]}"
                        if len(observation) > 1000:
                            success_msg += "...\n(observation truncated)"
                        state["messages"].append(HumanMessage(content=success_msg))

                        # Move to next step
                        state["artifacts"]["micro_step_index"] = step_idx + 1
                        state["artifacts"]["micro_step_retries"][retry_key] = 0
                        break
                    elif validation['should_retry'] and retry_count < RETRY_LIMIT_PER_STEP - 1:
                        # LLM says this is retryable
                        retry_count += 1
                        state["artifacts"]["micro_step_retries"][retry_key] = retry_count
                        print(f"  ⚠ LLM assessment: Should retry ({retry_count}/{RETRY_LIMIT_PER_STEP})")
                        
                        # Add feedback message for retry
                        feedback_msg = f"""[MICRO-STEP {step_idx + 1} RETRY {retry_count}/{RETRY_LIMIT_PER_STEP}]
Task: {micro_step['description']}
Expected: {micro_step.get('expected_outputs', 'Progress on this sub-task')}

Previous attempt observation:
{validation['error'][:500] if validation['error'] else observation[:500]}

LLM Assessment: {validation['feedback']}

Write corrected code for THIS STEP ONLY in <execute>...</execute> tags."""
                        state["messages"].append(HumanMessage(content=feedback_msg))
                        continue  # retry loop
                    else:
                        # LLM says this should not be retried
                        print(f"  ✗ Step {step_idx + 1} failed: {validation.get('stuck_reason', 'LLM assessment: should not retry')}")
                        state["artifacts"]["bailout_reason"] = validation.get('stuck_reason', 'Unfixable error in micro-step')
                        state["artifacts"]["bailout_step"] = step_idx
                        
                        # Add summary message
                        summary_msg = f"""[MICRO-PLANNING BAILOUT]
Failed at step {step_idx + 1}/{len(micro_plan)}: {micro_step['description']}
Reason: {state["artifacts"]["bailout_reason"]}
LLM Assessment: {validation['feedback']}
Completed {step_idx} of {len(micro_plan)} steps successfully.

Proceeding to assessment with partial results."""
                        state["messages"].append(AIMessage(content=summary_msg))
                        
                        # Build rich micro_plan_state and preserve remaining steps
                        self._build_micro_plan_state(
                            state, micro_plan, step_idx,
                            bailout_reason=state["artifacts"].get("bailout_reason", "Unfixable error"),
                            failed_step=micro_step,
                        )

                        # Consolidate and save state before bailout
                        self._consolidate_micro_execution_state(state)

                        # Route to assess with bailout info
                        self._roll_memory(state, "EXECUTE")
                        self._prune_history(state)
                        state["next_step"] = "assess"
                        return state
                
                # Check if we exhausted retries without the LLM approving continuation
                if retry_count >= RETRY_LIMIT_PER_STEP:
                    print(f"  ✗ Step {step_idx + 1} failed: max retries exceeded")
                    
                    # Get final assessment from LLM
                    try:
                        final_assessment = self._micro_assess_result(observation, micro_step, retry_count)
                    except Exception as _ae:
                        print(f"[WARNING] _micro_assess_result crashed: {_ae} — using empty assessment")
                        final_assessment = {'feedback': str(_ae)}
                    
                    state["artifacts"]["bailout_reason"] = f"Max retries ({RETRY_LIMIT_PER_STEP}) exceeded for step {step_idx + 1}"
                    state["artifacts"]["bailout_step"] = step_idx
                    
                    summary_msg = f"""[MICRO-PLANNING BAILOUT]
Failed at step {step_idx + 1}/{len(micro_plan)}: {micro_step['description']}
Reason: Max retries exceeded
Final LLM Assessment: {final_assessment.get('feedback', 'No assessment available')}
Completed {step_idx} of {len(micro_plan)} steps successfully.

Proceeding to assessment with partial results."""
                    state["messages"].append(AIMessage(content=summary_msg))
                    
                    # Build rich micro_plan_state and preserve remaining steps
                    self._build_micro_plan_state(
                        state, micro_plan, step_idx,
                        bailout_reason=state["artifacts"].get("bailout_reason", f"Max retries exceeded at step {step_idx + 1}"),
                        failed_step=micro_step,
                    )

                    # Consolidate and save state before bailout
                    self._consolidate_micro_execution_state(state)

                    self._roll_memory(state, "EXECUTE")
                    self._prune_history(state)
                    state["next_step"] = "assess"
                    return state
            
            # Check if we hit max steps limit
            if step_idx + 1 >= MAX_MICRO_STEPS and step_idx + 1 < len(micro_plan):
                print(f"  ⚠ Reached max micro-steps limit ({MAX_MICRO_STEPS})")
                state["artifacts"]["bailout_reason"] = f"Reached max micro-steps limit ({MAX_MICRO_STEPS})"
                state["artifacts"]["bailout_step"] = step_idx
            
            # All steps completed successfully (or hit limit)
            print(f"\n{'='*80}")
            print(f"MICRO-EXECUTION COMPLETE")
            print(f"Successfully completed {state['artifacts']['micro_step_index']} of {len(micro_plan)} steps")
            print(f"{'='*80}\n")
            
            # Transition message to cleanly break out of step generation code context
            state["messages"].append(HumanMessage(content="[MACRO-EXECUTE] All micro-steps completed. Transitioning to ASSESS phase. Review all previous state context."))

            # Consolidate and save final state
            self._consolidate_micro_execution_state(state)
            
            # Clean up micro-plan state for next EXECUTE phase.
            # If we bailed out due to the step-count limit, use _build_micro_plan_state
            # to record completed/remaining context for ASSESS and preserve remaining steps.
            bailout_reason = state["artifacts"].get("bailout_reason") or ""
            if "max micro-steps limit" in bailout_reason:
                # step_idx is the last completed step; remaining start at step_idx + 1
                self._build_micro_plan_state(
                    state, micro_plan, step_idx,
                    bailout_reason=bailout_reason,
                    failed_step=None,
                    remaining_from_idx=step_idx + 1,
                )
            else:
                # All steps truly finished — clear everything
                state["artifacts"]["micro_plan"] = None
                state["artifacts"]["micro_plan_state"] = None
                state["artifacts"]["micro_step_index"] = 0
                state["artifacts"]["micro_step_retries"] = {}
                state["artifacts"]["micro_accumulated_code"] = []
                state["artifacts"]["micro_accumulated_observations"] = []
            
            self._roll_memory(state, "EXECUTE")
            self._prune_history(state)
            
            # Route to assess
            if "artifacts" not in state:
                state["artifacts"] = {}
            assess_count = state["artifacts"].get("assess_iteration_count", 0)
            assess_count += 1
            state["artifacts"]["assess_iteration_count"] = assess_count
            
            if assess_count % 5 == 0:
                state["next_step"] = "mini_share"
            else:
                state["next_step"] = "assess"
            
            return state
        
        def _execute_micro_step(self, state: AgentState, micro_step: dict, retry_count: int) -> tuple[str, str, bool]:
            """Execute a single micro-step and return (code, observation, success)."""
            
            phase_hint = f"""
[PHASE: EXECUTE - MICRO-STEP]
OUTPUT DIRECTORY RULES — THE RUN DIRECTORY HAS 3 SUBDIRS AND NO FILES LIVE ANYWHERE ELSE:

  AUTHORIZED DESTINATIONS (the ONLY 3 valid places to write files):
    artifacts/  →  {self.run_artifacts_dir}
    figures/    →  {self.run_figures_dir}
    code/       →  {self.run_code_dir}

In every code block, resolve paths EXACTLY like this:
  import os, json, glob
  ARTIFACTS = os.environ.get("BIOPLEASE_ARTIFACTS_DIR", "{self.run_artifacts_dir}")
  FIGURES   = os.environ.get("BIOPLEASE_FIGURES_DIR",   "{self.run_figures_dir}")
  os.makedirs(ARTIFACTS, exist_ok=True)
  os.makedirs(FIGURES,   exist_ok=True)

• ALL data outputs (CSV, TSV, JSON, Parquet, PKL, TXT, HDF5 …) → ARTIFACTS
• ALL plots/images → FIGURES
• NEVER save any file to RUN_DIR itself, cwd, /tmp, or any other path
• The main run directory MUST remain clean — no stray files
• Always print after saving: print(f"Saved: {{path}}")
• To locate a previously generated file: glob.glob(os.path.join(ARTIFACTS, "**", "*keyword*"), recursive=True)

Figure saving — after saving a plot at `img_path`, also do BOTH:
  1) with open(os.path.join(FIGURES, "manifest.jsonl"), "a", encoding="utf-8") as _f:
         _f.write(json.dumps({{"path": img_path, "caption": caption, "label": "Figure"}}) + "\\n")
  2) print(f"<<FIGURE:{{img_path}}|{{caption}}>>"
"""
            
            # Add phase instructions to the history
            messages = [
                SystemMessage(content=phase_hint),
                SystemMessage(content=self._prompt_for("EXECUTE")),
            ] + state["messages"]
            
            llm = self._llm_for("EXECUTE")
            
            # Log
            self.phase_logger.set_phase("EXECUTE")
            self.phase_logger.log_prompt(messages)
            log_llm_event("execute_micro_step", messages)
            
            # Invoke LLM
            response = llm.invoke(messages)
            
            try:
                self._record_llm_usage(messages + [response], llm_obj=llm)
            except Exception:
                pass
            
            msg = str(response.content)
            
            # Process the response (validate, extract, execute)
            return _process_execute_response(self, state, msg, is_micro_step=True)
        
        def _process_execute_response(self, state: AgentState, msg: str, is_micro_step: bool = False) -> Union[AgentState, tuple[str, str, bool]]:
            """Process execute response: validate, extract code, execute. 
            
            Returns:
            - AgentState (for single-step mode): complete state with next_step set  
            - tuple[str, str, bool] (for micro-step mode): (code, observation, success) for accumulation
            """
            
            # Pre-parse validation: check response structure before attempting extraction
            structure_valid, structure_error = self._validate_response_structure(msg)
            if not structure_valid:
                print(f"[STRUCTURE ERROR] Response has structural issues: {structure_error}")
                if is_micro_step:
                    # For micro-steps, return error result (code, observation, success)
                    return ("", f"STRUCTURE ERROR: {structure_error}", False)
                else:
                    # For single-step, add feedback and retry
                    state["messages"].append(HumanMessage(
                        content=f"[STRUCTURE ERROR] {structure_error}\\n\\nREMINDER: Emit ONE <execute>CODE ONLY</execute> block. Put explanations BEFORE the execute tag, not inside it."
                    ))
                    state["next_step"] = "execute"
                    return state

            # Update State (only for single-step mode)
            if not is_micro_step and self.product_manager:
                self._ensure_mem(state)
                lts = state["artifacts"]["memory"].get("long_term_summary","")
                self.product_manager.end_phase_state("EXECUTE", msg, long_term_memory=lts)

            # Log with phase logger
            self.phase_logger.log_response(msg)
            log_llm_event("execute_response", msg)
            
            # Check for incomplete tags and fix them
            if "<execute>" in msg and "</execute>" not in msg:
                msg += "</execute>"
            
            state["messages"].append(AIMessage(content=msg.strip()))
            
            print(f"[GENERATE] response={msg}")

            execute_match = re.search(r"<execute>(.*?)</execute>", msg, re.DOTALL)
            

            if execute_match:
                pass
            else:
                print("parsing error...")
                if is_micro_step:
                    # For micro-steps, return parsing error (code, observation, success)
                    return ("", "PARSING ERROR: No <execute> tags found", False)
                else:
                    # For single-step, retry logic
                    tries = state["artifacts"].get("parse_retries", 0) if "artifacts" in state else 0
                    tries += 1
                    if "artifacts" not in state:
                        state["artifacts"] = {}
                    state["artifacts"]["parse_retries"] = tries

                    if tries >= 3:
                        print("Detected repeated parsing errors, ending conversation")
                        state["messages"].append(AIMessage(content="Terminating: model failed to emit <execute> after 3 retries."))
                        state["next_step"] = "end"
                    else:
                        state["messages"].append(HumanMessage(
                            content="[FORMAT ERROR] Emit exactly ONE of: <execute>...</execute> (code only). No other tags."
                        ))
                        state["next_step"] = "execute"
                    return state
            
            # Ensure run-scoped environment vars are present
            os.environ.setdefault("BIOPLEASE_DATA", self.path)
            if hasattr(self, "run_dir"):
                os.environ.setdefault("BIOPLEASE_RUN_DIR", self.run_dir)

            
            last_message = state["messages"][-1].content
            # Only add the closing tag if it's not already there
            if "<execute>" in last_message and "</execute>" not in last_message:
                last_message += "</execute>"

            execute_match = re.search(r"<execute>(.*?)</execute>", last_message, re.DOTALL)
            if execute_match:
                code = execute_match.group(1)
                
                # Pre-execution format validation: catch common mistakes before cleaning
                format_valid, format_error = self._validate_code_format(code)
                if not format_valid:
                    print(f"[FORMAT ERROR] Code block contains formatting issues: {format_error}")
                    if is_micro_step:
                        return (code, f"FORMAT ERROR: {format_error}", False)
                    else:
                        state["messages"].append(AIMessage(
                            content=f"<observation>FORMAT ERROR: {format_error}\n\nPlease emit ONLY executable code inside <execute> tags. No narrative text, markdown, or explanations inside the tags.</observation>"
                        ))
                        state["next_step"] = "execute"
                        return state
                
                # Clean the extracted code: remove markdown artifacts and strip whitespace
                code = self._clean_extracted_code(code)
                
                # Validate syntax before execution (for Python code)
                if not (code.strip().startswith("#!R") or code.strip().startswith("# R") or 
                        code.strip().startswith("#!BASH") or code.strip().startswith("# Bash") or
                        code.strip().startswith("#!CLI")):
                    syntax_valid, syntax_error = self._validate_python_syntax(code)
                    if not syntax_valid:
                        print(f"[SYNTAX ERROR] Code failed validation: {syntax_error}")
                        if is_micro_step:
                            return (code, f"SYNTAX ERROR: {syntax_error}", False)
                        else:
                            state["messages"].append(AIMessage(
                                content=f"<observation>SYNTAX ERROR: {syntax_error}\n\nFix the syntax and emit corrected code in <execute>...</execute> tags.</observation>"
                            ))
                            state["next_step"] = "execute"
                            return state

                # Snapshot files before execution to track new ones
                def _get_tracked_files():
                    found_files = set()
                    import os
                    for rdir in [getattr(self, "run_artifacts_dir", ""), getattr(self, "run_figures_dir", ""), getattr(self, "run_code_dir", "")]:
                        if rdir and os.path.exists(rdir):
                            for root, _, fnames in os.walk(rdir):
                                for fname in fnames:
                                    found_files.add(os.path.join(root, fname))
                    return found_files
                
                files_before = _get_tracked_files()

                # Set timeout duration (10 minutes = 600 seconds)
                timeout = self.timeout_seconds

                # Check if the code is R code
                if (
                    code.strip().startswith("#!R")
                    or code.strip().startswith("# R code")
                    or code.strip().startswith("# R script")
                ):
                    # Remove the R marker and run as R code
                    r_code = re.sub(r"^#!R|^# R code|^# R script", "", code, 1).strip()  # noqa: B034
                    result = run_with_timeout(run_r_code, [r_code], timeout=timeout)
                # Check if the code is a Bash script or CLI command
                elif (
                    code.strip().startswith("#!BASH")
                    or code.strip().startswith("# Bash script")
                    or code.strip().startswith("#!CLI")
                ):
                    # Handle both Bash scripts and CLI commands with the same function
                    if code.strip().startswith("#!CLI"):
                        # For CLI commands, extract the command and run it as a simple bash script
                        cli_command = re.sub(r"^#!CLI", "", code, 1).strip()  # noqa: B034
                        # Remove any newlines to ensure it's a single command
                        cli_command = cli_command.replace("\n", " ")
                        result = run_with_timeout(run_bash_script, [cli_command], timeout=timeout)
                    else:
                        # For Bash scripts, remove the marker and run as a bash script
                        bash_script = re.sub(r"^#!BASH|^# Bash script", "", code, 1).strip()  # noqa: B034
                        result = run_with_timeout(run_bash_script, [bash_script], timeout=timeout)
                # Otherwise, run as Python code
                else:
                    # Inject custom functions into the Python execution environment
                    self._inject_custom_functions_to_repl()
                    result = run_with_timeout(run_python_repl, [code], timeout=timeout)

                # Track new files
                files_after = _get_tracked_files()
                new_files = files_after - files_before
                if new_files:
                    if "created_files" not in state["artifacts"]:
                        state["artifacts"]["created_files"] = []
                    state["artifacts"]["created_files"].extend(list(new_files))
                    # Deduplicate just in case
                    state["artifacts"]["created_files"] = list(set(state["artifacts"]["created_files"]))
                    
                    nl = "\n"
                    result += f"\n\n[SYSTEM NOTE: The following new files were successfully saved to your run directory:\n{nl.join(new_files)}\n]"

                if len(result) > 10000:
                    result = (
                        "The output is too long to be added to context. Here are the first 10K characters...\n"
                        + result[:10000]
                    )
                observation = f"\n<observation>{result}</observation>"
                
                # For micro-steps, DON'T save state yet - accumulate instead
                if is_micro_step:
                    # Log observation to EXECUTE logs directory
                    self.phase_logger.log_observation(
                        content=result,
                        metadata={"type": "micro_step_execution"}
                    )
                    # Check for errors
                    execution_has_error = self._detect_execution_error(result)
                    # Return (code, observation, success) for accumulation
                    return (code, result, not execution_has_error)
                
                # For single-step mode, add to messages and save state
                state["messages"].append(AIMessage(content=observation.strip()))
                
                # Update state with the observation output (single-step only)
                if self.product_manager and self.product_manager.current_state:
                    current_exec_output = self.product_manager.current_state.execute_output or ""
                    # Append observation to the execute output
                    updated_output = current_exec_output + "\n\n" + observation.strip()
                    self.product_manager.current_state.execute_output = updated_output
                    self.product_manager.state_manager.save_state(self.product_manager.current_state)
                
                # For single-step: auto-retry mechanism
                execution_has_error = self._detect_execution_error(result)
                
                # Only auto-retry if enabled
                if execution_has_error and self.auto_retry_on_error:
                    # Initialize or get retry counter
                    if "artifacts" not in state:
                        state["artifacts"] = {}
                    exec_retries = state["artifacts"].get("execute_retries", 0)
                    max_exec_retries = self.max_execute_retries
                    
                    if exec_retries < max_exec_retries:
                        exec_retries += 1
                        state["artifacts"]["execute_retries"] = exec_retries
                        
                        print(f"[AUTO-RETRY] Execution error detected. Retry {exec_retries}/{max_exec_retries}")
                        
                        # Extract key parts of the error for focused debugging
                        error_lines = result.split('\n')
                        error_summary = '\n'.join(error_lines[:15])  # First 15 lines of error
                        if len(error_lines) > 15:
                            error_summary += '\n... (error continues)'
                        
                        # Add debugging hint with the failed code AND error for context
                        debug_hint = f"""[AUTO-DEBUG {exec_retries}/{max_exec_retries}] CODE EXECUTION FAILED

=== FAILED CODE ===
{code}

=== ERROR OUTPUT ===
{error_summary}

=== INSTRUCTIONS ===
The code above failed with the error shown. Analyze:
1. What is the SPECIFIC error (error type and message)?
2. Which LINE or OPERATION failed?
3. What is the ROOT CAUSE (missing variable, wrong type, file not found, etc.)?
4. How to FIX it (be specific)?

Now write CORRECTED code in <execute>...</execute> tags. 
Focus ONLY on fixing the error - don't change working parts.
If you need to debug, add print statements to check values."""
                        
                        state["messages"].append(HumanMessage(content=debug_hint))
                        
                        # Roll memory but DON'T prune during retries - keep full context
                        self._roll_memory(state, "EXECUTE")
                        # Skip pruning during retries to preserve debugging context
                        if exec_retries >= max_exec_retries:
                            self._prune_history(state)
                        
                        # Route back to EXECUTE for another attempt
                        state["next_step"] = "execute"
                        return state
                    else:
                        print(f"[AUTO-RETRY] Max retries ({max_exec_retries}) reached. Proceeding to assessment.")
                        # Reset retry counter for future execute phases
                        state["artifacts"]["execute_retries"] = 0
                        # Now prune history since retries are done
                        self._prune_history(state)
                else:
                    # Success - reset retry counter
                    if "artifacts" in state:
                        state["artifacts"]["execute_retries"] = 0
                
            self._roll_memory(state, "EXECUTE")
            self._prune_history(state)
                
            # After execution, decide whether to generate mid-run report
            # Only generate mini_share report every 5th assess iteration
            if "artifacts" not in state:
                state["artifacts"] = {}
            
            assess_count = state["artifacts"].get("assess_iteration_count", 0)
            assess_count += 1
            state["artifacts"]["assess_iteration_count"] = assess_count
            
            # Generate mid-run report every 5 iterations
            if assess_count % 5 == 0:
                state["next_step"] = "mini_share"
            else:
                state["next_step"] = "assess"
            
            return state


        def mini_share(state: AgentState) -> AgentState:
            """Generate a concise mid-run progress report for expert discussion."""
            state["phase"] = "MINI_SHARE"
            _announce_phase(state)
            
            # Start State
            if self.product_manager:
                self._ensure_mem(state)
                lts = state["artifacts"]["memory"].get("long_term_summary","")
                self.product_manager.start_phase_state("MINI_SHARE", long_term_memory=lts)

            # Aggregate recent conversation history for mid-run report
            all_msgs = [str(m.content) for m in state["messages"]]
            conversation_blob = "\n".join(all_msgs)

            # Inject State Context if available
            state_context_msg = ""
            if self.product_manager:
                ctx = self.product_manager.get_state_context()
                if ctx.get("formatted_state"):
                    state_context_msg = f"\n[PREVIOUS STATE]\n{ctx['formatted_state']}\n"

            # Create a detailed, in-depth progress report using SHARE LLM
            llm_mini_share = self._llm_for("SHARE")
            report_prompt = f"""
Write a comprehensive, in-depth mid-run progress report (1000-1500 words) for expert technical discussion.
This is a detailed interim checkpoint for specialists to analyze low-level implementation details.

== REQUIREMENTS ==
• DEEP TECHNICAL DETAIL - experts need granular information to discuss problems
• Include specific parameter values, hyperparameters, configurations, and settings
• Document exact data transformations, preprocessing steps, and pipeline stages
• Report full error messages, stack traces, and debugging observations
• Specify software versions, library dependencies, and environment details
• Include intermediate numerical results, statistics, and diagnostic metrics
• Be methodical and comprehensive - experts will scrutinize technical choices

== DETAILED STRUCTURE ==

1. **Current Objective & Context** (2-3 paragraphs)
   - Precise research question with mathematical/biological formulation
   - Hypotheses being tested with expected outcomes
   - Key assumptions and constraints

2. **Detailed Methods & Implementation** (extensive)
   - Algorithm specifics: equations, pseudocode, parameter settings
   - Data preprocessing: exact transformations, normalization methods, filtering criteria
   - Tool/software configuration: versions, flags, settings, random seeds
   - Code logic decisions: why specific approaches were chosen
   - Computational environment: hardware specs, memory usage, runtime

3. **Complete Results & Observations** (comprehensive)
   - ALL numerical outputs with precision (means ± std, CI, p-values)
   - Intermediate calculations and diagnostic checks
   - Data quality metrics (missing values, outliers, distributions)
   - Performance metrics at each stage
   - Unexpected behaviors or edge cases observed

4. **In-Depth Technical Issues & Debugging** (thorough)
   - FULL error messages and tracebacks (not summaries)
   - Failed approaches with exact reasons why they didn't work
   - Performance bottlenecks with profiling data
   - Memory/resource constraints encountered
   - Data quality problems discovered
   - Edge cases and boundary conditions

5. **Literature & Methods Context** (targeted)
   - Papers/methods directly relevant to current decisions
   - Alternative approaches considered and why rejected
   - Known limitations from prior work that apply here

6. **Next Steps & Open Questions** (specific)
   - Exact experiments planned with parameter ranges
   - Specific hypotheses to test
   - Technical uncertainties requiring expert input
   - Resource requirements (compute time, memory, data)

7. **Code & Data References**
   - Specific functions, classes, files modified or created
   - Data files read/written with paths and formats
   - Scripts executed with command-line arguments

== STYLE REQUIREMENTS ==
• Use technical language - assume expert audience
• Include exact values, not ranges or approximations
• Provide enough detail for reproduction
• Reference specific lines of code, equations, or data points
• Document every decision point and rationale

Do NOT include:
- Figure collection (figures will be in final paper only)

===== CONVERSATION LOG =====
{conversation_blob}
{state_context_msg}
===== END LOG =====
"""
            report_resp = llm_mini_share.invoke([HumanMessage(content=report_prompt)])
            try:
                self._record_llm_usage([HumanMessage(content=report_prompt)] + [report_resp], llm_obj=llm_mini_share)
            except Exception:
                pass
            
            report_text = str(report_resp.content).strip()

            # Save to run reports directory
            out_root = getattr(self, "run_dir", self.path)
            reports_dir = getattr(self, "run_reports_dir", os.path.join(out_root, "reports"))
            os.makedirs(reports_dir, exist_ok=True)

            ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            report_path = os.path.join(reports_dir, f"mini_report_{ts}.md")

            # Save the mid-run report
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"# Mid-Run Progress Report\n")
                f.write(f"Generated: {ts}\n\n")
                f.write(report_text)

            # Update State - pass only a reference to the report file, not the full content
            if self.product_manager:
                self._ensure_mem(state)
                lts = state["artifacts"]["memory"].get("long_term_summary","")
                # Pass only the file path reference instead of full report content
                self.product_manager.end_phase_state("MINI_SHARE", f"Mid-run report saved to {report_path}", long_term_memory=lts)

            # Update state
            if "artifacts" not in state or not isinstance(state["artifacts"], dict):
                state["artifacts"] = {}
            if "mini_reports" not in state["artifacts"]:
                state["artifacts"]["mini_reports"] = []
            state["artifacts"]["mini_reports"].append(report_path)

            state["messages"].append(AIMessage(content=f"[MINI_SHARE] Saved mid-run report to {report_path}"))

            # Roll memory and proceed to assess
            self._roll_memory(state, "MINI_SHARE")
            self._prune_history(state)
            
            state["next_step"] = "assess"
            return state

        def assess(state: AgentState) -> AgentState:

            state["phase"] = "ASSESS"
            _announce_phase(state)
            
            # Start State
            if self.product_manager:
                self._ensure_mem(state)
                lts = state["artifacts"]["memory"].get("long_term_summary","")
                self.product_manager.start_phase_state("ASSESS", long_term_memory=lts)

            # Inject State Context if available
            state_context_msg = ""
            if self.product_manager:
                ctx = self.product_manager.get_state_context()
                if ctx.get("formatted_state"):
                    state_context_msg = f"\n[PREVIOUS STATE]\n{ctx['formatted_state']}\n"

            messages = [SystemMessage(content=self._prompt_for("ASSESS") + state_context_msg)] + state["messages"]
            llm = self._llm_for("ASSESS")

            # Log with phase logger
            self.phase_logger.set_phase("ASSESS")
            self.phase_logger.log_prompt(messages)
            log_llm_event("assess_prompt", messages)

            response = llm.invoke(messages)
            # Handle thinking-mode responses: content may be a list of blocks
            # e.g. [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "..."}]
            _raw = response.content
            if isinstance(_raw, list):
                msg = "\n".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in _raw
                    if not (isinstance(block, dict) and block.get("type") == "thinking")
                ).strip()
            else:
                msg = str(_raw)

            # Update State
            if self.product_manager:
                self._ensure_mem(state)
                lts = state["artifacts"]["memory"].get("long_term_summary","")
                self.product_manager.end_phase_state("ASSESS", msg, long_term_memory=lts)

            # Log with phase logger
            self.phase_logger.log_response(msg)
            log_llm_event("assess_response", msg)
            
            state["messages"].append(AIMessage(content=msg.strip()))
            
            try:
                self._record_llm_usage(messages + [response], llm_obj=llm)
            except Exception:
                pass
            
            # new
            self._roll_memory(state, "ASSESS")
            self._prune_history(state)
            
            msg_lower = msg.lower()
            
            if "<goplan>" in msg_lower:
                state["next_step"] = "plan"
            elif "<goshare>" in msg_lower:
                state["next_step"] = "share"
            # New: robust parsing with retries
            else:
                print("parsing error...")
                print(msg)
                
                
                # Robust retry counter stored in state
                tries = state["artifacts"].get("parse_retries", 0) if "artifacts" in state else 0
                tries += 1
                if "artifacts" not in state:
                    state["artifacts"] = {}
                state["artifacts"]["parse_retries"] = tries

                if tries >= 3:
                    print("Detected repeated parsing errors, ending conversation")
                    state["messages"].append(AIMessage(content="Terminating: model failed to emit <goplan> or <goshare> after 3 retries."))
                    state["next_step"] = "end"
                else:
                    # Nudge once more with a precise instruction
                    state["messages"].append(HumanMessage(
                        content="[FORMAT ERROR] Emit EITHER <goplan> or <goshare> tag. Not multiple at the same time. Do not respond with messages without any tags. No empty messages."
                    ))
                    state["next_step"] = "assess"  # retry assess
            
            # Check if cost budget is exceeded (after routing decision)
            if self.cost_budget is not None and hasattr(self, "cost_manager") and self.cost_manager is not None:
                total_cost = getattr(self.cost_manager, "total_cost", None)
                if total_cost is not None and total_cost >= self.cost_budget:
                    print(f"[COST LIMIT] Budget exceeded: ${total_cost:.4f} >= ${self.cost_budget:.2f}. Routing to SHARE.")
                    state["messages"].append(HumanMessage(
                        content=f"[COST MANAGEMENT] Budget limit reached (${total_cost:.4f} / ${self.cost_budget:.2f}). "
                                f"Proceeding to SHARE phase to finalize results with current progress."
                    ))
                    state["next_step"] = "share"
            
            return state

           
            

        
            
            

            '''
            # If the optional self-critic is available, reuse it verbatim
            if self_critic:
                return execute_self_critic(state)

            # Otherwise, light fallback: if a <solution> exists, share; else think again
            all_msgs = [str(m.content) for m in state["messages"]]
            blob = "\n".join(all_msgs)
            if re.search(r"<solution>.*?</solution>", blob, re.DOTALL):
                state["next_step"] = "share"
            else:
                state["next_step"] = "generate"
            return state
            '''
        def share(state: AgentState) -> AgentState:
            state["phase"] = "SHARE"
            #_announce_phase(state)
            
            # Start State
            if self.product_manager:
                self._ensure_mem(state)
                lts = state["artifacts"]["memory"].get("long_term_summary","")
                self.product_manager.start_phase_state("SHARE", long_term_memory=lts)

            # Aggregate conversation history (messages) for logging
            all_msgs = [str(m.content) for m in state["messages"]]
            conversation_blob = "\n".join(all_msgs)

            # Inject State Context if available
            state_context_msg = ""
            if self.product_manager:
                ctx = self.product_manager.get_state_context()
                if ctx.get("formatted_state"):
                    state_context_msg = f"\n[PREVIOUS STATE]\n{ctx['formatted_state']}\n"

            # Try to extract any final <solution> block (optional)
            #m_sol = re.search(r"<solution>(.*?)</solution>", conversation_blob, re.DOTALL)
            #solution_text = m_sol.group(1).strip() if m_sol else ""

            # Compose a paper using the phase-specific LLM
            llm_share = self._llm_for("SHARE")
            paper_prompt = f"""
Write a full scientific paper based entirely on the following research process stream (conversation logs).
Adhere strictly to this structure and formatting:

You are a methodical scientific writer. Write a self-contained, conference-style paper that is approximately 7 pages (roughly 3,000–4,000 words) based ONLY on the MATERIALS section below. Do not invent facts. If a datum is missing, say so explicitly and proceed with clearly labeled assumptions.

== GLOBAL REQUIREMENTS ==
• Truthfulness & scope discipline: All claims must match the evidence in MATERIALS. Distinguish clearly between (a) observed results; (b) simulated or placeholder outputs; (c) hypotheses or aspirational goals. Never imply that simulated numbers are empirical.
• Rigor: Provide enough detail for replication; state assumptions, edge cases, and known failure modes.
• Clarity: Use precise, non-florid language suitable for a technical audience.

== LENGTH & STYLE ==
• Length target: 3,000–4,000 words (≈7 pages when typeset).
• Audience: reviewers with technical background.
• Tense: use past tense for what was done; present/future for interpretation and planned work.
• Figures/Tables: When referenced, include a placeholder caption and a TODO note (no images required).

== STRUCTURE (use headers EXACTLY as written) ==

Front matter
- Title
- Author information
- Abstract (250–300 words)
- Keywords

IMRaD
- Introduction (context and background)
- Literature review (prior work and current knowledge)
- Gap analysis (limitations in prior work your study addresses)
- Research question and hypothesis
- Methods
  - Study design
  - Participants/subjects (incl./excl. criteria)  [Write “Not applicable” if none.]
  - Materials & procedures
  - Ethical considerations
  - Statistical analysis (software/methods). If only simulated or qualitative, state that.
- Results
  - Objective reporting; reference figures/tables
  - Include ALL available quantitative results; if simulated, say “Simulated” and describe how generated
  - Report non-significant findings as well
- Discussion
  - Interpretation vs. hypotheses
  - Comparison to previous work
  - Relevance (clinical/basic/engineering)
  - Limitations (brief; a full “Limitations” section also appears later)
  - Future research

IMPORTANT:
• Base the paper only on the logs below; if data/values are missing, be explicit about assumptions or that a datum was not recorded.
• Use clear section headers exactly matching the spec above.
• Where figures/tables are referenced, include a short placeholder caption (e.g., "Figure 1. …") and a TODO note for generation.

===== BEGIN LOGS =====
{conversation_blob}
{state_context_msg}
===== END LOGS =====
"""
            paper_resp = llm_share.invoke([HumanMessage(content=paper_prompt)])
            try:
                self._record_llm_usage([HumanMessage(content=paper_prompt)] + [paper_resp], llm_obj=llm_share)
            except Exception:
                pass
            
            paper_text = str(paper_resp.content).strip()
            
            # --- Collect figures from manifest or from message markers
            

            out_root = os.path.join(self.path)
            fig_dir = getattr(self, "run_figures_dir", os.path.join(out_root, "figures"))
            manifest_path = os.path.join(fig_dir, "manifest.jsonl")
            fig_recs = []

            # Preferred: manifest.jsonl
            if os.path.isfile(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            fig_recs.append(rec)
                        except Exception:
                            pass

            # Fallback: scrape logs for <<FIGURE:path|caption>>
            if not fig_recs:
                import re
                pat = r"<<FIGURE:(?P<path>[^|>]+)\|(?P<caption>[^>]+)>>"
                for m_fig in re.finditer(pat, conversation_blob):
                    pth = m_fig.group("path").strip()
                    cap = m_fig.group("caption").strip()
                    label = cap.split(".")[0] if "." in cap else "Figure"
                    fig_recs.append({"path": pth, "caption": cap, "label": label})
                    
                    
            # --- Deduplicate figure records by absolute path
            seen = set()
            deduped = []
            for r in fig_recs:
                p = os.path.abspath(r.get("path", ""))
                if p and p not in seen:
                    seen.add(p)
                    deduped.append(r)
            fig_recs = deduped
 

            # Build Markdown for figures
            fig_md_lines = []
            for rec in fig_recs:
                # skip if path missing
                if not os.path.isfile(rec.get("path","")):
                    continue
                # make path relative and normalize slashes for Markdown
                rel = os.path.relpath(rec["path"], out_root).replace("\\", "/")
                label = rec.get("label", "Figure")
                caption = rec.get("caption", "")
                fig_md_lines.append(f'![{label}: {caption}]({rel})')

            figures_md = "\n\n".join(fig_md_lines)

            # Inject before End matter if available; else append at end
            if figures_md:
                # try to find a heading that starts with "End matter" (case-insensitive)
                import re as _re
                m_anchor = _re.search(r"\n\s*End matter\b", paper_text, flags=_re.IGNORECASE)
                if m_anchor:
                    idx = m_anchor.start()
                    paper_text = paper_text[:idx] + f"\nFigures\n-------\n{figures_md}\n" + paper_text[idx:]
                else:
                    paper_text += f"\n\nFigures\n-------\n{figures_md}\n"

            # new
            # --- Ensure artifacts dict exists
            if "artifacts" not in state or not isinstance(state["artifacts"], dict):
                state["artifacts"] = {}

            # --- Extract and persist <execute> code blocks
            code_blocks = []
            for i, m in enumerate(re.finditer(r"<execute>(.*?)</execute>", conversation_blob, re.DOTALL), start=1):
                code_txt = m.group(1).strip()
                if code_txt:
                    code_blocks.append((i, code_txt))

            # Choose a run-scoped code dir (sibling of papers/logs)
            out_root = os.path.join(self.path)
            code_dir = getattr(
                self, "run_code_dir",
                os.path.join(getattr(self, "run_dir", out_root), "code")
            )
            os.makedirs(code_dir, exist_ok=True)

            saved_code_paths = []
            ts_label = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            for i, code_txt in code_blocks:
                code_path = os.path.join(code_dir, f"step_{i:02d}.py")
                with open(code_path, "w", encoding="utf-8") as f:
                    f.write(f"# Extracted from <execute> block #{i} at {ts_label}\n")
                    f.write(code_txt + "\n")
                saved_code_paths.append(code_path)

            # Record code artifacts (relative paths for UI)
            rel_code_paths = [os.path.relpath(p, out_root).replace("\\", "/") for p in saved_code_paths]
            state["artifacts"]["code_paths"] = rel_code_paths

            # --- Append a Code Appendix to the paper (before End matter if possible)
            code_list_md = "\n".join(f"- `{p}`" for p in rel_code_paths) or "_(No <execute> blocks found)_"
            code_appendix = f"\n\nCode Appendix\n-------------\nSaved code blocks:\n{code_list_md}\n"

            m_anchor = re.search(r"\n\s*End matter\b", paper_text, flags=re.IGNORECASE)
            if m_anchor:
                idx = m_anchor.start()
                paper_text = paper_text[:idx] + code_appendix + paper_text[idx:]
            else:
                paper_text += code_appendix

            # new end
            # File destinations inside Biomni
            '''
            out_root = os.path.join(self.path)  # e.g., <path>/bioplease_data
            papers_dir = os.path.join(out_root, "papers")
            logs_dir = os.path.join(out_root, "logs")
            '''
            
            out_root   = getattr(self, "run_dir", self.path)  # prefer the *run* root
            papers_dir = getattr(self, "run_papers_dir", os.path.join(out_root, "papers"))
            logs_dir   = getattr(self, "run_logs_dir",   os.path.join(out_root, "logs"))

            # Source of truth for figures (writer puts them here)
            src_fig_dir = getattr(self, "run_figures_dir", os.path.join(out_root, "figures"))

            # Destination next to the TeX (LaTeX includes from here)
            figures_dir = os.path.join(papers_dir, "figures")

            os.makedirs(papers_dir, exist_ok=True)
            os.makedirs(logs_dir, exist_ok=True)
            os.makedirs(figures_dir, exist_ok=True)


            ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            paper_path_md = os.path.join(papers_dir, f"paper_{ts}.md")
            log_path_txt = os.path.join(logs_dir, f"log_{ts}.txt")

            # Save paper and logs
            with open(paper_path_md, "w", encoding="utf-8") as f:
                f.write(paper_text)

            with open(log_path_txt, "a", encoding="utf-8") as f:
                f.write(f"\n\n===== SHARE @ {ts} =====\n")
                f.write(conversation_blob)

            # Also persist any extracted <solution> as a compact note (optional)
            #if solution_text:
                # sol_note = os.path.join(papers_dir, f"solution_note_{ts}.txt")
                # with open(sol_note, "w", encoding="utf-8") as f:
                #     f.write(solution_text)

            # Record artifact paths
            state["artifacts"]["paper_path"] = paper_path_md
            state["artifacts"]["log_path"] = log_path_txt

            state["messages"].append(AIMessage(content=f"[SHARE] Saved paper to {paper_path_md} and logs to {log_path_txt}"))
            
            r'''
            # === Use custom LaTeX template if present ===
            tmpl_dir = os.path.join(self.path, "templates")
            user_tex = os.path.join(tmpl_dir, "agents4science_2025.tex")
            user_sty = os.path.join(tmpl_dir, "agents4science_2025.sty")

            pdf_path = self._ensure_pdf_from_share(paper_path_md)

            
            if os.path.exists(user_tex):
                os.makedirs(papers_dir, exist_ok=True)
                # Copy template files into this run’s papers dir (so latexmk sees them)
                import shutil, textwrap
                shutil.copyfile(user_tex, os.path.join(papers_dir, "agents4science_2025.tex"))
                if os.path.exists(user_sty):
                    shutil.copyfile(user_sty, os.path.join(papers_dir, "agents4science_2025.sty"))

                # 1) Gather figures and code excerpts
                figs = self._gather_figures_for_latex(figures_dir)          # you added this earlier
                code_items = self._select_code_snippets(work_dir=self.path) # you added this earlier

                # 2) Parse sections from the md paper you already created in SHARE
                arts = self._collect_run_artifacts()   # you added this earlier
                fields = self._paper_sections_from_md(arts.get("paper_md")) # you added this earlier

                # 3) Write a content file that your template can \input
                #    We’ll create latex-safe text for prose, but keep code verbatim for minted/listings.
                content_tex = os.path.join(papers_dir, "content_generated.tex")
                def sec(name, body):
                    return f"\\section{{{name}}}\n{self._escape_tex(body)}\n" if body else ""

                # Figures block
                fig_lines = []
                os.makedirs(os.path.join(papers_dir, "figures"), exist_ok=True)
                for i, f in enumerate(figs, 1):
                    try:
                        dst = os.path.join(papers_dir, "figures",
                                        f"fig_{i}{os.path.splitext(f['path'])[1].lower()}")
                        if not os.path.exists(dst):
                            shutil.copyfile(f["path"], dst)
                        cap = self._escape_tex(f.get("caption", ""))
                        label = self._escape_tex(f.get("label", "Figure"))
                        fig_lines.append(textwrap.dedent(f"""
                            \\begin{{figure}}[H]
                            \\centering
                            \\includegraphics[width=0.95\\linewidth]{{figures/{os.path.basename(dst)}}}
                            \\caption{{{label}. {cap}}}
                            \\end{{figure}}
                        """).strip())
                    except Exception:
                        pass
                if not fig_lines:
                    fig_lines.append("No figures were produced.")

                # Code block with minted/listings macro (`\\codeblock{python}{...}`) – your template can define it
                code_lines = []
                for item in code_items:
                    nm = self._escape_tex(item["name"])
                    code_lines.append(textwrap.dedent(f"""
                        \\subsection*{{{nm}}}
                        \\codeblock{{python}}{{
            {item["content"]}
                        }}
                    """).strip())
                if not code_lines:
                    code_lines.append("No code excerpts selected.")

                Path(content_tex).write_text(
                    "\n\n".join([
                        sec("Introduction", fields.get("introduction","")),
                        sec("Methods",      fields.get("methods","")),
                        sec("Results",      fields.get("results","")),
                        sec("Discussion",   fields.get("discussion","")),
                        "\\section{Figures}\n" + "\n\n".join(fig_lines),
                        "\\section{Code Excerpts}\n" + "\n\n".join(code_lines),
                        "\\section{References}\n" + self._escape_tex(fields.get("references","")),
                    ]),
                    encoding="utf-8"
                )

                # 4) Create a tiny wrapper main.tex that sets front-matter and \input{sci template}
                #    Your agents4science_2025.tex should include something like:
                #    %%CONTENT_HERE  (a marker)  OR  \\input{content_generated.tex} near the end.
                #    We’ll try to patch a marker; else we append before \\end{document}.
                main_tex = os.path.join(papers_dir, "main.tex")
                raw_tmpl = Path(os.path.join(papers_dir, "agents4science_2025.tex")).read_text(encoding="utf-8", errors="ignore")

                # Title/authors/abstract/keywords macros your template can use; safe to \def even if unused.
                frontmatter = f"""
            % === Auto front matter injected by SHARE ===
            \\def\\PaperTitle{{{self._escape_tex(fields.get('title','Untitled'))}}}
            \\def\\PaperAuthors{{{self._escape_tex(fields.get('authors','Anonymous'))}}}
            \\def\\PaperKeywords{{{self._escape_tex(fields.get('keywords',''))}}}
            \\def\\PaperAbstract{{{self._escape_tex(fields.get('abstract',''))}}}
            % ===========================================
            """
                patched = raw_tmpl

                # Preferred: replace marker if present
                if "%%CONTENT_HERE" in raw_tmpl:
                    patched = raw_tmpl.replace("%%CONTENT_HERE", "\\input{content_generated.tex}")
                    patched = frontmatter + patched
                else:
                    # Fallback: insert before \end{document}
                    import re
                    m = re.search(r"\\end{document}", raw_tmpl)
                    if m:
                        idx = m.start()
                        patched = frontmatter + raw_tmpl[:idx] + "\n\\input{content_generated.tex}\n" + raw_tmpl[idx:]
                    else:
                        # Last resort: just prepend front matter and append our input
                        patched = frontmatter + raw_tmpl + "\n\\input{content_generated.tex}\n"

                Path(main_tex).write_text(patched, encoding="utf-8")

                # 5) Compile with latexmk; minted supported if your template defines \\codeblock using minted
                pdf_path = self._compile_pdf_from_tex(main_tex)  # you added this earlier
                if pdf_path and os.path.exists(pdf_path):
                    print(f"[SHARE] Compiled PDF (template): {pdf_path}")
                else:
                    print("[SHARE] PDF compile failed; see latexmk logs in papers directory.")

            
            
            '''
            
            
            # Update State
            if self.product_manager:
                self._ensure_mem(state)
                lts = state["artifacts"]["memory"].get("long_term_summary","")
                self.product_manager.end_phase_state("SHARE", paper_text, long_term_memory=lts)

            # new
            self._roll_memory(state, "SHARE")
            self._prune_history(state)
            
            state["next_step"] = "overview_assess"
            return state
        

        def routing_function_execute(
            state: AgentState,
        ) -> Literal["execute", "mini_share", "assess"]:
            next_step = state.get("next_step")
            if next_step == "execute":
                return "execute"
            elif next_step == "mini_share":
                return "mini_share"
            elif next_step == "assess":
                return "assess"
            else:
                print(f"[EXECUTE ROUTER] unexpected next_step={str(next_step)}; defaulting to assess")
                return "assess"
            
        
        def routing_function_assess(
            state: AgentState,
        ) -> Literal["plan", "learn", "execute", "assess", "share"]:
            next_step = state.get("next_step")
            
            # FASTMODE: Strictly prevent backward edges
            if getattr(self, "fastmode", False):
                # In fastmode, assess can only go forward to share
                if next_step == "share":
                    print("[FASTMODE] ASSESS → SHARE (forward only)")
                    return "share"
                elif next_step in ["plan", "learn", "execute"]:
                    print(f"[FASTMODE] Blocking backward edge ASSESS → {next_step}, forcing SHARE instead")
                    return "share"
                elif next_step == "assess":
                    # Allow staying in assess if needed
                    return "assess"
                else:
                    print(f"[FASTMODE] ASSESS ROUTER unexpected next_step={str(next_step)}; forcing SHARE")
                    return "share"
            
            # Normal mode: allow all transitions
            if next_step == "plan":
                return "plan"
            elif next_step == "learn":
                return "learn"
            elif next_step == "execute":
                return "execute"
            elif next_step == "assess":
                return "assess"
            elif next_step == "share":
                return "share"
            else:
                print(f"[ASSESS ROUTER] unexpected next_step={str(next_step)}; defaulting to plan")
                return "plan"
            
        def routing_function_overview_assess(
            state: AgentState,
        ) -> Literal["plan", "execute", "share", "end", "overview_assess"]:
            next_step = state.get("next_step")
            
            # FASTMODE: Strictly prevent backward edges
            if getattr(self, "fastmode", False):
                # In fastmode, overview_assess can only go to end or stay in overview_assess
                if next_step == "end":
                    print("[FASTMODE] OVERVIEW_ASSESS → END")
                    return "end"
                elif next_step == "share":
                    print("[FASTMODE] OVERVIEW_ASSESS → SHARE")
                    return "share"
                elif next_step in ["plan", "execute"]:
                    print(f"[FASTMODE] Blocking backward edge OVERVIEW_ASSESS → {next_step}, forcing END instead")
                    return "end"
                elif next_step == "overview_assess":
                    return "overview_assess"
                else:
                    print(f"[FASTMODE] OVERVIEW_ASSESS ROUTER unexpected next_step={str(next_step)}; forcing END")
                    return "end"
            
            # Normal mode: allow all transitions
            if next_step == "plan":
                return "plan"
            elif next_step == "execute":
                return "execute"
            elif next_step == "share":
                return "share"
            elif next_step == "end":
                return "end"
            elif next_step == "overview_assess":
                return "overview_assess"
            else:
                print(f"[ASSESS ROUTER] unexpected next_step={str(next_step)}; defaulting to share")
                return "share"   
            


        
            
        
        '''
        def execute_self_critic(state: AgentState) -> AgentState:
            if self.critic_count < test_time_scale_round:
                # Generate feedback based on message history
                messages = state["messages"]
                feedback_prompt = f"""
                Here is a reminder of what is the user requested: {self.user_task}
                Examine the previous executions, reaosning, and solutions.
                Critic harshly on what could be improved?
                Be specific and constructive.
                Think hard what are missing to solve the task.
                No question asked, just feedbacks.
                """
                feedback = self.llm.invoke(messages + [HumanMessage(content=feedback_prompt)])

                # Add feedback as a new message
                state["messages"].append(
                    HumanMessage(
                        content=f"Wait... this is not enough to solve the task. Here are some feedbacks for improvement:\n{feedback.content}"
                    )
                )
                self.critic_count += 1
                state["next_step"] = "generate"
            else:
                state["next_step"] = "end"

            return state
        '''
        
        # Create the workflow
        workflow = StateGraph(AgentState)

        # Nodes 
        workflow.add_node("plan", plan)
        workflow.add_node("learn", learn)
        workflow.add_node("execute", execute)
        workflow.add_node("mini_share", mini_share)
        workflow.add_node("assess", assess)
        workflow.add_node("share", share)
        workflow.add_node("overview_assess", self.overview_assess)

        

        # Edges
        workflow.add_edge(START, "plan")
        workflow.add_edge("plan", "learn")
        workflow.add_edge("learn", "execute")
        workflow.add_conditional_edges(
            "execute",
            routing_function_execute,
            path_map={
                "execute": "execute",
                "mini_share": "mini_share",
                "assess": "assess",
            }
        )
        workflow.add_edge("mini_share", "assess")
        workflow.add_conditional_edges(
        "assess",
        routing_function_assess,
        path_map={
            "plan": "plan",
            "learn": "learn",
            "execute": "execute",
            "assess": "assess",
            "share": "share",
            
        }
        )
        # workflow.add_edge("share", "overview_assess")  # COMMENTED: Pause overview_assess chain
        workflow.add_edge("share", END)  # End after one SHARE instead
        
        workflow.add_conditional_edges(
        "overview_assess",
        routing_function_overview_assess,
        path_map={
            "plan": "plan",
            "execute": "execute",
            "share": "share",
            "end": END,
            "overview_assess": "overview_assess",
        }
        )

        # Compile
        self.app = workflow.compile()
        
        #print(self.app.get_graph().draw_mermaid())
        self.checkpointer = MemorySaver()
        self.app.checkpointer = self.checkpointer

        # Old
        '''
        # Create the workflow
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("generate", generate)
        workflow.add_node("execute", execute)

        if self_critic:
            workflow.add_node("self_critic", execute_self_critic)
            # Add conditional edges
            workflow.add_conditional_edges(
                "generate",
                routing_function,
                path_map={
                    "execute": "execute",
                    "generate": "generate",
                    "end": "self_critic",
                },
            )
            workflow.add_conditional_edges(
                "self_critic",
                routing_function_self_critic,
                path_map={"generate": "generate", "end": END},
            )
        else:
            # Add conditional edges
            workflow.add_conditional_edges(
                "generate",
                routing_function,
                path_map={"execute": "execute", "generate": "generate", "end": END},
            )
        workflow.add_edge("execute", "generate")
        workflow.add_edge(START, "generate")
        
        # Compile the workflow
        self.app = workflow.compile()
        self.checkpointer = MemorySaver()
        self.app.checkpointer = self.checkpointer
        # display(Image(self.app.get_graph().draw_mermaid_png()))
        '''

    def go(self, prompt):
        """Execute the agent with the given prompt.

        Args:
            prompt: The user's query

        """
        self.critic_count = 0
        self.user_task = prompt
        
        
        # --- Create a per-run directory structure ---
        # Run id uses UTC timestamp + slug of the prompt for human readability
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        def _slugify(value, allow_unicode=False):
            value = str(value)
            if allow_unicode:
                value = unicodedata.normalize('NFKC', value)
            else:
                value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
            value = re.sub(r'[^\w\s-]', '', value.lower())
            return re.sub(r'[-\s]+', '-', value).strip('-_')
        task_slug = _slugify(prompt)[:40] or "run"
        self.run_id = f"{ts}_{task_slug}"
        runs_root = os.path.join(self.path, "runs")
        self.run_dir = os.path.join(runs_root, self.run_id)
        # Subfolders for organization
        self.run_logs_dir = os.path.join(self.run_dir, "logs")
        self.run_papers_dir = os.path.join(self.run_dir, "papers")
        self.run_reports_dir = os.path.join(self.run_dir, "reports")
        self.run_figures_dir = os.path.join(self.run_dir, "figures")
        self.run_artifacts_dir = os.path.join(self.run_dir, "artifacts")
        self.run_code_dir = os.path.join(self.run_dir, "code")
        for d in [runs_root, self.run_dir, self.run_logs_dir, self.run_papers_dir, self.run_reports_dir, self.run_figures_dir, self.run_artifacts_dir, self.run_code_dir]:
            os.makedirs(d, exist_ok=True)

        # Propagate environment variables so EXECUTE code can discover the run directory
        os.environ["BIOPLEASE_DATA"] = self.path            # e.g., .../bioplease_data
        os.environ["BIOPLEASE_RUN_DIR"] = self.run_dir
        os.environ["BIOPLEASE_ARTIFACTS_DIR"] = self.run_artifacts_dir
        os.environ["BIOPLEASE_FIGURES_DIR"] = self.run_figures_dir
        os.environ["BIOPLEASE_CODE_DIR"] = self.run_code_dir

        # Initialize Phase Logger for organized logging
        self.phase_logger = PhaseLogger(
            base_logs_dir=self.run_logs_dir,
            enabled=True
        )
        print(f"[INFO] Phase-based logging enabled: {self.run_logs_dir}")

        # Initialize Unified Phase Logger for cumulative logging
        unified_log_file = os.path.join(self.run_dir, "unified_phase_log.txt")
        self.unified_phase_logger = UnifiedPhaseLogger(unified_log_file)
        self.unified_phase_logger.log_context(f"Run started for prompt: {prompt}")

        # Update Product Manager with run directory
        if self.product_manager:
            self.product_manager.set_run_directory(self.run_dir)
            self.product_manager.set_original_task(prompt)
            if self.cost_budget is not None:
                self.product_manager.set_cost_budget(self.cost_budget)

        # Initialize Communication Hub for inter-agent communication
        self.comm_hub = CommunicationHub(
            logs_dir=os.path.join(self.run_logs_dir, "communication")
        )
        print(f"[INFO] Communication hub initialized")

        # Reset cost manager at the start of each agent run to avoid cost accumulation across runs
        if hasattr(self, "cost_manager") and self.cost_manager is not None:
            self.cost_manager.reset()
            # Update Product Manager's cost_manager reference to ensure it tracks costs correctly
            if hasattr(self, "product_manager") and self.product_manager is not None:
                self.product_manager.cost_manager = self.cost_manager
                print(f"[INFO] Cost manager reset and synced with Product Manager")

        # Append a run index record
        run_index = os.path.join(runs_root, "index.jsonl")
        meta = {
            "run_id": self.run_id,
            "ts": ts,
            "task": prompt,
            "run_dir": self.run_dir,
        }
        try:
            with open(run_index, "a", encoding="utf-8") as f:
                f.write(json.dumps(meta) + "\n")
        except Exception:
            pass

        
        # New End
        

        if self.use_tool_retriever:
            # Gather all available resources
            # 1. Tools from the registry
            all_tools = self.tool_registry.tools if hasattr(self, "tool_registry") else []

            # 2. Data lake items with descriptions
            data_lake_path = self.path + "/data_lake"
            data_lake_content = glob.glob(data_lake_path + "/*")
            data_lake_items = [x.split("/")[-1] for x in data_lake_content]

            # Create data lake descriptions for retrieval
            data_lake_descriptions = []
            for item in data_lake_items:
                description = self.data_lake_dict.get(item, f"Data lake item: {item}")
                data_lake_descriptions.append({"name": item, "description": description})

            # Add custom data items to retrieval if they exist
            if hasattr(self, "_custom_data") and self._custom_data:
                for name, info in self._custom_data.items():
                    data_lake_descriptions.append({"name": name, "description": info["description"]})

            # 3. Libraries with descriptions - use library_content_dict directly
            library_descriptions = []
            for lib_name, lib_desc in self.library_content_dict.items():
                library_descriptions.append({"name": lib_name, "description": lib_desc})

            # Add custom software items to retrieval if they exist
            if hasattr(self, "_custom_software") and self._custom_software:
                for name, info in self._custom_software.items():
                    # Check if it's not already in the library descriptions to avoid duplicates
                    if not any(lib["name"] == name for lib in library_descriptions):
                        library_descriptions.append({"name": name, "description": info["description"]})

            # Use retrieval to get relevant resources
            resources = {
                "tools": all_tools,
                "data_lake": data_lake_descriptions,
                "libraries": library_descriptions,
            }

            # Use prompt-based retrieval with the agent's LLM
            selected_resources = self.retriever.prompt_based_retrieval(prompt, resources, llm=self.llm)
            print("Using prompt-based retrieval with the agent's LLM")

            # Extract the names from the selected resources for the system prompt
            selected_resources_names = {
                "tools": selected_resources["tools"],
                "data_lake": [],
                "libraries": [lib["name"] if isinstance(lib, dict) else lib for lib in selected_resources["libraries"]],
            }

            # Process data lake items to extract just the names
            for item in selected_resources["data_lake"]:
                if isinstance(item, dict):
                    selected_resources_names["data_lake"].append(item["name"])
                elif isinstance(item, str) and ": " in item:
                    # If the item already has a description, extract just the name
                    name = item.split(": ")[0]
                    selected_resources_names["data_lake"].append(name)
                else:
                    selected_resources_names["data_lake"].append(item)

            # Update the system prompt with the selected resources
            self.update_system_prompt_with_selected_resources(selected_resources_names)

        inputs = { # new
            "messages": [HumanMessage(content=prompt)],
            "next_step": "generate",
            "phase": "PLAN",
            "artifacts": {},
        }
        
        config = {"recursion_limit": 500, "configurable": {"thread_id": 42}}
        self.log = []

        for s in self.app.stream(inputs, stream_mode="values", config=config):
            message = s["messages"][-1]
            out = pretty_print(message)
            self.log.append(out)
            
            # Log each phase step to the unified log after every phase step
            if hasattr(self, "unified_phase_logger") and self.unified_phase_logger:
                phase_val = s.get("phase", "UNKNOWN")
                step_val = s.get("next_step", "UNKNOWN")
                content_val = str(message.content)
                meta_val = s.get("artifacts", {})
                self.unified_phase_logger.log_phase(
                    phase=phase_val,
                    step_type=step_val,
                    content=content_val,
                    metadata=meta_val
                )
        
        return self.log, message.content
    
    def reply(self, text, thread_id=42):
        from langchain_core.messages import HumanMessage
        out = self.app.invoke(
            {"messages": [HumanMessage(content=text)], "next_step": None},
            config={"configurable": {"thread_id": thread_id}},
        )
        return out["messages"][-1].content
    
    def update_system_prompt_with_selected_resources(self, selected_resources):
        """Update the system prompt with the selected resources."""
        # Extract tool descriptions for the selected tools
        tool_desc = {}
        for tool in selected_resources["tools"]:
            # Get the module name from the tool
            if isinstance(tool, dict):
                module_name = tool.get("module", None)

                # If module is not specified, try to find it in the module2api
                if not module_name and hasattr(self, "module2api"):
                    for mod, apis in self.module2api.items():
                        for api in apis:
                            if api.get("name") == tool.get("name"):
                                module_name = mod
                                # Update the tool with the module information
                                tool["module"] = module_name
                                break
                        if module_name:
                            break

                # If still not found, use a default
                if not module_name:
                    module_name = "bioplease.tool.scRNA_tools"  # Default to scRNA_tools as a fallback
                    tool["module"] = module_name
            else:
                module_name = getattr(tool, "module_name", None)

                # If module is not specified, try to find it in the module2api
                if not module_name and hasattr(self, "module2api"):
                    tool_name = getattr(tool, "name", str(tool))
                    for mod, apis in self.module2api.items():
                        for api in apis:
                            if api.get("name") == tool_name:
                                module_name = mod
                                # Set the module_name attribute
                                tool.module_name = module_name
                                break
                        if module_name:
                            break

                # If still not found, use a default
                if not module_name:
                    module_name = "bioplease.tool.scRNA_tools"  # Default to scRNA_tools as a fallback
                    tool.module_name = module_name

            if module_name not in tool_desc:
                tool_desc[module_name] = []

            # Add the tool to the appropriate module
            if isinstance(tool, dict):
                # Ensure the module is included in the tool description
                if "module" not in tool:
                    tool["module"] = module_name
                tool_desc[module_name].append(tool)
            else:
                # Convert tool object to dictionary
                tool_dict = {
                    "name": getattr(tool, "name", str(tool)),
                    "description": getattr(tool, "description", ""),
                    "parameters": getattr(tool, "parameters", {}),
                    "module": module_name,  # Explicitly include the module
                }
                tool_desc[module_name].append(tool_dict)

        # Prepare data lake items with descriptions
        data_lake_with_desc = []
        for item in selected_resources["data_lake"]:
            description = self.data_lake_dict.get(item, f"Data lake item: {item}")
            data_lake_with_desc.append({"name": item, "description": description})

        # Prepare custom resources for highlighting
        custom_tools = []
        if hasattr(self, "_custom_tools") and self._custom_tools:
            for name, info in self._custom_tools.items():
                custom_tools.append(
                    {
                        "name": name,
                        "description": info["description"],
                        "module": info["module"],
                    }
                )

        custom_data = []
        if hasattr(self, "_custom_data") and self._custom_data:
            for name, info in self._custom_data.items():
                custom_data.append({"name": name, "description": info["description"]})

        custom_software = []
        if hasattr(self, "_custom_software") and self._custom_software:
            for name, info in self._custom_software.items():
                custom_software.append({"name": name, "description": info["description"]})

        # new
        self._build_phase_prompts(
            tool_desc=tool_desc,
            data_lake_with_desc=data_lake_with_desc,
            library_content_list=selected_resources["libraries"],
            is_retrieval=True,
            self_critic=getattr(self, "self_critic", False),
            custom_tools=custom_tools if custom_tools else None,
            custom_data=custom_data if custom_data else None,
            custom_software=custom_software if custom_software else None,
        )

        # Print the raw system prompt for debugging
        # print("\n" + "="*20 + " RAW SYSTEM PROMPT FROM AGENT " + "="*20)
        # print(self.system_prompt)
        # print("="*70 + "\n")

    def result_formatting(self, output_class, task_intention):
        self.format_check_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are evaluateGPT, tasked with extract and parse the task output based on the history of an agent. "
                        "Review the entire history of messages provided. "
                        "Here is the task output requirement: \n"
                        f"'{task_intention.replace('{', '{{').replace('}', '}}')}'.\n"
                    ),
                ),
                ("placeholder", "{messages}"),
            ]
        )

        checker_llm = self.format_check_prompt | self.llm.with_structured_output(output_class)
        result = checker_llm.invoke({"messages": [("user", str(self.log))]}).dict()
        return result

    def _inject_custom_functions_to_repl(self):
        """Inject custom functions into the Python REPL execution environment.
        This makes custom tools available during code execution.
        """
        if hasattr(self, "_custom_functions") and self._custom_functions:
            # Access the persistent namespace used by run_python_repl
            from bioplease.tool.support_tools import _persistent_namespace

            # Inject all custom functions into the execution namespace
            for name, func in self._custom_functions.items():
                _persistent_namespace[name] = func

            # Also make them available in builtins for broader access
            import builtins

            if not hasattr(builtins, "_biomni_custom_functions"):
                builtins._biomni_custom_functions = {}
            builtins._biomni_custom_functions.update(self._custom_functions)

    def create_mcp_server(self, tool_modules=None):
        """
        Create an MCP server object that exposes internal Biomni tools.
        This gives you control over when and how to run the server.

        Args:
            tool_modules: List of module names to expose (default: all in self.module2api)

        Returns:
            FastMCP server object that you can run manually
        """
        import importlib
        import inspect
        from typing import Optional

        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("BiomniTools")
        modules = tool_modules or list(self.module2api.keys())

        registered_tools = 0

        for module_name in modules:
            try:
                # Import the actual module
                module = importlib.import_module(module_name)
                # Get tools for this module
                module_tools = self.module2api.get(module_name, [])

                for tool_schema in module_tools:
                    tool_name = tool_schema.get("name")
                    if not tool_name:
                        continue

                    try:
                        # Get the actual function
                        fn = getattr(module, tool_name, None)
                        if fn is None:
                            fn = getattr(self, "_custom_functions", {}).get(tool_name)

                        if fn is None:
                            print(f"Warning: Could not find function '{tool_name}' in module '{module_name}'")
                            continue

                        # Extract parameters from your specific schema format
                        required_params = tool_schema.get("required_parameters", [])
                        optional_params = tool_schema.get("optional_parameters", [])

                        # Generate the wrapper function
                        wrapper_func = self._generate_mcp_wrapper_from_biomni_schema(
                            fn, tool_name, required_params, optional_params
                        )

                        # Register with MCP
                        mcp.tool()(wrapper_func)
                        registered_tools += 1

                    except Exception as e:
                        print(f"Warning: Failed to register tool '{tool_name}': {e}")
                        continue

            except ImportError as e:
                print(f"Warning: Could not import module '{module_name}': {e}")
                continue

        print(f"Created MCP server with {registered_tools} tools")
        return mcp

    def _generate_mcp_wrapper_from_biomni_schema(self, original_func, func_name, required_params, optional_params):
        """Generate wrapper function based on Biomni schema format."""
        import inspect

        # Combine all parameters
        all_params = required_params + optional_params

        if not all_params:
            # No parameters
            def wrapper() -> dict:
                try:
                    result = original_func()
                    if isinstance(result, dict):
                        return result
                    return {"result": result}
                except Exception as e:
                    return {"error": str(e)}

            wrapper.__name__ = func_name
            wrapper.__doc__ = original_func.__doc__
            return wrapper

        else:
            # Has parameters
            def wrapper(**kwargs) -> dict:
                try:
                    # Build arguments dict
                    filtered_kwargs = {}

                    # Add required parameters
                    for param_info in required_params:
                        param_name = param_info["name"]
                        if param_name in kwargs and kwargs[param_name] is not None:
                            filtered_kwargs[param_name] = kwargs[param_name]

                    # Add optional parameters only if provided and not None
                    for param_info in optional_params:
                        param_name = param_info["name"]
                        if param_name in kwargs and kwargs[param_name] is not None:
                            filtered_kwargs[param_name] = kwargs[param_name]

                    result = original_func(**filtered_kwargs)
                    if isinstance(result, dict):
                        return result
                    return {"result": result}
                except Exception as e:
                    return {"error": str(e)}

            # Set function metadata
            wrapper.__name__ = func_name
            wrapper.__doc__ = original_func.__doc__

            # Create proper signature
            new_params = []

            # Map your types to Python types
            type_map = {"str": str, "int": int, "float": float, "bool": bool, "List[str]": list[str], "dict": dict}

            # Add required parameters
            for param_info in required_params:
                param_name = param_info["name"]
                param_type_str = param_info["type"]
                param_type = type_map.get(param_type_str, str)

                new_params.append(inspect.Parameter(param_name, inspect.Parameter.KEYWORD_ONLY, annotation=param_type))

            # Add optional parameters
            for param_info in optional_params:
                param_name = param_info["name"]
                param_type_str = param_info["type"]
                param_type = type_map.get(param_type_str, str)

                # Make it optional
                optional_type = param_type | None

                new_params.append(
                    inspect.Parameter(
                        param_name, inspect.Parameter.KEYWORD_ONLY, default=None, annotation=optional_type
                    )
                )

            # Set the signature
            wrapper.__signature__ = inspect.Signature(new_params, return_annotation=dict)

            return wrapper
