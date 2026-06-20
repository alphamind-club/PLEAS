import glob
import inspect
import os
import re
import sys
import time
import json
import unicodedata
from pathlib import Path
from typing import Any, Literal, TypedDict

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
    phase: Literal["PLAN","LEARN","EXECUTE","ASSESS","SHARE"]  # NEW
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

        """
        
        self.max_overview_loops = 3  # max iterations for OVERVIEW phase
        
        
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
        self.module2api = module2api
        self.use_tool_retriever = use_tool_retriever

        if self.use_tool_retriever:
            self.tool_registry = ToolRegistry(module2api)
            self.retriever = ToolRetriever()

        # Add timeout parameter
        self.timeout_seconds = timeout_seconds  # 10 minutes default timeout
        
        # --- Enhanced Memory System (NEW) ---
        # Import MemoryManager
        from bioplease.agent.memory import MemoryManager, MemoryConfig
        
        # Create memory configuration with customizable parameters
        self.memory_config = MemoryConfig(
            short_window=6,              # Recent messages to keep verbatim
            max_message_length=1200,     # Truncate very long messages
            summary_max_chars=2500,      # Max size for rolling summary
            summary_update_threshold=4,  # Compress every N messages
            compression_ratio=0.3,       # Target 30% compression
            max_total_tokens=8000,       # Token budget for memory
            auto_save=True,              # Auto-save memory snapshots
            save_directory=os.path.join(path, "bioplease_data", "memory_snapshots")
        )
        
        # Initialize memory manager (will be created per session)
        self.memory_manager = None
        
        # Legacy compatibility - these can still be adjusted
        self.short_window = self.memory_config.short_window
        self.summary_max_chars = self.memory_config.summary_max_chars
        
        self.configure()

        
        
        
        
        
        
        
        
    '''    
    def with_pleas(self, reviewer_model: str | None = None):
        """Attach a PLEAS runner to this agent instance."""
        from .pleas import PLEASRunner
        if reviewer_model:
            from ..llm import get_llm
            reviewer = get_llm(model=reviewer_model, temperature=0.2)
        else:
            reviewer = None
        self._pleas = PLEASRunner(llm=self.llm, reviewer_llm=reviewer)
        return self

    def run_pleas(self, task: str, budget: dict | None = None):
        """Run the full PLEAS pipeline and persist the report."""
        if not hasattr(self, "_pleas"):
            self.with_pleas()
        state = self._pleas.run(task=task, budget=budget)
        report_dir = os.path.join(self.path, "reports")
        os.makedirs(report_dir, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        with open(os.path.join(report_dir, f"pleas_{ts}.md"), "w") as f:
            f.write(state.report_md or "")
        with open(os.path.join(report_dir, f"pleas_{ts}.json"), "w") as f:
            json.dump(state.report_json or {}, f, indent=2)
        return state
        '''
        
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
                base_url = rest[0] if len(rest) > 0 else None
                api_key  = rest[1] if len(rest) > 1 else None
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
                    llm_name, source=source, base_url=base_url, api_key=api_key
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

        # Add custom resources to format dict if they exist
        if custom_tools_formatted:
            format_dict["custom_tools"] = "\n".join(custom_tools_formatted)
        if custom_data_formatted:
            format_dict["custom_data"] = "\n".join(custom_data_formatted)
        if custom_software_formatted:
            format_dict["custom_software"] = "\n".join(custom_software_formatted)

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
        # Base prompt
        prompt_modifier = """
You are a helpful, fully autonomous biomedical assistant assigned with the task of providing a plan towards problem-solving.
You are also the Plan agent of the PLAN-LEARN-EXECUTE-ASSESS-SHARE (or PLEAS) agentic framework.

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

        # Add custom resources to format dict if they exist
        if custom_tools_formatted:
            format_dict["custom_tools"] = "\n".join(custom_tools_formatted)
        if custom_data_formatted:
            format_dict["custom_data"] = "\n".join(custom_data_formatted)
        if custom_software_formatted:
            format_dict["custom_software"] = "\n".join(custom_software_formatted)

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
You are a helpful, fully autonomous biomedical assistant assigned with the task of collecting the necessary information and reading the proper papers for effective problem-solving.
You are also the LEARN agent of the PLAN-LEARN-EXECUTE-ASSESS-SHARE (or PLEAS) agentic framework.

[PHASE: LEARN]
To problem solve, you will need to collect, read, and understand a variety of tool functions, data, and softwares to assist you throughout the process.
You are the research and knowledge-gathering agent. Your job is to explore the necessary resources to fulfill the plan. You retrieve knowledge, summarize it, and ground the process — but you do not yet solve the problem directly.

Instructions:
1. Review the plan and determine which gaps in knowledge need to be filled.  
2. Search in the available data lake, libraries, or custom resources for relevant material.  
3. Collect definitions, formulas, domain facts, or background knowledge.  
4. Summarize findings concisely and clearly, with enough depth to support execution.  
5. If there are multiple conflicting sources, compare and highlight the differences.  
6. Conclude with a short list of insights and resources ready for the next phase.

Rules:
- Do not invent facts — stay grounded in provided resources.  
- Do not run code or produce final solutions.  
- Focus on retrieving, interpreting, and organizing knowledge for later use.
- Try to learn the information for each task one step at a time. Do not try to learn everything at once.

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

        # Add custom resources to format dict if they exist
        if custom_tools_formatted:
            format_dict["custom_tools"] = "\n".join(custom_tools_formatted)
        if custom_data_formatted:
            format_dict["custom_data"] = "\n".join(custom_data_formatted)
        if custom_software_formatted:
            format_dict["custom_software"] = "\n".join(custom_software_formatted)

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

Instructions:
1. Re-check the PLAN and LEARN summaries to ensure alignment.  
2. Write clear, efficient code or step-by-step calculations.  
3. Use the appropriate tools and libraries.  
4. Only produce `<execute>...</execute>` blocks for code execution — no explanatory text inside them.  
5. After execution, output structured results that can be assessed.  
6. If an error occurs, explain it briefly and suggest corrections.
7. DO NOT ASK THE USER FOR FEEDBACK. You must solve the task autonomously in the best interest of rigor.

Rules:
- Only code and computation happen here — no extra commentary outside results.  
- Do not re-plan or re-learn; only act on what is prepared.  
- Precision and correctness are the priority.
- YOU CANNOT DIRECTLY INTERACT WITH THE USER AND ASK FOR FEEDBACK.
- Try to solve the plan one task at a time. Do not try to solve everything at once.

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

        # Add custom resources to format dict if they exist
        if custom_tools_formatted:
            format_dict["custom_tools"] = "\n".join(custom_tools_formatted)
        if custom_data_formatted:
            format_dict["custom_data"] = "\n".join(custom_data_formatted)
        if custom_software_formatted:
            format_dict["custom_software"] = "\n".join(custom_software_formatted)

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

Instructions:
1. Compare the execution output with the plan objectives.  
2. Check for correctness, completeness, and consistency.  
3. Identify errors, missing pieces, or improvements.  
4. Decide whether the workflow should:
   - Go back to PLAN (if the approach itself is flawed),  
   - Go back to LEARN (if more knowledge is needed),  
   - Go back to EXECUTE (if code/results need fixing),  
   - Or move to SHARE (if the solution is ready).  
5. Justify your routing decision clearly.
6. DO NOT ASK THE USER FOR FEEDBACK. Make the decision independently in the best interest of rigor.
7. Try to route to plan after each successfiul task completion within plan. Try to update the plan so the user can see the progress. Only route to learn and execute if there are problems or bugs within those sections.

Rules:
- DO NOT WRITE ANY NEW CODE AND DO NOT CHANGE OR CONTINUE ANY CODE.
- Be critical but constructive.
- DO NOT WRITE FOLLOW UP QUESTIONS
- Route decisively: <goplan>, <golearn>, <goexecute>, or <goshare>. THESE ARE THE ONLY OPTIONS.
- YOU MUST PICK ONE OF THE ROUTES! THIS IS ABOSLUTELY MANDATORY.
- YOU CANNOT DIRECTLY END. IF YOU WANT TO END OR FINISH, YOU MUST ROUTE TO SHARE. IF YOU DON'T ROUTE TO SHARE, YOU CANNOT END OR FINISH SINCE YOU WILL LOOP INDEFINITELY.
- YOU CANNOT CONTINUE TO THE NEXT STEP WITHOUT ASSESSING.
- YOU CANNOT DIRECTLY INTERACT WITH THE USER AND ASK FOR FEEDBACK.

In each response, you must include EITHER <goplan>, <golearn>, <goexecute>, or <goshare> tag. Not multiple at the same time. YOU DO NOT NEED TO CLOSE THE <goplan>, <golearn>, <goexecute>, OR <goshare> TAGS. Do not respond with messages without any tags. No empty messages.
If you provide more than one tag, even if you mentioned them in passing, you will misroute yourself. So be careful and only provide one tag.
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

        # Add custom resources to format dict if they exist
        if custom_tools_formatted:
            format_dict["custom_tools"] = "\n".join(custom_tools_formatted)
        if custom_data_formatted:
            format_dict["custom_data"] = "\n".join(custom_data_formatted)
        if custom_software_formatted:
            format_dict["custom_software"] = "\n".join(custom_software_formatted)

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
        self.system_prompts["PLAN"] = self._plan_system_prompt(
            tool_desc=tool_desc,
            data_lake_content=data_lake_with_desc,
            library_content_list=library_content_list,
            self_critic=self_critic,
            is_retrieval=is_retrieval,
            custom_tools=custom_tools,
            custom_data=custom_data,
            custom_software=custom_software,
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

        log_llm_event("overview_assess_PROMPT", prompt)
    
        ai = llm.invoke([SystemMessage(content="You refine outputs into concrete, actionable fixes."),
                        HumanMessage(content=prompt)])
        feedback = str(getattr(ai, "content", ai))

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
        """Initialize memory manager for this state if not already present."""
        arts = state.setdefault("artifacts", {})
        
        # Initialize MemoryManager if not present
        if "memory_manager" not in arts:
            from bioplease.agent.memory import MemoryManager
            arts["memory_manager"] = MemoryManager(
                config=self.memory_config,
                llm=self.llm
            )
        
        # Legacy compatibility: also keep the old memory dict structure
        mem = arts.setdefault("memory", {})
        mem.setdefault("long_term_summary", "")
        mem.setdefault("events", [])
        
        return arts["memory_manager"]

    def _summarize_messages_for_long_term(self, prev_summary: str, new_msgs: list[BaseMessage]) -> str:
        """
        Condense recent dialog into a compact summary (legacy method).
        Now uses MemoryManager internally for better compression.
        """
        if not new_msgs:
            return prev_summary or ""
        
        # Use the new MemoryManager for summarization
        from bioplease.agent.memory import MemoryManager
        temp_manager = MemoryManager(config=self.memory_config, llm=self.llm)
        temp_manager.long_term_summary = prev_summary
        
        # Generate summary
        new_summary = temp_manager._generate_summary(new_msgs)
        
        return new_summary

    def _roll_memory(self, state, phase_tag: str):
        """
        Update long-term summary with newest messages using MemoryManager.
        
        Args:
            state: Agent state containing messages
            phase_tag: Current phase (PLAN, LEARN, EXECUTE, etc.)
        """
        # Get or create memory manager
        memory_manager = self._ensure_mem(state)
        
        # Get non-system messages to process
        msgs = [m for m in state.get("messages", []) if not isinstance(m, SystemMessage)]
        
        # Add messages to memory manager (it will handle compression)
        if msgs:
            # Only add new messages that aren't already in memory
            current_count = len(memory_manager.short_term_messages)
            new_messages = msgs[current_count:] if current_count < len(msgs) else msgs[-1:]
            
            for msg in new_messages:
                memory_manager.add_message(msg, phase=phase_tag)
        
        # Add event breadcrumb
        memory_manager.add_event(
            phase=phase_tag,
            note=f"Rolled memory at {phase_tag}",
            metadata={"message_count": len(msgs)}
        )
        
        # Update legacy memory dict for backward compatibility
        mem = state["artifacts"]["memory"]
        mem["long_term_summary"] = memory_manager.long_term_summary
        mem["events"] = memory_manager.events

    def _prune_history(self, state):
        """
        Prune message history using MemoryManager for intelligent compression.
        
        This method now uses the MemoryManager to keep an optimal number of
        recent messages while compressing older ones into long-term memory.
        """
        memory_manager = self._ensure_mem(state)
        
        # Get all messages
        msgs = state.get("messages", [])
        
        # Separate system and non-system messages
        sys_msgs = [m for m in msgs if isinstance(m, SystemMessage)]
        non_sys = [m for m in msgs if not isinstance(m, SystemMessage)]
        
        # If we have too many messages, force compression
        if len(non_sys) > self.memory_config.short_window + self.memory_config.summary_update_threshold:
            # Update memory manager with all messages
            memory_manager.short_term_messages = non_sys
            memory_manager.compress_to_long_term()
            
            # Update state with compressed messages
            non_sys = memory_manager.short_term_messages
        
        # Reconstruct messages: system messages + recent non-system
        state["messages"] = sys_msgs + non_sys[-self.memory_config.short_window:]
        
        # Log memory stats
        stats = memory_manager.get_stats()
        if stats["over_budget"]:
            print(f"⚠️  Memory over budget: ~{stats['estimated_tokens']} tokens")
            # Force aggressive compression
            memory_manager.force_compression(target_messages=max(3, self.memory_config.short_window // 2))
            state["messages"] = sys_msgs + memory_manager.short_term_messages

    # ==================== NEW: Memory Management Utilities ====================
    
    def get_memory_stats(self, state=None) -> dict:
        """
        Get current memory statistics.
        
        Args:
            state: Optional agent state. If None, uses instance memory_manager
            
        Returns:
            Dictionary with memory statistics
        """
        if state and "artifacts" in state and "memory_manager" in state["artifacts"]:
            memory_manager = state["artifacts"]["memory_manager"]
        elif self.memory_manager:
            memory_manager = self.memory_manager
        else:
            return {"error": "No memory manager initialized"}
        
        return memory_manager.get_stats()
    
    def save_memory_snapshot(self, state, filepath: str = None) -> str:
        """
        Save current memory state to disk.
        
        Args:
            state: Agent state
            filepath: Optional custom filepath
            
        Returns:
            Path to saved file
        """
        memory_manager = self._ensure_mem(state)
        memory_manager.save(filepath)
        return filepath or "auto-saved"
    
    def load_memory_snapshot(self, state, filepath: str):
        """
        Load memory from a saved snapshot.
        
        Args:
            state: Agent state
            filepath: Path to memory snapshot file
        """
        memory_manager = self._ensure_mem(state)
        memory_manager.load(filepath)
        
        # Update state messages from loaded memory
        state["messages"] = memory_manager.get_context_messages(include_summary=False)
        
        # Update legacy memory dict
        mem = state["artifacts"]["memory"]
        mem["long_term_summary"] = memory_manager.long_term_summary
        mem["events"] = memory_manager.events
    
    def clear_memory(self, state, keep_long_term: bool = False):
        """
        Clear memory for a fresh start.
        
        Args:
            state: Agent state
            keep_long_term: Whether to keep the long-term summary
        """
        memory_manager = self._ensure_mem(state)
        memory_manager.clear(keep_long_term=keep_long_term)
        
        # Clear state messages
        state["messages"] = []
        
        # Update legacy memory dict
        mem = state["artifacts"]["memory"]
        mem["long_term_summary"] = memory_manager.long_term_summary
        mem["events"] = []
    
    def configure_memory(
        self,
        short_window: int = None,
        summary_max_chars: int = None,
        max_total_tokens: int = None,
        compression_ratio: float = None,
        auto_save: bool = None,
    ):
        """
        Update memory configuration parameters.
        
        Args:
            short_window: Number of recent messages to keep
            summary_max_chars: Max characters in long-term summary
            max_total_tokens: Token budget for memory
            compression_ratio: Target compression ratio (0.0-1.0)
            auto_save: Whether to auto-save memory snapshots
        """
        if short_window is not None:
            self.memory_config.short_window = short_window
            self.short_window = short_window
        
        if summary_max_chars is not None:
            self.memory_config.summary_max_chars = summary_max_chars
            self.summary_max_chars = summary_max_chars
        
        if max_total_tokens is not None:
            self.memory_config.max_total_tokens = max_total_tokens
        
        if compression_ratio is not None:
            self.memory_config.compression_ratio = compression_ratio
        
        if auto_save is not None:
            self.memory_config.auto_save = auto_save
        
        print(f"✓ Memory configured: window={self.memory_config.short_window}, "
              f"max_tokens={self.memory_config.max_total_tokens}, "
              f"compression={self.memory_config.compression_ratio}")
    
    def print_memory_summary(self, state):
        """
        Print a human-readable summary of current memory state.
        
        Args:
            state: Agent state
        """
        memory_manager = self._ensure_mem(state)
        stats = memory_manager.get_stats()
        
        print("\n" + "="*60)
        print("MEMORY STATUS")
        print("="*60)
        print(f"Short-term messages: {stats['short_term_count']}")
        print(f"Long-term summary length: {stats['long_term_length']} chars")
        print(f"Total compressions: {stats['compressions']}")
        print(f"Estimated tokens: ~{stats['estimated_tokens']}")
        print(f"Token budget: {self.memory_config.max_total_tokens}")
        
        if stats['over_budget']:
            print("⚠️  STATUS: OVER BUDGET")
        else:
            usage_pct = (stats['estimated_tokens'] / self.memory_config.max_total_tokens) * 100
            print(f"✓ STATUS: OK ({usage_pct:.1f}% of budget)")
        
        print(f"\nEvents tracked: {stats['events_count']}")
        
        if memory_manager.long_term_summary:
            print("\nLong-term summary preview:")
            preview = memory_manager.long_term_summary[:200]
            print(f"  {preview}...")
        
        print("="*60 + "\n")

    # ==================== END: Memory Management Utilities ====================
    
    
    
    
    
    
    
    
    

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
            memory_hint = SystemMessage(content=f"[LONG-TERM SUMMARY]\n{lts}")

            messages = [memory_hint, SystemMessage(content=self._prompt_for("PLAN"))] + state["messages"]
            llm = self._llm_for("PLAN")
            log_llm_event("plan_prompt", messages)
            response = llm.invoke(messages)
            msg = str(response.content).strip()
            log_llm_event("plan_response", msg)

            state["messages"].append(AIMessage(content=msg))

            # <<< NEW: prune short-term >>>
            self._prune_history(state)

            state["next_step"] = "learn"
            return state


        def learn(state: AgentState) -> AgentState:
            state["phase"] = "LEARN"
            _announce_phase(state)

            self._ensure_mem(state)
            lts = state["artifacts"]["memory"].get("long_term_summary","")
            memory_hint = SystemMessage(content=f"[LONG-TERM SUMMARY]\n{lts}")

            messages = [memory_hint, SystemMessage(content=self._prompt_for("LEARN"))] + state["messages"]
            log_llm_event("learn_prompt", messages)
            llm = self._llm_for("LEARN")
            response = llm.invoke(messages)
            msg = str(response.content).strip()
            log_llm_event("learn_response", msg)
            state["messages"].append(AIMessage(content=msg))

            # <<< NEW >>>
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
            
            phase_hint = f"""
[PHASE: EXECUTE]
Additional Dynamic Rules:
• Save all plots/figures into this directory:
  {self.run_figures_dir}
• After saving each plot at variable `img_path`, also do BOTH:
  1) Append to a manifest (create if missing):
     import json, os
     os.makedirs("{self.run_figures_dir}", exist_ok=True)
     with open(os.path.join("{self.run_figures_dir}", "manifest.jsonl"), "a", encoding="utf-8") as _f:
         _f.write(json.dumps({{"path": img_path, "caption": caption, "label": "Figure"}}) + "\\n")
  2) Print a log marker so SHARE can harvest it:
     <<FIGURE:{self.run_figures_dir}/{{os.path.basename(img_path)}}|{{caption}}>>
"""

            
            messages = [SystemMessage(content=phase_hint)] + [SystemMessage(content=self._prompt_for("EXECUTE"))] + state["messages"]
            llm = self._llm_for("EXECUTE")
            log_llm_event("execute_prompt", messages)
            response = llm.invoke(messages)
            
            msg = str(response.content)
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
                # Robust retry counter stored in state
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
                    # Nudge once more with a precise instruction
                    state["messages"].append(HumanMessage(
                        content="[FORMAT ERROR] Emit exactly ONE of: <execute>...</execute> (code only). No other tags."
                    ))
                    state["next_step"] = "execute"
                    return state  # retry execute
            
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

                if len(result) > 10000:
                    result = (
                        "The output is too long to be added to context. Here are the first 10K characters...\n"
                        + result[:10000]
                    )
                observation = f"\n<observation>{result}</observation>"
                state["messages"].append(AIMessage(content=observation.strip()))
                
            self._prune_history(state)
                
            # After execution, always go to ASSESS
            state["next_step"] = "assess"
            return state


        def assess(state: AgentState) -> AgentState:

            state["phase"] = "ASSESS"
            _announce_phase(state)

            messages = [SystemMessage(content=self._prompt_for("ASSESS"))] + state["messages"]
            llm = self._llm_for("ASSESS")
            log_llm_event("assess_prompt", messages)
            response = llm.invoke(messages)
            msg = str(response.content)
            log_llm_event("assess_response", msg)
            state["messages"].append(AIMessage(content=msg.strip()))
            
            # new
            self._roll_memory(state, "ASSESS")
            self._prune_history(state)
            
            msg_lower = msg.lower()
            
            if "<goplan>" in msg_lower:
                state["next_step"] = "plan"
            elif "<golearn>" in msg_lower:
                state["next_step"] = "learn"
            elif "<goexecute>" in msg_lower:
                state["next_step"] = "execute"
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
                    state["messages"].append(AIMessage(content="Terminating: model failed to emit <goplan>, <golearn>, <goexecute>, or <goshare> after 3 retries."))
                    state["next_step"] = "end"
                else:
                    # Nudge once more with a precise instruction
                    state["messages"].append(HumanMessage(
                        content="[FORMAT ERROR] Emit EITHER <goplan>, <golearn>, <goexecute>, <goshare> tag. Not multiple at the same time. Do not respond with messages without any tags. No empty messages."
                    ))
                    state["next_step"] = "assess"  # retry assess
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

            # Aggregate conversation history (messages) for logging
            all_msgs = [str(m.content) for m in state["messages"]]
            conversation_blob = "\n".join(all_msgs)

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
===== END LOGS =====
"""
            paper_resp = llm_share.invoke([HumanMessage(content=paper_prompt)])
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

            
            
            
            # new
            self._prune_history(state)
            
            
            state["next_step"] = "overview_assess"
            return state
        

        def routing_function_execute(
            state: AgentState,
        ) -> Literal["execute", "assess"]:
            next_step = state.get("next_step")
            if next_step == "execute":
                return "execute"
            elif next_step == "assess":
                return "assess"
            else:
                print(f"[ASSESS ROUTER] unexpected next_step={str(next_step)}; defaulting to assess")
                return "assess"
            
        
        def routing_function_assess(
            state: AgentState,
        ) -> Literal["plan", "learn", "execute", "assess", "share"]:
            next_step = state.get("next_step")
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
            "assess": "assess",
        }
        )  
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
        workflow.add_edge("share", "overview_assess")
        
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
        self.run_figures_dir = os.path.join(self.run_dir, "figures")
        self.run_artifacts_dir = os.path.join(self.run_dir, "artifacts")
        for d in [runs_root, self.run_dir, self.run_logs_dir, self.run_papers_dir, self.run_figures_dir, self.run_artifacts_dir]:
            os.makedirs(d, exist_ok=True)

        # Propagate environment variables so EXECUTE code can discover the run directory
        os.environ["BIOPLEASE_DATA"] = self.path            # e.g., .../bioplease_data
        os.environ["BIOPLEASE_RUN_DIR"] = self.run_dir

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
