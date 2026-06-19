# bioplease/agent/pleas.py
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

# Reuse your existing safe executors & helpers
from ..utils import (
    run_r_code,          # R executor
    run_bash_script,     # Bash executor
    run_cli_command,     # CLI executor
    run_with_timeout,    # Timeout wrapper for Python exec
)

# ------------------------------
# Pydantic schemas + JSON helpers
# ------------------------------

class PlanStepSchema(BaseModel):
    id: str
    goal: str
    method: str
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)

class PlanSchema(BaseModel):
    decisions: list[str]
    assumptions: list[str]
    risks: list[str]
    steps: list[PlanStepSchema]
    success_criteria: list[str]

class EvidenceSchema(BaseModel):
    source_id: str
    url: str
    quote: str
    quality: str                  # "peer-reviewed" | "preprint" | "official-doc"
    checked: bool
    conflicts_with: list[str] = Field(default_factory=list)
    resolution_note: str = ""

def _extract_json_loose(text: str) -> dict | list:
    """Fallback: pull a JSON object/array out of free-form text."""
    if not text:
        raise ValueError("Empty model response")
    # Strip common markdown fences
    text = re.sub(r"^```(?:json|JSON)?\s*|```$", "", text.strip(), flags=re.MULTILINE)
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find the first {...} or [...]
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON found in response")
    candidate = m.group(1)
    # Gentle single-quote to double-quote substitution if needed
    if "'" in candidate and '"' not in candidate:
        candidate = candidate.replace("'", '"')
    return json.loads(candidate)

# ------------------------------
# Dataclasses for run state
# ------------------------------

@dataclass
class Evidence:
    source_id: str
    url: str
    quote: str
    quality: str
    checked: bool
    conflicts_with: List[str] = field(default_factory=list)
    resolution_note: str = ""

@dataclass
class PlanStep:
    id: str
    goal: str
    method: str
    inputs: List[str]
    outputs: List[str]
    success_criteria: List[str]

@dataclass
class Plan:
    decisions: List[str]
    assumptions: List[str]
    risks: List[str]
    steps: List[PlanStep]
    success_criteria: List[str]

@dataclass
class ExecutionRecord:
    step_id: str
    kind: str               # "python" | "bash" | "r" | "cli"
    content: str            # code or command
    runtime_s: float
    exit_code: int
    stdout: str
    stderr: str
    artifacts: List[str]
    seed: Optional[int] = None
    env: Dict[str, str] = field(default_factory=dict)

@dataclass
class Assessment:
    rubric_scores: Dict[str, int]
    rationale: str
    passed: bool
    reviewer_model: str

@dataclass
class PLEASState:
    task: str
    plan: Optional[Plan] = None
    evidence: List[Evidence] = field(default_factory=list)
    executions: List[ExecutionRecord] = field(default_factory=list)
    assessment: Optional[Assessment] = None
    report_md: Optional[str] = None
    report_json: Optional[Dict[str, Any]] = None
    budget: Dict[str, float] = field(default_factory=lambda: {"tokens": 100_000, "seconds": 900, "usd": 2.0})

# ------------------------------
# PLEAS Runner
# ------------------------------

class PLEASRunner:
    """
    Formal scientific AI Agent runner following PLEAS:
    - PLAN: rigorous, auditable plan (decisions/assumptions/risks/steps/success_criteria)
    - LEARN: literature verification (trusted sources, conflicts resolved)
    - EXECUTE: cost-managed, reproducible runs (R/Bash/CLI/Python with logs)
    - ASSESS: independent model rubric-based review
    - SHARE: standardized report + feedback schema
    """
    def __init__(self, llm: BaseChatModel, reviewer_llm: Optional[BaseChatModel] = None):
        self.llm = llm
        self.reviewer_llm = reviewer_llm or llm  # In production, prefer a different model/provider here.

    # ---------- PLAN ----------
    def plan(self, state: PLEASState) -> PLEASState:
        prompt = f"""
You are drafting a rigorous, auditable PLAN for this task: {state.task}

Return ONLY a JSON object with keys:
- decisions[]: explicit choices that shape the approach
- assumptions[]: testable assumptions
- risks[]: key risks and mitigations (short)
- steps[]: list of steps, each with {{id, goal, method, inputs[], outputs[], success_criteria[]}}
- success_criteria[]: measurable success definitions

No extra prose, no markdown fences.
"""
        # Prefer structured output; fallback to loose extraction
        try:
            parser_llm = self.llm.with_structured_output(PlanSchema)
            parsed: PlanSchema = parser_llm.invoke([HumanMessage(content=prompt)])
            plan_obj = parsed.model_dump()
        except Exception:
            raw = self.llm.invoke([HumanMessage(content=prompt)]).content
            plan_obj = _extract_json_loose(raw)

        # Minimal schema checks
        for key in ["decisions", "assumptions", "risks", "steps", "success_criteria"]:
            if key not in plan_obj or not plan_obj[key]:
                raise ValueError(f"PLAN missing '{key}'")

        state.plan = Plan(
            decisions=plan_obj["decisions"],
            assumptions=plan_obj["assumptions"],
            risks=plan_obj["risks"],
            steps=[PlanStep(**s) for s in plan_obj["steps"]],
            success_criteria=plan_obj["success_criteria"],
        )
        return state

    # ---------- LEARN ----------
    def learn(self, state: PLEASState) -> PLEASState:
        plan_focus = (state.plan.decisions if state.plan else []) + (state.plan.assumptions if state.plan else [])
        prompt = f"""
Identify authoritative sources supporting or contradicting these items:
{plan_focus}

Return ONLY a JSON array of evidence objects:
[
  {{
    "source_id": "short label",
    "url": "https://...",
    "quote": "verbatim or near-verbatim support/contradiction",
    "quality": "peer-reviewed|preprint|official-doc",
    "checked": true,
    "conflicts_with": ["source_id_2"],
    "resolution_note": "If conflicts exist, explain the resolution briefly"
  }},
  ...
]
No markdown, no commentary.
"""
        try:
            parser_llm = self.llm.with_structured_output(list[EvidenceSchema])  # type: ignore[arg-type]
            parsed_list = parser_llm.invoke([HumanMessage(content=prompt)])
            ev_list = [e.model_dump() for e in parsed_list]
        except Exception:
            raw = self.llm.invoke([HumanMessage(content=prompt)]).content
            ev_list = _extract_json_loose(raw)

        if not isinstance(ev_list, list) or not ev_list:
            raise ValueError("LEARN produced no evidence")
        for e in ev_list:
            for req in ["source_id", "url", "quote", "quality", "checked"]:
                if req not in e:
                    raise ValueError(f"Evidence missing '{req}'")

        state.evidence = [Evidence(**e) for e in ev_list]
        return state

    # ---------- EXECUTE ----------
    def execute(self, state: PLEASState, seed: int = 1337) -> PLEASState:
        if not state.plan or not state.plan.steps:
            return state

        spent_s = 0.0
        for step in state.plan.steps:
            if spent_s >= state.budget["seconds"]:
                break

            ask = f"""
Propose one executable for step {step.id} (goal: {step.goal}; method: {step.method}).
Return ONLY JSON: {{"kind": "bash|cli|r|python", "content": "<code_or_command>", "artifacts": []}}.
Be minimal, deterministic; respect inputs/outputs.
"""
            try:
                tool_choice = self.llm.with_structured_output(dict).invoke([HumanMessage(content=ask)])
                tool = tool_choice
            except Exception:
                raw = self.llm.invoke([HumanMessage(content=ask)]).content
                tool = _extract_json_loose(raw)

            t0 = time.time()
            kind = str(tool.get("kind", "python")).lower()
            content = str(tool.get("content", ""))

            if kind == "bash":
                out = run_bash_script(content)
                exit_code = 0 if not str(out).startswith("Error") else 1
                stdout, stderr = (str(out), "") if exit_code == 0 else ("", str(out))
            elif kind == "cli":
                out = run_cli_command(content)
                exit_code = 0 if not str(out).startswith("Error") else 1
                stdout, stderr = (str(out), "") if exit_code == 0 else ("", str(out))
            elif kind == "r":
                out = run_r_code(content)
                exit_code = 0 if not str(out).startswith("Error") else 1
                stdout, stderr = (str(out), "") if exit_code == 0 else ("", str(out))
            else:
                code = content

                def _py():
                    loc: Dict[str, Any] = {}
                    exec(code, {}, loc)  # In production, sandbox this
                    return str(loc.get("result", ""))

                out = run_with_timeout(_py, timeout=120)
                exit_code = 0 if not str(out).startswith("ERROR") else 1
                stdout, stderr = (str(out), "") if exit_code == 0 else ("", str(out))

            dt = time.time() - t0
            spent_s += dt

            state.executions.append(
                ExecutionRecord(
                    step_id=step.id,
                    kind=kind,
                    content=content,
                    runtime_s=dt,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr,
                    artifacts=tool.get("artifacts", []) if isinstance(tool, dict) else [],
                    seed=seed,
                    env=dict(os.environ),
                )
            )

            if spent_s >= state.budget["seconds"]:
                break

        return state

    # ---------- ASSESS ----------
    def assess(self, state: PLEASState, pass_threshold: float = 0.7) -> PLEASState:
        rubric_keys = ["correctness", "evidence_quality", "reproducibility", "cost_efficiency", "clarity"]
        prompt = f"""
You are an INDEPENDENT reviewer. Score each 0-5:
{rubric_keys}.
Consider evidence, execution logs, budgets, and success_criteria.
Return ONLY JSON:
{{
  "scores": {{"correctness":0, "evidence_quality":0, "reproducibility":0, "cost_efficiency":0, "clarity":0}},
  "rationale": "brief justification",
  "passed": true
}}
"""
        try:
            parsed = self.reviewer_llm.with_structured_output(dict).invoke([HumanMessage(content=prompt)])
            review = parsed
        except Exception:
            raw = self.reviewer_llm.invoke([HumanMessage(content=prompt)]).content
            review = _extract_json_loose(raw)

        scores = review.get("scores", {})
        if isinstance(scores, list):
            scores = {k: int(scores[i]) for i, k in enumerate(rubric_keys)}
        else:
            scores = {k: int(scores.get(k, 0)) for k in rubric_keys}

        passed = bool(review.get("passed", False))
        rationale = str(review.get("rationale", ""))

        # Try to name the reviewer model (best-effort)
        reviewer_model = getattr(self.reviewer_llm, "model", None) or getattr(self.reviewer_llm, "model_name", "unknown")
        state.assessment = Assessment(
            rubric_scores=scores,
            rationale=rationale,
            passed=passed,
            reviewer_model=str(reviewer_model),
        )
        return state

    # ---------- SHARE ----------
    def share(self, state: PLEASState) -> PLEASState:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        title = f"PLEAS Report — {ts}"

        md = [f"# {title}", "## Task", state.task, "## Plan"]
        if state.plan:
            md += [
                f"- **Decisions:** {state.plan.decisions}",
                f"- **Assumptions:** {state.plan.assumptions}",
                f"- **Risks:** {state.plan.risks}",
                "### Steps:",
            ] + [f"  - {s.id}: {s.goal} via {s.method}" for s in state.plan.steps]

        md += ["## Evidence"] + [
            f"- [{e.source_id}] {e.url}\n  > {e.quote} ({e.quality})"
            + (f"\n  Conflicts: {e.conflicts_with} — {e.resolution_note}" if e.conflicts_with else "")
            for e in state.evidence
        ]

        md += ["## Executions"] + [
            (
                f"- Step {x.step_id} [{x.kind}] exit={x.exit_code} time={x.runtime_s:.2f}s\n"
                f"```{('' if x.kind!='python' else 'python')}\n{x.content}\n```\n"
                f"stdout:\n```\n{x.stdout[:4000]}\n```"
                + (f"\nstderr:\n```\n{x.stderr[:2000]}\n```" if x.stderr else "")
            )
            for x in state.executions
        ]

        if state.assessment:
            md += [
                "## Assessment",
                f"- Scores: {state.assessment.rubric_scores}",
                f"- Passed: {state.assessment.passed}",
                f"- Rationale: {state.assessment.rationale}",
            ]

        md += [
            "## Feedback (fill):",
            "- Was the plan appropriate?",
            "- Did citations feel trustworthy?",
            "- Were outputs reproducible?",
            "- What to improve next time?",
        ]
        state.report_md = "\n\n".join(md)

        state.report_json = {
            "task": state.task,
            "plan": state.plan.__dict__ if state.plan else None,
            "evidence": [e.__dict__ for e in state.evidence],
            "executions": [x.__dict__ for x in state.executions],
            "assessment": state.assessment.__dict__ if state.assessment else None,
            "feedback_schema": {
                "fields": [
                    {"name": "plan_quality", "type": "int", "0-5": True},
                    {"name": "citation_trust", "type": "int", "0-5": True},
                    {"name": "reproducibility", "type": "int", "0-5": True},
                    {"name": "comments", "type": "string"},
                ]
            },
        }
        return state

    # ---------- FULL RUN ----------
    def run(self, task: str, budget: Optional[Dict[str, float]] = None) -> PLEASState:
        state = PLEASState(task=task)
        if budget:
            state.budget.update(budget)

        state = self.plan(state)
        state = self.learn(state)
        state = self.execute(state)
        state = self.assess(state)
        state = self.share(state)
        return state
