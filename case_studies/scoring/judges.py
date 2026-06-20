"""
Multi-model judge interface.

Each judge independently scores all blinded responses for a given task
using the rubric criteria.  Judges operate with "high thinking" / extended
reasoning enabled so the evaluation itself benefits from deep
chain-of-thought before emitting a score.

Supported providers
-------------------
* OpenAI  (GPT-5.5-thinking-extended)
* Anthropic  (Claude Opus 4.8 with extended thinking)
* Google  (Gemini 3.1 Pro with thinking mode)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from .blinding import BlindedResponse
from .config import JudgeModel, RETRY_ATTEMPTS, RETRY_BACKOFF_S, REQUEST_TIMEOUT_S
from .rubric import BenchmarkTask

log = logging.getLogger(__name__)


@dataclass
class JudgeScore:
    judge_id: str
    task_id: str
    response_label: str
    score: int            # 0, 1, or 2
    rationale: str


# ── Prompt construction ──────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert biomedical-informatics evaluator for a blinded \
scientific-workflow benchmark.  You will receive a TASK description, \
EXPECTED output, SCORING CRITERIA (0/1/2 rubric), and a RESPONSE to score.

IMPORTANT BLINDING RULES:
- You do NOT know which system produced this response.
- Judge ONLY on factual correctness, tool/database usage, and reasoning quality.
- Do NOT attempt to infer or guess the identity of the system.
- Apply the rubric strictly and consistently.

OUTPUT FORMAT — return valid JSON and nothing else:
{
  "score": <0 or 1 or 2>,
  "rationale": "<one-paragraph justification citing specific rubric criteria>"
}
"""


def _build_user_prompt(
    task: BenchmarkTask,
    response: BlindedResponse,
) -> str:
    return f"""\
=== TASK ({task.task_id}) ===
{task.task_text}

=== EXPECTED OUTPUT ===
{task.expected_output}

=== SCORING CRITERIA ===
{task.scoring_criteria}

=== {response.label} (TO SCORE) ===
{response.text}

Score this response according to the rubric.  Return JSON only."""


# ── Provider dispatch ────────────────────────────────────────────────────────

def _call_openai(model: JudgeModel, system: str, user: str) -> str:
    import openai
    client = openai.OpenAI(api_key=model.api_key, timeout=REQUEST_TIMEOUT_S)
    params: dict[str, Any] = {
        "model": model.model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **model.extra_params,
    }
    resp = client.chat.completions.create(**params)
    return resp.choices[0].message.content or ""


def _call_anthropic(model: JudgeModel, system: str, user: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=model.api_key)
    params: dict[str, Any] = {
        "model": model.model_id,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "max_tokens": model.extra_params.get("max_tokens", 16_384),
    }
    thinking_cfg = model.extra_params.get("thinking")
    if thinking_cfg:
        params["thinking"] = thinking_cfg
    resp = client.messages.create(**params)
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


def _call_google(model: JudgeModel, system: str, user: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=model.api_key)
    gen_model = genai.GenerativeModel(
        model.model_id,
        system_instruction=system,
        generation_config=genai.types.GenerationConfig(
            temperature=model.extra_params.get("temperature", 0.0),
            max_output_tokens=model.extra_params.get("max_output_tokens", 16_384),
        ),
    )
    thinking_cfg = model.extra_params.get("thinking_config")
    gen_kwargs: dict[str, Any] = {}
    if thinking_cfg:
        gen_kwargs["thinking_config"] = thinking_cfg
    resp = gen_model.generate_content(user, **gen_kwargs)
    return resp.text or ""


_DISPATCH = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "google": _call_google,
}


# ── Scoring logic ────────────────────────────────────────────────────────────

def _parse_score(raw: str) -> tuple[int, str]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        import re
        m = re.search(r'"score"\s*:\s*(\d)', raw)
        if m:
            score = int(m.group(1))
            r = re.search(r'"rationale"\s*:\s*"([^"]+)"', raw)
            return score, (r.group(1) if r else "parse-fallback")
        raise ValueError(f"Cannot parse judge output: {raw[:200]}")
    score = int(obj["score"])
    if score not in (0, 1, 2):
        raise ValueError(f"Score {score} not in {{0,1,2}}")
    return score, obj.get("rationale", "")


def score_one(
    judge: JudgeModel,
    task: BenchmarkTask,
    response: BlindedResponse,
) -> JudgeScore:
    """Have a single judge score a single blinded response with retries."""
    system_prompt = _SYSTEM_PROMPT
    user_prompt = _build_user_prompt(task, response)
    call_fn = _DISPATCH.get(judge.provider)
    if call_fn is None:
        raise ValueError(f"Unknown provider: {judge.provider}")

    last_err: Exception | None = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            raw = call_fn(judge, system_prompt, user_prompt)
            score, rationale = _parse_score(raw)
            return JudgeScore(
                judge_id=judge.judge_id,
                task_id=task.task_id,
                response_label=response.label,
                score=score,
                rationale=rationale,
            )
        except Exception as exc:
            last_err = exc
            log.warning(
                "Judge %s attempt %d/%d failed for %s/%s: %s",
                judge.judge_id, attempt + 1, RETRY_ATTEMPTS,
                task.task_id, response.label, exc,
            )
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_BACKOFF_S * (2 ** attempt))

    return JudgeScore(
        judge_id=judge.judge_id,
        task_id=task.task_id,
        response_label=response.label,
        score=-1,
        rationale=f"SCORING FAILED after {RETRY_ATTEMPTS} attempts: {last_err}",
    )
