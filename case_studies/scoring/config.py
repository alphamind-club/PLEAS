"""
Judge model configurations and API setup for blinded scoring.

Three independent judge models score every response without knowing
which system produced it.  Each judge uses "high thinking" / extended
reasoning so the evaluation itself benefits from deep chain-of-thought.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class JudgeModel:
    judge_id: str
    provider: str
    model_id: str
    api_key_env: str
    extra_params: dict[str, Any] = field(default_factory=dict)

    @property
    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise EnvironmentError(
                f"Missing API key: set {self.api_key_env} in your environment."
            )
        return key


# ── Judge Panel ──────────────────────────────────────────────────────────────

JUDGE_PANEL: list[JudgeModel] = [
    JudgeModel(
        judge_id="claude-opus-48",
        provider="anthropic",
        model_id="claude-opus-4-8",
        api_key_env="ANTHROPIC_API_KEY",
        extra_params={
            "max_tokens": 16_384,
            "thinking": {
                "type": "enabled",
                "budget_tokens": 10_000,
            },
        },
    ),
    JudgeModel(
        judge_id="claude-sonnet-46",
        provider="anthropic",
        model_id="claude-sonnet-4-6",
        api_key_env="ANTHROPIC_API_KEY",
        extra_params={
            "max_tokens": 16_384,
            "thinking": {
                "type": "enabled",
                "budget_tokens": 10_000,
            },
        },
    ),
    JudgeModel(
        judge_id="gemini-25-pro",
        provider="google",
        model_id="gemini-2.5-pro",
        api_key_env="GOOGLE_API_KEY",
        extra_params={
            "temperature": 0.0,
            "max_output_tokens": 16_384,
            "thinking_config": {"thinking_budget": 10_000},
        },
    ),
]


# ── Scoring session defaults ─────────────────────────────────────────────────

SCORING_ROUNDS: int = 1          # increase for repeated-measure reliability
MAX_CONCURRENT_JUDGES: int = 3   # parallel API calls per task
REQUEST_TIMEOUT_S: int = 120
RETRY_ATTEMPTS: int = 3
RETRY_BACKOFF_S: float = 2.0
