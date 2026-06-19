"""Simple CostManager for token/cost/time estimation and usage tracking.

Drop-in utility for rough budget tracking. This is intentionally small and
self-contained so it doesn't force new dependencies (no tokenizers).

Usage:
    from bioplease.agent.cost_manager import CostManager
    cm = CostManager()
    toks = cm.estimate_tokens("Hello world")
    cm.record_usage("gpt-4o-mini", toks)
    print(cm.get_report())

"""
from __future__ import annotations
import time
from math import ceil
from typing import Dict, Optional


class CostManager:
    """Simple token / cost / time estimator + usage tracker.

    - estimate_tokens(text): heuristic tokens = ceil(len(text)/4)
    - estimate_cost(tokens, model): cost in USD (uses price_per_1k_tokens)
    - estimate_time_seconds(tokens, model): estimate seconds (uses ms_per_1k_tokens)
    - record_usage(model, tokens): track totals
    - get_report(): aggregated totals and per-model breakdown
    """

    DEFAULT_MODELS = {
        # ===== OpenAI Models =====
        
        # GPT-4o family (latest stable, Nov 2024)
        "gpt-4o": {
            "price_per_1k_input": 0.0025,
            "price_per_1k_output": 0.01,
            "ms_per_1k": 300,
        },
        "gpt-4o-2024-11-20": {  # Latest stable version
            "price_per_1k_input": 0.0025,
            "price_per_1k_output": 0.01,
            "ms_per_1k": 300,
        },
        "gpt-4o-mini": {
            "price_per_1k_input": 0.00015,
            "price_per_1k_output": 0.0006,
            "ms_per_1k": 150,
        },
        "gpt-4o-mini-2024-07-18": {  # Specific stable version
            "price_per_1k_input": 0.00015,
            "price_per_1k_output": 0.0006,
            "ms_per_1k": 150,
        },
        
        # GPT-4 Turbo
        "gpt-4-turbo": {
            "price_per_1k_input": 0.01,
            "price_per_1k_output": 0.03,
            "ms_per_1k": 350,
        },
        "gpt-4-turbo-2024-04-09": {
            "price_per_1k_input": 0.01,
            "price_per_1k_output": 0.03,
            "ms_per_1k": 350,
        },
        
        # o1 reasoning models
        "o1": {
            "price_per_1k_input": 0.015,
            "price_per_1k_output": 0.06,
            "ms_per_1k": 400,
        },
        "o1-preview": {  # High-performance reasoning
            "price_per_1k_input": 0.015,
            "price_per_1k_output": 0.06,
            "ms_per_1k": 400,
        },
        "o1-mini": {
            "price_per_1k_input": 0.003,
            "price_per_1k_output": 0.012,
            "ms_per_1k": 250,
        },
        "o3-mini": {  # Latest reasoning model (Jan 2025)
            "price_per_1k_input": 0.0011,  # $1.10 per million
            "price_per_1k_output": 0.0044,  # $4.40 per million
            "ms_per_1k": 200,
        },

        # Newer OpenAI / GPT-5 family
        "gpt-5": {
            # Input cost: $1.25 / million → 0.00125 / 1k tokens :contentReference[oaicite:0]{index=0}
            "price_per_1k_input": 1.25 / 1_000,
            # Output cost: $10.00 / million → 0.01 / 1k tokens :contentReference[oaicite:1]{index=1}
            "price_per_1k_output": 10.00 / 1_000,
            # Latency: average ~10.28 seconds for unspecified prompt size → ≈ 10,280 ms for ~1000 tokens :contentReference[oaicite:2]{index=2}
            "ms_per_1k": 10_280,
        },
        "gpt-5-mini": {
            # Input: $0.25 / million → 0.00025 /1k; Output: $2.00 / million → 0.002 /1k :contentReference[oaicite:3]{index=3}
            "price_per_1k_input": 0.25 / 1_000,
            "price_per_1k_output": 2.00 / 1_000,
            # Latency estimate (scaled down): ~4,530 ms (4.53 s in sources) :contentReference[oaicite:4]{index=4}
            "ms_per_1k": 4_530,
        },
        "gpt-5-nano": {
            # Input: $0.05 / million → 0.00005 /1k; Output: $0.40 / million → 0.00040 /1k :contentReference[oaicite:5]{index=5}
            "price_per_1k_input": 0.05 / 1_000,
            "price_per_1k_output": 0.40 / 1_000,
            # Latency: ~3.13 s → ≈ 3,130 ms for ~1000 tokens :contentReference[oaicite:6]{index=6}
            "ms_per_1k": 3_130,
        },
        "gpt-5-codex": {
            # Assume same costs as gpt-5 for now
            "price_per_1k_input": 1.25 / 1_000,
            "price_per_1k_output": 10.00 / 1_000,
            # We might assume same latency as gpt-5
            "ms_per_1k": 10_280,
        },

        # ===== Google Gemini Models =====
        
        # Gemini 2.0 (latest, Dec 2024)
        "gemini-2.0-flash-exp": {
            "price_per_1k_input": 0.0,  # Free during preview
            "price_per_1k_output": 0.0,
            "ms_per_1k": 500,
        },
        "gemini-exp-1206": {
            "price_per_1k_input": 0.0,  # Free during preview  
            "price_per_1k_output": 0.0,
            "ms_per_1k": 800,
        },
        
        # Gemini 1.5 (stable production)
        "gemini-1.5-pro": {
            "price_per_1k_input": 0.00125,  # $1.25 per million
            "price_per_1k_output": 0.005,    # $5.00 per million
            "ms_per_1k": 2000,
        },
        "gemini-1.5-pro-002": {
            "price_per_1k_input": 0.00125,
            "price_per_1k_output": 0.005,
            "ms_per_1k": 2000,
        },
        "gemini-1.5-flash": {
            "price_per_1k_input": 0.000075,  # $0.075 per million
            "price_per_1k_output": 0.0003,    # $0.30 per million
            "ms_per_1k": 800,
        },
        "gemini-1.5-flash-002": {
            "price_per_1k_input": 0.000075,
            "price_per_1k_output": 0.0003,
            "ms_per_1k": 800,
        },
        "gemini-1.5-flash-8b": {
            "price_per_1k_input": 0.0000375,  # $0.0375 per million
            "price_per_1k_output": 0.00015,    # $0.15 per million
            "ms_per_1k": 400,
        },

        # Google / Gemini 2.5 family
        "gemini-2.5-flash": {
            # Pricing: Input $0.30 / million → 0.00030 /1k; Output $2.50 / million → 0.00250 /1k :contentReference[oaicite:7]{index=7}
            "price_per_1k_input": 0.30 / 1_000,
            "price_per_1k_output": 2.50 / 1_000,
            # Latency: TTFT ~0.38s, throughput suggests ~0.76s full for some output lengths :contentReference[oaicite:8]{index=8}
            # So for 1000 tokens, assume ~760 ms (~0.76 s)
            "ms_per_1k": 760,
        },
        "gemini-2.5-pro": {
            # A source claims “charges $2.50 per 1M input and $15.00 per 1M output” :contentReference[oaicite:9]{index=9}
            "price_per_1k_input": 2.50 / 1_000,
            "price_per_1k_output": 15.00 / 1_000,
            # Latency: reports of very high latency for large prompts; let's assume ~2-5 s scale → ~2,500 ms for 1000 tokens (optimistic)
            "ms_per_1k": 2_500,
        },
        "gemini-2.5-flash-lite": {
            # Pricing: $0.10 / million input, $0.40 / million output :contentReference[oaicite:10]{index=10}
            "price_per_1k_input": 0.10 / 1_000,
            "price_per_1k_output": 0.40 / 1_000,
            # Latency: “Flash-Lite is 1.5× faster than 2.0 Flash” (which had moderate latency) :contentReference[oaicite:11]{index=11}
            # Let’s assume ~600 ms per 1k tokens
            "ms_per_1k": 600,
        },        # Gemini 3.x family
        "gemini-3.1-pro-preview": {
            # Estimated pricing based on Gemini 3 Pro lineage
            # Input: $3.50 / million → 0.0035 /1k; Output: $14.00 / million → 0.014 /1k
            "price_per_1k_input": 3.50 / 1_000,
            "price_per_1k_output": 14.00 / 1_000,
            # Latency estimate for a large Pro model: ~3,000 ms per 1k tokens
            "ms_per_1k": 3_000,
        },
        "gemini-3.1-flash-preview": {
            # Estimated pricing based on Gemini 3 Flash lineage
            # Input: $0.40 / million; Output: $3.00 / million
            "price_per_1k_input": 0.40 / 1_000,
            "price_per_1k_output": 3.00 / 1_000,
            "ms_per_1k": 800,
        },
        "gemini-3.0-pro": {
            "price_per_1k_input": 3.50 / 1_000,
            "price_per_1k_output": 14.00 / 1_000,
            "ms_per_1k": 3_000,
        },
        "gemini-3.0-flash": {
            "price_per_1k_input": 0.40 / 1_000,
            "price_per_1k_output": 3.00 / 1_000,
            "ms_per_1k": 800,
        },        # Claude models
        "claude-opus-4": {
            "price_per_1k_input": 15.0 / 1000.0,
            "price_per_1k_output": 75.0 / 1000.0,
            "ms_per_1k": 6000,   # ~6 seconds
        },
        "claude-opus-4-5": {
            "price_per_1k_input": 5.0 / 1000.0,
            "price_per_1k_output": 25.0 / 1000.0,
            "ms_per_1k": 6000,   # same as base Opus estimate
        },
        "claude-sonnet-4-5": {
            "price_per_1k_input": 3.0 / 1000.0,
            "price_per_1k_output": 15.0 / 1000.0,
            "ms_per_1k": 4000,   # ~4 seconds
        },
        "claude-sonnet-4": {
            "price_per_1k_input": 3.0 / 1000.0,
            "price_per_1k_output": 15.0 / 1000.0,
            "ms_per_1k": 4500,   # a bit slower than 4-5 in many tasks
        },
        "claude-haiku-4-5": {
            "price_per_1k_input": 0.001,
            "price_per_1k_output": 0.005,
            "ms_per_1k": 200,
        },
        
        # Claude 3.7 (latest, Dec 2024)
        "claude-3-7-sonnet": {
            "price_per_1k_input": 0.003,
            "price_per_1k_output": 0.015,
            "ms_per_1k": 250,
        },
        "claude-3-7-sonnet-20241220": {  # Dated version
            "price_per_1k_input": 0.003,
            "price_per_1k_output": 0.015,
            "ms_per_1k": 250,
        },
        
        # Claude 3.5 family
        "claude-3-5-sonnet": {
            "price_per_1k_input": 0.003,
            "price_per_1k_output": 0.015,
            "ms_per_1k": 300,
        },
        "claude-3-5-sonnet-20241022": {  # Latest stable
            "price_per_1k_input": 0.003,
            "price_per_1k_output": 0.015,
            "ms_per_1k": 300,
        },
        "claude-3-5-sonnet-20240620": {  # Earlier version
            "price_per_1k_input": 0.003,
            "price_per_1k_output": 0.015,
            "ms_per_1k": 350,
        },
        "claude-3-5-haiku": {
            "price_per_1k_input": 0.0008,  # $0.80 per million
            "price_per_1k_output": 0.004,   # $4.00 per million
            "ms_per_1k": 200,
        },
        "claude-3-5-haiku-20241022": {
            "price_per_1k_input": 0.0008,
            "price_per_1k_output": 0.004,
            "ms_per_1k": 200,
        },
        
        # Claude 3 original family
        "claude-3-opus": {
            "price_per_1k_input": 0.015,
            "price_per_1k_output": 0.075,
            "ms_per_1k": 500,
        },
        "claude-3-opus-20240229": {
            "price_per_1k_input": 0.015,
            "price_per_1k_output": 0.075,
            "ms_per_1k": 500,
        },
        "claude-3-sonnet-20240229": {
            "price_per_1k_input": 0.003,
            "price_per_1k_output": 0.015,
            "ms_per_1k": 300,
        },
        "claude-3-haiku-20240307": {
            "price_per_1k_input": 0.00025,
            "price_per_1k_output": 0.00125,
            "ms_per_1k": 150,
        },

        # ===== MiniMax Models =====
        # Pricing from: https://platform.minimax.io/docs/guides/pricing-paygo
        "MiniMax-M2.7": {
            "price_per_1k_input": 0.14 / 1_000,
            "price_per_1k_output": 0.56 / 1_000,
            "ms_per_1k": 167,
        },
        "MiniMax-Text-01": {
            "price_per_1k_input": 0.14 / 1_000,   # Approx estimate based on M2.7 pricing trend
            "price_per_1k_output": 0.56 / 1_000,
            "ms_per_1k": 167,
        },
        "minimax-text-01": {
            "price_per_1k_input": 0.14 / 1_000,
            "price_per_1k_output": 0.56 / 1_000,
            "ms_per_1k": 167,
        },
        "MiniMax-M2.5": {
            "price_per_1k_input": 0.30 / 1_000,   # $0.30 per million
            "price_per_1k_output": 1.20 / 1_000,  # $1.20 per million
            "ms_per_1k": 167,  # ~60 tps → ~16.7ms/token → 167ms/1k
        },
        "MiniMax-M2.5-highspeed": {
            "price_per_1k_input": 0.60 / 1_000,   # $0.60 per million
            "price_per_1k_output": 2.40 / 1_000,  # $2.40 per million
            "ms_per_1k": 100,  # ~100 tps → 10ms/token → 100ms/1k
        },
        "MiniMax-M2.1": {
            "price_per_1k_input": 0.30 / 1_000,
            "price_per_1k_output": 1.20 / 1_000,
            "ms_per_1k": 167,
        },
        "MiniMax-M2.1-highspeed": {
            "price_per_1k_input": 0.60 / 1_000,
            "price_per_1k_output": 2.40 / 1_000,
            "ms_per_1k": 100,
        },
        "MiniMax-M2": {
            "price_per_1k_input": 0.30 / 1_000,
            "price_per_1k_output": 1.20 / 1_000,
            "ms_per_1k": 167,
        },
        "M2-her": {
            "price_per_1k_input": 0.30 / 1_000,
            "price_per_1k_output": 1.20 / 1_000,
            "ms_per_1k": 167,
        },
    }



    def __init__(self, models: Optional[Dict[str, Dict]] = None):
        # copy defaults
        self.models = dict(self.DEFAULT_MODELS)
        if models:
            for k, v in models.items():
                self.models[k] = {**self.models.get(k, {}), **v}
        # usage tracking
        # model -> {tokens:int, cost:float, time_s:float, calls:int}
        self.usage: Dict[str, Dict] = {}

    def normalize_model_name(self, model: str) -> str:
        """Strip date suffixes from model names for consistent lookups.
        
        Examples:
            claude-sonnet-4-5-20250929 -> claude-sonnet-4-5
            gpt-4o-2024-11-20 -> gpt-4o
            gemini-1.5-pro-002 -> gemini-1.5-pro-002 (no date pattern)
        """
        import re
        # Pattern: hyphen followed by 8 digits (YYYYMMDD) or similar date formats
        # Matches: -20250929, -2024-11-20, etc.
        normalized = re.sub(r'-\d{8}$', '', model)  # Strip -YYYYMMDD at end
        normalized = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', normalized)  # Strip -YYYY-MM-DD at end
        
        # If normalized version exists in models, use it; otherwise return original
        if normalized in self.models:
            return normalized
        return model

    def add_model(self, name: str, price_per_1k_input: float, price_per_1k_output: float, ms_per_1k: float = 100.0):
        self.models[name] = {
            "price_per_1k_input": float(price_per_1k_input),
            "price_per_1k_output": float(price_per_1k_output),
            "ms_per_1k": float(ms_per_1k)
        }

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        # rough heuristic: 1 token ≈ 4 chars
        return max(1, int(ceil(len(text) / 4)))

    def estimate_cost(self, tokens: int, model: str, input_tokens: int = None, output_tokens: int = None) -> float:
        model = self.normalize_model_name(model)  # Normalize before lookup
        m = self.models.get(model, {})
        # If input/output tokens provided, use separate pricing
        if input_tokens is not None and output_tokens is not None:
            price_in = float(m.get("price_per_1k_input", m.get("price_per_1k", 0.0)))
            price_out = float(m.get("price_per_1k_output", m.get("price_per_1k", 0.0)))
            cost_in = (input_tokens / 1000.0) * price_in
            cost_out = (output_tokens / 1000.0) * price_out
            return cost_in + cost_out
        # Fallback: treat all tokens as input
        price_per_1k = float(m.get("price_per_1k_input", m.get("price_per_1k", 0.0)))
        return (tokens / 1000.0) * price_per_1k

    def estimate_time_seconds(self, tokens: int, model: str) -> float:
        model = self.normalize_model_name(model)  # Normalize before lookup
        m = self.models.get(model, {})
        ms_per_1k = float(m.get("ms_per_1k", 100.0))
        # Return seconds
        return (tokens / 1000.0) * (ms_per_1k / 1000.0)

    def record_usage(self, model: str, input_tokens: int = 0, output_tokens: int = 0):
        """Record a call with separate input/output token counts and pricing."""
        tokens = int(input_tokens) + int(output_tokens)
        cost = self.estimate_cost(tokens, model, input_tokens=input_tokens, output_tokens=output_tokens)
        time_s = self.estimate_time_seconds(tokens, model)
        rec = self.usage.setdefault(
            model, {"input_tokens": 0, "output_tokens": 0, "tokens": 0, "cost": 0.0, "time_s": 0.0, "calls": 0}
        )
        rec["input_tokens"] += int(input_tokens)
        rec["output_tokens"] += int(output_tokens)
        rec["tokens"] += int(tokens)
        rec["cost"] += float(cost)
        rec["time_s"] += float(time_s)
        rec["calls"] += 1
        return {"model": model, "input_tokens": int(input_tokens), "output_tokens": int(output_tokens), "tokens": tokens, "cost": cost, "time_s": time_s}

    @property
    def total_cost(self) -> float:
        """Return the total cost so far."""
        return sum(v["cost"] for v in self.usage.values())

    def get_report(self) -> Dict:
        total_tokens = sum(v["tokens"] for v in self.usage.values())
        total_cost = self.total_cost
        total_time_s = sum(v["time_s"] for v in self.usage.values())
        return {
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "total_time_seconds": total_time_s,
            "by_model": dict(self.usage),
        }

    def reset(self):
        self.usage = {}
