#!/usr/bin/env python3
"""
Blinded Multi-Judge AI Scoring — Main Entry Point

Orchestrates the full pipeline:
  1. Load model answers from 3 system files
  2. Generate blinding session and manifest
  3. Send blinded responses to each judge model in parallel
  4. Collect scores, de-anonymise, aggregate
  5. Compute inter-rater reliability (Fleiss' kappa, Krippendorff's alpha)
  6. Write results to CSV, JSON, and human-readable report

Usage
-----
    python -m CaseStudy.C15_blinded_ai_scoring.run_blinded_scoring \\
        --system1 model_answers/system_1_answers.json \\
        --system2 model_answers/system_2_answers.json \\
        --system3 model_answers/system_3_answers.json \\
        --output  results/

Environment variables required:
    OPENAI_API_KEY      — for GPT-5.5-thinking-extended judge
    ANTHROPIC_API_KEY   — for Claude Opus 4.8 judge
    GOOGLE_API_KEY      — for Gemini 3.1 Pro judge
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .aggregator import deblind_and_aggregate, write_results
from .blinding import (
    BlindingManifest,
    blind_answers,
    create_session,
)
from .config import JUDGE_PANEL, MAX_CONCURRENT_JUDGES, SCORING_ROUNDS
from .judges import JudgeScore, score_one
from .rubric import TASK_INDEX, TASKS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("blinded_scoring")


def load_answers(path: Path) -> dict[str, str]:
    """Load a system's answers from JSON.

    Expected format:
    {
      "system_id": "BioClaw",
      "answers": {
        "T01": "full text of system answer for T01 ...",
        "T02": "...",
        ...
        "T24": "..."
      }
    }
    """
    with open(path) as f:
        data = json.load(f)
    system_id = data["system_id"]
    answers = data["answers"]
    missing = [t.task_id for t in TASKS if t.task_id not in answers]
    if missing:
        log.warning("System %s missing answers for: %s", system_id, missing)
    return {system_id: answers}


def run(
    system_files: list[Path],
    output_dir: Path,
    session_id: str | None = None,
    rounds: int = SCORING_ROUNDS,
    dry_run: bool = False,
) -> None:
    session_id = create_session(session_id)
    log.info("Scoring session: %s", session_id)

    # ── Load all system answers ──────────────────────────────────────────
    all_systems: dict[str, dict[str, str]] = {}
    for path in system_files:
        loaded = load_answers(path)
        all_systems.update(loaded)
    system_ids = sorted(all_systems.keys())
    log.info("Systems loaded: %s", system_ids)

    if len(system_ids) != 3:
        log.error("Expected 3 systems, got %d. Aborting.", len(system_ids))
        sys.exit(1)

    # ── Blinding ─────────────────────────────────────────────────────────
    manifest = BlindingManifest(
        session_id=session_id,
        created_utc=datetime.now(timezone.utc).isoformat(),
    )
    blinded_by_task: dict[str, list] = {}

    for task in TASKS:
        task_answers = {sid: all_systems[sid].get(task.task_id, "[NO ANSWER]") for sid in system_ids}
        blinded, record = blind_answers(task.task_id, task_answers, session_id)
        blinded_by_task[task.task_id] = blinded
        manifest.task_records.append(record)

    manifest_path = output_dir / "blinding_manifest.json"
    manifest.save(manifest_path)
    log.info("Blinding manifest sealed: %s", manifest_path)

    if dry_run:
        log.info("DRY RUN — manifest written, no API calls made.")
        return

    # ── Scoring ──────────────────────────────────────────────────────────
    all_scores: list[JudgeScore] = []
    total_calls = len(TASKS) * 3 * len(JUDGE_PANEL) * rounds
    log.info(
        "Starting scoring: %d tasks x 3 systems x %d judges x %d round(s) = %d API calls",
        len(TASKS), len(JUDGE_PANEL), rounds, total_calls,
    )

    futures = []
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JUDGES) as pool:
        for _round in range(rounds):
            for task in TASKS:
                for response in blinded_by_task[task.task_id]:
                    for judge in JUDGE_PANEL:
                        fut = pool.submit(score_one, judge, task, response)
                        futures.append(fut)

        completed = 0
        for fut in as_completed(futures):
            completed += 1
            try:
                result = fut.result()
                all_scores.append(result)
                if completed % 10 == 0 or completed == total_calls:
                    log.info("Progress: %d / %d calls completed", completed, total_calls)
            except Exception as exc:
                log.error("Scoring call failed: %s", exc)

    log.info("All scoring calls completed. De-anonymising...")

    # ── Aggregation ──────────────────────────────────────────────────────
    task_results, summaries, reliability = deblind_and_aggregate(all_scores, manifest)
    write_results(output_dir, task_results, summaries, reliability, all_scores)

    log.info("Results written to %s", output_dir)
    log.info(
        "Inter-rater reliability: Fleiss' kappa = %.4f, Krippendorff alpha = %.4f",
        reliability["fleiss_kappa"],
        reliability["krippendorff_alpha"],
    )
    for sid in sorted(summaries, key=lambda k: -summaries[k].total_majority):
        s = summaries[sid]
        pct = round(100.0 * s.total_majority / s.total_max, 1)
        log.info("  %s: %d/48 (%.1f%%)", s.system_id, s.total_majority, pct)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blinded multi-judge AI scoring for the PLEAS 24-task benchmark",
    )
    parser.add_argument(
        "--system1", type=Path, required=True,
        help="Path to system 1 answers JSON file",
    )
    parser.add_argument(
        "--system2", type=Path, required=True,
        help="Path to system 2 answers JSON file",
    )
    parser.add_argument(
        "--system3", type=Path, required=True,
        help="Path to system 3 answers JSON file",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results"),
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--session-id", type=str, default=None,
        help="Override session UUID (for reproducibility)",
    )
    parser.add_argument(
        "--rounds", type=int, default=SCORING_ROUNDS,
        help="Number of scoring rounds per judge (default: 1)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate manifest only, no API calls",
    )
    args = parser.parse_args()

    run(
        system_files=[args.system1, args.system2, args.system3],
        output_dir=args.output,
        session_id=args.session_id,
        rounds=args.rounds,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
