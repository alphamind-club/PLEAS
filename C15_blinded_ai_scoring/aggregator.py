"""
Score aggregation with inter-rater reliability statistics.

After all judges have scored every blinded response, this module:
1.  De-anonymises scores using the blinding manifest.
2.  Computes per-task and per-system aggregates (majority-vote, mean).
3.  Computes inter-rater reliability: Fleiss' kappa and Krippendorff's alpha.
4.  Produces final CSV and JSON result files.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .blinding import BlindingManifest
from .judges import JudgeScore
from .rubric import CATEGORIES, TASKS


@dataclass
class TaskResult:
    task_id: str
    category: str
    difficulty: str
    system_id: str
    judge_scores: dict[str, int] = field(default_factory=dict)
    majority_score: int = -1
    mean_score: float = -1.0

    def compute_aggregates(self) -> None:
        valid = [s for s in self.judge_scores.values() if s >= 0]
        if not valid:
            return
        self.mean_score = round(sum(valid) / len(valid), 3)
        counts = defaultdict(int)
        for s in valid:
            counts[s] += 1
        self.majority_score = max(counts, key=lambda k: (counts[k], k))


@dataclass
class SystemSummary:
    system_id: str
    total_majority: int = 0
    total_max: int = 0
    mean_total: float = 0.0
    category_scores: dict[str, int] = field(default_factory=dict)
    complete: int = 0
    partial: int = 0
    failed: int = 0


# ── Fleiss' kappa ────────────────────────────────────────────────────────────

def fleiss_kappa(ratings_matrix: list[list[int]], k: int = 3) -> float:
    """Compute Fleiss' kappa for inter-rater reliability.

    Parameters
    ----------
    ratings_matrix : list[list[int]]
        Each row is a subject (task-system pair).  Each column is a category
        count:  how many raters assigned that category.  Categories are 0, 1, 2.
    k : int
        Number of categories (default 3 for scores 0/1/2).

    Returns
    -------
    float
        Fleiss' kappa value (-1 to 1).  1 = perfect agreement.
    """
    n = len(ratings_matrix)
    if n == 0:
        return 0.0
    N = sum(ratings_matrix[0])  # number of raters per subject
    if N <= 1:
        return 1.0

    p_j = []
    for j in range(k):
        total = sum(row[j] for row in ratings_matrix)
        p_j.append(total / (n * N))

    P_e = sum(p ** 2 for p in p_j)

    P_i_list = []
    for row in ratings_matrix:
        P_i = (sum(r ** 2 for r in row) - N) / (N * (N - 1)) if N > 1 else 1.0
        P_i_list.append(P_i)

    P_bar = sum(P_i_list) / n

    if abs(1 - P_e) < 1e-12:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


# ── Krippendorff's alpha (ordinal, simplified) ──────────────────────────────

def krippendorff_alpha(values_by_unit: list[list[int]]) -> float:
    """Compute Krippendorff's alpha for ordinal data.

    Parameters
    ----------
    values_by_unit : list[list[int]]
        Each inner list contains the ratings for one unit (task-system pair)
        from all raters.  Values in {0, 1, 2} or -1 for missing.

    Returns
    -------
    float
        Alpha value.  1 = perfect reliability, 0 = chance.
    """
    all_values: list[int] = []
    for unit_vals in values_by_unit:
        all_values.extend(v for v in unit_vals if v >= 0)
    if not all_values:
        return 0.0

    n_total = len(all_values)
    categories = sorted(set(all_values))
    if len(categories) <= 1:
        return 1.0

    D_o = 0.0
    n_units = 0
    for unit_vals in values_by_unit:
        valid = [v for v in unit_vals if v >= 0]
        m = len(valid)
        if m < 2:
            continue
        n_units += 1
        for i in range(m):
            for j in range(i + 1, m):
                D_o += (valid[i] - valid[j]) ** 2
        D_o_denom = m * (m - 1) / 2
        if D_o_denom > 0:
            pass  # already accumulated in quadratic form

    if n_units == 0:
        return 0.0

    total_pairs = 0
    for unit_vals in values_by_unit:
        valid = [v for v in unit_vals if v >= 0]
        m = len(valid)
        total_pairs += m * (m - 1) // 2

    if total_pairs == 0:
        return 0.0
    D_o_norm = D_o / total_pairs

    mean_val = sum(all_values) / n_total
    D_e = sum((v - mean_val) ** 2 for v in all_values) / (n_total - 1) if n_total > 1 else 1.0

    if D_e < 1e-12:
        return 1.0
    return 1.0 - D_o_norm / D_e


# ── Aggregation pipeline ────────────────────────────────────────────────────

def deblind_and_aggregate(
    scores: list[JudgeScore],
    manifest: BlindingManifest,
) -> tuple[list[TaskResult], dict[str, SystemSummary], dict[str, float]]:
    """De-anonymise scores and compute all aggregates.

    Returns
    -------
    task_results : list[TaskResult]
    system_summaries : dict[str, SystemSummary]
    reliability : dict with 'fleiss_kappa' and 'krippendorff_alpha'
    """
    task_map = {t.task_id: t for t in TASKS}
    record_map = {r.task_id: r for r in manifest.task_records}

    keyed: dict[tuple[str, str], TaskResult] = {}

    for js in scores:
        record = record_map[js.task_id]
        system_id = record.reverse[js.response_label]
        task_obj = task_map[js.task_id]

        key = (js.task_id, system_id)
        if key not in keyed:
            keyed[key] = TaskResult(
                task_id=js.task_id,
                category=task_obj.category,
                difficulty=task_obj.difficulty,
                system_id=system_id,
            )
        keyed[key].judge_scores[js.judge_id] = js.score

    task_results = list(keyed.values())
    for tr in task_results:
        tr.compute_aggregates()

    # System summaries
    system_ids = sorted({tr.system_id for tr in task_results})
    summaries: dict[str, SystemSummary] = {}
    for sid in system_ids:
        s = SystemSummary(system_id=sid, total_max=48)
        cat_scores: dict[str, int] = {c: 0 for c in CATEGORIES}
        for tr in task_results:
            if tr.system_id != sid:
                continue
            s.total_majority += tr.majority_score
            s.mean_total += tr.mean_score
            cat_scores[tr.category] = cat_scores.get(tr.category, 0) + tr.majority_score
            if tr.majority_score == 2:
                s.complete += 1
            elif tr.majority_score == 1:
                s.partial += 1
            else:
                s.failed += 1
        s.mean_total = round(s.mean_total, 2)
        s.category_scores = cat_scores
        summaries[sid] = s

    # Inter-rater reliability
    judge_ids = sorted({js.judge_id for js in scores})
    ratings_matrix: list[list[int]] = []
    values_by_unit: list[list[int]] = []

    for tr in task_results:
        counts = [0, 0, 0]
        unit_vals: list[int] = []
        for jid in judge_ids:
            s = tr.judge_scores.get(jid, -1)
            if s >= 0:
                counts[s] += 1
                unit_vals.append(s)
        if sum(counts) > 0:
            ratings_matrix.append(counts)
            values_by_unit.append(unit_vals)

    reliability = {
        "fleiss_kappa": round(fleiss_kappa(ratings_matrix), 4),
        "krippendorff_alpha": round(krippendorff_alpha(values_by_unit), 4),
        "num_judges": len(judge_ids),
        "num_task_system_pairs": len(ratings_matrix),
        "judge_ids": judge_ids,
    }

    return task_results, summaries, reliability


# ── Output writers ───────────────────────────────────────────────────────────

def write_results(
    results_dir: Path,
    task_results: list[TaskResult],
    summaries: dict[str, SystemSummary],
    reliability: dict[str, Any],
    all_scores: list[JudgeScore],
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    # Per-task CSV
    with open(results_dir / "blinded_task_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        judge_ids = sorted({s.judge_id for s in all_scores})
        header = [
            "task_id", "category", "difficulty", "system_id",
            *[f"score_{jid}" for jid in judge_ids],
            "majority_score", "mean_score",
        ]
        w.writerow(header)
        for tr in sorted(task_results, key=lambda r: (r.task_id, r.system_id)):
            row = [
                tr.task_id, tr.category, tr.difficulty, tr.system_id,
                *[tr.judge_scores.get(jid, -1) for jid in judge_ids],
                tr.majority_score, tr.mean_score,
            ]
            w.writerow(row)

    # System summary CSV
    with open(results_dir / "blinded_system_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "system_id", "total_majority", "total_max", "percent",
            "mean_total", "complete", "partial", "failed",
            *CATEGORIES,
        ])
        for sid in sorted(summaries):
            s = summaries[sid]
            pct = round(100.0 * s.total_majority / s.total_max, 1) if s.total_max else 0.0
            w.writerow([
                s.system_id, s.total_majority, s.total_max, pct,
                s.mean_total, s.complete, s.partial, s.failed,
                *[s.category_scores.get(c, 0) for c in CATEGORIES],
            ])

    # Full judge rationales (JSON)
    with open(results_dir / "judge_rationales.json", "w") as f:
        json.dump([asdict(s) for s in all_scores], f, indent=2)

    # Reliability report
    with open(results_dir / "inter_rater_reliability.json", "w") as f:
        json.dump(reliability, f, indent=2)

    # Human-readable summary
    with open(results_dir / "scoring_report.txt", "w") as f:
        f.write("=" * 72 + "\n")
        f.write("BLINDED MULTI-JUDGE SCORING REPORT\n")
        f.write("PLEAS 24-Task Biomedical Scientific Workflow Benchmark\n")
        f.write("=" * 72 + "\n\n")

        f.write("INTER-RATER RELIABILITY\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Judges:              {reliability['num_judges']}\n")
        f.write(f"  Judge IDs:           {', '.join(reliability['judge_ids'])}\n")
        f.write(f"  Fleiss' kappa:       {reliability['fleiss_kappa']}\n")
        f.write(f"  Krippendorff alpha:  {reliability['krippendorff_alpha']}\n")
        f.write(f"  Scored pairs:        {reliability['num_task_system_pairs']}\n\n")

        f.write("SYSTEM SUMMARY (majority-vote scores)\n")
        f.write("-" * 72 + "\n")
        f.write(f"{'System':<25} {'Score':>6} {'%':>7}  {'C':>3} {'P':>3} {'F':>3}\n")
        f.write("-" * 72 + "\n")
        for sid in sorted(summaries, key=lambda k: -summaries[k].total_majority):
            s = summaries[sid]
            pct = round(100.0 * s.total_majority / s.total_max, 1)
            f.write(
                f"{s.system_id:<25} {s.total_majority:>3}/48 {pct:>6.1f}%  "
                f"{s.complete:>3} {s.partial:>3} {s.failed:>3}\n"
            )

        f.write("\n\nPER-CATEGORY BREAKDOWN (majority-vote)\n")
        f.write("-" * 72 + "\n")
        for cat in CATEGORIES:
            f.write(f"\n  {cat}:\n")
            for sid in sorted(summaries):
                cs = summaries[sid].category_scores.get(cat, 0)
                f.write(f"    {sid:<25} {cs}/8\n")

        f.write("\n\n" + "=" * 72 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 72 + "\n")
