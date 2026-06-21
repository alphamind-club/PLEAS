# C1 - Token Reduction Analysis

Source files:

- `raw_token_counts_from_logs.csv`
- `C1_token_reduction_summary.csv`
- `ablation_table.csv`

The available raw token-count file contains per-call input-token counts from
log/screenshots, not a paired 12-task task-level experiment.

## Descriptive Result

| Condition | Calls | Total input tokens | Average input tokens / call |
|-----------|------:|-------------------:|----------------------------:|
| Baseline | 18 | 1,868,947 | 103,830.4 |
| BioPLEASE | 18 | 646,825 | 35,934.7 |

Descriptive per-call input-token reduction:

```text
(103,830.4 - 35,934.7) / 103,830.4 = 65.4%
```

Paper-ready wording:

> In screenshot-derived call logs, BioPLEASE reduced average input context from
> 103,830.4 to 35,934.7 tokens per call, a 65.4% descriptive reduction.

Note: both conditions increase over time as context accumulates. The baseline
grows rapidly (24K → 195K) while BioPLEASE grows slowly (20K → 52K),
indicating that phase isolation and trace compression bound the growth rate
even though some accumulation still occurs.

Do not cite Wilcoxon significance, paired task-level statistics, or a 12-task
paired result from this artifact alone. Those require a paired task-level CSV
and a statistical test script.
