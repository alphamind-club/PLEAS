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
| BioPLEASE | 14 | 347,921 | 24,851.5 |

Descriptive per-call input-token reduction:

```text
(103,830.4 - 24,851.5) / 103,830.4 = 76.1%
```

Paper-ready wording:

> In screenshot-derived call logs, BioPLEASE reduced average input context from
> 103,830.4 to 24,851.5 tokens per call, a 76.1% descriptive reduction.

Do not cite Wilcoxon significance, paired task-level statistics, or a 12-task
paired result from this artifact alone. Those require a paired task-level CSV
and a statistical test script.
