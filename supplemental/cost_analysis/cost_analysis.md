# C8 - Cost Analysis: Baseline vs BioPLEASE

Source files:

- `per_call_costs.csv`
- `cost_summary.csv`
- `../cost_tracking/cost_manager.py`

Pricing model used for this analysis:

- Model: `claude-sonnet-4-5`
- Input: `$3.00 / 1M tokens`
- Output: `$15.00 / 1M tokens`

## Per-Call Averages

| Condition | Calls | Avg input tokens | Avg output tokens | Avg cost / call |
|-----------|------:|-----------------:|------------------:|----------------:|
| Baseline | 18 | 103,830.4 | 3,772.5 | $0.3681 |
| BioPLEASE | 14 | 24,851.5 | 1,569.6 | $0.0981 |

Per-call cost reduction:

```text
(0.3681 - 0.0981) / 0.3681 = 73.3%
```

## Full Logged Run Cost

| Condition | Calls | Total input tokens | Total output tokens | Total cost |
|-----------|------:|-------------------:|--------------------:|-----------:|
| Baseline | 18 | 1,868,947 | 67,905 | $6.6254 |
| BioPLEASE | 14 | 347,921 | 21,974 | $1.3734 |

Logged-run cost reduction:

```text
(6.6254 - 1.3734) / 6.6254 = 79.3%
```

## Paper-Ready Wording

> In the logged single-model cost trace, BioPLEASE reduced total estimated cost
> from $6.6254 to $1.3734, a 79.3% reduction under `claude-sonnet-4-5` pricing.
> Average cost per call fell from $0.3681 to $0.0981, a 73.3% reduction.

Do not cite older dollar figures such as `$0.84` vs `$0.31` from this artifact
unless a separate mixed-model cost trace is added. This C8 folder supports the
single-model logged-run numbers above.

## Reproducible Formula

```python
P_IN = 3.0 / 1_000_000
P_OUT = 15.0 / 1_000_000
cost = input_tokens * P_IN + output_tokens * P_OUT
```
