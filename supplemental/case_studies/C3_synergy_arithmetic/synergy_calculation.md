# C3 - Synergy Arithmetic (Exploratory)

**Status:** Exploratory observation, not a primary contribution of the paper.
This can be dropped from v21 without weakening the core claims.

## Unit Convention

C2 is expressed as average input tokens per call.

## The Calculation

BioPLEASE has two main token-reduction mechanisms:

- **F5 Compression**: ablating this gives 67,320 tokens/call, a 35.2% reduction.
- **F3 Phase Isolation**: ablating this gives 58,441 tokens/call, a 43.7% reduction.

If these two forces were independent, the combined reduction would be
multiplicative:

```text
expected = baseline * (1 - r_F5) * (1 - r_F3)
         = 103,830.4 * (1 - 0.3516) * (1 - 0.4371)
         = 103,830.4 * 0.6484 * 0.5629
         = 37,891.1 tokens/call
         -> Expected reduction: 63.5%
```

**Actual full BioPLEASE: 24,851.5 tokens/call, a 76.1% reduction.**

```text
Synergy = 76.1% - 63.5% = 12.6 percentage points above the multiplicative expectation
```

This matches the paper's stated roughly 12 percentage point synergy.

## Why the Synergy Exists

Phase isolation (F3) shrinks each phase's input context before F5 compression
runs. This means F5 has less raw text to carry forward, so its relative
reduction compounds on top of a smaller base. The forces are not independent;
they interact positively.

## Numbers Verified

| Quantity | Confirmed value | Source |
|----------|----------------:|--------|
| Baseline | 103,830.4 tokens/call | API log screenshot (Sept 2025) |
| No Compression (F5 off) | 67,320 tokens/call | Paper-stated in `ablation_table.csv` |
| No Phase Isolation (F3 off) | 58,441 tokens/call | Paper-stated in `ablation_table.csv` |
| Full BioPLEASE | 24,851.5 tokens/call | API log screenshot (Dec 2025) |
| F5 reduction | 35.2% | (103,830.4 - 67,320) / 103,830.4 |
| F3 reduction | 43.7% | (103,830.4 - 58,441) / 103,830.4 |
| Expected multiplicative | 63.5% | 103,830.4 * 0.6484 * 0.5629 = 37,891.1 tokens/call |
| Actual full reduction | 76.1% | (103,830.4 - 24,851.5) / 103,830.4 |
| Synergy delta | **12.6 pp** | 76.1% - 63.5% |

All numbers are internally consistent with the tokens-per-call values in
`ablation_table.csv`.

## Recommendation for v21

Keep this as a one-paragraph note in the results section labelled
"Exploratory Observation." Do not present it as a tested hypothesis. The
ablation data (C2) is the real contribution here; the synergy observation is a
post-hoc interpretation of those numbers.

If the paper rounds the baseline to 103,753, update the arithmetic here to match,
but ensure it is consistent with `ablation_table.csv` and
`raw_token_counts_from_logs.csv`.
