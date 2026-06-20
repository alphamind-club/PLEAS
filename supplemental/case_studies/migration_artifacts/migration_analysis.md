# C12 - Memory Architecture Migration: Line-Count Analysis

Source: `migration_all_changes.diff`.

## Summary

| Metric | Value |
|--------|------:|
| Files changed | 12 |
| Lines added | 623 |
| Lines removed | 247 |
| Net change | 376 |
| Added / removed ratio | 2.52x |
| Old total lines across changed files | 32,794 |
| Estimated old lines preserved | 32,547 |
| Estimated old-line preservation | 99.2% |

The preservation estimate uses this definition:

```text
old_line_preservation = (old_total_lines - removed_lines) / old_total_lines
```

This is a diff-level preservation measure. It does not mean the migration was
small in absolute terms; it means most old lines in the touched files were not
deleted.

## Per-File Detail

Use `migration_line_counts.csv` for the file-by-file table. The largest touched
file is `bioplease/agent/a1.py`, with:

- old total lines: 7,201
- added lines: 200
- removed lines: 78
- estimated old-line preservation: 98.9%

## Paper-Ready Wording

Supported:

> The migration touched 12 files, adding 623 lines and removing 247 lines.
> Across the changed files, 32,547 of 32,794 old lines were preserved under a
> diff-level old-line preservation definition (99.2%).

Also supported:

> The main orchestrator file, `bioplease/agent/a1.py`, preserved 7,123 of 7,201
> old lines under the same definition (98.9%) while adding 200 lines and
> removing 78.

Avoid:

- Calling the migration "minimal" without qualification.
- Claiming exact unchanged-line ratios for files not included in the diff.
- Treating added-line volume as negligible; the migration added 623 lines.
