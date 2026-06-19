# Force-Aligned Integration Tests (IT-F Suite)

These tests validate the five structural forces (F1-F5) of the PLEAS specification as described in **Section V-A** of the paper. All tests run with **zero API calls** using lightweight stubs.

## Quick Start

```bash
# From the repository root
pip install pydantic pytest
pytest tests/test_it_forces.py -v
```

**Expected:** 23 tests pass in under 1 second.

## Test Coverage

| Force | Tests | Count | Behavior Validated |
|---|---|---|---|
| F1 | IT-F1-01 to 04 | 4 | State isolation; per-phase directories |
| F2 | IT-F2-01 to 06 | 6 | Schema accept/reject payloads |
| F3 | IT-F3-01 to 05 | 5 | Plan schema guards; empty-step rejection |
| F4 | IT-F4-01 to 04 | 4 | Retry counter; budget-exhaustion stop |
| F5 | IT-F5-01 to 04 | 4 | Long-term memory; JSON round-trip |

## Representative Examples

- **IT-F1-03:** Verifies `StateManager` writes PLAN and EXECUTE outputs to separate filesystem subdirectories
- **IT-F2-02:** Verifies `EvidenceSchema` raises a Pydantic `ValidationError` when `source_id` is absent
- **IT-F4-03:** Verifies `execute()` runs no plan steps when the seconds budget equals zero
- **IT-F5-02:** Verifies a `long_term_memory` string survives a JSON serialization round-trip without loss

## Additional Tests

- `test_state_manager.py` — Unit tests for phase-scoped state persistence
- `test_cost_manager.py` — Token/cost estimation tests
- `smoke_test.py` — Import and basic functionality smoke test
- `trigger_force_violations.py` — Demonstrates behavior when forces are violated
