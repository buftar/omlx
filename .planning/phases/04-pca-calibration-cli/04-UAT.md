---
status: complete
phase: 04-pca-calibration-cli
source: [04-01-SUMMARY.md, 04-02-SUMMARY.md, 04-03-SUMMARY.md]
started: 2026-03-22T22:38:00Z
updated: 2026-03-22T22:42:00Z
---

## Current Test

[testing complete]

## Tests

### 1. CLI calibrate-kv command registered
expected: Run `uv run omlx calibrate-kv --help` — shows help text including all 4 flags: --n-components, --n-groups, --bits-per-token, --output. Exits with code 0.
result: pass

### 2. Unit test suite GREEN
expected: Run `uv run pytest tests/test_calibrator.py -m "not slow" -v` — all 12 tests PASS with no failures or errors.
result: pass

### 3. Calibrator module imports cleanly
expected: Run `uv run python -c "from omlx.compression.calibrator import run_calibration, strip_rope_from_keys, compute_pca_basis, save_calibration_bundle, load_calibration_bundle, assign_layer_groups, align_bases_to_reference; print('OK')"` — prints "OK" with no ImportError.
result: pass

### 4. Built-in calibration prompts
expected: Run `uv run python -c "from omlx.compression.calibrator import CALIBRATION_PROMPTS; print(len(CALIBRATION_PROMPTS))"` — prints "25". No external download required.
result: pass

## Summary

total: 4
passed: 4
issues: 0
pending: 0
skipped: 0

## Gaps

[none yet]
