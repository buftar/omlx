---
phase: 04-pca-calibration-cli
plan: 03
subsystem: compression
tags: [pca, calibration, kvtc, mlx-lm, numpy, tqdm, cli]

# Dependency graph
requires:
  - phase: 04-pca-calibration-cli/04-02
    provides: strip_rope_from_keys, compute_pca_basis, assign_layer_groups, save/load_calibration_bundle
  - phase: 03-kvtc-compression
    provides: _dp_allocate_bits from kvtc.py
provides:
  - run_calibration(): full KV prefill, RoPE strip, PCA SVD, DP allocation, bundle save pipeline
  - CALIBRATION_PROMPTS: 25 built-in diverse prompts, no external download needed
  - TestCLIDispatch with real monkeypatch mock assertions (CAL-01 GREEN)
affects:
  - phase-05-integration (uses kv_pca_calibration.npz as KVTCCompressor input)
  - phase-06-ssd-integration (calibration output feeds compressed KV storage)

# Tech tracking
tech-stack:
  added: [tqdm (optional graceful fallback), pathlib, unittest.mock]
  patterns: [lazy import inside run_calibration avoids circular imports, graceful ImportError fallbacks for optional deps]

key-files:
  created: []
  modified:
    - omlx/compression/calibrator.py
    - tests/test_calibrator.py

key-decisions:
  - "_dp_allocate_bits called with 4 required args: singular_values, bits_per_token, n_tok (group row count), n_components -- matches actual kvtc.py signature which differs from plan's 2-arg example"
  - "test_cli_help_registered uses 'uv run omlx calibrate-kv --help' not 'python -m omlx' since omlx package has no __main__.py"
  - "tqdm import wrapped in try/except -- graceful fallback to identity wrapper keeps calibrator usable without tqdm installed"
  - "mlx_lm import wrapped in try/except at module level -- raises RuntimeError with clear install hint inside run_calibration()"

patterns-established:
  - "Pattern: Import-time fallback defs for optional/private dependencies (tqdm, _dp_allocate_bits)"
  - "Pattern: run_calibration() resolves HuggingFace hub cache path via HF_HOME env var with fallback to ~/.cache/huggingface"

requirements-completed: [CAL-01, CAL-05]

# Metrics
duration: 20min
completed: 2026-03-22
---

# Phase 4 Plan 03: run_calibration() Full Pipeline Summary

**Full KV calibration pipeline: model prefill with 25 built-in prompts, RoPE stripping, per-group SVD PCA, DP bit allocation via kvtc, and kv_pca_calibration.npz bundle output**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-03-22T00:00:00Z
- **Completed:** 2026-03-22T00:20:00Z
- **Tasks:** 2 of 3 (Task 3 is checkpoint:human-verify)
- **Files modified:** 2

## Accomplishments

- Implemented `run_calibration()` replacing the `NotImplementedError` stub with the full 7-step pipeline
- Added 25 built-in `CALIBRATION_PROMPTS` covering instruction, factual, code, and math domains
- Updated `TestCLIDispatch` from a stub `pytest.raises(NotImplementedError)` to real monkeypatch mock assertions verifying all kwargs forwarded correctly
- All 12 unit tests GREEN (`pytest tests/test_calibrator.py -m "not slow"`)
- CLI `omlx calibrate-kv --help` shows all 4 flags correctly

## Task Commits

1. **Task 1: Implement run_calibration()** - `d33438a` (feat)
2. **Task 2: Update TestCLIDispatch with mock** - `084e990` (feat)

## Files Created/Modified

- `/Users/tonysina/projects/omlx/omlx/compression/calibrator.py` - Added CALIBRATION_PROMPTS, pathlib/tqdm/mlx_lm imports, full run_calibration() implementation
- `/Users/tonysina/projects/omlx/tests/test_calibrator.py` - Replaced TestCLIDispatch stub with mock-based dispatch test + CLI help subprocess test

## Decisions Made

- `_dp_allocate_bits` takes 4 required args in actual kvtc.py signature (`singular_values, bits_per_token_budget, n_tokens, n_components`), not 2 as shown in plan. Used `k_all.shape[0]` as `n_tokens` (total accumulated token rows per group).
- `test_cli_help_registered` uses `uv run omlx calibrate-kv --help` because `omlx` package has no `__main__.py` so `python -m omlx` fails with exit code 1.
- `tqdm` and `mlx_lm` imports wrapped in `try/except` at module level for graceful fallback when not installed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected _dp_allocate_bits call signature**
- **Found during:** Task 1 (run_calibration() implementation)
- **Issue:** Plan showed `_dp_allocate_bits(k_sv, bits_per_token)` with 2 args, but actual function requires 4: `(singular_values, bits_per_token_budget, n_tokens, n_components)`
- **Fix:** Called with `_dp_allocate_bits(k_sv, bits_per_token, n_tok, n_components)` where `n_tok = k_all.shape[0]`
- **Files modified:** omlx/compression/calibrator.py
- **Verification:** Unit tests pass, no TypeError at import or call time
- **Committed in:** d33438a (Task 1 commit)

**2. [Rule 1 - Bug] Fixed test_cli_help_registered invocation**
- **Found during:** Task 2 (TestCLIDispatch update)
- **Issue:** Plan used `python -m omlx` but omlx has no `__main__.py`; subprocess returned exit code 1
- **Fix:** Changed to `uv run omlx calibrate-kv --help` which uses the installed entry point
- **Files modified:** tests/test_calibrator.py
- **Verification:** `test_cli_help_registered` passes, asserts show `--n-components` and `--bits-per-token` in stdout
- **Committed in:** 084e990 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - Bug)
**Impact on plan:** Both fixes necessary for correctness. No scope creep.

## Issues Encountered

None beyond the auto-fixed deviations above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `run_calibration()` is fully functional end-to-end for models accessible via mlx_lm
- Optional smoke test: `uv run python -c "from omlx.compression.calibrator import run_calibration; run_calibration('mlx-community/Qwen2.5-7B-Instruct-4bit', n_components=8, output_path='/tmp/cal_test')"` (requires model on disk)
- Phase 5 integration can proceed: kv_pca_calibration.npz schema is stable and load_calibration_bundle() returns KVTCCompressor-compatible pca_bundle

---
*Phase: 04-pca-calibration-cli*
*Completed: 2026-03-22*
