---
phase: 03-kvtc-compression
plan: "02"
subsystem: compression
tags: [kvtc, pca, quantization, zstandard, mlx, numpy]

requires:
  - phase: 03-kvtc-compression-01
    provides: KVTCCompressor class stub with constructor and NotImplementedError compress/decompress

provides:
  - _split_tokens: sinks/body/window token exemption slicing per KVTC-05/06 spec
  - _calibrate_onthefly: on-the-fly PCA basis via svd_f32, returns (basis, mean, singular_values)
  - _dp_allocate_bits: greedy DP bit allocation, all values >= min_bits, numpy uint8 output
  - _lloyd_codebook: percentile-init + 3-iteration Lloyd 1D quantization codebook
  - _compress_zstd / _decompress_zstd: zstandard level-3 byte compression helpers
affects: [03-03-kvtc-compression, phase-04-calibration-cli, phase-06-integration]

tech-stack:
  added: [zstandard, numpy (used directly in kvtc.py)]
  patterns:
    - Module-level private functions (not methods) so tests can import them directly
    - _mx_materialize called before np.array() conversion of mx.array (MLX lazy graph flush)
    - svd_f32 wrapper mandatory -- linalg_utils.py is the only file allowed to call mx.linalg.svd
    - Greedy DP allocation: fill highest-SV components first within budget

key-files:
  created: []
  modified:
    - omlx/compression/kvtc.py

key-decisions:
  - "Comments referencing forbidden linalg patterns must avoid the exact regex -- test_no_bare_linalg_calls scans full file text including docstrings, so even negative-example comments must be reworded"
  - "Six primitives implemented as module-level functions (not methods) so test isolation is possible without constructing a KVTCCompressor"
  - "_dp_allocate_bits uses int(max_bits) - int(alloc[comp_idx]) to avoid numpy uint8 overflow when computing extra budget per component"

patterns-established:
  - "Private compression helpers in kvtc.py are module-level to enable direct unit testing without the full compressor"
  - "All SVD paths go through svd_f32 -- never direct mx.linalg calls outside linalg_utils.py"

requirements-completed: [KVTC-01, KVTC-02, KVTC-03]

duration: 3min
completed: 2026-03-19
---

# Phase 03 Plan 02: KVTC Compression Primitives Summary

**Six private compression helpers (_split_tokens, _calibrate_onthefly, _dp_allocate_bits, _lloyd_codebook, _compress_zstd, _decompress_zstd) implemented in kvtc.py using svd_f32 wrapper, greedy DP allocation, and zstandard level-3 entropy coding**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-03-19T10:54:12Z
- **Completed:** 2026-03-19T10:57:23Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Implemented all 6 private compression primitives as module-level functions in omlx/compression/kvtc.py
- PCA calibration uses svd_f32 with mandatory _mx_materialize before numpy bridge (MLX lazy graph compliance)
- DP bit allocation guarantees every component receives >= min_bits (1) and distributes remaining budget to highest-SV components first
- Lloyd codebook uses percentile-based initialization + 3 refinement iterations for stable 1D quantization
- zstd wrappers use level=3 per the research spike baseline
- compress()/decompress() stubs still raise NotImplementedError (wired in Plan 03 as designed)
- CI lint gate (test_no_bare_linalg_calls) passes: zero actual mx.linalg.svd calls outside linalg_utils

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement private compression primitives** - `a2ed66b` (feat)
2. **Task 2: Lint gate fix (Rule 1 auto-fix)** - `1244b6e` (fix)

**Plan metadata:** (docs commit — pending)

## Files Created/Modified
- `omlx/compression/kvtc.py` - Added 6 module-level private helpers + imports (struct, numpy, zstandard, svd_f32)

## Decisions Made
- Six helpers implemented as module-level functions (not methods) so unit tests can import them directly without constructing KVTCCompressor
- `_dp_allocate_bits` uses explicit `int()` casts when computing extra budget to avoid numpy uint8 underflow/overflow
- Comments that reference the forbidden `mx.linalg.svd` pattern (even in negative-example form) must be reworded — the CI lint test performs full-text regex scan including docstrings

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed docstring comments triggering CI lint gate**
- **Found during:** Task 2 (lint gate verification)
- **Issue:** Two docstring comments in `_calibrate_onthefly` contained the literal string `mx.linalg.svd` as a negative example ("NEVER call mx.linalg.svd"). The `test_no_bare_linalg_calls` test in test_linalg_utils.py scans the full file text using regex, causing a false-positive CI failure.
- **Fix:** Rewrote both comments to describe the prohibition without embedding the forbidden pattern.
- **Files modified:** omlx/compression/kvtc.py
- **Verification:** `uv run python -m pytest tests/test_linalg_utils.py -v` -- 21/21 passed including test_no_bare_linalg_calls
- **Committed in:** `1244b6e` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Necessary for CI correctness. No scope creep -- the fix was a one-line comment rewrite.

## Issues Encountered
- None beyond the auto-fixed lint deviation above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 6 primitives are available for Plan 03 (03-03) to wire into compress()/decompress()
- `_calibrate_onthefly` provides on-the-fly PCA for the testing path (no bundle required)
- compress()/decompress() still raise NotImplementedError -- Plan 03 wires them up
- CI gates: test_no_bare_linalg_calls GREEN, test_compressor_constructs_without_bundle GREEN, test_compress_raises_not_implemented GREEN

---
*Phase: 03-kvtc-compression*
*Completed: 2026-03-19*

## Self-Check: PASSED

- FOUND: .planning/phases/03-kvtc-compression/03-02-SUMMARY.md
- FOUND: commit a2ed66b (feat: implement private compression primitives)
- FOUND: commit 1244b6e (fix: lint gate comment fix)
- FOUND: omlx/compression/kvtc.py
