---
phase: 01-linalg-foundation
plan: 01
subsystem: compression
tags: [mlx, linalg, scipy, float32, svd, pinv, qr, nnls]

# Dependency graph
requires: []
provides:
  - "omlx/compression/__init__.py: Empty package root for compression subpackage"
  - "omlx/compression/linalg_utils.py: Float32-safe SVD, pseudoinverse, QR, and NNLS wrappers"
  - "tests/test_linalg_utils.py: Full unit + lint gate tests for all wrappers"
affects:
  - "02-svd-compressor"
  - "03-quantized-compressor"
  - "04-sliding-window"
  - "All phases that import from omlx.compression"

# Tech tracking
tech-stack:
  added:
    - "scipy>=1.7.0 (NNLS solver via scipy.optimize.nnls)"
  patterns:
    - "_ensure_f32: Private cast helper pattern for float16/bfloat16 -> float32"
    - "stream=mx.cpu: All MLX linalg calls route to CPU stream to avoid GPU materialization errors"
    - "scipy numpy bridge: mx.array -> np.float64 -> scipy -> mx.float32 for NNLS"

key-files:
  created:
    - "omlx/compression/__init__.py"
    - "omlx/compression/linalg_utils.py"
    - "tests/test_linalg_utils.py"
  modified:
    - "pyproject.toml (added scipy>=1.7.0 dependency)"
    - "omlx/scheduler.py (removed stale make_presence_penalty import)"

key-decisions:
  - "omlx/compression/__init__.py is intentionally empty (license header only) — no re-exports. Downstream callers must import from linalg_utils directly."
  - "All wrappers always return float32 — callers own their dtype conversion, not the helpers"
  - "stream=mx.cpu used on all svd/pinv/qr calls — GPU stream raises at graph materialization time in MLX 0.31.x"
  - "mx.linalg.norm is float16-safe and must NOT be wrapped or appear in lint gate pattern"

patterns-established:
  - "_ensure_f32 pattern: Cast float16/bfloat16 to float32 before any MLX linalg op that rejects half-precision"
  - "Lint gate pattern: test_no_bare_linalg_calls scans omlx/ for r'mx\\.linalg\\.(svd|pinv)\\b' outside linalg_utils.py"
  - "scipy bridge pattern: numpy bridge with explicit float64 cast for scipy, result returned as mx.float32"

requirements-completed: [MATH-01, MATH-02, MATH-03]

# Metrics
duration: 14min
completed: 2026-03-18
---

# Phase 1 Plan 01: Linalg Foundation Summary

**Float32-safe MLX linalg wrappers (svd_f32, pinv_f32, qr_f32, nnls_solve) that cast float16/bfloat16 inputs and route all ops through stream=mx.cpu, eliminating silent NaN failures for all downstream compression phases**

## Performance

- **Duration:** 14 min
- **Started:** 2026-03-18T11:06:48Z
- **Completed:** 2026-03-18T11:21:25Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments
- Created `omlx/compression/` package with locked-down empty `__init__.py`
- Implemented 5 symbols in `linalg_utils.py`: `_ensure_f32`, `svd_f32`, `pinv_f32`, `qr_f32`, `nnls_solve` — all with stream=mx.cpu routing
- 21 tests passing including lint gate that prevents bare `mx.linalg.svd/pinv` calls outside the module
- scipy declared as a project dependency in pyproject.toml

## Task Commits

Each task was committed atomically:

1. **Task 1: Create test scaffold** - `d385086` (test)
2. **Task 2: Implement omlx/compression package and linalg_utils.py** - `dad6cfa` (feat)
3. **Task 3: Declare scipy dependency in pyproject.toml** - `c6619ed` (chore)

## Files Created/Modified
- `omlx/compression/__init__.py` - Empty package root (license header only, no re-exports)
- `omlx/compression/linalg_utils.py` - Float32-safe linalg wrappers and scipy NNLS bridge
- `tests/test_linalg_utils.py` - 21 unit tests + lint gate (test_no_bare_linalg_calls)
- `pyproject.toml` - Added scipy>=1.7.0 to [project.dependencies]
- `omlx/scheduler.py` - Removed stale `make_presence_penalty` import (auto-fix)

## Decisions Made
- `omlx/compression/__init__.py` is intentionally empty — no re-exports, locked decision per plan
- All wrappers always return float32; callers own their dtype decisions
- `mx.linalg.norm` intentionally excluded from lint gate and wrappers — it is float16-safe

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Removed stale `make_presence_penalty` import from scheduler.py**
- **Found during:** Task 2 (running tests after implementing linalg_utils)
- **Issue:** `omlx/scheduler.py` imported `make_presence_penalty` from `mlx_lm.sample_utils`, which no longer exports that symbol in the installed version. This caused `ImportError` when importing any `omlx.*` subpackage, making it impossible to run tests.
- **Fix:** Removed `make_presence_penalty` from the import line (it was imported but unused in all call sites)
- **Files modified:** `omlx/scheduler.py`
- **Verification:** `from omlx.compression.linalg_utils import svd_f32` succeeds without error
- **Committed in:** `dad6cfa` (Task 2 commit)

**2. [Rule 1 - Bug] Fixed test_reconstruction to use MLX full-U SVD shape**
- **Found during:** Task 2 (GREEN phase test run)
- **Issue:** MLX 0.31.x SVD returns full U (m x m), thin S (k,), thin Vt (k x n). Test naively wrote `U @ mx.diag(S) @ Vt` which fails because U is (3,3) and diag(S) is (2,2) — incompatible shapes.
- **Fix:** Changed to `U[:, :k] @ mx.diag(S) @ Vt` where k = S.shape[0]
- **Files modified:** `tests/test_linalg_utils.py`
- **Verification:** 21/21 tests pass after fix
- **Committed in:** `dad6cfa` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 blocking import error, 1 test bug)
**Impact on plan:** Both fixes were necessary for test execution. No scope creep. The scheduler fix resolves a pre-existing broken import that was blocking all omlx package tests.

## Issues Encountered
- Pre-existing `conftest.py` imports via `omlx.request` would re-trigger the scheduler import chain. Resolved by fixing the root cause (removing stale import) rather than using `--noconftest` workaround.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All linalg helpers ready for import by phases 2, 3, 4
- `omlx.compression` package structure established — downstream phases add modules alongside `linalg_utils.py`
- Lint gate active — CI will catch any bare `mx.linalg.svd/pinv` calls added outside the wrapper module
- Deferred: `make_logits_processors(presence_penalty=..., frequency_penalty=...)` kwargs in scheduler.py lines 1508-1518 and 1585-1596 will fail at runtime (those kwargs were removed from mlx_lm's API). Logged to deferred-items — out of scope for this phase.

---
*Phase: 01-linalg-foundation*
*Completed: 2026-03-18*
