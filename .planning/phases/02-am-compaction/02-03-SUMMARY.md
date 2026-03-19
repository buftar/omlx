---
phase: 02-am-compaction
plan: 03
subsystem: compression
tags: [mlx, attention-matching, entropy, non-uniform-budgets, kv-cache, compaction, numpy]

# Dependency graph
requires:
  - phase: 02-am-compaction
    plan: 02
    provides: "AMCompactor class and generate_reference_queries helper with uniform budget logic"

provides:
  - "omlx/compression/am.py: AMCompactor._compute_head_budgets() entropy-proportional non-uniform head budgets"
  - "omlx/compression/am.py: compact() updated to use per-head budgets from _compute_head_budgets"
  - "omlx/compression/am.py: generate_reference_queries docstring updated with production-path note"
  - "tests/test_am.py: 4 new direct unit tests for _compute_head_budgets method"

affects:
  - "03-kvtc-compressor"
  - "05-pipeline-assembly"
  - "All phases that consume AMCompactedCache output"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "_compute_head_budgets: entropy-proportional budget formula using numpy; proportions derived from self._head_entropy stored at __init__ time"
    - "Head concatenation padding: when per-head budgets differ, shorter heads zero-padded to max_budget before mx.concatenate along head axis"
    - "Budget rounding correction: diff applied to highest-entropy head to ensure sum(budgets) == n_heads * floor(T/ratio)"

key-files:
  created: []
  modified:
    - "omlx/compression/am.py"
    - "tests/test_am.py"

key-decisions:
  - "Zero-padding approach for non-uniform head budget concatenation: shorter heads padded with zeros to max_budget before mx.concatenate to preserve [1, n_heads, budget, head_dim] shape contract"
  - "int(seq_len / ratio) used in _compute_head_budgets (floor semantics) matching TestHeadBudgets::test_uniform_budgets_correct which uses math.floor -- consistent with plan spec"
  - "4 direct _compute_head_budgets unit tests added (not in canonical test file) to verify method behavior explicitly rather than relying on trivially-passing diagnostics-gated tests"

patterns-established:
  - "Entropy proportions stored at __init__: self._head_entropy captured once; _compute_head_budgets uses stored value, does not recompute from data"
  - "Non-uniform budget formula: proportions = ent / ent.sum(); budgets = max(n_sinks, round(proportions * total)); rounding error += highest-entropy head"

requirements-completed: [AM-05, AM-06, AM-07]

# Metrics
duration: 9min
completed: 2026-03-19
---

# Phase 2 Plan 03: AM Compaction Summary

**Entropy-proportional non-uniform head budgets via _compute_head_budgets, completing all AM requirements (AM-01 through AM-08) with 25/25 tests passing**

## Performance

- **Duration:** 9 min
- **Started:** 2026-03-19T02:52:00Z
- **Completed:** 2026-03-19T02:55:29Z
- **Tasks:** 2 (TDD: RED + GREEN for each)
- **Files modified:** 2

## Accomplishments

- Added `_compute_head_budgets(seq_len, ratio, n_heads) -> list[int]` to `AMCompactor`
  - Uniform mode (head_entropy=None): `max(n_sink_tokens, floor(T/ratio))` per head
  - Entropy mode: budgets proportional to stored `self._head_entropy` with np.maximum(n_sink_tokens) floor and rounding correction on highest-entropy head
- Updated `compact()` to call `_compute_head_budgets` and route `head_budgets[h]` to each `_compact_head` call
- Added zero-padding in concatenation path for non-uniform budgets (heads padded to `max_budget` before `mx.concatenate`)
- Updated `generate_reference_queries` docstring to match plan spec exactly (added production-path note for `queries=None`)
- Added 4 direct unit tests for `_compute_head_budgets` method to provide non-trivial verification

## Task Commits

1. **Task 1 RED: failing _compute_head_budgets tests** - `bb28bc4` (test)
2. **Task 1 GREEN: _compute_head_budgets implementation + compact() update** - `2ca0ea2` (feat)
3. **Task 2 feat: generate_reference_queries docstring** - `cf801b7` (feat)

## Files Created/Modified

- `omlx/compression/am.py` - Added `_compute_head_budgets()` method, updated `compact()` budget logic, updated `generate_reference_queries` docstring
- `tests/test_am.py` - Added 4 direct unit tests for `_compute_head_budgets`

## Decisions Made

- Zero-padding approach: when heads have different budgets (non-uniform entropy case), shorter heads are padded with zeros to `max_budget` before `mx.concatenate`. This preserves the `[1, n_heads, budget, head_dim]` shape contract without breaking existing shape invariant tests (which all use `head_entropy=None` uniform path).
- `int(seq_len / ratio)` (floor) used for uniform_budget inside `_compute_head_budgets` -- matches the `math.floor` in `test_uniform_budgets_correct`.
- Direct unit tests added for `_compute_head_budgets` because the original `TestHeadBudgets` tests only checked `diagnostics` (which is None), making them trivially passing without any real behavior verification.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Non-uniform head budgets caused mx.concatenate shape mismatch**
- **Found during:** Task 1 GREEN verification
- **Issue:** `mx.concatenate(compacted_keys_per_head, axis=1)` failed with `ValueError` when per-head budgets differ (e.g., [19, 4, 28, 13] from entropy [1.68, 0.34, 2.47, 1.12]). Shapes were `(1,1,19,128), (1,1,4,128), (1,1,28,128), (1,1,13,128)` -- incompatible along axis 2 when concatenating on axis 1.
- **Fix:** Added zero-padding loop before concatenation: each head is padded to `max_budget` when budgets are non-uniform.
- **Files modified:** `omlx/compression/am.py`
- **Commit:** `2ca0ea2`

---

**Total deviations:** 1 (auto-fixed bug in concatenation path for non-uniform budgets)
**Impact on plan:** Fix is backward-compatible -- uniform budget path (head_entropy=None) is unaffected.

## Self-Check: PASSED

- `omlx/compression/am.py` exists and contains `_compute_head_budgets`
- `tests/test_am.py` exists and contains `test_compute_head_budgets_uniform`
- All commits exist: bb28bc4, 2ca0ea2, cf801b7
- 25/25 test_am.py tests pass
- Lint gate clean: test_no_bare_linalg_calls passes
- Import smoke test: `from omlx.compression.am import AMCompactor, AMCompactedCache, generate_reference_queries` succeeds

## Next Phase Readiness

- All AM requirements complete: AM-01 through AM-08
- AMCompactor with full entropy-proportional non-uniform budgets ready for Phase 3 (KVTC compressor) and Phase 5 (pipeline assembly)
- Phase 2 is feature-complete and ready for /gsd:verify-work

---
*Phase: 02-am-compaction*
*Completed: 2026-03-19*
