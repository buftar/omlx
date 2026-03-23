---
phase: 05-pipeline-assembly
plan: "02"
subsystem: compression
tags: [pipeline, kv-cache, am-compaction, kvtc, rope-stripping, tdd, green-state]

# Dependency graph
requires:
  - phase: 02-am-compaction
    provides: AMCompactor, AMCompactedCache — compact() input/output types
  - phase: 03-kvtc-compression
    provides: KVTCCompressor — compress()/decompress() delegation targets
  - phase: 04-pca-calibration-cli
    provides: strip_rope_from_keys — RoPE inversion at mx->numpy bridge
  - phase: 05-pipeline-assembly-01
    provides: KVCachePipeline stub + PipelineBlob dataclass + 8 RED test scaffold
provides:
  - Full KVCachePipeline implementation in omlx/compression/pipeline.py
  - compact(), compress(), decompress(), _strip_rope() all implemented and GREEN
  - 8 fast tests GREEN; 67 total compression tests GREEN
affects: [06-ssd-integration, 08-observability]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-path pipeline: compact()=AMCompactedCache (in-memory), compress()=PipelineBlob (SSD eviction)"
    - "RoPE bridge: _mx_materialize() graph flush -> np.array(float32) -> strip -> mx.array(float16)"
    - "_rope_params=(theta, traditional) stored only when bundle_path set; None skips stripping"
    - "compaction_ratio computed from actual compacted_seq_len, not constructor target ratio"

key-files:
  created: []
  modified:
    - omlx/compression/pipeline.py
    - tests/test_pipeline.py

key-decisions:
  - "compaction_ratio = original_seq_len / actual_compacted_seq_len (not 1/am_ratio) — reports observed ratio for Phase 8 observability"
  - "test_round_trip_cosine_sim compares compacted values vs decompressed values, not original vs decompressed — AM compaction changes token count so shapes must match"

patterns-established:
  - "Pipeline as pure delegation layer — no math in pipeline.py, all computation delegated to am.py, kvtc.py, calibrator.py"
  - "RoPE stripping is a no-op on bundle_path=None testing path — consistent with AMCompactor(head_entropy=None) and KVTCCompressor(pca_bundle=None) None contracts"

requirements-completed: [PIPE-01, PIPE-02, PIPE-03, PIPE-04, PIPE-05]

# Metrics
duration: 2min
completed: 2026-03-23
---

# Phase 5 Plan 02: Pipeline Assembly Implementation Summary

**KVCachePipeline fully implemented with AM->RoPE-strip->kvtc eviction path and direct AM compact() memory-pressure path; 8 fast tests GREEN, 67 total compression tests GREEN**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-23T03:36:23Z
- **Completed:** 2026-03-23T03:39:04Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments

- `compact()` implemented: delegates to `AMCompactor` with `effective_ratio = ratio or self._am_ratio`
- `compress()` implemented: AM compact -> `_strip_rope()` -> `KVTCCompressor.compress()` -> `PipelineBlob` with actual ratio
- `decompress()` implemented: `KVTCCompressor.decompress(blob.compressed)` returning `(layers, blob.logical_seq_len)`
- `_strip_rope()` implemented: no-op when `_rope_params is None`; mx-to-numpy-to-strip-to-mx conversion for bundle path
- Fixed `test_round_trip_cosine_sim` to compare compacted values vs decompressed values (shape compatibility fix)

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement compact(), compress(), decompress(), _strip_rope()** - `518511e` (feat)

## Files Created/Modified

- `omlx/compression/pipeline.py` — Four methods implemented; `_rope_params` set in `__init__` for both bundle_path branches
- `tests/test_pipeline.py` — `test_round_trip_cosine_sim` fixed to compare compacted vs decompressed values

## Decisions Made

- `compaction_ratio` is computed from `original_seq_len / compacted.layers[0][0].shape[2]` — not from the constructor `am_ratio` param — because AM applies floor + sink token clamping that can deviate from the nominal ratio
- `test_round_trip_cosine_sim` compares `compacted.layers[0][1]` (values post-AM, same token count as decompressed) vs `v_restored` — comparing `v_orig` (300 tokens) vs `v_restored` (~75 tokens) would raise a ValueError from incompatible flat shapes

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_round_trip_cosine_sim shape mismatch**
- **Found during:** Task 1 (implement GREEN phase)
- **Issue:** The Wave 0 test scaffold compared `v_orig` (original 300-token values) vs `v_restored` (compacted ~75-token values). After `.ravel()`, shapes are `(153600,)` vs `(38400,)` — `np.dot()` raises `ValueError: shapes not aligned`.
- **Fix:** Replaced `v_orig = cache[0][1]` with a direct call to `pipeline.compact(cache)` to get `v_compacted = compacted.layers[0][1]` (post-AM, matching token count). Plan spec states "cosine_sim(decompressed_values, compacted_values)" — the fix aligns the test with the spec.
- **Files modified:** `tests/test_pipeline.py`
- **Verification:** cosine_sim passes with value well above 0.9 threshold; all 8 fast tests GREEN
- **Committed in:** `518511e` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Bug was in the Wave 0 scaffold; the fix aligns the test with the plan spec's stated comparison (compacted vs decompressed, not original vs decompressed). No scope creep.

## Issues Encountered

None beyond the auto-fixed cosine sim shape mismatch.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `KVCachePipeline` is fully callable: `compact()` for memory pressure, `compress()` for SSD eviction, `decompress()` for cache restore
- Phase 6 (SSD integration) can wire `pipeline.compact()` and `pipeline.compress()` to the omlx memory monitor and eviction queue
- Phase 8 (observability) can read `PipelineBlob.compaction_ratio` for actual ratio telemetry
- Pre-existing test suite (59 tests across test_am, test_kvtc, test_calibrator) fully GREEN; no regressions

---
*Phase: 05-pipeline-assembly*
*Completed: 2026-03-23*

## Self-Check: PASSED

- omlx/compression/pipeline.py: FOUND
- tests/test_pipeline.py: FOUND
- .planning/phases/05-pipeline-assembly/05-02-SUMMARY.md: FOUND
- Commit 518511e (feat: implement pipeline methods): FOUND
