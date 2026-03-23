---
phase: 05-pipeline-assembly
plan: "01"
subsystem: compression
tags: [pipeline, kv-cache, am-compaction, kvtc, dataclass, tdd, red-state]

# Dependency graph
requires:
  - phase: 02-am-compaction
    provides: AMCompactor, AMCompactedCache — compact() output type
  - phase: 03-kvtc-compression
    provides: KVTCCompressor — compress()/decompress() used in pipeline eviction path
  - phase: 04-pca-calibration-cli
    provides: load_calibration_bundle — optional bundle_path constructor arg
provides:
  - KVCachePipeline stub class in omlx/compression/pipeline.py
  - PipelineBlob dataclass with compressed, logical_seq_len, compaction_ratio fields
  - test_pipeline.py scaffold with 8 fast RED tests + 1 slow skipped test
affects: [05-pipeline-assembly-02, 06-ssd-integration, 08-observability]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Wave 0 RED scaffold: stub raises NotImplementedError, tests call directly and fail — same pattern as phases 2, 3, 4"
    - "Lazy imports inside KVCachePipeline.__init__() body for AMCompactor, KVTCCompressor, load_calibration_bundle — avoids circular import risk"
    - "Two-path API: compact() for memory pressure (no bytes), compress() for eviction (SSD blob)"
    - "PipelineBlob is a plain dataclass — self-describing, no MLX arrays, safe for serialization"

key-files:
  created:
    - omlx/compression/pipeline.py
    - tests/test_pipeline.py
  modified: []

key-decisions:
  - "rope_theta and rope_traditional are constructor params on KVCachePipeline, not stored in calibration bundle — matches RESEARCH.md open question 1 resolution"
  - "PipelineBlob.compaction_ratio stores actual AM ratio achieved, not target — supports Phase 8 observability without extra instrumentation"
  - "TestSlowQwen.test_qwen_round_trip uses pytest.skip() body so it is collected and skip-reported (not xfail) — cleaner Wave 2 story"
  - "_mx_materialize = mx.eval alias uses noqa: S307 to document it is MLX graph materialization, not Python built-in eval"

patterns-established:
  - "Two-path compression API: compact() = in-memory, compress() = SSD eviction — established here, consumed by Phase 6"
  - "PipelineBlob dataclass as the serialization boundary — callers see typed fields, not raw bytes"

requirements-completed: [PIPE-01, PIPE-02, PIPE-03, PIPE-04, PIPE-05]

# Metrics
duration: 3min
completed: 2026-03-23
---

# Phase 5 Plan 01: Pipeline Assembly Scaffold Summary

**KVCachePipeline stub + PipelineBlob dataclass with 8 RED tests establishing the two-path compact/compress/decompress contract for Wave 1 implementation**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-23T03:30:44Z
- **Completed:** 2026-03-23T03:33:43Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- `omlx/compression/pipeline.py` created with `PipelineBlob` dataclass and `KVCachePipeline` stub; imports cleanly, constructor accepts all 6 params
- `tests/test_pipeline.py` scaffold with `TestCompress` (2), `TestRoundTrip` (3), `TestTriggerSemantics` (3), and `TestSlowQwen` (1 skipped) — all 8 fast tests RED with exit code 1
- Pre-existing test suite (59 tests across test_am, test_kvtc, test_calibrator) remains GREEN with no regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Create pipeline.py stub** - `53f0af5` (feat)
2. **Task 2: Create test_pipeline.py scaffold** - `7c49472` (test)

## Files Created/Modified

- `omlx/compression/pipeline.py` — KVCachePipeline stub (lazy imports, two-path API, all methods raise NotImplementedError) + PipelineBlob dataclass
- `tests/test_pipeline.py` — 9-test scaffold covering PIPE-01 through PIPE-05

## Decisions Made

- `rope_theta` and `rope_traditional` are constructor params on `KVCachePipeline`, not stored in the calibration bundle — matches RESEARCH.md open question 1 resolution
- `PipelineBlob.compaction_ratio` stores the actual AM ratio achieved (not target) to support Phase 8 observability without extra instrumentation
- `TestSlowQwen.test_qwen_round_trip` uses `pytest.skip()` so it appears as a skip (not xfail) in Wave 2 output
- `_mx_materialize = mx.eval  # noqa: S307` documents that this is MLX graph materialization, not Python's built-in `eval`

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Wave 0 RED state confirmed: `pipeline.py` imports cleanly, `KVCachePipeline()` constructs, all 8 fast tests fail with `NotImplementedError` (exit code 1)
- Plan 05-02 (Wave 1 implementation) can proceed immediately: contract is fully established
- Pre-existing tests GREEN — no regressions introduced

---
*Phase: 05-pipeline-assembly*
*Completed: 2026-03-23*

## Self-Check: PASSED

- omlx/compression/pipeline.py: FOUND
- tests/test_pipeline.py: FOUND
- .planning/phases/05-pipeline-assembly/05-01-SUMMARY.md: FOUND
- Commit 53f0af5 (feat: pipeline.py stub): FOUND
- Commit 7c49472 (test: test_pipeline.py scaffold): FOUND
