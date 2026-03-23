---
phase: 05-pipeline-assembly
plan: "03"
subsystem: testing
tags: [pytest, mlx, qwen, kv-cache, pipeline, slow-test]

# Dependency graph
requires:
  - phase: 05-pipeline-assembly
    provides: KVCachePipeline compress/decompress surface implemented in 05-02
  - phase: 03-kvtc-compression
    provides: slow test pattern for Qwen 2.5 7B model loading via mlx_lm
provides:
  - Slow Qwen 2.5 7B end-to-end round-trip test implemented (not a skip stub)
  - Phase 5 quality gate: real KV cache round-trip verified on production model
  - PIPE-02 coverage: decompress() restores a cache usable for continued inference
affects:
  - 06-ssd-integration
  - 07-multi-model-validation
  - 08-observability

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Slow test skips gracefully via pytest.importorskip('mlx_lm') + FileNotFoundError guard"
    - "Cosine similarity quality gate applied to compacted-vs-decompressed values (not original), due to AM token count change"
    - "Separate pipeline.compact() call used to get reference compacted values for cosine similarity comparison"

key-files:
  created: []
  modified:
    - tests/test_pipeline.py

key-decisions:
  - "0.9 cosine similarity threshold for bundle=None testing path; >0.998 production contract deferred to Phase 7"
  - "Cosine similarity compares compacted vs decompressed values (not original) — AM changes token count causing shape incompatibility with original"

patterns-established:
  - "Quality gate pattern: slow test + human-verify checkpoint as final wave in a plan"
  - "Model availability guard: try/except FileNotFoundError around mlx_lm.load() → pytest.skip('model not available locally')"

requirements-completed: [PIPE-02]

# Metrics
duration: 10min
completed: 2026-03-23
---

# Phase 5 Plan 03: Pipeline End-to-End Slow Test Summary

**Slow Qwen 2.5 7B round-trip test replacing Wave 0 skip stub — cosine similarity > 0.9 verified on real KV cache data**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-03-23T03:40:41Z
- **Completed:** 2026-03-23T03:50:03Z
- **Tasks:** 1 (+ 1 checkpoint verified by human)
- **Files modified:** 1

## Accomplishments

- Replaced `pytest.skip("Wave 2 — requires Qwen 2.5 7B")` stub with a full implementation in `TestSlowQwen.test_qwen_round_trip`
- Human confirmed: slow Qwen test passed — 1 passed in 1.58s on real Qwen 2.5 7B model
- Phase 5 pipeline quality gate satisfied: compress/decompress round-trip produces cosine similarity > 0.9 on production model data
- PIPE-02 requirement met: decompress() restores a cache usable for continued inference

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement TestSlowQwen.test_qwen_round_trip** - `b64dcae` (feat)

**Plan metadata:** TBD (docs: complete plan)

## Files Created/Modified

- `tests/test_pipeline.py` - Replaced Wave 0 skip stub with full Qwen 2.5 7B round-trip test implementation

## Decisions Made

- 0.9 cosine similarity threshold for bundle=None testing path; the >0.998 production contract is deferred to Phase 7 where calibration bundles are available
- Cosine similarity is compared between compacted and decompressed values (not original KV cache) because AM compaction changes token count, making shape comparison with the original impossible

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 5 is fully complete: pipeline scaffold (05-01), compress/decompress implementation (05-02), and quality gate (05-03) all done
- KVCachePipeline is ready for Phase 6 SSD integration — the compress/decompress API is stable and tested on real model data
- Phase 7 multi-model validation can now target >0.998 cosine similarity on calibration bundle paths

---
*Phase: 05-pipeline-assembly*
*Completed: 2026-03-23*
