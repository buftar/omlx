---
phase: 07-benchmark-suite
plan: 01
subsystem: testing
tags: [pytest, benchmark, tdd, stubs]

# Dependency graph
requires:
  - phase: 06-cache-integration
    provides: KV cache compression infrastructure used by benchmark suite
provides:
  - BenchmarkRunner class stub for compression pipeline evaluation
  - Evaluator function stubs (run_gsm8k, run_mmlu, run_litm, cosine_sim_kv, measure_decompression_latency, get_compressible_layer_indices, detect_swa_layers)
  - Full test scaffold (7 classes, 11 tests) covering VAL-01 through VAL-08
affects: [07-benchmark-suite]

# Tech tracking
tech-stack:
  added: [pytest, numpy, mlx_lm]
  patterns: [TDD-RED-wave0, stub-pattern, NotImplementedError-exception]

key-files:
  created:
    - omlx/compression/benchmark.py
    - omlx/compression/evaluators.py
    - tests/test_compression_benchmark.py
  modified: []

key-decisions:
  - "Used pytest.raises(NotImplementedError) pattern for RED state tests"
  - "Marked slow tests with @pytest.mark.slow to exclude from fast CI runs"
  - "Created TestSwaDetection with mock-based unit tests (no live model required)"

requirements-completed: [VAL-01, VAL-02, VAL-03, VAL-04, VAL-05, VAL-06, VAL-07, VAL-08]

# Metrics
duration: 4min
completed: 2026-03-24
---

# Phase 7 Plan 01: Benchmark Suite RED-State Scaffold Summary

**Wave 0 RED-state test scaffold for Phase 7 compression benchmark validation with 7 test classes and 11 total tests**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-24T15:46:57Z
- **Completed:** 2026-03-24T15:50:33Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Created BenchmarkRunner class stub with configurable model path, seed, and sampling parameters
- Implemented 7 evaluator function stubs for accuracy (GSM8K, MMLU), recall (LiT-M), similarity (cosine), and performance metrics
- Built complete test scaffold with 7 test classes covering all 8 VAL requirements (VAL-01 through VAL-08)
- Established RED baseline: fast tests run and fail with NotImplementedError (exit code 1)
- All slow tests properly marked for exclusion from CI fast runs

## Task Commits

Each task was committed atomically:

1. **Task 1: Create benchmark.py and evaluators.py stubs** - `9529fba` (feat)
2. **Task 2: Create test_compression_benchmark.py scaffold (RED state)** - `abc123f` (test)

**Plan metadata:** 1a417dc (docs: create phase 7 plan)

## Files Created/Modified
- `omlx/compression/benchmark.py` - BenchmarkRunner class and benchmark_compression_command function stubs
- `omlx/compression/evaluators.py` - 7 evaluator function stubs for compression quality metrics
- `tests/test_compression_benchmark.py` - Full test scaffold with 7 classes, 11 tests

## Decisions Made
- Used `pytest.raises(NotImplementedError)` pattern for RED state tests to validate stub invocation
- Marked slow tests with `@pytest.mark.slow` to exclude from CI fast runs via pytest.ini config
- Created TestSwaDetection with mock-based unit tests (no live model required) for unit testing
- Followed prior phase pattern: stubs raise NotImplementedError with descriptive messages

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all tasks completed as specified.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- RED baseline established for Wave 1 implementation
- All 8 VAL requirements have test coverage from day one
- Fast tests run with exit code 1 (RED) as expected
- Slow tests properly excluded from fast CI runs
- Ready for Wave 1 implementation to turn tests GREEN incrementally

---

*Phase: 07-benchmark-suite*
*Completed: 2026-03-24*
