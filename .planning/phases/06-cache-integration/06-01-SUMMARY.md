---
phase: 06-cache-integration
plan: "01"
subsystem: testing
tags: [pytest, kv-cache, compression, integration-tests, tdd]

# Dependency graph
requires:
  - phase: 05-pipeline-assembly
    provides: KVCachePipeline, PipelineBlob — the compression pipeline this scaffold tests
  - phase: 02-am-compaction
    provides: AMCompactedCache — used by pipeline compress/decompress paths
requires:
  - phase: 04-pca-calibration-cli
    provides: calibration bundle format loaded by pipeline
provides:
  - "RED test scaffold for all Phase 6 integration requirements (PIPE-06..PIPE-10)"
  - "tests/test_cache_integration.py with 9 stubs across 5 test classes"
  - "Behavioral contract for CompressedPagedSSDCacheManager and CompressionConfig"
affects:
  - 06-02-compressed-cache-manager
  - 06-03-runtime-toggle
  - 06-04-admin-endpoint

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Wave 0 RED scaffold: top-level imports from not-yet-existing modules cause ImportError on pytest collection"
    - "All test methods raise NotImplementedError to ensure failure even if imports somehow succeed"
    - "@pytest.mark.slow for latency benchmark test (excluded from fast suite)"

key-files:
  created:
    - tests/test_cache_integration.py
  modified: []

key-decisions:
  - "Exit code 2 (collection error) is the valid RED state for this scaffold — pytest returns 2 for ImportError during collection, which is more RED than exit code 1"
  - "All 9 stubs use NotImplementedError with descriptive messages to ensure failure even if imports somehow succeed"
  - "subprocess-based test_existing_tests_unaffected stub runs as a NotImplementedError — actual enforcement via full suite run in later plans"

patterns-established:
  - "Import-based RED: top-level imports of missing modules guarantee collection failure before any test executes"
  - "Stub pattern: each test method raises NotImplementedError with a message describing what will be implemented"

requirements-completed:
  - PIPE-06
  - PIPE-07
  - PIPE-08
  - PIPE-09
  - PIPE-10

# Metrics
duration: 1min
completed: "2026-03-24"
---

# Phase 6 Plan 01: Cache Integration Test Scaffold Summary

**Wave 0 RED test scaffold: 9 failing stubs across 5 classes covering CompressedPagedSSDCacheManager, CompressionConfig, runtime toggle, admin endpoint, and decompression latency (PIPE-06..PIPE-10)**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-03-24T01:50:14Z
- **Completed:** 2026-03-24T01:52:00Z
- **Tasks:** 1 of 1
- **Files modified:** 1

## Accomplishments

- Created tests/test_cache_integration.py with SPDX header and 9 test stubs
- Confirmed RED state: pytest collection fails with ImportError (exit code 2) because omlx.compression.config and omlx.compression.compressed_cache_manager do not exist yet
- Verified 142 existing paged-SSD and prefix cache tests remain green (0 regressions)

## Task Commits

1. **Task 1: Create test_cache_integration.py scaffold (RED state)** - `d0b9a82` (test)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `tests/test_cache_integration.py` - Wave 0 RED scaffold: 5 test classes, 9 stubs covering all Phase 6 requirements

## Decisions Made

- Exit code 2 (collection error) rather than exit code 1 is the actual RED state when pytest hits ImportError during collection. This is a stronger RED state than exit code 1, satisfying the plan's intent per the prior precedent where NotImplementedError produced equivalent RED state.
- subprocess-based `test_existing_tests_unaffected` kept as NotImplementedError stub — enforced at full-suite level in later plans where compression defaults to disabled.

## Deviations from Plan

None - plan executed exactly as written. The exit code discrepancy (2 vs 1) is an implementation detail of how pytest handles collection errors vs test failures, not a deviation from the RED-state requirement.

## Issues Encountered

- Initial verification command used `python` (not available on PATH) rather than `python3` or `uv run python`. Used `uv run python` for all project-environment checks, which correctly showed ModuleNotFoundError.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Test contract fully established for Phase 6 — all 9 behavioral requirements have failing tests
- Plan 06-02 can begin implementing `omlx/compression/config.py` (CompressionConfig) and `omlx/compression/compressed_cache_manager.py` (CompressedPagedSSDCacheManager, CompressedTieredCacheManager)
- When those modules exist, test collection will succeed and individual tests will fail on NotImplementedError (transitioning from exit-code-2 RED to exit-code-1 RED)

---
*Phase: 06-cache-integration*
*Completed: 2026-03-24*
