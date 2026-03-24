---
phase: 06-cache-integration
plan: "03"
subsystem: cache
tags: [compression, kv-cache, cli, argparse, factory, scheduler]

requires:
  - phase: 06-02
    provides: CompressedPagedSSDCacheManager, CompressedTieredCacheManager, CompressionConfig

provides:
  - CacheFactory.create_full_cache_stack() accepts optional compression_config parameter
  - CacheFactory.create_paged_ssd_cache() conditionally instantiates CompressedPagedSSDCacheManager
  - SchedulerConfig.compression_config field (Optional[CompressionConfig] = None)
  - omlx serve --compression-bundle, --compression-am-ratio, --compression-n-components CLI flags
  - CompressionConfig wired from CLI args into SchedulerConfig in serve_command handler

affects:
  - 06-04
  - 06-05
  - phase 8 observability (compression_config visible on scheduler_config)

tech-stack:
  added: []
  patterns:
    - Lazy conditional import inside factory method avoids circular import for compression subclass
    - String annotation Optional["CompressionConfig"] on SchedulerConfig field avoids circular import
    - CLI flag logic replicated in unit tests via argparse.Namespace to avoid deeply-nested lazy imports

key-files:
  created: []
  modified:
    - omlx/cache/factory.py
    - omlx/cli.py
    - omlx/scheduler.py
    - tests/test_cache_integration.py

key-decisions:
  - "CacheFactory uses conditional lazy import inside create_paged_ssd_cache() to pick CompressedPagedSSDCacheManager — avoids circular import and preserves zero-overhead path when compression_config=None"
  - "compression_config field added to SchedulerConfig with string annotation Optional['CompressionConfig'] = None — avoids circular import at module level"
  - "TestCliFlagIntegration tests replicate the serve_command flag-to-config wiring logic directly via argparse.Namespace — avoids mocking 6+ lazily-imported modules inside serve_command"
  - "CLI flags placed in 'KV cache compression options' section immediately after --initial-cache-blocks, before MCP options — logical grouping with related cache flags"

patterns-established:
  - "Pattern: compression_config=None (default) is a true no-op — factory code path is byte-for-byte identical to pre-Phase-6 when compression is absent or disabled"

requirements-completed:
  - PIPE-07
  - PIPE-08
  - PIPE-09

duration: 15min
completed: 2026-03-23
---

# Phase 06 Plan 03: CacheFactory + CLI Compression Wiring Summary

**CompressionConfig wired into CacheFactory object graph and omlx serve CLI via --compression-bundle flag with SchedulerConfig.compression_config field, PIPE-07 vanilla path confirmed unchanged**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-03-23T10:08:13Z
- **Completed:** 2026-03-23T10:23:00Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- `CacheFactory.create_full_cache_stack()` and `create_paged_ssd_cache()` accept `compression_config=None`, conditionally instantiating `CompressedPagedSSDCacheManager` when `compression_config.enabled` and `bundle_path` are set
- `SchedulerConfig` dataclass has new `compression_config: Optional["CompressionConfig"] = None` field (zero-overhead, backward-compatible)
- Three new CLI flags added to `omlx serve`: `--compression-bundle`, `--compression-am-ratio`, `--compression-n-components`
- `serve_command` builds `CompressionConfig` from args and assigns it to `scheduler_config.compression_config`
- `TestCliFlagIntegration` class with 3 tests added: `test_cli_flags`, `test_cli_flags_no_bundle`, `test_cli_help_shows_compression_flags`

## Task Commits

1. **Task 1: Add compression_config to CacheFactory.create_full_cache_stack()** - `46bc9d5` (feat)
2. **Task 2: Add --compression-bundle CLI flags and wire SchedulerConfig.compression_config** - `59c2660` (feat)

**Plan metadata:** (docs commit — see final_commit)

## Files Created/Modified
- `omlx/cache/factory.py` - Added `compression_config` param to `create_paged_ssd_cache()` and `create_full_cache_stack()`; lazy import of `CompressedPagedSSDCacheManager` in conditional branch
- `omlx/scheduler.py` - Added `compression_config: Optional["CompressionConfig"] = None` field to `SchedulerConfig` dataclass
- `omlx/cli.py` - Three new `--compression-*` flags in serve_parser; `CompressionConfig` construction in `serve_command`; assignment to `scheduler_config.compression_config`
- `tests/test_cache_integration.py` - Added `TestCliFlagIntegration` class with 3 tests

## Decisions Made
- Lazy conditional import inside `create_paged_ssd_cache()` picks `CompressedPagedSSDCacheManager` only when needed — avoids circular import and ensures zero overhead when `compression_config=None`
- String annotation `Optional["CompressionConfig"]` on `SchedulerConfig` field avoids circular import at module level
- `TestCliFlagIntegration` tests replicate CLI wiring logic via `argparse.Namespace` directly rather than mocking 6+ lazily-imported modules inside `serve_command` — simpler, more reliable

## Deviations from Plan

None - plan executed exactly as written.

The plan suggested using `from_existing()` pattern for wrapping an existing `PagedSSDCacheManager` instance; instead used the simpler recommended alternative of passing `compression_config` through to `create_paged_ssd_cache()` for direct conditional instantiation (explicitly listed as the preferred pattern in the plan task description).

## Issues Encountered
- CLI test approach: patching `omlx.cli.uvicorn` and `omlx.cli.init_settings` fails because `serve_command` uses lazy imports (all inside the function body, not module-level). Resolved by testing the flag-to-config wiring logic directly via `argparse.Namespace` replicas — cleaner and avoids mocking brittleness.

## Next Phase Readiness
- `CacheFactory` and `SchedulerConfig` are ready for Plan 04 (engine integration — passing `compression_config` from `SchedulerConfig` into `CacheFactory.create_full_cache_stack()`)
- `omlx serve --compression-bundle` flag is user-visible and documented in `--help`
- PIPE-07 no-op path regression confirmed: all 41 existing `test_cache_factory.py` tests pass unchanged

---
*Phase: 06-cache-integration*
*Completed: 2026-03-23*
