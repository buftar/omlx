---
phase: 06-cache-integration
plan: "04"
subsystem: api
tags: [fastapi, admin, compression, runtime-toggle, pydantic]

# Dependency graph
requires:
  - phase: 06-02
    provides: CompressionConfig dataclass with set_enabled() and _lock
  - phase: 06-03
    provides: SchedulerConfig.compression_config field and CacheFactory wiring
provides:
  - POST /admin/api/compression/config endpoint for runtime compression toggle
  - _get_compression_config module-level accessor in admin/routes.py
  - compression_config_getter lambda wired into server.py set_admin_getters()
affects: [phase-07-gemma-validation, phase-08-observability]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Admin route accessor pattern extended with compression_config_getter=None default
    - Named helper function _get_compression_config_from_pool in server.py for safe pool access

key-files:
  created: []
  modified:
    - omlx/admin/routes.py
    - omlx/server.py
    - tests/test_cache_integration.py

key-decisions:
  - "compression_config_getter added as optional keyword-only parameter with None default — all existing set_admin_getters() call sites unaffected"
  - "Named helper _get_compression_config_from_pool in server.py uses getattr guard for scheduler_config.compression_config — safe against pre-Plan-03 engine pools"
  - "test_admin_endpoint uses TestClient with dependency_overrides[require_admin] bypass — avoids standing up full server while still testing route logic and config mutation"

patterns-established:
  - "Admin runtime toggle pattern: module-level None accessor + set_admin_getters parameter + HTTPException 400 when not configured"

requirements-completed: [PIPE-08]

# Metrics
duration: 3min
completed: 2026-03-23
---

# Phase 06 Plan 04: Admin Compression Config Endpoint Summary

**POST /admin/api/compression/config endpoint wired to live CompressionConfig via set_admin_getters() getter pattern, enabling runtime toggle without server restart (PIPE-08)**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-03-23T20:09:00Z
- **Completed:** 2026-03-23T20:12:32Z
- **Tasks:** 1
- **Files modified:** 3

## Accomplishments

- Added `_get_compression_config` module-level accessor to `omlx/admin/routes.py` following existing `_get_global_settings` pattern
- Extended `set_admin_getters()` with `compression_config_getter=None` keyword parameter, keeping all existing call sites backward-compatible
- Added `CompressionConfigRequest` Pydantic model and `POST /admin/api/compression/config` route protected by `require_admin`
- Added `_get_compression_config_from_pool` helper in `omlx/server.py` with `getattr` guard, wired into `set_admin_getters()` call
- Replaced test stub with real `TestAdminEndpoint.test_admin_endpoint` — GREEN, tests both `enabled` toggle and `am_ratio` mutation

## Task Commits

Each task was committed atomically:

1. **Task 1: Add POST /api/compression/config admin endpoint and wire getter call-site** - `adc859c` (feat)

## Files Created/Modified

- `omlx/admin/routes.py` - Added `_get_compression_config` accessor, extended `set_admin_getters()`, added `CompressionConfigRequest` model and route
- `omlx/server.py` - Added `_get_compression_config_from_pool` helper and wired `compression_config_getter` into `set_admin_getters()` call
- `tests/test_cache_integration.py` - Replaced `test_admin_endpoint` stub with functional TestClient-based test

## Decisions Made

- `compression_config_getter=None` default parameter keeps all existing callers of `set_admin_getters()` working without changes
- Named helper function in server.py (not inline lambda) for readability and safe `getattr` access on `scheduler_config`
- `test_admin_endpoint` uses `app.dependency_overrides[require_admin]` to bypass auth in unit test, avoiding need to mock the full server startup chain

## Deviations from Plan

None — plan executed exactly as written. The test stub update was implicit in the "test_admin_endpoint GREEN" done criterion.

## Issues Encountered

- One pre-existing test failure (`test_stats_response_does_not_include_raw_api_key`) confirmed out-of-scope via `git stash` check — unrelated to this plan's changes.

## Next Phase Readiness

- PIPE-08 complete: operators can toggle compression at runtime via `POST /admin/api/compression/config`
- Phase 07 (Gemma validation) and Phase 08 (observability) can consume the live compression state via the same getter pattern

---
*Phase: 06-cache-integration*
*Completed: 2026-03-23*
