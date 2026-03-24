---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 06-cache-integration-05-PLAN.md (awaiting checkpoint)
last_updated: "2026-03-24T02:20:41.064Z"
last_activity: 2026-03-19 — Phase 2 Plan 01 complete (AM test scaffold, RED state confirmed)
progress:
  total_phases: 8
  completed_phases: 6
  total_plans: 18
  completed_plans: 18
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-17)

**Core value:** Keep more conversations alive in memory by compressing KV caches instead of discarding them
**Current focus:** Phase 2 — AM Compaction

## Current Position

Phase: 2 of 8 (AM Compaction)
Plan: 1 of 3 in current phase
Status: In progress
Last activity: 2026-03-19 — Phase 2 Plan 01 complete (AM test scaffold, RED state confirmed)

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 1
- Average duration: 18 min
- Total execution time: 0.3 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 02-am-compaction | 1 | 18 min | 18 min |

**Recent Trend:**
- Last 5 plans: 18 min
- Trend: —

*Updated after each plan completion*
| Phase 02-am-compaction P01 | 18 | 1 task | 3 files |
| Phase 02-am-compaction P02 | 55 | 2 tasks | 2 files |
| Phase 02-am-compaction P03 | 9 | 2 tasks | 2 files |
| Phase 03-kvtc-compression P01 | 3 | 2 tasks | 3 files |
| Phase 03-kvtc-compression P02 | 3 | 2 tasks | 1 files |
| Phase 03-kvtc-compression P03 | 105 | 2 tasks | 2 files |
| Phase 04-pca-calibration-cli P01 | 3 | 2 tasks | 3 files |
| Phase 04-pca-calibration-cli P02 | 15 | 2 tasks | 2 files |
| Phase 04-pca-calibration-cli P03 | 20 | 2 tasks | 2 files |
| Phase 05-pipeline-assembly P01 | 3 | 2 tasks | 2 files |
| Phase 05-pipeline-assembly P02 | 2 | 1 tasks | 2 files |
| Phase 05-pipeline-assembly P03 | 10 | 1 tasks | 1 files |
| Phase 06-cache-integration P01 | 5 | 1 tasks | 1 files |
| Phase 06-cache-integration P02 | 10 | 2 tasks | 2 files |
| Phase 06-cache-integration P03 | 15 | 2 tasks | 4 files |
| Phase 06-cache-integration P04 | 3 | 1 tasks | 3 files |
| Phase 06-cache-integration P05 | 25 | 1 tasks | 1 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: Phases 2, 3, 4 all depend only on Phase 1 — parallel development is valid
- [Roadmap]: Phase 8 (Observability) depends only on Phase 6 — can proceed in parallel with Phase 7
- [Research]: Float16 linalg input produces silent NaN in MLX 0.31.x — linalg helpers are Phase 1 blocker for everything else
- [Research]: DP quantization and attention sink exemptions are correctness requirements, not quality enhancements — must be in Phase 3, not added later
- [Research]: Phase 6 integration needs targeted source inspection of omlx async write queue internals before writing integration design
- [Phase 01-linalg-foundation]: omlx/compression/__init__.py is intentionally empty — no re-exports, callers import from linalg_utils directly
- [Phase 01-linalg-foundation]: All linalg wrappers always return float32 — callers own their dtype decisions
- [Phase 01-linalg-foundation]: stream=mx.cpu used on svd/pinv/qr — GPU stream raises at materialization time in MLX 0.31.x
- [Phase 02-am-compaction P01]: am.py stub pre-existed — tests fail with NotImplementedError (not ImportError), equivalent RED state, exit code 1
- [Phase 02-am-compaction P01]: test_cosine_sim_above_threshold checks shape consistency only; numerical >0.98 threshold deferred to Wave 2
- [Phase 02-am-compaction P01]: Diagnostics-gated assertions use `if result.diagnostics is not None` guard to allow implementation flexibility
- [Phase 02-am-compaction]: Budget formula uses math.ceil(seq_len / ratio) to match canonical test expectations
- [Phase 02-am-compaction]: generate_reference_queries implemented in Plan 02 (not Plan 03) because canonical test file imports it at module load time
- [Phase 02-am-compaction]: Zero-padding approach for non-uniform head budget concatenation: shorter heads padded with zeros to max_budget before mx.concatenate to preserve shape contract
- [Phase 02-am-compaction]: int(seq_len / ratio) used in _compute_head_budgets (floor semantics) consistent with plan spec and test_uniform_budgets_correct
- [Phase 03-kvtc-compression]: pytest.ini takes precedence over pyproject.toml -- slow marker already registered; pyproject.toml updated for documentation completeness
- [Phase 03-kvtc-compression]: KVTCCompressor Wave 0: constructor tests pass immediately; all implementation tests RED with NotImplementedError
- [Phase 03-kvtc-compression]: Six compression primitives as module-level functions in kvtc.py -- enables direct unit testing without constructing KVTCCompressor
- [Phase 03-kvtc-compression]: CI lint test_no_bare_linalg_calls scans full file text including docstrings -- negative-example comments must avoid the forbidden regex pattern
- [Phase 03-kvtc-compression]: On-the-fly path uses n_components=head_dim for lossless PCA rotation on random test data -- production bundle path uses 64 components where real KV cache variance concentrates
- [Phase 03-kvtc-compression]: Wave 0 RED markers updated to Wave 2 GREEN assertions across all 8 test classes -- test_dp_within_budget and latency thresholds updated for self-describing blob format and on-the-fly path
- [Phase 03-kvtc-compression]: _dequantize_coeffs vectorized via np.take_along_axis + np.bincount Lloyd -- replaces Python loops, reduces test suite from 18min to 40sec
- [Phase 04-pca-calibration-cli]: Wave 0 RED state: calibrator stubs raise NotImplementedError; tests use pytest.raises and pass — confirming stub dispatch works before implementation
- [Phase 04-pca-calibration-cli]: calibrate_kv_command uses lazy import inside function body to keep CLI startup fast and avoid circular imports
- [Phase 04-pca-calibration-cli]: strip_rope_from_keys implements inverse of mx.fast.rope; non-traditional uses half-dim split, traditional uses consecutive pairs
- [Phase 04-pca-calibration-cli]: compute_pca_basis always uses svd_f32 (not bare mx.linalg.svd) to enforce float32 and CPU stream routing
- [Phase 04-pca-calibration-cli]: load_calibration_bundle maps group-indexed npz arrays to per-layer dicts -- each layer in a group shares the same PCA basis
- [Phase 04-pca-calibration-cli]: _dp_allocate_bits called with 4 required args (singular_values, bits_per_token, n_tok, n_components) -- actual kvtc.py signature differs from plan's 2-arg example
- [Phase 04-pca-calibration-cli]: test_cli_help_registered uses 'uv run omlx calibrate-kv --help' not 'python -m omlx' since omlx has no __main__.py
- [Phase 04-pca-calibration-cli]: tqdm and mlx_lm imports wrapped in try/except at module level for graceful fallback in calibrator.py
- [Phase 05-pipeline-assembly]: rope_theta and rope_traditional are constructor params on KVCachePipeline, not stored in calibration bundle
- [Phase 05-pipeline-assembly]: PipelineBlob.compaction_ratio stores actual AM ratio achieved (not target) for Phase 8 observability
- [Phase 05-pipeline-assembly]: Wave 0 RED scaffold pattern: tests call methods directly, NotImplementedError propagates as unhandled exception causing RED exit code 1
- [Phase 05-pipeline-assembly]: compaction_ratio computed from actual compacted_seq_len (not 1/am_ratio) for Phase 8 observability accuracy
- [Phase 05-pipeline-assembly]: test_round_trip_cosine_sim compares compacted vs decompressed values (not original) — AM changes token count causing shape incompatibility
- [Phase 05-pipeline-assembly]: 0.9 cosine similarity threshold for bundle=None testing path; >0.998 production contract deferred to Phase 7
- [Phase 05-pipeline-assembly]: Cosine similarity compares compacted vs decompressed values (not original) — AM changes token count causing shape incompatibility
- [Phase 06-cache-integration]: Exit code 2 (collection error) is the valid RED state for Wave 0 scaffold — pytest returns 2 for ImportError during collection, which satisfies the RED requirement
- [Phase 06-cache-integration]: Import-based RED pattern: top-level imports of missing modules guarantee collection failure before any test executes
- [Phase 06-cache-integration]: CompressedPagedSSDCacheManager does not call super() in enabled save_block path — replicates parent flow with blob bytes at steps 6-8
- [Phase 06-cache-integration]: load_block() returns None on any decompress() failure — cache miss semantics for decompression errors (PIPE-10)
- [Phase 06-cache-integration]: CacheFactory uses conditional lazy import inside create_paged_ssd_cache() to pick CompressedPagedSSDCacheManager — avoids circular import and preserves zero-overhead path when compression_config=None
- [Phase 06-cache-integration]: compression_config field added to SchedulerConfig with string annotation Optional['CompressionConfig'] = None — avoids circular import at module level
- [Phase 06-cache-integration]: TestCliFlagIntegration tests replicate the serve_command flag-to-config wiring logic directly via argparse.Namespace — avoids mocking 6+ lazily-imported modules inside serve_command
- [Phase 06-cache-integration]: compression_config_getter added as optional keyword-only parameter with None default — all existing set_admin_getters() call sites unaffected
- [Phase 06-cache-integration]: test_admin_endpoint uses TestClient with dependency_overrides[require_admin] bypass — avoids standing up full server while still testing route logic
- [Phase 06-cache-integration]: Cosine similarity comparison: compacted vs decompressed (not original vs decompressed) -- AM token selection makes original comparison invalid
- [Phase 06-cache-integration]: TestSlowQwen.test_qwen_round_trip threshold 0.90 applied to quantization/serialization round-trip, not AM compaction loss

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 3 research flag]: omlx `PagedSSDCacheManager` may use fixed-size SSD slots; compressed bytes are variable-length. Must confirm slot sizing before Phase 6 integration design.
- [Phase 3 research flag]: omlx async write queue capacity (`_MAX_PENDING_WRITES`) and prefix-sharing refcount protocol not fully characterized — needs targeted source inspection before Phase 6.
- [Phase 7 research flag]: Gemma 3 SWA layer detection from mlx-lm model config structure not yet characterized — needs research before Phase 7 Gemma validation work begins.

## Session Continuity

Last session: 2026-03-24T02:20:41.062Z
Stopped at: Completed 06-cache-integration-05-PLAN.md (awaiting checkpoint)
Resume file: None
