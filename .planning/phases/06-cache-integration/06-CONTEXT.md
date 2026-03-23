# Phase 6: Cache Integration - Context

**Gathered:** 2026-03-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire `KVCachePipeline` (from Phase 5) into omlx's existing cache system using a decorator/subclass pattern. No modification to `CacheManager` ABC. Two integration points: (1) `compact()` on GPU memory pressure via `TieredCacheManager`, and (2) `compress()` on SSD eviction via `PagedSSDCacheManager`. Full config control via `CompressionConfig` with runtime toggle.

</domain>

<decisions>
## Implementation Decisions

### Eviction-path hook (compress())
- Subclass `PagedSSDCacheManager` as `CompressedPagedSSDCacheManager` in `omlx/compression/`
- Override `save_block()` to call `pipeline.compress(cache_data)` on the inference thread (before enqueueing to `_write_queue`) — MLX-safe, background thread stays MLX-free
- Override `load_block()` to call `pipeline.decompress()` after loading, checking `file_metadata["compressed"] == "true"` flag
- When compression disabled: `CacheFactory` creates vanilla `PagedSSDCacheManager` — no overhead, strict PIPE-07 no-op path
- Same block hash as uncompressed; `"compressed": "true"` flag added to safetensors `file_metadata` (the Dict[str, str] header)
- No timeout/fallback — compress always on the save path (not the decode path)
- On `decompress()` failure: log error, return cache miss — inference continues with fresh KV generation

### Memory-pressure hook (compact())
- Subclass `TieredCacheManager` as `CompressedTieredCacheManager`
- Override `check_memory_pressure()`: call `_compact_hot_blocks()` (LRU selection) first, then fall through to existing eviction logic
- Eligible blocks: LRU blocks in hot cache (same selection policy as eviction)
- Compacted `AMCompactedCache` replaces original block in `PagedCacheManager` — immediate memory recovery, no new storage path
- Runs synchronously during `check_memory_pressure()` — already off the hot decode path

### CompressionConfig
- New file: `omlx/compression/config.py` — standalone dataclass, not mixed into `CacheConfig`
- Fields: `enabled: bool = False`, `bundle_path: Optional[str] = None`, `am_ratio: float = 4.0`, `n_sink_tokens: int = 4`, `sliding_window: int = 128`, plus threading lock for runtime toggle
- Runtime toggle: thread-safe `enabled` flag (threading.Event or lock-protected bool) — admin endpoint flips it
- `am_ratio` / `n_components` in `CompressionConfig` override bundle defaults — deployments can tune without re-calibrating

### CacheFactory wiring
- `CacheFactory.create_full_cache_stack()` gets `compression_config: Optional[CompressionConfig] = None` as a separate argument — `CacheConfig` stays unchanged
- When `compression_config` is set and `enabled=True` and `bundle_path` is set: factory creates `CompressedPagedSSDCacheManager` and `CompressedTieredCacheManager`
- When `compression_config=None` or `enabled=False`: factory creates vanilla versions — zero behavioral change

### CLI flags
- `--compression-bundle path.npz` enables compression (absence = disabled) — consistent with `--paged-ssd-cache-dir` opt-in pattern
- `--compression-am-ratio 4.0` and `--compression-n-components N` as optional overrides
- Example: `uv run omlx serve Qwen/Qwen2.5-7B-Instruct --paged-ssd-cache-dir /tmp/cache --compression-bundle ~/.omlx/qwen2.5.npz`

### Block metadata schema
- `file_metadata["compressed"] = "true"` — added to safetensors header dict
- `file_metadata["logical_seq_len"] = str(logical_seq_len)` — stored in safetensors header only (not in `PagedSSDBlockMetadata` index dataclass)
- Absent = uncompressed — backward compatible, no migration needed

### Admin API
- `POST /api/compression/config` in `omlx/admin/routes.py` — follows pattern of `/api/global-settings`
- Body: `{"enabled": bool, "am_ratio": Optional[float]}` — supports both toggle and runtime ratio update
- Protected by existing `require_admin` dependency
- Toggle affects new `save_block()` calls only — existing compressed/uncompressed SSD blocks stay as-is

### Testing
- New `tests/test_cache_integration.py` — fast tests using synthetic KV tensors through real `CompressedPagedSSDCacheManager` (tmp_path fixture, no model load)
- Covers: compress/decompress roundtrip, metadata flags, no-op path (compression disabled), runtime toggle, decompression failure → cache miss
- Existing `tests/test_paged_ssd_cache.py` and `tests/test_prefix_cache.py` run unchanged (compression disabled in factory) — enforces PIPE-07
- One `@pytest.mark.slow` test: real Qwen 2.5 7B → generate KV cache → save_block → load_block → verify cosine similarity and inference can continue

### Claude's Discretion
- Internal `_compact_hot_blocks()` implementation — exactly how many LRU blocks to target per pressure check
- Thread lock granularity in CompressionConfig (threading.Event vs RLock)
- Error message text and validation details
- Whether to add GET /api/compression/status stub (no stats in Phase 6, Phase 8 adds stats — stub returns config only)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `omlx/compression/pipeline.py`: `KVCachePipeline(bundle_path, am_ratio, n_sink_tokens, sliding_window)` — Phase 6 instantiates from `CompressionConfig` fields. `compact()` and `compress()` are the two integration methods.
- `omlx/cache/paged_ssd_cache.py`: `PagedSSDCacheManager.save_block()` and `load_block()` — override points. `file_metadata: Dict[str, str]` is the safetensors header dict where `"compressed"` and `"logical_seq_len"` flags go.
- `omlx/cache/tiered_manager.py`: `TieredCacheManager.check_memory_pressure()` currently returns False (no-op) — override point for `compact()`.
- `omlx/cache/factory.py`: `CacheFactory.create_full_cache_stack()` — add `compression_config` argument, switch types when enabled.
- `omlx/admin/routes.py`: `router = APIRouter(prefix="/admin")` with `@router.post("/api/...")` pattern and `require_admin` dependency — add `/api/compression/config` here.

### Established Patterns
- `bundle=None` fallback for testing — `KVCachePipeline(bundle_path=None)` works for tests without real bundle
- File-based SSD storage (not fixed slots) — variable-length compressed blobs are fine, no slot sizing concern
- `file_metadata` as `Dict[str, str]` in safetensors header — add `"compressed"` and `"logical_seq_len"` following the `"layer_cache_types"` pattern
- Background `_write_queue` thread is MLX-free by design — compression must happen on inference thread before enqueue
- `@pytest.mark.slow` for real-model tests — precedent from Phase 3 and 5
- `# SPDX-License-Identifier: Apache-2.0` on all new `.py` files

### Integration Points
- `omlx/compression/` — new files: `config.py`, `compressed_cache_manager.py` (or inline in `pipeline.py`)
- `omlx/cache/factory.py` — `create_full_cache_stack()` gets `compression_config` arg, creates subclasses when enabled
- `omlx/admin/routes.py` — `POST /api/compression/config` endpoint
- `omlx/cli.py` — `--compression-bundle`, `--compression-am-ratio`, `--compression-n-components` flags wired to `CompressionConfig`
- `tests/test_cache_integration.py` — new test file

</code_context>

<specifics>
## Specific Ideas

- The two-subclass pattern (`CompressedPagedSSDCacheManager` + `CompressedTieredCacheManager`) keeps the integration surgical — no existing classes modified, CacheManager ABC untouched
- `CacheFactory` as the switch point is clean: when `compression_config` is absent/disabled, the factory creates vanilla classes and the execution path is byte-for-byte identical to pre-Phase-6 behavior
- `file_metadata` already stores per-block string metadata in the safetensors header — `"compressed": "true"` and `"logical_seq_len": "8192"` follow the same pattern as `"layer_cache_types"` and shape strings already stored there

</specifics>

<deferred>
## Deferred Ideas

- GET /api/compression/status with full stats (bytes_compressed, blocks_compressed, compression_ratio_mean) — Phase 8 Observability
- Prefix-sharing refcount interaction with compacted blocks — researcher should inspect `BlockAwarePrefixCache` refcount protocol; may be a Phase 6 blocker to characterize
- Variable-length blob SSD slot concern from STATE.md: confirmed non-issue — PagedSSDCacheManager uses file-based (not fixed-slot) storage, variable-length blobs are already the norm

</deferred>

---

*Phase: 06-cache-integration*
*Context gathered: 2026-03-23*
