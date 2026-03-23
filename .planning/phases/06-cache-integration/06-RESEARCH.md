# Phase 6: Cache Integration - Research

**Researched:** 2026-03-23
**Domain:** Python decorator/subclass pattern, FastAPI admin route, threading safety, safetensors metadata, MLX thread boundaries
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Eviction-path hook (compress())**
- Subclass `PagedSSDCacheManager` as `CompressedPagedSSDCacheManager` in `omlx/compression/`
- Override `save_block()` to call `pipeline.compress(cache_data)` on the inference thread (before enqueueing to `_write_queue`) — MLX-safe, background thread stays MLX-free
- Override `load_block()` to call `pipeline.decompress()` after loading, checking `file_metadata["compressed"] == "true"` flag
- When compression disabled: `CacheFactory` creates vanilla `PagedSSDCacheManager` — no overhead, strict PIPE-07 no-op path
- Same block hash as uncompressed; `"compressed": "true"` flag added to safetensors `file_metadata` (the `Dict[str, str]` header)
- No timeout/fallback — compress always on the save path (not the decode path)
- On `decompress()` failure: log error, return cache miss — inference continues with fresh KV generation

**Memory-pressure hook (compact())**
- Subclass `TieredCacheManager` as `CompressedTieredCacheManager`
- Override `check_memory_pressure()`: call `_compact_hot_blocks()` (LRU selection) first, then fall through to existing eviction logic
- Eligible blocks: LRU blocks in hot cache (same selection policy as eviction)
- Compacted `AMCompactedCache` replaces original block in `PagedCacheManager` — immediate memory recovery, no new storage path
- Runs synchronously during `check_memory_pressure()` — already off the hot decode path

**CompressionConfig**
- New file: `omlx/compression/config.py` — standalone dataclass, not mixed into `CacheConfig`
- Fields: `enabled: bool = False`, `bundle_path: Optional[str] = None`, `am_ratio: float = 4.0`, `n_sink_tokens: int = 4`, `sliding_window: int = 128`, plus threading lock for runtime toggle
- Runtime toggle: thread-safe `enabled` flag (threading.Event or lock-protected bool) — admin endpoint flips it
- `am_ratio` / `n_components` in `CompressionConfig` override bundle defaults — deployments can tune without re-calibrating

**CacheFactory wiring**
- `CacheFactory.create_full_cache_stack()` gets `compression_config: Optional[CompressionConfig] = None` as a separate argument — `CacheConfig` stays unchanged
- When `compression_config` is set and `enabled=True` and `bundle_path` is set: factory creates `CompressedPagedSSDCacheManager` and `CompressedTieredCacheManager`
- When `compression_config=None` or `enabled=False`: factory creates vanilla versions — zero behavioral change

**CLI flags**
- `--compression-bundle path.npz` enables compression (absence = disabled) — consistent with `--paged-ssd-cache-dir` opt-in pattern
- `--compression-am-ratio 4.0` and `--compression-n-components N` as optional overrides
- Example: `uv run omlx serve Qwen/Qwen2.5-7B-Instruct --paged-ssd-cache-dir /tmp/cache --compression-bundle ~/.omlx/qwen2.5.npz`

**Block metadata schema**
- `file_metadata["compressed"] = "true"` — added to safetensors header dict
- `file_metadata["logical_seq_len"] = str(logical_seq_len)` — stored in safetensors header only (not in `PagedSSDBlockMetadata` index dataclass)
- Absent = uncompressed — backward compatible, no migration needed

**Admin API**
- `POST /api/compression/config` in `omlx/admin/routes.py` — follows pattern of `/api/global-settings`
- Body: `{"enabled": bool, "am_ratio": Optional[float]}` — supports both toggle and runtime ratio update
- Protected by existing `require_admin` dependency
- Toggle affects new `save_block()` calls only — existing compressed/uncompressed SSD blocks stay as-is

**Testing**
- New `tests/test_cache_integration.py` — fast tests using synthetic KV tensors through real `CompressedPagedSSDCacheManager` (tmp_path fixture, no model load)
- Covers: compress/decompress roundtrip, metadata flags, no-op path (compression disabled), runtime toggle, decompression failure → cache miss
- Existing `tests/test_paged_ssd_cache.py` and `tests/test_prefix_cache.py` run unchanged (compression disabled in factory) — enforces PIPE-07
- One `@pytest.mark.slow` test: real Qwen 2.5 7B → generate KV cache → save_block → load_block → verify cosine similarity and inference can continue

### Claude's Discretion
- Internal `_compact_hot_blocks()` implementation — exactly how many LRU blocks to target per pressure check
- Thread lock granularity in CompressionConfig (threading.Event vs RLock)
- Error message text and validation details
- Whether to add GET /api/compression/status stub (no stats in Phase 6, Phase 8 adds stats — stub returns config only)

### Deferred Ideas (OUT OF SCOPE)
- GET /api/compression/status with full stats (bytes_compressed, blocks_compressed, compression_ratio_mean) — Phase 8 Observability
- Prefix-sharing refcount interaction with compacted blocks — researcher should inspect `BlockAwarePrefixCache` refcount protocol; may be a Phase 6 blocker to characterize
- Variable-length blob SSD slot concern from STATE.md: confirmed non-issue — PagedSSDCacheManager uses file-based (not fixed-slot) storage, variable-length blobs are already the norm
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PIPE-06 | Compression integrates with omlx cache system without modifying the CacheManager ABC | Two-subclass decorator pattern confirmed; `CacheManager` ABC (`omlx/cache/interface.py`) has `fetch`, `store`, `evict`, `clear`, `get_stats`, `size`, `max_size` — `PagedSSDCacheManager` overrides `save_block`/`load_block` which are not ABC methods, so subclassing freely extends without touching interface |
| PIPE-07 | Existing cache behavior unchanged when compression is disabled | `CacheFactory.create_full_cache_stack()` currently returns vanilla instances; adding `compression_config=None` default means all existing callers unaffected; test suite (2808 fast tests including 70 paged_ssd_cache tests + 72 prefix_cache tests) must stay green |
| PIPE-08 | Compression can be enabled/disabled at runtime via config flags | `CompressionConfig.enabled` with threading lock; `POST /api/compression/config` admin endpoint follows established `require_admin` + `GlobalSettingsRequest` Pydantic model pattern in `routes.py` |
| PIPE-09 | Target compression ratios are configurable per deployment | `CompressionConfig.am_ratio` and `n_components` fields; CLI `--compression-am-ratio` and `--compression-n-components`; `KVCachePipeline(am_ratio=...)` already accepts these as constructor args |
| PIPE-10 | Decompression latency under 10ms per layer for 8K context sequences | Decompression path is `KVTCCompressor.decompress()` → numpy array reconstruction; the 8K token case after AM 4x compaction = 2K physical tokens; `_arrays_from_tensors_raw` reconstructs from raw bytes on inference thread — no I/O involved in hot-cache case |
</phase_requirements>

---

## Summary

Phase 6 wires the existing `KVCachePipeline` (from Phase 5) into omlx's cache system via two new subclasses — `CompressedPagedSSDCacheManager` and `CompressedTieredCacheManager` — plus `CompressionConfig`, CLI flags, and an admin API endpoint. No existing classes are modified. The factory switch pattern ensures the no-op path is byte-for-byte identical to pre-Phase-6 behavior.

The key insight from source inspection is that `save_block()` in `PagedSSDCacheManager` already performs all MLX operations (materialization, `_extract_tensor_bytes`) on the inference thread and then enqueues raw bytes to `_write_queue` for a background thread that is MLX-free. This is exactly where `pipeline.compress()` should be called — after MLX materialization but before the enqueue. The compressed bytes replace the per-layer tensors_raw dict with a single compressed blob stored under a known key (e.g., `"compressed_blob"`), with `"compressed": "true"` and `"logical_seq_len"` added to the metadata dict.

The `check_memory_pressure()` method in `TieredCacheManager` currently returns `False` unconditionally (paged SSD-only mode assumes no GPU pressure from cache data). The `CompressedTieredCacheManager` override calls `_compact_hot_blocks()` first, then calls `super().check_memory_pressure()`.

**Primary recommendation:** Implement both subclasses in a single new file `omlx/compression/compressed_cache_manager.py`. Keep `config.py` separate. Wire the factory, CLI, and admin endpoint as three additional focused tasks.

---

## Standard Stack

### Core (all already present in project)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python `threading` | stdlib | `CompressionConfig` runtime toggle (RLock or Event) | Already used in `paged_ssd_cache.py` for `_pending_write_hashes_lock`, `_hot_cache_lock` |
| `dataclasses` | stdlib | `CompressionConfig` definition | Same pattern as `CacheConfig` in `factory.py` |
| `FastAPI` + `pydantic.BaseModel` | project dep | Admin endpoint request model | Established pattern: `GlobalSettingsRequest(BaseModel)` + `@router.post("/api/global-settings")` |
| `argparse` | stdlib | CLI flags | `cli.py` uses `argparse` (not Click) — `add_argument("--compression-bundle", ...)` |
| `queue.Queue` | stdlib | Existing `_write_queue` background writer | Not added by Phase 6 — background thread must stay MLX-free (existing constraint) |

### Supporting (already in compression/)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `KVCachePipeline` | Phase 5 output | Compress/decompress/compact | Instantiated from `CompressionConfig` fields in both subclasses |
| `PipelineBlob` | Phase 5 output | Self-describing compressed byte container | Serialized as single `"compressed_blob"` tensor-like entry in the safetensors file |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Subclass `PagedSSDCacheManager` | Wrap with composition | Subclassing requires fewer changes — existing `from_config()`/`__init__` flows remain intact; composition would require re-implementing all public methods |
| `threading.RLock` for CompressionConfig | `threading.Event` | RLock allows atomic read+write of both `enabled` and `am_ratio`; Event is simpler but only signals one boolean |

**Installation:** No new packages needed. All dependencies already in project.

---

## Architecture Patterns

### Recommended File Structure

```
omlx/
├── compression/
│   ├── __init__.py                      # unchanged (intentionally empty)
│   ├── config.py                        # NEW: CompressionConfig dataclass
│   ├── compressed_cache_manager.py      # NEW: both subclasses
│   ├── pipeline.py                      # existing
│   ├── am.py                            # existing
│   ├── kvtc.py                          # existing
│   ├── linalg_utils.py                  # existing
│   └── calibrator.py                    # existing
├── cache/
│   └── factory.py                       # MODIFIED: add compression_config arg
├── admin/
│   └── routes.py                        # MODIFIED: add /api/compression/config
└── cli.py                               # MODIFIED: add --compression-bundle flags
tests/
└── test_cache_integration.py            # NEW: fast integration tests
```

### Pattern 1: save_block() Thread Model

The full `save_block()` flow in `PagedSSDCacheManager` is:

1. Check existing index / hot cache (early return on hit)
2. Check queue capacity
3. Build `arrays` dict from `cache_data`; call `mx.eval(*arrays.values())` — inference thread, MLX allowed
4. Call `_extract_tensor_bytes()` on each array — inference thread, extracts raw bytes from materialized arrays
5. Build `cache_entry` and enqueue to `_write_queue` (background writer thread is MLX-free)

The subclass override inserts compression between steps 3 and 5:

- Call `pipeline.compress(cache_data)` — returns `PipelineBlob` — all MLX ops happen here on the inference thread
- Encode `blob.compressed` (bytes) as `mx.array(np.frombuffer(blob_bytes, dtype=np.uint8))`; materialize it
- Call `_extract_tensor_bytes(blob_mx)` — extracts raw bytes from the uint8 array
- Build `tensors_raw = {"compressed_blob": (raw_bytes, "U8", [N])}` — single-tensor representation
- Add `"compressed": "true"` and `"logical_seq_len": str(blob.logical_seq_len)` to metadata dict
- Enqueue `(block_hash, tensors_raw, metadata, file_path)` to `_write_queue` — background writer handles it unchanged

**Key constraint:** `pipeline.compress()` calls AM compaction and KVTC quantization, both using MLX operations. This is safe on the inference thread. It must NOT be called from the `_write_queue` background thread.

### Pattern 2: Blob Storage in safetensors

`PagedSSDCacheManager` stores tensors as `Dict[str, Tuple[bytes, str, List[int]]]` internally (tensors_raw format), serialized to safetensors by the background writer via `_write_safetensors_no_mx()`.

The compressed blob fits this format as a single `mx.uint8` array:
- The dtype `mx.uint8` is mapped to safetensors dtype `"U8"` via `_MX_TO_ST_DTYPE`
- `_extract_tensor_bytes()` supports uint8 (uses `memoryview()` directly)
- Shape `[N]` where N = len(blob.compressed) in bytes
- Stored under key `"compressed_blob"` — distinct from the per-layer `"layer_N_keys"` / `"layer_N_values"` naming convention

### Pattern 3: load_block() Override — Reading file_metadata

`load_block()` returns only `Optional[List[Any]]` — it discards `file_metadata` after reconstruction. Two options for the override:

**Option A (cleaner):** Override `load_block()` to call `mx.load(file_path, return_metadata=True)` directly when the file is on disk, check `file_metadata["compressed"]`, and branch:
- If `"compressed" == "true"`: reconstruct the uint8 blob array, decode to bytes, call `pipeline.decompress()`
- If not present: call `_reconstruct_cache_data()` as usual

For hot-cache hits (block is in `_hot_cache` in-memory): check `entry['file_metadata']['compressed']` directly from the stored cache entry before any disk I/O.

**Option B:** Override `load_block()` to call `self.load_block_with_metadata()`. This method returns `Tuple[Optional[List[Any]], Optional[Dict[str, Any]]]` but the returned `metadata_dict` only populates fixed keys (`"layer_meta_states"`, etc.) — custom keys like `"compressed"` are NOT automatically forwarded.

**Recommendation:** Use Option A. Check `_hot_cache_get(block_hash)` first to handle hot-cache path. For disk path, call `mx.load(..., return_metadata=True)` directly. This mirrors what the parent already does in `load_block()` at line 1285.

### Pattern 4: CompressionConfig with Thread-Safe Toggle

```python
# Source: threading patterns from omlx/cache/paged_ssd_cache.py
import threading
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class CompressionConfig:
    enabled: bool = False
    bundle_path: Optional[str] = None
    am_ratio: float = 4.0
    n_sink_tokens: int = 4
    sliding_window: int = 128
    _lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False, compare=False, init=False
    )

    def is_enabled(self) -> bool:
        with self._lock:
            return self.enabled

    def set_enabled(self, value: bool) -> None:
        with self._lock:
            self.enabled = value

    def set_am_ratio(self, value: float) -> None:
        with self._lock:
            self.am_ratio = value
```

**Note on dataclass `_lock`:** The `field(default_factory=threading.RLock, init=False)` pattern avoids passing the lock through `__init__` while ensuring each instance gets its own lock. `repr=False` and `compare=False` prevent lock objects from appearing in equality checks and repr output.

### Pattern 5: CacheFactory Extension

```python
# Source: omlx/cache/factory.py create_full_cache_stack() lines 191-236
@staticmethod
def create_full_cache_stack(
    config: CacheConfig,
    model: Any = None,
    num_layers: Optional[int] = None,
    compression_config: Optional["CompressionConfig"] = None,  # NEW arg
) -> dict:
    paged_ssd_cache = None
    if config.paged_ssd_cache_dir is not None:
        use_compression = (
            compression_config is not None
            and compression_config.is_enabled()
            and compression_config.bundle_path is not None
        )
        if use_compression:
            from ..compression.compressed_cache_manager import CompressedPagedSSDCacheManager
            pipeline = KVCachePipeline(
                bundle_path=compression_config.bundle_path,
                am_ratio=compression_config.am_ratio,
                n_sink_tokens=compression_config.n_sink_tokens,
                sliding_window=compression_config.sliding_window,
            )
            paged_ssd_cache = CompressedPagedSSDCacheManager(
                pipeline=pipeline,
                compression_config=compression_config,
                cache_dir=cache_dir,
                max_size_bytes=config.max_paged_ssd_cache_size,
            )
        else:
            paged_ssd_cache = PagedSSDCacheManager(...)
```

`CacheConfig` stays unchanged. All existing callers of `create_full_cache_stack()` that do not pass `compression_config` get the vanilla path via the default `None`.

### Pattern 6: Admin Route

```python
# Source: pattern from omlx/admin/routes.py @router.post("/api/global-settings")
class CompressionConfigRequest(BaseModel):
    enabled: bool
    am_ratio: Optional[float] = None

@router.post("/api/compression/config")
async def update_compression_config(
    request: CompressionConfigRequest,
    is_admin: bool = Depends(require_admin),
):
    compression_config = _get_compression_config()  # access via server state
    if compression_config is None:
        raise HTTPException(status_code=404, detail="Compression not configured")
    compression_config.set_enabled(request.enabled)
    if request.am_ratio is not None:
        compression_config.set_am_ratio(request.am_ratio)
    return {"success": True, "enabled": compression_config.enabled}
```

The `_get_compression_config()` helper follows the same pattern as `_get_global_settings()` in `routes.py` — reads from `_server_state`. The `CompressionConfig` instance should be stored on `scheduler_config` (alongside `paged_ssd_cache_dir` and `paged_ssd_cache_max_size`), accessible via `_server_state.engine_pool._scheduler_config.compression_config`.

### Anti-Patterns to Avoid

- **Calling pipeline.compress() from the background writer thread:** The `_write_queue` consumer thread is explicitly MLX-free by design (comment at `paged_ssd_cache.py` line 1038). Any MLX operation there causes Metal GPU resource contention with the inference thread. Compression MUST happen on the inference thread inside `save_block()`.
- **Modifying `CacheManager` ABC or `PagedSSDCacheManager` directly:** PIPE-06 is an explicit prohibition. All changes go in new subclasses.
- **Storing `logical_seq_len` in `PagedSSDBlockMetadata`:** The dataclass is part of the SSD index scanned at startup; adding fields to it changes the on-disk scan format and breaks `from_dict()` for existing cache files. Store only in `file_metadata` (safetensors header).
- **Calling `super().load_block()` and expecting to detect compression from return value:** `load_block()` returns `Optional[List[Any]]` with no metadata. The override must access `file_metadata` directly via `mx.load()` or the hot-cache entry.
- **Using `save_block()`'s return value to signal compression:** The method returns `bool`. Callers already use it this way; compression must not change this contract.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Thread-safe boolean flag | Custom atomic wrapper | `threading.RLock` + guarded bool | stdlib, same pattern as `_pending_write_hashes_lock` in `paged_ssd_cache.py` |
| Safetensors uint8 blob storage | Custom binary format | Existing `_write_safetensors_no_mx()` + `_extract_tensor_bytes()` + uint8 array | `_MX_TO_ST_DTYPE` already maps `mx.uint8` to `"U8"`; file format handles arbitrary dtypes |
| Async admin endpoint validation | Custom auth middleware | `Depends(require_admin)` | Established pattern across all protected admin routes |
| CLI option parsing | argparse subparser clone | `parser.add_argument("--compression-bundle", ...)` | Consistency with existing `--paged-ssd-cache-dir` opt-in pattern in `cli.py` |

**Key insight:** The hardest part of this integration is blob storage. Encoding `PipelineBlob.compressed` as a `mx.uint8` array makes the existing safetensors serialization pipeline handle it with zero custom I/O code — the background writer does not need to know about compression at all.

---

## Common Pitfalls

### Pitfall 1: MLX in Background Writer Thread
**What goes wrong:** Calling `pipeline.compress()` (or any `mx.*` operation) from the `_write_queue` consumer thread causes Metal GPU resource contention with the inference thread, resulting in deadlocks or silent corruption.
**Why it happens:** The background writer was explicitly designed to be MLX-free. Comment at `paged_ssd_cache.py` line 1038: "The background writer thread then writes the safetensors file using pure Python I/O — no mx/Metal API calls needed."
**How to avoid:** Call `pipeline.compress()` inside `save_block()` override on the inference thread, before `_extract_tensor_bytes()`. Only enqueue raw bytes.
**Warning signs:** Test hangs, Metal API errors from non-main thread, or intermittent inference deadlocks.

### Pitfall 2: file_metadata Not Accessible via load_block()
**What goes wrong:** Override of `load_block()` calls `super().load_block()` expecting to detect compression from the returned value — but `super().load_block()` returns only `Optional[List[Any]]`, with no metadata.
**Why it happens:** `load_block()` at line 1285 calls `mx.load(..., return_metadata=True)` and uses `file_metadata` internally but does not return it.
**How to avoid:** Call `mx.load()` directly in the override, or check `_hot_cache_get(block_hash)['file_metadata']` for hot-cache hits.
**Warning signs:** Compressed blocks are returned as raw uint8 arrays to the inference engine, causing shape mismatches.

### Pitfall 3: CacheFactory create_paged_ssd_cache() Bypasses Compression
**What goes wrong:** `CacheFactory.create_full_cache_stack()` calls `create_paged_ssd_cache()` internally (line 219). If only `create_full_cache_stack()` is updated, direct callers of `create_paged_ssd_cache()` get vanilla instances.
**Why it happens:** The factory has multiple independent creation methods.
**How to avoid:** Update both `create_paged_ssd_cache()` and `create_full_cache_stack()`, or clearly document that `compression_config` only applies via `create_full_cache_stack()`.
**Warning signs:** CLI works but direct factory method callers don't exercise compression.

### Pitfall 4: PagedSSDBlockMetadata Field Addition Breaks Startup Scan
**What goes wrong:** Adding `compressed: bool` or `logical_seq_len: int` to `PagedSSDBlockMetadata` breaks `_scan_existing_files()` — existing cache files on disk lack these fields, causing `from_dict()` to raise `TypeError`.
**Why it happens:** The startup scanner at line 758 reconstructs metadata from every `.safetensors` file in the cache directory.
**How to avoid:** Store `"compressed"` and `"logical_seq_len"` only in `file_metadata` (the safetensors header string dict). Never add them to `PagedSSDBlockMetadata`.
**Warning signs:** Server crashes at startup when a cache directory with pre-Phase-6 blocks exists.

### Pitfall 5: Hot Cache Read-Back for Compressed Blocks
**What goes wrong:** After `save_block()` enqueues the compressed block, it is also stored in `_hot_cache` with `entry['file_metadata']['compressed'] == 'true'`. The `load_block()` override must handle this hot-cache path separately from the disk path.
**Why it happens:** `paged_ssd_cache.py` checks hot cache first on `load_block()` (line 1248-1263). If the block was just saved and is still in hot cache, no disk I/O occurs.
**How to avoid:** In the override's `load_block()`, check `entry = self._hot_cache_get(block_hash)` first. If the entry's `file_metadata` has `"compressed": "true"`, decompress from the stored tensors_raw before returning.
**Warning signs:** Cache hits on newly-saved compressed blocks return raw uint8 arrays instead of reconstructed KV tensors.

### Pitfall 6: Runtime Toggle Creates Mixed Block Populations
**What goes wrong:** `enabled` flipped to `False` while `save_block()` is in-flight creates a transition where some blocks are compressed and some are not. This is actually fine by design — each block carries its own `"compressed"` flag. The risk is reading `enabled` non-atomically mid-call.
**Why it happens:** Multi-threaded inference; multiple requests may call `save_block()` concurrently.
**How to avoid:** Read `compression_config.is_enabled()` once at the start of `save_block()` under the lock. Per-block decisions are atomic. Mixed compressed/uncompressed blocks in the same cache are fine — `load_block()` checks the per-block `"compressed"` flag.
**Warning signs:** Spurious decompression failures on blocks written during toggle transitions (if `enabled` is read twice with inconsistent results).

---

## Code Examples

### CompressionConfig skeleton

```python
# Source: based on threading patterns in omlx/cache/paged_ssd_cache.py
# SPDX-License-Identifier: Apache-2.0
import threading
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class CompressionConfig:
    enabled: bool = False
    bundle_path: Optional[str] = None
    am_ratio: float = 4.0
    n_sink_tokens: int = 4
    sliding_window: int = 128
    _lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False, compare=False, init=False
    )

    def is_enabled(self) -> bool:
        with self._lock:
            return self.enabled

    def set_enabled(self, value: bool) -> None:
        with self._lock:
            self.enabled = value

    def set_am_ratio(self, value: float) -> None:
        with self._lock:
            self.am_ratio = value
```

### Blob encoding — uint8 array path

```python
# Source: based on _extract_tensor_bytes() and _MX_TO_ST_DTYPE in paged_ssd_cache.py
# mx.uint8 is mapped to safetensors dtype "U8" — supported by _write_safetensors_no_mx()
import mlx.core as mx
import numpy as np

def encode_blob_as_tensor(blob_bytes: bytes) -> "mx.array":
    """Convert PipelineBlob.compressed bytes to mx.uint8 array.
    Must be called on the inference thread before _extract_tensor_bytes().
    """
    np_blob = np.frombuffer(blob_bytes, dtype=np.uint8)
    blob_mx = mx.array(np_blob, dtype=mx.uint8)
    mx.eval(blob_mx)  # noqa: S307 — MLX tensor materialization, not Python eval
    return blob_mx  # Shape: (len(blob_bytes),)

def decode_blob_from_array(blob_mx) -> bytes:
    """Reconstruct bytes from uint8 mx.array."""
    return np.array(blob_mx).tobytes()
```

### save_block() override skeleton

```python
# Source: structure mirrors omlx/cache/paged_ssd_cache.py save_block() lines 905-1115
def save_block(self, block_hash, cache_data, token_count,
               model_name="", layer_cache_types=None, layer_meta_states=None) -> bool:
    if not self._compression_config.is_enabled():
        return super().save_block(
            block_hash, cache_data, token_count, model_name,
            layer_cache_types, layer_meta_states
        )

    # Check existing index (early return on duplicate)
    if self._index.contains(block_hash):
        self._index.touch(block_hash)
        self._stats["hits"] += 1
        return True

    # --- COMPRESSION ON INFERENCE THREAD (MLX ops allowed here) ---
    try:
        blob = self._pipeline.compress(cache_data)  # AM + KVTC — MLX ops
    except Exception as exc:
        logger.warning(f"Compression failed for {block_hash.hex()[:16]}: {exc}; saving uncompressed")
        return super().save_block(
            block_hash, cache_data, token_count, model_name,
            layer_cache_types, layer_meta_states
        )

    # Encode blob as uint8 tensor and extract raw bytes (inference thread)
    blob_np = np.frombuffer(blob.compressed, dtype=np.uint8)
    blob_mx = mx.array(blob_np, dtype=mx.uint8)
    mx.eval(blob_mx)  # noqa: S307 — MLX tensor materialization
    tensors_raw = {"compressed_blob": _extract_tensor_bytes(blob_mx)}

    metadata = {
        "block_hash": block_hash.hex(),
        "token_count": str(token_count),
        "num_layers": str(len(cache_data)),
        "model_name": model_name,
        "created_at": str(time.time()),
        "compressed": "true",
        "logical_seq_len": str(blob.logical_seq_len),
    }
    if layer_cache_types:
        metadata["layer_cache_types"] = json.dumps(layer_cache_types)

    # Remainder mirrors parent: file_path, block_metadata, hot_cache_put / enqueue
    file_path = self._get_file_path(block_hash)
    # ... (block_metadata creation, hot_cache put, _write_queue.put_nowait)
```

### load_block() override skeleton

```python
# Source: based on load_block() structure in paged_ssd_cache.py lines 1224-1322
def load_block(self, block_hash: bytes):
    # --- Hot cache path ---
    entry = self._hot_cache_get(block_hash)
    if entry is not None:
        file_meta = entry.get('file_metadata', {})
        if file_meta.get('compressed') != 'true':
            # Not compressed — reconstruct normally
            arrays = self._arrays_from_tensors_raw(entry['tensors_raw'])
            return self._reconstruct_cache_data(
                arrays, file_meta, entry['num_layers'], entry.get('layer_cache_types')
            )
        # Compressed — decode and decompress
        return self._decompress_from_tensors_raw(entry['tensors_raw'], file_meta)

    # --- Disk path ---
    metadata = self._index.get(block_hash)
    if metadata is None:
        self._stats["misses"] += 1
        return None

    file_path = metadata.file_path
    if not file_path.exists():
        self._index.remove(block_hash)
        self._stats["misses"] += 1
        return None

    try:
        arrays, file_metadata = mx.load(str(file_path), return_metadata=True)
        if file_metadata.get('compressed') != 'true':
            # Not compressed — normal reconstruction
            return self._reconstruct_cache_data(arrays, file_metadata, metadata.num_layers)
        # Compressed — decompress
        return self._decompress_from_arrays(arrays, file_metadata)
    except Exception:
        return None

def _decompress_from_arrays(self, arrays, file_metadata):
    """Decompress blob from mx.load() output arrays."""
    try:
        blob_mx = arrays["compressed_blob"]
        blob_bytes = np.array(blob_mx).tobytes()
        logical_seq_len = int(file_metadata.get("logical_seq_len", 0))
        from omlx.compression.pipeline import PipelineBlob
        blob = PipelineBlob(compressed=blob_bytes, logical_seq_len=logical_seq_len, compaction_ratio=1.0)
        layers, _ = self._pipeline.decompress(blob)
        return layers
    except Exception as exc:
        logger.error(f"Decompression failed: {exc}")
        return None  # Cache miss — inference regenerates
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| SSD cache reads in executor thread | Reads on inference thread (Metal-safe) | Phase 3 decision | Background thread is MLX-free; compression must stay on inference thread too |
| Fixed safetensors key set per layer | Extensible `file_metadata` dict for arbitrary string flags | From genesis | `"compressed"`, `"logical_seq_len"` follow `"layer_cache_types"` pattern |
| `check_memory_pressure()` always returns False | Stub returns False — override point for Phase 6 | Current state | `CompressedTieredCacheManager` can add compaction without changing vanilla behavior |

**Confirmed non-issues (from STATE.md blockers):**
- Fixed SSD slots for variable-length blobs: `PagedSSDCacheManager` uses file-per-block storage, not fixed slots. Variable-length blobs are the norm.
- `_MAX_PENDING_WRITES` capacity: computed dynamically from system RAM (`min(256, int(total_gb / 2))`, min 32). On M3 Max 128GB: 64 queue slots. Deep enough that synchronous compression on the inference thread does not risk queue starvation under normal conditions.

---

## Open Questions

1. **Does `load_block_with_metadata()` forward custom file_metadata keys?**
   - What we know: `load_block_with_metadata()` at lines 1424-1432 explicitly handles `"layer_meta_states"` from `file_metadata` but the returned `metadata_dict` is built from fixed fields only (`"layer_meta_states"`, `"layer_meta_states"` from file_metadata). Custom keys like `"compressed"` are not automatically forwarded.
   - What's unclear: Whether the returned `metadata_dict` includes all file_metadata keys or just the ones the method explicitly handles.
   - Recommendation: Use Option A (call `mx.load()` directly in the override). Do not rely on `load_block_with_metadata()` to forward custom metadata keys. This avoids the ambiguity entirely.

2. **Where does `CompressionConfig` attach to the server state for admin route access?**
   - What we know: `routes.py` accesses the engine pool via `_server_state.engine_pool`. The `_apply_cache_settings_runtime()` function (line 468) accesses `pool._scheduler_config` directly.
   - Recommendation: Add `compression_config: Optional[CompressionConfig] = None` to `scheduler_config` (the `SchedulerConfig` dataclass), following `paged_ssd_cache_dir` and `paged_ssd_cache_max_size`. The admin route accesses it via `_server_state.engine_pool._scheduler_config.compression_config`. This is consistent with how cache settings are currently managed.

3. **How many blocks should `_compact_hot_blocks()` process per pressure check?**
   - What we know: `check_memory_pressure()` is called periodically. Compaction is synchronous (by design, per CONTEXT.md — already off the hot decode path). `KVCachePipeline.compact()` is fast for individual blocks.
   - Recommendation: Start with N = 4 blocks per pressure check call. This is Claude's Discretion per CONTEXT.md. 4 provides measurable memory recovery per invocation while keeping the synchronous overhead bounded. The number can be tuned later.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 |
| Config file | `pytest.ini` (takes precedence over `pyproject.toml`) |
| Quick run command | `uv run pytest tests/test_cache_integration.py -m "not slow" -x` |
| Full suite command | `uv run pytest tests/ -m "not slow" --ignore=tests/test_updater.py -x` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PIPE-06 | Subclass overrides work without modifying ABC | unit | `uv run pytest tests/test_cache_integration.py::TestCompressedSSDCacheManager -x` | Wave 0 |
| PIPE-06 | `CacheManager` ABC unchanged after Phase 6 | unit | `uv run pytest tests/test_cache_integration.py::TestABCUnchanged -x` | Wave 0 |
| PIPE-07 | Existing tests pass with compression disabled | regression | `uv run pytest tests/test_paged_ssd_cache.py tests/test_prefix_cache.py -x` | Exists |
| PIPE-07 | Factory with no `compression_config` creates vanilla classes | unit | `uv run pytest tests/test_cache_integration.py::TestFactoryNoOpPath -x` | Wave 0 |
| PIPE-08 | Runtime toggle disables compression mid-session | unit | `uv run pytest tests/test_cache_integration.py::TestRuntimeToggle -x` | Wave 0 |
| PIPE-09 | am_ratio and n_components override pipeline defaults | unit | `uv run pytest tests/test_cache_integration.py::TestCompressionConfig -x` | Wave 0 |
| PIPE-10 | Decompression under 10ms per layer for 8K context | perf | `uv run pytest tests/test_cache_integration.py::TestDecompressionLatency -x` | Wave 0 |
| PIPE-06+07+08 | Full round-trip with real Qwen 2.5 7B | slow | `uv run pytest tests/test_cache_integration.py::TestQwenRoundTrip -m slow -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/test_cache_integration.py -m "not slow" -x`
- **Per wave merge:** `uv run pytest tests/ -m "not slow" --ignore=tests/test_updater.py -x`
- **Phase gate:** Full fast suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_cache_integration.py` — covers PIPE-06, PIPE-07, PIPE-08, PIPE-09, PIPE-10
- [ ] `omlx/compression/config.py` — `CompressionConfig` dataclass
- [ ] `omlx/compression/compressed_cache_manager.py` — both subclasses

*(No new framework installs needed — pytest 9.0.2 already present)*

---

## Sources

### Primary (HIGH confidence)
- Direct source inspection: `/Users/tonysina/projects/omlx/omlx/cache/paged_ssd_cache.py` — `save_block()` lines 905-1115, `load_block()` lines 1224-1322, `load_block_with_metadata()` lines 1324-1449, `_write_queue` architecture lines 576-580, `_MAX_PENDING_WRITES` lines 53-70, dtype maps `_MX_TO_ST_DTYPE` lines 93-127
- Direct source inspection: `/Users/tonysina/projects/omlx/omlx/cache/tiered_manager.py` — `check_memory_pressure()` lines 150-162, full class structure
- Direct source inspection: `/Users/tonysina/projects/omlx/omlx/cache/factory.py` — `CacheFactory.create_full_cache_stack()` lines 191-236, `CacheConfig` lines 23-47
- Direct source inspection: `/Users/tonysina/projects/omlx/omlx/cache/interface.py` — `CacheManager` ABC lines 15-119
- Direct source inspection: `/Users/tonysina/projects/omlx/omlx/compression/pipeline.py` — `KVCachePipeline` full class, `PipelineBlob` dataclass, `compress()`, `decompress()`, `compact()` APIs
- Direct source inspection: `/Users/tonysina/projects/omlx/omlx/admin/routes.py` — `require_admin`, `APIRouter`, `GlobalSettingsRequest` pattern, `@router.post("/api/global-settings")` at line 1691
- Direct source inspection: `/Users/tonysina/projects/omlx/omlx/cli.py` — argparse pattern for `--paged-ssd-cache-dir` and cache CLI flags

### Secondary (MEDIUM confidence)
- pytest collection output: 70 tests in `test_paged_ssd_cache.py`, 72 in `test_prefix_cache.py`, 2808 fast tests total in suite

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already present in project; no new dependencies
- Architecture: HIGH — override points, threading model, safetensors serialization path verified line-by-line from source
- Pitfalls: HIGH — pitfalls 1, 2, 4 directly observable from source structure; pitfalls 3, 5, 6 are design-level deductions from verified source patterns
- Open questions: MEDIUM — questions 1 and 2 need one targeted read to confirm; question 3 is Claude's Discretion per CONTEXT.md

**Research date:** 2026-03-23
**Valid until:** 2026-04-22 (stable Python patterns, 30 days)
