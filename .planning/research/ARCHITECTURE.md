# Architecture Research

**Domain:** KV cache compression pipeline integration (Apple Silicon inference server)
**Researched:** 2026-03-18
**Confidence:** HIGH — based on direct inspection of omlx source, spike prototypes, and paper findings

## Standard Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Inference Request Path                        │
│                                                                        │
│   engine_core.py (EngineCore)                                         │
│       └── Scheduler → BatchGenerator (mlx-lm, manages GPU tensors)   │
│                             │ prefill KV cache tensors                │
│                             ▼                                          │
│                   ┌─────────────────────┐                             │
│                   │  AM Compactor        │  ← NEW (memory pressure)   │
│                   │  omlx/compression/   │                             │
│                   │  am_compactor.py     │                             │
│                   └──────────┬──────────┘                             │
│                              │ compacted KV tensors                   │
│                              ▼                                          │
├──────────────────────────────────────────────────────────────────────┤
│                        Cache Coordination Layer                        │
│                                                                        │
│   BlockAwarePrefixCache (prefix_cache.py)                             │
│       ├── PagedCacheManager (paged_cache.py)  [block metadata only]  │
│       └── TieredCacheManager (tiered_manager.py)                     │
│                   │                 │                                  │
│           eviction path        restore path                           │
│                   │                 │                                  │
│                   ▼                 ▲                                  │
│          ┌──────────────────────────────────┐                         │
│          │  kvtc Compressor                  │  ← NEW (eviction path) │
│          │  omlx/compression/                │                        │
│          │  kvtc_compressor.py               │                        │
│          └────────────┬─────────────────────┘                        │
│                       │ compressed bytes (zstd envelope)              │
│                       ▼                                                │
├──────────────────────────────────────────────────────────────────────┤
│                          Cold Storage Layer                            │
│                                                                        │
│   PagedSSDCacheManager (paged_ssd_cache.py)                           │
│       ├── safetensors block files  (uncompressed, existing)           │
│       └── .kvtc block files        (compressed, new)                  │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                     Calibration / Offline Path                         │
│                                                                        │
│   omlx/compression/pca_calibrator.py  (one-time, per-model)          │
│   CLI: omlx calibrate-kv <model>                                      │
│       └── stores PCA matrices alongside model weights                  │
└──────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Location |
|-----------|----------------|----------|
| `AMCompactor` | Token compaction via NNLS beta-fitting + OLS value-fitting. Reduces KV token count (4x target) under memory pressure. Stateless per call. | `omlx/compression/am_compactor.py` (new) |
| `KVTCCompressor` | Byte-level storage compression via PCA projection + DP quantization + zstd. Operates on eviction path before SSD write; decompresses on restore. | `omlx/compression/kvtc_compressor.py` (new) |
| `PCACalibrator` | One-time calibration: runs SVD over representative activations, computes per-layer PCA matrices. Writes calibration bundle to disk. | `omlx/compression/pca_calibrator.py` (new) |
| `CompressionConfig` | Enable/disable flags, compaction ratio, PCA component count, quantization bits, calibration path. Loaded by factory, passed to compressors. | `omlx/compression/config.py` (new) |
| `TieredCacheManager` | Coordinates GPU↔SSD eviction. Gains awareness of compression: calls KVTCCompressor before writing to SSD, calls decompressor on restore. | `omlx/cache/tiered_manager.py` (modified) |
| `BlockAwarePrefixCache` | Prefix lookup and block lifecycle. Calls AMCompactor when memory monitor signals pressure (new hook needed). | `omlx/cache/prefix_cache.py` (modified) |
| `MemoryMonitor` | GPU memory pressure detection via MLX Metal API. Already reports utilization; compression adds a callback interface for AM trigger. | `omlx/memory_monitor.py` (lightly modified) |
| `PagedSSDCacheManager` | SSD block persistence. Needs format flag in block metadata to distinguish compressed vs uncompressed blocks. | `omlx/cache/paged_ssd_cache.py` (lightly modified) |

## Recommended Project Structure

```
omlx/
├── compression/                  # All new compression code lives here
│   ├── __init__.py               # Public API exports
│   ├── config.py                 # CompressionConfig dataclass
│   ├── am_compactor.py           # AMCompactor: NNLS + OLS token compaction
│   ├── kvtc_compressor.py        # KVTCCompressor: PCA + DP quant + zstd
│   ├── pca_calibrator.py         # PCACalibrator: offline SVD calibration
│   └── head_budgets.py           # Non-uniform head budget precomputation
├── cache/
│   ├── tiered_manager.py         # Modified: kvtc on eviction/restore path
│   ├── prefix_cache.py           # Modified: AM trigger hook
│   └── paged_ssd_cache.py        # Modified: compressed block format flag
└── memory_monitor.py             # Modified: pressure callback interface
```

### Structure Rationale

- **`omlx/compression/`**: Isolated from cache machinery. Compressors are pure functions over MLX tensors — no cache state, no block management. This boundary means they can be tested and benchmarked independently, and disabled cleanly via config without touching cache logic.
- **Modification surface is narrow**: Only `tiered_manager.py`, `prefix_cache.py`, `paged_ssd_cache.py`, and `memory_monitor.py` need changes. All new logic goes in `omlx/compression/`.

## Architectural Patterns

### Pattern 1: Decorator / Pass-Through on Eviction and Restore

**What:** `TieredCacheManager.evict_blocks_to_cold()` and `restore_block_from_cold()` are the natural interception points. Compression wraps the SSD write; decompression wraps the SSD read. If compression is disabled (config flag false), the path is identical to today's behavior.

**When to use:** Always — this is the primary integration pattern.

**Trade-offs:** Eviction path can absorb compression latency (background write queue already exists in `PagedSSDCacheManager`). Restore path is latency-sensitive (TTFT); 1.5ms/layer from spike is acceptable but must be measured end-to-end.

**Example (pseudocode):**
```python
# tiered_manager.py evict_blocks_to_cold modification
def _write_block_to_ssd(self, block, kv_tensors):
    if self._compression_config and self._compression_config.kvtc_enabled:
        compressed = self._kvtc_compressor.compress(kv_tensors)
        self._paged_ssd_cache_manager.write_compressed(block.block_hash, compressed)
    else:
        self._paged_ssd_cache_manager.write(block.block_hash, kv_tensors)
```

### Pattern 2: Memory Pressure Callback for AM

**What:** `MemoryMonitor` gains a callback interface. When utilization exceeds threshold, it calls registered handlers. `BlockAwarePrefixCache` registers an AM compaction callback that compacts LRU entries in the hot cache.

**When to use:** AM only runs under memory pressure, not on every request. The callback pattern avoids polling and keeps MemoryMonitor unaware of compression details.

**Trade-offs:** AM compaction time (~0.4s for 28-layer model) blocks the compaction thread. Since omlx serializes all MLX GPU ops onto a single thread (`get_mlx_executor()`), AM must run on that thread or use CPU stream ops only. The spike shows NNLS uses scipy/numpy (CPU), and the OLS uses `mx.linalg.pinv` with `stream=mx.cpu` — both are safe.

**Example (pseudocode):**
```python
# memory_monitor.py addition
def register_pressure_callback(self, callback: Callable[[float], None]) -> None:
    self._pressure_callbacks.append(callback)

def _check_and_notify(self, utilization: float) -> None:
    if utilization > self._pressure_threshold:
        for cb in self._pressure_callbacks:
            cb(utilization)
```

### Pattern 3: PCA Calibration Bundle as Model Artifact

**What:** PCA matrices (one per layer, keys and values separately) are computed once by `omlx calibrate-kv <model>` and stored as a `.npz` or safetensors bundle alongside the model. `KVTCCompressor` loads them at startup and holds them in memory.

**When to use:** Before first use of kvtc. `KVTCCompressor` raises a clear error if calibration file not found, rather than silently falling back to uncompressed writes.

**Trade-offs:** ~28 layers × 2 (K+V) × PCA matrix bytes is small (a few MB). Loading at startup is fine. Calibration takes ~4s per model (spike result) — one-time cost.

## Data Flow

### Eviction Path (AM then kvtc)

```
Memory pressure detected (MemoryMonitor callback)
    ↓
AMCompactor.compact(kv_tensors, budget=seq_len/4)
    → NNLS beta-fitting (scipy, CPU)
    → OLS value-fitting (mx.linalg.pinv, CPU stream)
    → returns compacted_keys [heads, budget, head_dim],
               compacted_values [heads, budget, head_dim]
    ↓
[compacted KV replaces hot cache entry in BlockAwarePrefixCache]
    ↓
Eviction to SSD triggered (TieredCacheManager.evict_blocks_to_cold)
    ↓
KVTCCompressor.compress(compacted_kv_tensors, pca_bundle)
    → PCA projection (mx.linalg operations, float32 cast required)
    → DP quantization per component (4-bit or configured bits)
    → zstd entropy coding
    → returns bytes envelope with header (format version, shape, quant params)
    ↓
PagedSSDCacheManager.write_compressed(block_hash, bytes_envelope)
    → async background write queue (existing machinery)
    → metadata flag: compressed=True, format=kvtc_v1
```

### Restore Path (decompress then use)

```
Cache miss on SSD block (TieredCacheManager.restore_block_from_cold)
    ↓
PagedSSDCacheManager.read(block_hash)
    → reads file, checks metadata flag
    → if compressed=True: returns bytes envelope
    → if compressed=False: returns raw tensors (existing path)
    ↓
KVTCCompressor.decompress(bytes_envelope, pca_bundle)
    → zstd decode
    → DP dequantize
    → PCA back-project (mx.matmul with stored V^T)
    → returns KV tensors (float16, original dtype)
    [Target: <10ms/layer — spike measured 1.5ms/layer]
    ↓
[KV tensors loaded into BatchGenerator for prefill continuation]
```

### Calibration Path (offline, one-time)

```
omlx calibrate-kv <model_path>
    ↓
PCACalibrator.run(model, calibration_corpus)
    → prefill calibration prompts (diverse, ~1000 tokens each)
    → extract KV tensors per layer
    → mx.linalg.svd (float32 cast, stream=mx.cpu)
    → select top-k components (k=64 default, configurable)
    → store per-layer projection matrices
    ↓
write calibration bundle: <model_path>/kv_pca_calibration.npz
```

### Key Data Flows Summary

1. **Normal path (no pressure, no eviction):** Compression components are not invoked. Zero overhead.
2. **AM path (memory pressure, no SSD eviction):** AMCompactor runs in-place on hot cache tensors. KVTCCompressor not involved.
3. **kvtc path (SSD eviction, no prior AM compaction):** KVTCCompressor intercepts eviction write. SSD stores compressed bytes.
4. **Full pipeline (AM then eviction):** AM runs first, then kvtc compresses the already-compacted tensors. 16x combined compression (4x token × ~4x byte).
5. **Restore path (cache miss):** Decompress from SSD, no AM involvement (decompressed tokens stay at compacted count until next prefill).

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| Single model, single user | Current architecture sufficient. Calibration once, compression per eviction. |
| Multiple concurrent requests, single model | AM compaction and kvtc must serialize on the MLX executor thread. Queue depth already managed by `PagedSSDCacheManager._MAX_PENDING_WRITES`. |
| Multiple models loaded simultaneously | Each model needs its own calibration bundle. `KVTCCompressor` scoped per-model. PCA matrices per-model avoid cross-contamination. |

### Scaling Priorities

1. **First bottleneck:** AM compaction time (~0.4s for 28-layer model) blocking inference during memory pressure events. Mitigation: run AM on background executor thread using CPU-only ops (already the case for NNLS/pinv) — does not require the MLX Metal stream.
2. **Second bottleneck:** SSD write throughput under high eviction rates. Mitigation: existing async background write queue in `PagedSSDCacheManager` already decouples this from the inference path.

## Anti-Patterns

### Anti-Pattern 1: Compressing Inside `PagedSSDCacheManager`

**What people do:** Put `KVTCCompressor` inside `PagedSSDCacheManager.store()` directly, since that's where SSD writes happen.

**Why it's wrong:** `PagedSSDCacheManager` is generic storage infrastructure. Encoding knowledge of PCA calibration bundles and model-specific compression into it violates its responsibility boundary. It also makes the compression path hard to disable cleanly and impossible to test without SSD setup.

**Do this instead:** Intercept at `TieredCacheManager` eviction/restore methods. These are already the semantic "cold storage boundary" and have model context available.

### Anti-Pattern 2: Running AM on the MLX Metal Stream

**What people do:** Call `am_compact_layer()` inside the main inference loop on the GPU execution thread, since it uses MLX operations.

**Why it's wrong:** The spike confirms all AM linalg ops require `stream=mx.cpu` (pinv) or CPU-via-numpy (NNLS). Running on the Metal stream causes hangs or incorrect results. Additionally, omlx serializes all Metal ops onto a single thread (`get_mlx_executor()`); AM compaction during inference would stall generation.

**Do this instead:** AM compaction runs on a separate CPU executor. All AM operations use `mx.cpu` stream or numpy/scipy. Compacted tensors are then transferred back to GPU memory as regular MLX arrays.

### Anti-Pattern 3: Compressing Attention Sink and Sliding Window Tokens

**What people do:** Apply AM compaction uniformly to all token positions, including the first few tokens (attention sinks) and the most recent window.

**Why it's wrong:** Attention sinks (first 1-4 tokens) receive disproportionate attention weight — compacting them causes large quality degradation. Sliding window tokens in SWA architectures (Gemma) are critical for recent-context recall.

**Do this instead:** AM compaction must exempt the first N tokens (configurable, default 4) and optionally the last W tokens. kvtc has the same concern for attention sink positions but is less sensitive since it compresses dimensions not token count.

### Anti-Pattern 4: PCA Calibration from Production Traffic Only

**What people do:** Calibrate PCA matrices using the first N requests processed in production.

**Why it's wrong:** Early production traffic is not representative. PCA quality depends on diversity of the calibration corpus. Poor calibration → poor reconstruction → silent quality regression.

**Do this instead:** Use a dedicated calibration CLI (`omlx calibrate-kv`) with a curated prompt set covering the expected input distribution. Ship calibration as a documented step in model deployment, not an automatic background process.

## Integration Points

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `MemoryMonitor` → `BlockAwarePrefixCache` | Callback registration + invocation | New callback interface needed in `MemoryMonitor`. Threshold configurable. |
| `BlockAwarePrefixCache` → `AMCompactor` | Direct method call, synchronous | AMCompactor is stateless; receives (keys, values, queries, budget) → returns compacted tensors. |
| `TieredCacheManager` → `KVTCCompressor` | Direct method call on eviction/restore | `TieredCacheManager.__init__` receives optional `KVTCCompressor`; None = compression disabled. |
| `KVTCCompressor` → `PagedSSDCacheManager` | Passes bytes envelope; SSD manager needs a `write_compressed` / `read_raw` variant | Or: SSD manager detects format from file metadata. Prefer explicit API. |
| `PCACalibrator` → `KVTCCompressor` | Bundle file on disk (`.npz`). `KVTCCompressor` loads at init. | Path in `CompressionConfig`. Error at startup if missing and kvtc enabled. |
| `CompressionConfig` → All compressors | Passed at construction via `CacheFactory` | Config loaded from omlx config file / CLI flags. |

### Build Order (Phase Dependency)

The component dependencies imply this build sequence:

1. **`omlx/compression/` scaffolding + `CompressionConfig`** — No dependencies on existing code. Enables all subsequent work to import from a stable location.

2. **`KVTCCompressor` (compress/decompress, without integration)** — Depends on PCA calibration bundle format only. Can be developed and tested standalone using spike code as reference.

3. **`PCACalibrator` + `omlx calibrate-kv` CLI** — Depends on `KVTCCompressor` bundle format. Must be done before end-to-end kvtc integration tests.

4. **`PagedSSDCacheManager` format flag** — Small, isolated change. Add `compressed` flag to `PagedSSDBlockMetadata`. Can be done in parallel with step 2-3.

5. **`TieredCacheManager` eviction/restore hooks** — Depends on `KVTCCompressor` (step 2) and `PagedSSDCacheManager` format flag (step 4). This is the kvtc integration point.

6. **`AMCompactor`** — Depends on spike prototype only. No integration dependencies. Can be developed in parallel with steps 2-5.

7. **`MemoryMonitor` pressure callback + `BlockAwarePrefixCache` AM hook** — Depends on `AMCompactor` (step 6). This is the AM integration point.

8. **`head_budgets.py` + non-uniform budget precomputation** — Depends on `AMCompactor`. Enhancement, not MVP.

9. **End-to-end benchmark suite and validation** — Depends on all above. Final gate before PR.

## Sources

- Direct inspection: `omlx/cache/interface.py`, `tiered_manager.py`, `paged_ssd_cache.py`, `prefix_cache.py`, `paged_cache.py`, `factory.py`, `memory_monitor.py`, `engine_core.py`
- Spike prototypes: `docs/research/kv-cache-compression/spike_am.py`, `spike_kvtc.py`, `spike_combined.py`
- Spike results: `docs/research/kv-cache-compression/SPIKE-RESULTS.md`
- Project context: `.planning/PROJECT.md`
- Papers: Zweiger et al. arXiv:2602.16284 (AM); Staniszewski & Łańcucki ICLR 2026 (kvtc)

---
*Architecture research for: omlx KV cache compression pipeline (AM + kvtc)*
*Researched: 2026-03-18*
