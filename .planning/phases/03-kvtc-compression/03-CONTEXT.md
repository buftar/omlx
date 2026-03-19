# Phase 3: kvtc Compression - Context

**Gathered:** 2026-03-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement `omlx/compression/kvtc.py` — a stateless `KVTCCompressor` class that compresses a full model's KV cache to a compact byte blob via PCA projection → DP quantization → zstd entropy coding, and decompresses back to usable tensors. This is pure in-memory tensor math. No file I/O, no cache system integration, no pipeline wiring. Phase 5 (pipeline assembly) wires triggers. Phase 4 (calibration CLI) generates the PCA bundle this phase consumes.

</domain>

<decisions>
## Implementation Decisions

### PCA scope
- **Cross-layer PCA with Procrustes alignment** — not per-layer (spike simplification is intentionally not carried forward)
- Separate K-basis and V-basis — keys and values have different statistics and need tailored bases
- **Layer grouping is configurable** — n_groups is a calibration-time parameter (not hard-coded to 3). The compressor receives a list of (basis, layer_indices) and looks up which basis applies per layer. Compressor is group-count-agnostic.
- Procrustes alignment runs **during calibration only** (Phase 4). Compressor receives pre-aligned bases — no Procrustes logic in the compression hot path.

### KVTCCompressor API
- Constructor: `KVTCCompressor(pca_bundle=None, n_sink_tokens=4, sliding_window=128)`
  - `pca_bundle`: Optional pre-loaded bundle from Phase 4 calibration. When `None` and `compress()` is called, falls back to on-the-fly PCA from the input (testing-only path — documented as lower quality).
  - `n_sink_tokens=4`: Matches Phase 2 AMCompactor and KVTC-05 requirement (first 4 tokens exempt).
  - `sliding_window=128`: Last 128 tokens exempt from compression per KVTC-06.
- `compress(kv_cache) -> bytes`: Takes list of `(keys, values)` tuples, returns a self-describing blob.
- `decompress(blob) -> list[tuple[mx.array, mx.array]]`: Blob is self-describing — contains codebooks, group map, all metadata needed for decompression. No bundle required at decompress time (enables SSD storage without keeping bundle in memory).

### Testability (without Phase 4)
- On-the-fly PCA fallback: when `pca_bundle=None`, `compress()` computes PCA from the input KV cache itself. Same pattern as `AMCompactor(head_entropy=None)`. Explicitly documented as testing-only — no quality guarantees.
- Test fixtures use **actual model KV caches** (Qwen 2.5 7B), marked `@pytest.mark.slow`. No synthetic tensor tests for the integration path — want to catch real RoPE and shape issues.
- One `@pytest.mark.slow` integration test: extract real KV cache → strip RoPE → compress → decompress → verify cosine similarity round-trip.

### DP quantization
- **Proper variable-bit DP allocation** per KVTC-02 — not uniform 4-bit (spike simplification not carried forward).
  - Each PCA component gets its own bit width (e.g., 6 bits for component 0, 2 bits for component 64).
  - DP minimizes reconstruction MSE subject to a target **bits-per-token** budget (configurable parameter on KVTCCompressor).
  - Lloyd's codebook per component used for quantization levels within the allocated bit width.
- **DP bit_alloc table is pre-baked in the calibration bundle** — Phase 4 runs DP once and stores the table. Compression is a lookup at runtime, not recomputed per call.
- Testing fallback (on-the-fly mode): use fixed 4-bit uniform quantization when no calibration bundle available (acceptable for testing only).

### RoPE stripping
- **Caller's responsibility** — `KVTCCompressor.compress()` assumes keys have already had RoPE stripped before the call. No frequency tables inside `kvtc.py`.
- Rationale: Keeps KVTCCompressor model-agnostic. Phase 4 (calibration) and Phase 5 (pipeline) will handle stripping before calling compress().
- Phase 3 scope includes one `@pytest.mark.slow` integration test that: extracts real Qwen 2.5 7B KV cache, strips RoPE, compresses, decompresses, and verifies round-trip cosine similarity. This validates correctness before Phase 5 wires it.

### GQA handling (KVTC-07)
- Compressor operates on KV heads only — not query heads.
- Shape contract: input tensors are `[1, n_kv_heads, seq_len, head_dim]`. Compressor treats `n_kv_heads` as the head dimension and never assumes n_kv_heads == n_query_heads.
- Verified correct on Qwen 2.5 7B (4 KV heads, 8 query heads in GQA config).

### Claude's Discretion
- Exact self-describing blob format (header layout, serialization scheme)
- Exact DP algorithm implementation (standard greedy DP or scipy minimizer)
- Internal batching strategy for per-layer compression
- Error message text and validation details

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `omlx/compression/linalg_utils.py`: `svd_f32` for PCA calibration fallback (float32 + CPU stream handled internally). Direct import: `from omlx.compression.linalg_utils import svd_f32`.
- `docs/research/kv-cache-compression/spike_kvtc.py`: complete prototype for `pca_calibrate()`, `pca_project()`, `pca_reconstruct()`, `dp_quantize()`, `compress_zstd()`, `decompress_zstd()`, and full pipeline. Phase 3 formalizes these patterns.
- `spike_kvtc_results.json`: confirmed 6.8x ratio, cos=0.98+, 1.5ms decompression on 501-token Qwen 2.5 7B. Baseline to beat with cross-layer PCA.
- `omlx/compression/am.py`: AMCompactor establishes the dataclass output pattern and optional-bundle design — KVTCCompressor follows the same idiom.

### Established Patterns
- MLX KV cache shape: `[1, n_kv_heads, seq_len, head_dim]`
- All linalg ops route through `linalg_utils.py` — no bare `mx.linalg.svd` outside that module (CI lint gate enforces this)
- MLX graph materialization calls are aliased in am.py to document intent and avoid security scanner confusion — same pattern in kvtc.py
- `__init__.py` stays empty — callers import: `from omlx.compression.kvtc import KVTCCompressor`
- Tests live in `tests/test_kvtc.py` matching `omlx/compression/kvtc.py`

### Integration Points
- `kvtc.py` is a sibling of `am.py` and `linalg_utils.py` in `omlx/compression/`
- Phase 4 calibration CLI generates a bundle with: `K_bases` (per-group), `V_bases` (per-group), `means` (per-layer, K and V), `group_map` (layer_idx to group_idx), `bit_alloc` (per-group, per-K/V, per-component), `n_components` (per group)
- Phase 5 (pipeline assembly) calls `compress(kv_cache)` after AMCompactor and `decompress(blob)` on cache miss

</code_context>

<specifics>
## Specific Ideas

- The self-describing blob must carry enough to decompress without the bundle in memory — this is critical for SSD cold storage semantics (Phase 6: cache eviction stores blob, retrieval decompresses without looking up bundle)
- Spike's Lloyd's codebook is already proven — keep it for quantization levels, just vary the bit width per component based on DP allocation
- `sliding_window=128` default matches the paper's w=128. Last 128 tokens are stored uncompressed alongside the blob (exempt from PCA+quantization path)
- Attention sinks (first `n_sink_tokens=4`) also stored uncompressed — consistent with Phase 2 AMCompactor's `n_sink_tokens=4`

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 03-kvtc-compression*
*Context gathered: 2026-03-19*
