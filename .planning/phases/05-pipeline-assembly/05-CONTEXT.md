# Phase 5: Pipeline Assembly - Context

**Gathered:** 2026-03-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement `omlx/compression/pipeline.py` — a `KVCachePipeline` class that composes `AMCompactor` and `KVTCCompressor` into a single compress/decompress surface with two distinct call paths: `compact()` for memory-pressure-triggered AM-only compaction, and `compress()` for eviction-path AM→kvtc full pipeline. No omlx cache system wiring in this phase — that is Phase 6. Trigger semantics are tested here via mock; actual hook registration happens in Phase 6.

</domain>

<decisions>
## Implementation Decisions

### Constructor
- `KVCachePipeline(bundle_path=None, am_ratio=4.0, n_sink_tokens=4, sliding_window=128)`
- Path-based: pipeline loads the `.npz` calibration bundle itself, constructs both `AMCompactor` and `KVTCCompressor` internally. Phase 6 passes one path.
- When `bundle_path=None`: both compressors use their bundle=None fallbacks (testing-only path, documented as lower quality).

### Method surface — two call paths
- **`compact(kv_cache, queries=None) -> AMCompactedCache`** — Memory-pressure path. Runs AM compaction only. Returns `AMCompactedCache` (same as `AMCompactor.compact()` output). Phase 6 triggers this on GPU memory pressure.
- **`compress(kv_cache, queries=None) -> PipelineBlob`** — Eviction path. Runs full AM→kvtc pipeline: compact first, then kvtc compress. Returns `PipelineBlob`. Phase 6 triggers this on eviction to SSD.
- **`decompress(blob: PipelineBlob) -> tuple[list[tuple[mx.array, mx.array]], int]`** — Returns `(compacted_layers, logical_seq_len)`. Compacted layers are at reduced token count (not original). `logical_seq_len` is the original T, preserved for RoPE position continuity.

### compress() input and RoPE handling
- `compress(kv_cache, queries=None)` — accepts raw KV cache (same shape as `AMCompactor.compact()` input).
- Pipeline handles RoPE stripping internally before passing keys to `KVTCCompressor`. Stripping uses `strip_rope_from_keys()` from `calibrator.py` with params loaded from the bundle.
- When `bundle_path=None` (testing): RoPE stripping is skipped. Keys passed to kvtc as-is. Consistent with `AMCompactor(head_entropy=None)` and `KVTCCompressor(pca_bundle=None)` patterns.

### PipelineBlob dataclass
- `PipelineBlob(compressed: bytes, logical_seq_len: int, compaction_ratio: float)`
- `compressed`: self-describing kvtc blob (contains all decompression metadata — no bundle needed at decompress time, per Phase 3 design).
- `logical_seq_len`: original sequence length T before AM compaction — preserved for RoPE position indices in continued inference.
- `compaction_ratio`: actual AM ratio achieved (for observability / Phase 8).

### Decompress fidelity
- `decompress()` returns the compacted cache at reduced token count — AM compaction is lossy, no reconstruction to original token count.
- Contract: decompressed cache achieves >0.998 cosine similarity (AM quality guarantee) at compacted token count. Suitable for continued inference.
- `logical_seq_len` flows through the blob so callers can set correct RoPE position offsets.

### Trigger semantics (Phase 5 scope)
- Phase 5 delivers the callable pipeline only. No callback registration, no ABC, no hook points.
- PIPE-03/04/05 trigger behavior is tested in Phase 5 via a mock memory monitor passed as an optional dep to tests — pipeline fires `compact()` above threshold and not below.
- Phase 6 registers the pipeline's `compact()` and `compress()` with omlx's actual memory monitor and eviction path.

### Testing
- **Fast (CI):** Synthetic round-trip: `compress(synthetic_kv_cache)` → `PipelineBlob` → `decompress()` → cosine similarity check + `logical_seq_len` match. No model loading required.
- **Slow (`@pytest.mark.slow`):** Load Qwen 2.5 7B, generate real KV cache, run full `compress()` → `decompress()` round-trip, verify cosine similarity and that inference can continue. Consistent with Phase 3 slow test precedent.
- Mock-based trigger test: mock memory monitor fires above/below threshold; verify `compact()` is called only above threshold.

### Claude's Discretion
- Exact `PipelineBlob` serialization if needed for blob-within-blob storage
- Internal batching strategy across layers
- Error message text and edge-case validation
- Whether `compact()` accepts the same `ratio` override as constructor default

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `omlx/compression/am.py`: `AMCompactor(head_entropy=None, n_sink_tokens=4)` and `AMCompactedCache` dataclass — pipeline wraps these directly. `generate_reference_queries()` helper also available.
- `omlx/compression/kvtc.py`: `KVTCCompressor(pca_bundle=None, n_sink_tokens=4, sliding_window=128)` — `compress(kv_cache) -> bytes` and `decompress(blob) -> list[tuple]`. Phase 5 upgrades the return type to `PipelineBlob`.
- `omlx/compression/calibrator.py`: `load_calibration_bundle(path)` for loading the `.npz` file, `strip_rope_from_keys(keys, rope_theta, traditional, offset)` for RoPE inversion before kvtc. Both are direct imports for the pipeline.
- `omlx/compression/linalg_utils.py`: no direct use in pipeline (linalg is internal to compressors).

### Established Patterns
- `bundle=None` fallback for testing — both `AMCompactor` and `KVTCCompressor` follow this; pipeline extends the pattern.
- `@pytest.mark.slow` for real-model tests — `tests/test_kvtc.py` precedent.
- `__init__.py` stays empty — callers import: `from omlx.compression.pipeline import KVCachePipeline`.
- `_mx_materialize = mx.eval` alias for graph materialization — follow same pattern in pipeline.py.
- `# SPDX-License-Identifier: Apache-2.0` required as first line of any new `.py` file.

### Integration Points
- `pipeline.py` is a sibling of `am.py`, `kvtc.py`, `calibrator.py` in `omlx/compression/`.
- Phase 6 (Cache Integration) will import `KVCachePipeline` and register `compact()` with omlx memory monitor and `compress()` with the SSD eviction path.
- `PipelineBlob.compressed` field must be the same self-describing blob format that `KVTCCompressor.decompress()` already handles — no format change needed.

</code_context>

<specifics>
## Specific Ideas

- Two-method design (`compact()` vs `compress()`) maps directly to the two-tier trigger model: memory pressure → AM-only hot cache compaction; eviction → full cold storage compression. Phase 6 can register them independently.
- `PipelineBlob` bridges the AM and kvtc layers: `compressed` is the kvtc blob, `logical_seq_len` comes from `AMCompactedCache.logical_seq_len`, `compaction_ratio` is computed from the physical token reduction.
- The mock memory monitor in Phase 5 tests should be a simple callable with configurable threshold, not a full mock of the omlx `MemoryMonitor` class — Phase 6 handles the real integration.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 05-pipeline-assembly*
*Context gathered: 2026-03-22*
