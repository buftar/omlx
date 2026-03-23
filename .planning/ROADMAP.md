# Roadmap: KV Cache Compression Pipeline

## Overview

This roadmap builds a two-stage KV cache compression pipeline for omlx on Apple Silicon. Work begins with a correctness-critical math foundation (float32 linalg helpers, scipy NNLS bridge), then builds AM compaction and kvtc compression as independent pure-function modules, then unlocks end-to-end compression via the PCA calibration CLI, then wires everything into omlx's existing tiered cache infrastructure as a non-breaking decorator, and finally validates the complete pipeline against quality thresholds and four model families. The result: multiplicative compression (token reduction x byte compression) that keeps more conversations alive in memory without breaking existing cache behavior when disabled.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Linalg Foundation** - Float32-safe MLX linalg helpers and scipy NNLS bridge that all compressors depend on (completed 2026-03-18)
- [x] **Phase 2: AM Compaction** - Stateless attention-matching token compactor with NNLS/OLS fitting and non-uniform head budgets (completed 2026-03-19)
- [x] **Phase 3: kvtc Compression** - PCA-based byte-level storage compressor with DP quantization, zstd entropy coding, and exemption logic (completed 2026-03-19)
- [x] **Phase 4: PCA Calibration CLI** - `omlx calibrate-kv <model>` command that generates PCA bundle required for kvtc (completed 2026-03-23)
- [x] **Phase 5: Pipeline Assembly** - AM-to-kvtc combined pipeline with full compress/decompress round-trip and trigger wiring (completed 2026-03-23)
- [ ] **Phase 6: Cache Integration** - Decorator-pattern integration into omlx cache system with async boundary and config flags
- [ ] **Phase 7: Benchmark Suite** - Task-accuracy-gated benchmark harness with multi-model validation across four model families
- [ ] **Phase 8: Observability** - Server metrics, admin UI stats, and feature documentation

## Phase Details

### Phase 1: Linalg Foundation
**Goal**: All MLX linear algebra operations are float32-safe and CPU-stream-routed so downstream compressors cannot produce silent NaN failures
**Depends on**: Nothing (first phase)
**Requirements**: MATH-01, MATH-02, MATH-03
**Success Criteria** (what must be TRUE):
  1. Calling `svd_f32` or `pinv_f32` with float16 input raises a clear error or automatically casts to float32 — no silent NaN production
  2. All linalg helper calls route to `stream=mx.cpu` — no bare `mx.linalg.*` calls exist outside the helper module
  3. The scipy NNLS wrapper accepts an MLX tensor, calls `scipy.optimize.nnls` via numpy bridge, and returns an MLX tensor
  4. A CI lint gate confirms no bare `mx.linalg.svd` or `mx.linalg.pinv` calls exist outside `omlx/compression/linalg_utils.py`
**Plans**: 1 plan

Plans:
- [x] 01-01-PLAN.md — Create omlx/compression package with float32-safe linalg wrappers and scipy NNLS bridge

### Phase 2: AM Compaction
**Goal**: A stateless `AMCompactor` produces compacted KV caches that preserve attention output quality at the configured token ratio
**Depends on**: Phase 1
**Requirements**: AM-01, AM-02, AM-03, AM-04, AM-05, AM-06, AM-07, AM-08
**Success Criteria** (what must be TRUE):
  1. `AMCompactor.compact(kv_cache, ratio=4)` returns a cache with physical token count reduced to ~25% of input while preserving logical sequence length T for correct RoPE position indices
  2. Attention output cosine similarity between compacted and original cache is >0.998 on Qwen 2.5 7B at 4x compaction
  3. Non-uniform head budgets are computed once per model and reused across compactions — later layers receive larger budgets matching spike observations (per-head entropy 0.34-2.47)
  4. Reference queries are generated via repeat-prefill strategy and used in NNLS beta-fitting
  5. Beta values are box-constrained to [-3, 3] and keys with beta < -7 are pruned when using OMP path
**Plans**: 3 plans

Plans:
- [ ] 02-01-PLAN.md — Wave 0: test scaffold for tests/test_am.py (all AM requirement stubs, RED state)
- [ ] 02-02-PLAN.md — Wave 1: AMCompactedCache dataclass, HighestAttnKeys selection, per-head compaction pipeline (AM-01..AM-04, AM-08)
- [ ] 02-03-PLAN.md — Wave 2: non-uniform head budgets and generate_reference_queries helper (AM-05, AM-06, AM-07)

### Phase 3: kvtc Compression
**Goal**: A stateless `KVTCCompressor` produces byte-level compressed representations of KV cache tensors that decompress within the latency target
**Depends on**: Phase 1
**Requirements**: KVTC-01, KVTC-02, KVTC-03, KVTC-04, KVTC-05, KVTC-06, KVTC-07
**Success Criteria** (what must be TRUE):
  1. `KVTCCompressor.compress(kv_cache)` and `.decompress(blob)` form a correct round-trip: reconstructed tensors are usable for attention computation
  2. First 4 tokens (attention sinks) and last 128 tokens (sliding window) are excluded from compression — confirmed by unit test that ablates exemption and verifies quality collapse
  3. GQA models compress KV heads (not query heads) — head count verified correct on Qwen 2.5 7B
  4. DP bit-allocation runs at compression time and produces variable bit widths per PCA component under a global budget
  5. Decompression completes in under 10ms per layer for 8K context on M3 Max (measured in unit test)
**Plans**: 3 plans

Plans:
- [ ] 03-01-PLAN.md -- Wave 0: dependency setup (zstandard, slow marker), kvtc.py stub, test scaffold RED state
- [ ] 03-02-PLAN.md -- Wave 1: private compression primitives (_split_tokens, _calibrate_onthefly, _dp_allocate_bits, _lloyd_codebook, zstd wrappers)
- [ ] 03-03-PLAN.md -- Wave 2: wire compress() and decompress(), blob format, full GREEN test suite

### Phase 4: PCA Calibration CLI
**Goal**: Users can run a one-time calibration command that generates the PCA bundle required for kvtc to function
**Depends on**: Phase 1
**Requirements**: CAL-01, CAL-02, CAL-03, CAL-04, CAL-05
**Success Criteria** (what must be TRUE):
  1. `omlx calibrate-kv <model>` runs to completion and writes `kv_pca_calibration.npz` alongside model weights containing keys: V, mu, bit_alloc, group_sizes, head_entropy
  2. Reconstruction cosine similarity is flat across token positions (no degradation with position), confirming RoPE was correctly stripped before SVD
  3. Head entropy sensitivity curves are present in the calibration bundle and load correctly into `AMCompactor` for non-uniform budget computation
  4. Calibration completes in under 10 minutes for Qwen 2.5 7B on M3 Max
  5. Running calibration twice on the same model produces a deterministic or near-deterministic bundle (randomized SVD seed is fixed)
**Plans**: 3 plans

Plans:
- [ ] 04-01-PLAN.md — Wave 0: test scaffold (tests/test_calibrator.py RED), calibrator.py stub, cli.py calibrate-kv wiring
- [ ] 04-02-PLAN.md — Wave 1: implement calibrator primitives (strip_rope, compute_pca_basis, bundle I/O, Procrustes) — unit tests GREEN for CAL-02/03/04
- [ ] 04-03-PLAN.md — Wave 2: implement run_calibration() full pipeline, TestCLIDispatch GREEN, human verify

### Phase 5: Pipeline Assembly
**Goal**: AM and kvtc are composed into a single compress/decompress pipeline with correct trigger semantics and a verified end-to-end round-trip
**Depends on**: Phase 2, Phase 3, Phase 4
**Requirements**: PIPE-01, PIPE-02, PIPE-03, PIPE-04, PIPE-05
**Success Criteria** (what must be TRUE):
  1. `pipeline.compress(kv_cache)` applies AM compaction first then kvtc compression, achieving a combined ratio greater than either stage alone
  2. `pipeline.decompress(blob)` restores a KV cache that can be used for continued inference without visible quality degradation
  3. Memory-pressure signal triggers AM compaction via a registered callback — compaction fires when pressure exceeds threshold and not otherwise
  4. Eviction-path signal triggers kvtc compression — compression fires on eviction to SSD and not on normal reads or writes
  5. Decompression fires on cache miss when restoring from SSD and the restored cache is usable for the requesting session
**Plans**: 3 plans

Plans:
- [ ] 05-01-PLAN.md — Wave 0: pipeline.py stub + test_pipeline.py scaffold (RED state)
- [ ] 05-02-PLAN.md — Wave 1: KVCachePipeline full implementation (all fast tests GREEN)
- [ ] 05-03-PLAN.md — Wave 2: Qwen slow test + human verification

### Phase 6: Cache Integration
**Goal**: The compression pipeline is wired into omlx's existing cache system as a non-breaking decorator with async execution and full config control
**Depends on**: Phase 5
**Requirements**: PIPE-06, PIPE-07, PIPE-08, PIPE-09, PIPE-10
**Success Criteria** (what must be TRUE):
  1. All existing omlx cache tests pass unchanged with compression disabled — no behavioral change on the no-op path
  2. The `CacheManager` ABC is unmodified — compression is added via decorator/hook pattern only
  3. Compression operations never execute synchronously on the decode thread — TTFT profile shows no compression in the decode trace
  4. Setting `compression.enabled = false` in config at runtime disables all compression and decompression with no restart required
  5. Target compression ratio and component count are configurable per deployment via `CompressionConfig`
  6. Decompression latency is under 10ms per layer for 8K context sequences in integration tests
**Plans**: TBD

### Phase 7: Benchmark Suite
**Goal**: The compression pipeline has a reproducible benchmark harness that enforces task-accuracy quality thresholds and validates across four model families
**Depends on**: Phase 6
**Requirements**: VAL-01, VAL-02, VAL-03, VAL-04, VAL-05, VAL-06, VAL-07, VAL-08
**Success Criteria** (what must be TRUE):
  1. `omlx benchmark-compression` runs end-to-end and produces a report with: compression ratio, per-stage cosine similarity, decompression latency, perplexity, LITM recall accuracy, GSM8K accuracy, MMLU accuracy
  2. AM at 4x compaction scores >0.998 cosine similarity on Qwen 2.5 7B
  3. kvtc at 16x combined ratio scores within 1 point of vanilla on GSM8K, MMLU, and LITM on Qwen 2.5 7B
  4. Pipeline is validated on Qwen 2.5 7B, Llama 3.x 8B, Gemma 3 (with SWA layer detection), and DeepSeek R1
  5. Gemma 3 SWA layers are detected from model config and skipped during compression — no quality collapse at SWA layers
  6. Benchmark results are deterministic and reproducible from a single CLI invocation with a fixed random seed
**Plans**: TBD

### Phase 8: Observability
**Goal**: Operators can monitor compression effectiveness and quality via server metrics and the omlx admin UI, and users have documentation to onboard
**Depends on**: Phase 6
**Requirements**: OBS-01, OBS-02, OBS-03, OBS-04
**Success Criteria** (what must be TRUE):
  1. Compression ratio, compaction ratio, and decompression latency are emitted as server metrics and visible in any metrics scraper
  2. Cache hit/miss rates before and after compression are tracked separately so operators can distinguish compression-induced misses
  3. All three metric types (compression ratio, decompression latency, cache hit/miss) are visible in the omlx admin UI dashboard without additional configuration
  4. Documentation covers: architecture overview, configuration reference, calibration workflow, and troubleshooting guide for common failure modes (missing bundle, float16 input, latency regression)
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
Note: Phases 2, 3, and 4 depend only on Phase 1 and can be developed in parallel. Phase 8 depends only on Phase 6 and can proceed independently of Phase 7.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Linalg Foundation | 1/1 | Complete   | 2026-03-18 |
| 2. AM Compaction | 3/3 | Complete   | 2026-03-19 |
| 3. kvtc Compression | 3/3 | Complete   | 2026-03-19 |
| 4. PCA Calibration CLI | 3/3 | Complete   | 2026-03-23 |
| 5. Pipeline Assembly | 3/3 | Complete   | 2026-03-23 |
| 6. Cache Integration | 0/TBD | Not started | - |
| 7. Benchmark Suite | 0/TBD | Not started | - |
| 8. Observability | 0/TBD | Not started | - |
