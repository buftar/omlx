# Requirements: KV Cache Compression Pipeline

**Defined:** 2026-03-18
**Core Value:** Keep more conversations alive in memory by compressing KV caches instead of discarding them

## v1 Requirements

Requirements for the KV cache compression milestone. Each maps to roadmap phases.

### Linalg Foundation

- [x] **MATH-01**: Linalg safety layer wraps MLX ops with automatic float32 cast and CPU stream routing
- [x] **MATH-02**: scipy NNLS wrapper for β-fitting with numpy↔MLX bridge
- [x] **MATH-03**: OLS value-fitting via `mx.linalg.pinv` with correct stream/dtype handling

### AM Compaction

- [x] **AM-01**: User's KV cache is compacted to a target ratio using HighestAttnKeys selection
- [x] **AM-02**: β bias vector is fitted via NNLS to preserve attention mass after compaction
- [x] **AM-03**: Compacted value matrix Cᵥ is fitted via OLS to preserve attention outputs
- [x] **AM-04**: Compacted cache retains logical sequence length T with physical size t for correct RoPE phases
- [x] **AM-05**: Non-uniform head budgets are precomputed per model based on per-head entropy sensitivity
- [x] **AM-06**: Head budget schedule is stored alongside model and reused across compactions
- [x] **AM-07**: Reference queries are generated via repeat-prefill strategy for compaction optimization
- [x] **AM-08**: β values are box-constrained ∈ [−3, 3] for HighestAttnKeys; keys with β < −7 are pruned for OMP

### kvtc Compression

- [x] **KVTC-01**: Cross-layer PCA basis V^T is computed from calibration data with RoPE embeddings stripped
- [x] **KVTC-02**: DP algorithm allocates optimal bit widths per PCA component under a global bit budget
- [x] **KVTC-03**: Quantized PCA coefficients are entropy-coded with zstd for final compressed representation
- [x] **KVTC-04**: Decompression restores KV cache tensors from compressed bitstream for attention computation
- [x] **KVTC-05**: First s=4 tokens (attention sinks) are exempt from compression
- [x] **KVTC-06**: Last w=128 tokens (sliding window) are exempt from compression
- [x] **KVTC-07**: GQA models are handled correctly (compress KV heads, not query heads)

### Calibration

- [x] **CAL-01**: User can run `omlx calibrate-kv <model>` to generate PCA basis for any supported model
- [x] **CAL-02**: Calibration uses randomized SVD on a representative dataset (~200K tokens)
- [x] **CAL-03**: PCA basis V^T, mean µ, and DP bit allocation table are stored alongside model weights
- [x] **CAL-04**: Head entropy sensitivity curves are computed and stored for AM non-uniform budgets
- [x] **CAL-05**: Calibration completes in under 10 minutes for models up to 12B parameters on Apple Silicon

### Pipeline Integration

- [x] **PIPE-01**: AM→kvtc combined pipeline compresses KV cache with multiplicative ratio (token reduction × byte compression)
- [x] **PIPE-02**: Full round-trip compress→decompress restores a cache usable for continued inference
- [x] **PIPE-03**: AM compaction is triggered automatically when GPU memory pressure exceeds threshold
- [x] **PIPE-04**: kvtc compression is triggered on cache eviction to SSD cold storage
- [x] **PIPE-05**: Decompression is triggered on cache miss when restoring from SSD
- [x] **PIPE-06**: Compression integrates with omlx cache system without modifying the CacheManager ABC
- [x] **PIPE-07**: Existing cache behavior is unchanged when compression is disabled (no-op path)
- [x] **PIPE-08**: Compression can be enabled/disabled at runtime via config flags
- [x] **PIPE-09**: Target compression ratios are configurable per deployment
- [x] **PIPE-10**: Decompression latency is under 10ms per layer for 8K context sequences

### Validation

- [x] **VAL-01**: Benchmark suite measures compression ratio, cosine similarity, downstream task accuracy, and decompression latency
- [x] **VAL-02**: AM compaction at 4× maintains >0.998 attention output cosine similarity
- [x] **VAL-03**: kvtc at 16× stays within 1 point of vanilla on standard benchmarks (GSM8K, MMLU, LITM)
- [x] **VAL-04**: Pipeline is validated against Qwen 2.5 7B (GQA, spiked)
- [x] **VAL-05**: Pipeline is validated against Llama 3.x 8B (GQA, popular baseline)
- [x] **VAL-06**: Pipeline is validated against Gemma 3 variants (SWA handling)
- [x] **VAL-07**: Pipeline is validated against DeepSeek R1 (long reasoning chains)
- [x] **VAL-08**: Benchmark results are reproducible via a single CLI command

### Observability

- [x] **OBS-01**: Compression ratio, compaction ratio, and decompression latency are exposed as server metrics
- [x] **OBS-02**: Cache hit/miss rates post-compression are tracked and reported
- [x] **OBS-03**: Compression stats are visible in the omlx admin UI dashboard
- [x] **OBS-05**: Admin UI dashboard with compression settings and stats cards (Wave 1)
- [x] **OBS-04**: Feature documentation covers architecture, configuration, and calibration workflow

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Advanced Compaction

- **AM-ADV-01**: OMP key selection for highest-quality compaction (offline/batch use cases)
- **AM-ADV-02**: Self-study query generation for reference queries (offline, highest quality)
- **AM-ADV-03**: Asymmetric key/value compression ratios for kvtc (task-specific tuning)

### Distribution

- **DIST-01**: Pre-bundled PCA matrices for popular models (Qwen, Llama, Gemma)
- **DIST-02**: Chunked compaction for very long contexts (>32K tokens)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Online/streaming compaction during generation | Only preliminary result in AM paper; RoPE phase handling unresolved for streaming |
| Direct key optimization (Cₖ not restricted to originals) | Requires gradient descent; eliminates closed-form speed advantage |
| Inference in compressed PCA domain | Requires model weight modifications; kvtc paper defers as future work |
| MLA (Multi-head Latent Attention) support | Both papers assume MHA/GQA; MLA derivations don't exist yet |
| L2 regularization on AM β or Cᵥ | AM paper shows degradation at all λ > 0 |
| Per-prompt PCA calibration | Kills compression ratio (1.3–12.4× vs 60–88× one-time) per kvtc paper |
| Training-aware compaction / model modifications | omlx runs off-the-shelf models without modification |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| MATH-01 | Phase 1 | Complete |
| MATH-02 | Phase 1 | Complete |
| MATH-03 | Phase 1 | Complete |
| AM-01 | Phase 2 | Complete |
| AM-02 | Phase 2 | Complete |
| AM-03 | Phase 2 | Complete |
| AM-04 | Phase 2 | Complete |
| AM-05 | Phase 2 | Complete |
| AM-06 | Phase 2 | Complete |
| AM-07 | Phase 2 | Complete |
| AM-08 | Phase 2 | Complete |
| KVTC-01 | Phase 3 | Complete |
| KVTC-02 | Phase 3 | Complete |
| KVTC-03 | Phase 3 | Complete |
| KVTC-04 | Phase 3 | Complete |
| KVTC-05 | Phase 3 | Complete |
| KVTC-06 | Phase 3 | Complete |
| KVTC-07 | Phase 3 | Complete |
| CAL-01 | Phase 4 | Complete |
| CAL-02 | Phase 4 | Complete |
| CAL-03 | Phase 4 | Complete |
| CAL-04 | Phase 4 | Complete |
| CAL-05 | Phase 4 | Complete |
| PIPE-01 | Phase 5 | Complete |
| PIPE-02 | Phase 5 | Complete |
| PIPE-03 | Phase 5 | Complete |
| PIPE-04 | Phase 5 | Complete |
| PIPE-05 | Phase 5 | Complete |
| PIPE-06 | Phase 6 | Complete |
| PIPE-07 | Phase 6 | Complete |
| PIPE-08 | Phase 6 | Complete |
| PIPE-09 | Phase 6 | Complete |
| PIPE-10 | Phase 6 | Complete |
| VAL-01 | Phase 7 | Complete |
| VAL-02 | Phase 7 | Complete |
| VAL-03 | Phase 7 | Complete |
| VAL-04 | Phase 7 | Complete |
| VAL-05 | Phase 7 | Complete |
| VAL-06 | Phase 7 | Complete |
| VAL-07 | Phase 7 | Complete |
| VAL-08 | Phase 7 | Complete |
| OBS-01 | Phase 11 | Complete |
| OBS-02 | Phase 11 | Complete |
| OBS-03 | Phase 11 | Complete |
| OBS-04 | Phase 9 | Complete |
| OBS-05 | Phase 11 | Complete |

**Coverage:**
- v1 requirements: 46 total
- Mapped to phases: 46
- Unmapped: 0

---
*Requirements defined: 2026-03-18*
*Last updated: 2026-03-26 - OBS-01..OBS-05 moved to Phase 11 for tech debt cleanup*
