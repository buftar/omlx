# KV Cache Compression Pipeline

## What This Is

A two-stage KV cache compression pipeline for omlx that combines Attention Matching (AM) for runtime token compaction with KV Transform Coding (kvtc) for storage compression. AM reduces the number of tokens in the hot GPU cache under memory pressure; kvtc compresses KV caches to compact byte representations for SSD cold storage on eviction. Together they deliver multiplicative compression (token reduction × byte compression) on Apple Silicon via MLX.

## Core Value

Keep more conversations alive in memory simultaneously by compressing KV caches instead of discarding them — preserving model quality while dramatically reducing memory and storage footprint.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] AM closed-form compaction (NNLS β-fitting, OLS value-fitting, key selection)
- [ ] kvtc PCA-based storage compression (cross-layer PCA, DP quantization, zstd entropy coding)
- [ ] Combined AM→kvtc pipeline with end-to-end compress/decompress
- [ ] Integration with omlx cache system (research to determine optimal integration point)
- [ ] Memory-pressure trigger for AM compaction
- [ ] Eviction-path trigger for kvtc cold storage compression
- [ ] Decompression on cache miss (restore from SSD)
- [ ] PCA calibration CLI (`omlx calibrate-kv <model>`)
- [ ] Non-uniform head budgets for AM (precomputed per model)
- [ ] Sliding window / attention sink exemptions for kvtc
- [ ] Config flags for enable/disable, compression ratios
- [ ] Stats and metrics exposed in admin UI
- [ ] Benchmark suite for compression ratio, quality, and latency
- [ ] Documentation for the feature and calibration workflow
- [ ] Validation against Qwen 2.5 7B, Llama 3.x 8B, Gemma 3 variants, DeepSeek R1

### Out of Scope

- Pre-calibrated PCA bundles for popular models — follow-up after CLI calibration is stable
- Inference in compressed PCA domain (decompression required before attention)
- Direct key optimization for AM (keys restricted to subset of originals, per paper)
- MLA architecture support (DeepSeek-V2 style multi-head latent attention)
- Online/streaming compaction during generation (offline between inference phases only)
- Training-aware compaction or model weight modifications

## Context

**Research papers:**
- Fast KV Compaction via Attention Matching (Zweiger et al., MIT, arXiv:2602.16284, Feb 2026) — closed-form AM compaction achieving 50× with Pareto-dominant quality vs baselines
- KV Cache Transform Coding (Staniszewski & Łańcucki, NVIDIA/Warsaw, ICLR 2026) — 20× lossless-quality storage compression via cross-layer PCA + DP quantization + DEFLATE

**Research spike (on this branch):**
- Working MLX prototypes for both AM and kvtc at `docs/research/kv-cache-compression/`
- Spike validated: 4× AM compaction (cos=0.999), 6.8× kvtc (cos=0.98+), 16× combined
- MLX blockers resolved: SVD needs float32 cast + CPU stream, no lstsq (use pinv), float16 softmax overflow

**Apple Silicon adaptations needed:**
- Replace nvCOMP (NVIDIA) with zstd for entropy coding
- Use MLX linalg: `mx.linalg.svd`, `mx.linalg.pinv` (CPU stream), `mx.linalg.qr`
- scipy NNLS for β-fitting (via numpy interop) or pure MLX alternative

**Existing omlx cache architecture:**
- `omlx/cache/` — paged_cache, paged_ssd_cache, prefix_cache, hybrid_cache, tiered_manager
- `CacheManager` ABC with fetch/store/evict/clear interface
- Memory monitor and process memory enforcer already exist
- Tiered manager handles GPU↔SSD promotion/demotion

## Constraints

- **Platform**: Apple Silicon only (M-series GPUs via MLX) — no CUDA dependencies
- **Framework**: MLX for all tensor operations; numpy/scipy for one-off calibration steps
- **Compatibility**: Must not break existing cache behavior when compression is disabled
- **Quality**: AM compaction at 4× must maintain >0.998 attention output cosine similarity; kvtc at 16× must stay within 1 point of vanilla on standard benchmarks
- **Latency**: Decompression must be fast enough to not dominate TTFT on cache miss (<10ms per layer target)
- **License**: Apache-2.0 (all new files need SPDX header)
- **Fork workflow**: All work on `feature/kv-cache-compression` branch, PR to upstream jundot/omlx

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| AM first, then kvtc in pipeline | AM reduces data size before kvtc processes it — multiplicative gains | — Pending |
| zstd instead of nvCOMP/DEFLATE | nvCOMP is NVIDIA-only; zstd is fast, cross-platform, well-supported on macOS | — Pending |
| CLI calibration per-model (not pre-bundled) | Simpler v1, avoids distribution/versioning of PCA matrices | — Pending |
| Memory pressure triggers AM; eviction triggers kvtc | Two-tier approach matches hot/cold cache distinction already in omlx | — Pending |
| Validate across 4 model families | Qwen (spiked), Llama (popular), Gemma (SWA), DeepSeek R1 (reasoning) — covers key architectures | — Pending |

---
*Last updated: 2026-03-17 after initialization*
