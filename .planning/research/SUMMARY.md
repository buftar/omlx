# Project Research Summary

**Project:** omlx KV Cache Compression Pipeline (AM + kvtc)
**Domain:** Two-stage KV cache compression for Apple Silicon LLM inference
**Researched:** 2026-03-18
**Confidence:** HIGH

## Executive Summary

This project implements a two-stage KV cache compression pipeline for the omlx Apple Silicon inference server, combining Attention Matching (AM) token compaction with KV Cache Transform Coding (kvtc) storage compression. Research is grounded in two primary papers (AM: arXiv:2602.16284, Feb 2026; kvtc: ICLR 2026 arXiv:2511.01815v2) and a validated spike on Qwen2.5-7B-Instruct-4bit on an M3 Max that confirmed 4x token compaction (cos=0.9987), 6.8x storage compression, and 16x combined compression. The pipeline is not an experiment — core math, platform constraints, and critical quality thresholds are fully characterized. The entire stack runs on MLX 0.31.1 with two new production dependencies (scipy and zstandard), integrating cleanly into omlx's existing tiered cache infrastructure.

The recommended build sequence is: linalg safety layer first (float32 casting + CPU stream wrappers), then AM and kvtc math independently in parallel, then the PCA calibration CLI, then omlx cache integration as a decorator with full async boundary, and finally a benchmark suite using task-accuracy metrics (not cosine similarity) as the quality gate. This ordering is driven by hard dependency constraints: kvtc requires pre-computed PCA calibration; AM must precede kvtc in the pipeline to get multiplicative compression gains; and async integration must be a design-time decision, not a retrofit. The modification surface on existing omlx code is narrow: only `tiered_manager.py`, `prefix_cache.py`, `paged_ssd_cache.py`, and `memory_monitor.py` need changes.

The dominant risks are platform-specific (float16 linalg ops fail silently in MLX, SVD/pinv require explicit CPU stream) and architectural (compression must not run synchronously on the decode thread, compression must be a pure decorator around existing cache managers so existing behavior is preserved when disabled). Secondary risks are correctness constraints with catastrophic failure modes: omitting attention sink exemption collapses LITM from 90.2 to 0.0 at 64x CR; omitting DP quantization collapses LITM from 99.4 to 13.1. Both are non-negotiable and must be built in from Phase 1, not added as quality fixes.

---

## Key Findings

### Recommended Stack

All core infrastructure is already in omlx or available via minimal additions. The only net-new production dependencies are `zstandard>=0.23.0` (best lossless entropy codec per kvtc paper Table B.8, beating DEFLATE) and `scipy>=1.15.0` (for `scipy.optimize.nnls` in AM beta-fitting — ~1ms/head, no viable MLX-native alternative). MLX 0.31.1 handles all tensor operations. Numpy 1.26.4 serves as the bridge layer for scipy calls and tobytes packing. Critical platform constraint: `mx.linalg.svd` and `mx.linalg.pinv` are CPU-only in MLX 0.31.x (Metal GPU SVD PR #2290 closed stale, Aug 2025) — both require explicit `stream=mx.cpu` and float32 inputs or they produce silent failures.

**Core technologies:**
- `mlx 0.31.1`: All tensor operations — the only viable tensor framework for Apple Silicon GPU (unified memory, zero-copy)
- `mlx-lm 0.30.7`: KV cache extraction via `make_prompt_cache` and `KVCache`/`QuantizedKVCache` — already a direct omlx dependency; do not reimplement
- `scipy 1.15.2`: `scipy.optimize.nnls` for AM beta-fitting — no MLX-native NNLS exists; spike measured ~1ms/head, fast enough; add explicitly to pyproject.toml
- `numpy 1.26.4`: Bridge layer for scipy interop and byte packing — pin to 1.x, numpy 2.x interop with MLX is unvalidated
- `zstandard 0.23.0`: Entropy coding — top codec in kvtc paper, cross-platform, replaces NVIDIA-only nvCOMP

### Expected Features

**Must have (table stakes — v1 milestone):**
- AM closed-form token compaction (NNLS beta-fitting + OLS value-fitting + HighestAttnKeys key selection)
- Non-uniform head budgets precomputed per model — single largest quality gain in AM ablations; build with AM, not after
- kvtc PCA-based storage compression (cross-layer SVD calibration + DP bit-allocation quantization + zstd)
- Attention sink exemption (first 4 tokens) and sliding window exemption (last 128 tokens) — correctness requirements, not optional; omit them and the model collapses at high CRs
- PCA calibration CLI (`omlx calibrate-kv <model>`) — kvtc cannot run without it; one-time per model, ~4s on Qwen 2.5 7B
- AM→kvtc combined pipeline with end-to-end compress/decompress round-trip
- Memory-pressure trigger for AM (hook into existing omlx MemoryMonitor)
- Eviction-path trigger for kvtc (hook into existing TieredCacheManager)
- Decompression on cache miss (<10ms/layer target; spike achieved 1.5ms/layer)
- Cache interface integration — compression is a decorator; existing behavior unchanged when disabled
- Enable/disable config flags for safe rollout and debugging
- Benchmark suite with compression ratio, cosine similarity, and downstream task accuracy (perplexity is insufficient — generative eval required per kvtc paper appendix A)

**Should have (v1.x — after core is stable):**
- Asymmetric key/value compression ratios (keys need higher precision; add if benchmarks show keys are the quality bottleneck)
- Stats and metrics in admin UI (compression ratio, cache hit rate, decompression latency)
- Repeat-prefill reference query strategy for AM (upgrade from context-prefill if benchmarks confirm quality benefit)
- Validation on Gemma 3 (SWA layers require per-layer attention type detection) and DeepSeek R1

**Defer (v2+):**
- Self-study query generation for AM (139s/60K tokens; viable only for offline batch, not interactive)
- OMP key selection (104–565s per compaction; offline batch only)
- Pre-bundled PCA matrices for popular models (distribution/versioning complexity; build calibration CLI first)
- Chunked compaction for very long contexts

**Anti-features to avoid explicitly:**
- Per-prompt PCA: reduces real CR to 1.3–12.4x vs 60-88x for one-time PCA (kvtc paper B.11)
- Online/streaming compaction during generation: preliminary result in AM paper only; RoPE phase handling complex and error-prone
- L2 regularization on AM beta or Cv: AM paper appendix confirms this degrades at all lambda > 0
- MLA support (DeepSeek-V2): both papers' derivations assume standard MHA/GQA; defer until MLA-specific papers publish

### Architecture Approach

The compression pipeline lives entirely in a new `omlx/compression/` module (five files: `config.py`, `am_compactor.py`, `kvtc_compressor.py`, `pca_calibrator.py`, `head_budgets.py`). Compressors are pure functions over MLX tensors — no cache state, no block management — enabling independent testing and clean disable paths. Integration with existing omlx cache machinery touches only four files via a decorator pattern: `tiered_manager.py` (kvtc on eviction/restore), `prefix_cache.py` (AM memory-pressure callback), `paged_ssd_cache.py` (format flag for compressed blocks), and `memory_monitor.py` (pressure callback interface). The base `CacheManager` ABC does not change.

**Major components:**
1. `AMCompactor` (`omlx/compression/am_compactor.py`) — stateless token compaction; NNLS beta-fitting (scipy, CPU) + OLS value-fitting (mx.linalg.pinv, CPU stream) + HighestAttnKeys selection; triggered by memory pressure callback
2. `KVTCCompressor` (`omlx/compression/kvtc_compressor.py`) — byte-level storage compression; PCA projection + DP quantization + zstd; intercepts TieredCacheManager eviction/restore path; requires preloaded PCA bundle
3. `PCACalibrator` (`omlx/compression/pca_calibrator.py`) — one-time offline calibration; SVD over cross-layer representative activations (float32, CPU stream, RoPE-stripped keys); writes `kv_pca_calibration.npz` alongside model weights
4. `CompressionConfig` (`omlx/compression/config.py`) — enable/disable flags, compaction ratio, PCA component count, quantization bits, calibration path; passed via factory to all compressors
5. `head_budgets.py` — non-uniform head budget precomputation; head sensitivity measured once per model; budget schedule assigns more to later layers (counter-intuitive but validated in spike: per-head entropy 0.34–2.47)

### Critical Pitfalls

1. **Float16 linalg input — silent NaN production** — MLX SVD and pinv silently produce garbage on float16 input; confirmed in spike. Mitigation: build linalg helper functions (`svd_f32`, `pinv_f32`) that enforce float32 casting before every call; make this the first thing built in Phase 1.

2. **Missing `stream=mx.cpu` on linalg calls** — Metal GPU does not support SVD/pinv in MLX 0.31.x; omitting the stream causes hangs or wrong results. Mitigation: centralized helpers at a single call site; CI gate that no bare `mx.linalg.*` calls exist outside helpers.

3. **Omitting attention sink (first 4) and sliding window (last 128) exemptions** — kvtc paper: LITM drops from 90.2 to 0.0 at 64x CR without sink exemption; GSM8K drops from 57.2 to 1.6. These are correctness requirements, not quality enhancements. Mitigation: non-optional parameters with hardcoded safe defaults enforced at the API boundary.

4. **Synchronous compression on the decode thread** — spike measured 0.26s/layer combined; 28 layers = 7+ second decode stall. Mitigation: async job queue in TieredCacheManager; compression enqueued and returned immediately; profiled in CI.

5. **RoPE embeddings not stripped before PCA calibration** — position-dependent variation dominates principal components; reconstruction degrades with token position. Mitigation: apply inverse RoPE rotation to keys before building calibration matrix; verify by plotting reconstruction cosine by token position (must be flat).

6. **DP quantization skipped in kvtc** — ablation B.9: LITM drops from 99.4 to 13.1 without DP. Not optional. Mitigation: implement as part of core compression path in Phase 1; bit-allocation table must be computed at calibration and loaded at startup.

7. **Cosine similarity as sole quality gate** — combined pipeline end-to-end cosine of 0.72 is consistent with within-1-point task accuracy; using cosine as acceptance criterion would block a working pipeline. Mitigation: benchmark suite with perplexity + task accuracy must exist before integration sign-off.

---

## Implications for Roadmap

Based on research, the natural phase structure follows component dependency order: math safety layer first (blocks everything), then independent AM and kvtc math in parallel, then calibration CLI (unblocks end-to-end kvtc), then omlx integration (decorator pattern, async boundary), then benchmark validation as the final gate.

### Phase 1: Math Foundation — Linalg Safety Layer + AM Core + kvtc Core

**Rationale:** All subsequent work depends on correct float32-enforced linalg helpers. AM and kvtc math are independent of each other and of omlx cache machinery — they can be built and unit-tested as pure functions before any integration. The most critical pitfalls (float16 linalg, missing CPU stream, missing DP quantization, sink/window exemptions) must all be addressed here. Starting with math-only code without integration complexity allows rapid iteration and testing.

**Delivers:** `AMCompactor` with NNLS/OLS/HighestAttnKeys + non-uniform head budgets; `KVTCCompressor` compress/decompress with cross-layer PCA projection + DP quantization + zstd; linalg helper layer; `CompressionConfig`; unit tests for each compressor with float16 input rejection, sink/window exemption ablations, GQA head count correctness on Qwen 2.5 7B.

**Addresses:** AM token compaction, non-uniform head budgets, kvtc storage compression, sink + window exemptions, GQA-aware compression, DP quantization.

**Avoids:** Float16 linalg (helper layer), missing CPU stream (helper layer), DP quantization omission, sink/window exemption omission, GQA head count error.

**Stack:** mlx 0.31.1 (CPU stream ops), scipy 1.15.2 (NNLS), numpy 1.26.4 (bridge), zstandard 0.23.0 (entropy coding).

**Research flag:** Standard patterns — spike already validated the math; paper ablations give exact numbers to reproduce. No additional research needed.

### Phase 2: PCA Calibration CLI

**Rationale:** kvtc cannot run end-to-end without a precomputed PCA basis. The calibration CLI is the unblocking dependency for Phase 3 integration. It is also the phase where the RoPE-stripping pitfall must be addressed. Calibration is intentionally a one-time offline step; implementing it correctly here prevents the per-prompt PCA anti-pattern from ever appearing in the integration.

**Delivers:** `PCACalibrator` with SVD-based cross-layer PCA (float32, CPU stream, RoPE-stripped keys); `omlx calibrate-kv <model>` CLI command; calibration bundle format (`kv_pca_calibration.npz` with V, mu, bit_alloc, group_sizes keys); verification that reconstruction cosine is flat across token positions (RoPE correctness check); calibration runtime target ~4s for Qwen 2.5 7B.

**Addresses:** PCA calibration CLI, one-time PCA approach, calibration bundle as model artifact.

**Avoids:** RoPE embeddings in calibration data (must strip before SVD), per-prompt PCA anti-pattern, synchronous calibration on first request.

**Stack:** `mx.linalg.svd` (float32 cast, CPU stream), numpy (calibration data accumulation, `np.savez_compressed`).

**Research flag:** Standard patterns — kvtc paper §3.1 gives exact calibration procedure; spike validated timing. No additional research needed.

### Phase 3: omlx Cache Integration (Decorator + Async)

**Rationale:** Integration is deliberately last in the math/calibration sequence so the compressor implementations are stable before touching existing cache code. The decorator pattern (no changes to `CacheManager` ABC) and async boundary (compression never on decode thread) are design-time decisions that must be locked in before any integration code is written. All existing omlx cache tests must pass unchanged with compression disabled — this is a hard acceptance criterion.

**Delivers:** `TieredCacheManager` eviction/restore hooks calling `KVTCCompressor`; `BlockAwarePrefixCache` AM compaction callback on memory pressure; `MemoryMonitor` pressure callback registration interface; `PagedSSDCacheManager` compressed block format flag; async compression job queue; enable/disable config flags with no-op disable path; end-to-end compress/decompress round-trip integration test; TTFT profile confirming compression does not appear in decode trace.

**Addresses:** Cache interface integration, memory-pressure trigger, eviction-path trigger, decompression on cache miss, enable/disable flags.

**Avoids:** Compression inside PagedSSDCacheManager (wrong responsibility boundary), AM on MLX Metal stream (must use CPU executor), synchronous compression on decode thread, breaking existing cache behavior when disabled, prefix-sharing refcount violation.

**Stack:** Existing omlx infrastructure (tiered_manager, prefix_cache, paged_ssd_cache, memory_monitor) with narrow modifications.

**Research flag:** Needs research into omlx's async write queue capacity (`PagedSSDCacheManager._MAX_PENDING_WRITES`) and prefix-sharing refcount protocol before writing integration code — these are internal implementation details not covered in research files.

### Phase 4: Benchmark Suite + Quality Validation

**Rationale:** The benchmark suite is the acceptance gate, not a nice-to-have. Cosine similarity alone is an invalid quality metric (combined pipeline end-to-end cosine of 0.72 is consistent with within-1-point task accuracy). Benchmarks must exist before any production sign-off. This is also the phase where multi-model validation happens — Qwen 2.5 7B is the primary spike model; Llama 3.x 8B, Gemma 3, DeepSeek R1 add coverage breadth.

**Delivers:** Benchmark harness covering: compression ratio (target 16x combined), cosine similarity per compressor stage (AM: >0.998, kvtc: >0.95), decompression latency (<10ms/layer), perplexity on fixed test set, LITM/recall accuracy, TTFT regression check; validated results on Qwen 2.5 7B matching paper numbers; SWA layer detection and skip for Gemma 3; multi-model validation report.

**Addresses:** Benchmark suite, stats visibility, Gemma 3 + DeepSeek R1 validation, asymmetric key/value compression (if benchmarks reveal need).

**Avoids:** Cosine as sole quality gate, SWA layer compaction on Gemma, treating one benchmark pass as sign-off without task-accuracy metrics.

**Research flag:** Gemma 3 SWA layer detection requires reading Gemma model config structure in mlx-lm — may need targeted research. Standard benchmark patterns otherwise.

### Phase Ordering Rationale

- Math-first ordering avoids integration complexity during the most algorithmically sensitive work. Float16 failures and numerical correctness bugs are much easier to debug in isolated unit tests than in integrated cache paths.
- Calibration CLI before integration prevents the per-prompt PCA anti-pattern from ever being considered as a fallback. KVTCCompressor raises a hard error if calibration file is missing — the CLI must exist first.
- Integration as a decorator (not modifying the ABC) means Phase 3 is safe to attempt once Phases 1-2 are complete. The narrow modification surface (4 files) limits regression risk.
- Benchmark suite last is intentional: it validates the complete system, not individual components. Using task accuracy as the acceptance criterion requires the full pipeline to be integrated before benchmarks are meaningful.
- AM and kvtc math can be developed in parallel within Phase 1 (no dependency between them); `PCACalibrator` can also begin in parallel with `KVTCCompressor` since their interface contract (bundle format) can be defined upfront.

### Research Flags

Phases needing deeper research during planning:
- **Phase 3 (integration):** omlx async write queue internals and prefix-sharing refcount protocol — internal implementation details not fully characterized in research files; needs targeted source inspection before writing integration design
- **Phase 4 (benchmarks, Gemma):** Gemma 3 SWA layer detection from mlx-lm model config structure — needed before implementing the layer-type classifier

Phases with standard patterns (skip research-phase):
- **Phase 1 (math layer):** Spike already validated AM and kvtc math; paper ablations give reproduction targets; linalg constraints fully characterized
- **Phase 2 (calibration CLI):** kvtc paper §3.1 gives exact calibration procedure; spike measured timing; bundle format is fully specified

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All versions verified against installed conda environment; spike validated MLX 0.31.0 (0.31.1 is non-breaking patch); PyPI versions confirmed; platform constraints (SVD CPU-only, float16 failures) confirmed empirically in spike |
| Features | HIGH | Derived from two primary papers with ablation tables; spike confirms core numbers; feature dependencies explicit and paper-validated; anti-features backed by ablation results (not opinion) |
| Architecture | HIGH | Based on direct inspection of omlx source and spike prototypes; component boundaries match existing cache infrastructure; build order derived from hard dependency graph |
| Pitfalls | HIGH | Pitfalls 1-3 confirmed empirically in spike (float16, softmax overflow, CPU stream); Pitfalls 4-6 backed by specific paper ablation numbers (LITM collapse quantified); Pitfall 11 (RoPE) documented in kvtc paper §3.1 |

**Overall confidence:** HIGH

### Gaps to Address

- **numpy 2.x compatibility:** omlx pyproject.toml pins `numpy>=1.24.0` which admits numpy 2.x. numpy 2.x has breaking changes in memory layout that may affect `.tobytes()` and MLX interop patterns. Pin to `numpy>=1.24.0,<2.0` in compression module until validated. Low urgency (numpy 1.26.4 is installed), but the open upper bound is a latent risk.

- **PCA bundle versioning:** Research recommends storing the calibration bundle alongside model weights with no versioning scheme in v1. This is acceptable for v1 but must be addressed before any distribution of pre-bundled matrices or multi-version model deployments. Flag in Phase 2 design doc.

- **SWA layer detection API in mlx-lm:** Gemma 3 requires per-layer attention type detection. The mlx-lm model config structure for this is not fully characterized in research. Needs targeted source inspection before Phase 4 Gemma validation work begins.

- **Calibration corpus coverage:** Research notes the kvtc paper used code-only calibration data that retained long-context retrieval ability at 16x — but the paper also specifies ~1.5 minutes on H100 for 160K calibration tokens. The spike ran fewer tokens. Calibration corpus size and diversity requirements for production-quality PCA matrices need validation during Phase 2.

- **Variable-length page slots in PagedSSDCacheManager:** Compressed bytes are variable-length; the existing SSD manager may use fixed-size slots. Integration research flagged this but did not resolve it. Must be confirmed during Phase 3 design.

---

## Sources

### Primary (HIGH confidence)
- Spike results: `/Users/tonysina/projects/omlx/docs/research/kv-cache-compression/SPIKE-RESULTS.md` — empirical validation on M3 Max, MLX 0.31.0, Qwen2.5-7B-Instruct-4bit
- AM paper: Zweiger, Fu, Guo, Kim (MIT), "Fast KV Compaction via Attention Matching", arXiv:2602.16284, Feb 2026 — ablations for solver comparison, head budgets, query strategies, regularization
- kvtc paper: Staniszewski & Łańcucki (NVIDIA/University of Warsaw), "KV Cache Transform Coding", ICLR 2026, arXiv:2511.01815v2 — ablations B.3 (sinks), B.7 (window), B.8 (codec comparison), B.9 (DP quantization), B.10 (cross-layer PCA), B.11 (per-prompt PCA)
- MLX linalg docs: https://ml-explore.github.io/mlx/build/html/python/linalg.html — full API listing
- MLX SVD GPU PR (closed stale): https://github.com/ml-explore/mlx/pull/2290 — CPU-only constraint confirmed
- MLX PyPI: https://pypi.org/project/mlx/ — version 0.31.1, March 12 2026
- omlx source: `omlx/cache/interface.py`, `tiered_manager.py`, `paged_ssd_cache.py`, `prefix_cache.py`, `paged_cache.py`, `factory.py`, `memory_monitor.py`, `engine_core.py`

### Secondary (MEDIUM confidence)
- python-zstandard PyPI: https://pypi.org/project/zstandard/ — stable 0.25.0; installed 0.23.0 confirmed working
- Python 3.14 compression.zstd: https://docs.python.org/3/library/compression.zstd.html — future migration path; not applicable to omlx's 3.10–3.13 support matrix

---
*Research completed: 2026-03-18*
*Ready for roadmap: yes*
