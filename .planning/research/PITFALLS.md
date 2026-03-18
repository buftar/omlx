# Pitfalls Research

**Domain:** KV cache compression pipeline on Apple Silicon (MLX) — AM + kvtc two-stage
**Researched:** 2026-03-18
**Confidence:** HIGH (spike results + paper ablations + MLX blockers already encountered)

---

## Critical Pitfalls

### Pitfall 1: Running linalg ops in float16

**What goes wrong:**
SVD, pinv, and QR decompositions produce NaN or wildly inaccurate results when the input KV cache is float16. The MLX SVD kernel silently produces garbage rather than erroring cleanly.

**Why it happens:**
KV caches are stored as float16 to save memory. It is natural to pass them directly to linalg ops. The spike hit this immediately — SVD on float16 tensors fails without a cast.

**How to avoid:**
Establish a single rule at the lowest level: all linalg ops cast to float32 before the call and cast results back to float16 afterward. Encapsulate this in helper functions (`svd_f32`, `pinv_f32`) so no call site can forget. Keep storage dtype (float16) and compute dtype (float32) as separate concepts in the type signatures.

**Warning signs:**
- Cosine similarities far from expected (e.g., 0.4 where 0.98 is expected)
- NaN appearing in PCA components or beta values
- NNLS converging but producing zero-weight solutions

**Phase to address:**
Phase 1 (kvtc calibration + AM core math) — must be the first line of the implementation before any other linalg work.

---

### Pitfall 2: Using float16 softmax in attention reconstruction

**What goes wrong:**
The AM value-fitting step and the kvtc reconstruction quality check compute attention scores via `q @ k.T / sqrt(d)`. With float16, scores for long sequences exceed the float16 max (~65504), softmax denominator overflows to inf, outputs become NaN.

**Why it happens:**
The spike specifically found this: float16 softmax overflows on realistic sequence lengths. Attention is written with float16 inputs because that matches the cache dtype, but the intermediate exponentials overflow.

**How to avoid:**
Cast queries and keys to float32 before computing attention scores in any compression-adjacent code path. The production model does this internally via scaled-dot-product attention kernels, but compression code written from scratch does not get that handling for free.

**Warning signs:**
- NaN in attention output tensors during AM fitting
- OLS solving for Cv produces zero or infinite matrix
- Beta values all becoming identical

**Phase to address:**
Phase 1 (AM core — beta fitting and value fitting).

---

### Pitfall 3: Forgetting the CPU stream requirement for MLX linalg

**What goes wrong:**
`mx.linalg.svd` and `mx.linalg.pinv` silently dispatch to GPU by default. On Apple Silicon the Metal GPU does not support these kernels; the ops either hang, produce wrong results, or raise a cryptic runtime error.

**Why it happens:**
MLX's default stream is the GPU. Most tensor ops work on GPU. Developers write `mx.linalg.svd(X)` expecting it to work like `mx.linalg.norm(X)` — it does not.

**How to avoid:**
Always pass `stream=mx.cpu` to every `mx.linalg.*` call. Wrap all linalg in centralized helpers that enforce the stream. Add a CI assertion that these helpers are the only linalg call sites.

**Warning signs:**
- Calls that never return (hang) or return immediately with wrong shapes
- Intermittent crashes on larger matrices that work on small ones

**Phase to address:**
Phase 1 — linalg helper layer must be the first thing built.

---

### Pitfall 4: Omitting attention sinks and sliding window from kvtc compression

**What goes wrong:**
Compressing the first 4 tokens (attention sinks) at high compression ratios causes total model collapse on retrieval tasks. The kvtc paper shows LITM drops from 90.2 to 0.0 and GSM8K drops from 57.2 to 1.6 on Llama 3.1 8B when sinks are not exempted at 64× CR. The sliding window exemption is nearly as critical.

**Why it happens:**
Attention sinks hold disproportionate attention mass. Their PCA reconstruction error is also highest — they are outliers relative to the cross-layer structure. An implementer writes a clean loop over all tokens without special-casing boundaries.

**How to avoid:**
Implement the exemption as a non-optional parameter with a hardcoded safe default: `sink_tokens=4`, `window_tokens=128`. Never compress these tokens. Enforce this at the API boundary, not as a caller-chosen option. The spike prototype must reproduce this before integration begins.

**Warning signs:**
- Quality degrades sharply at compression ratios above 16×
- Retrieval-style benchmarks fail much harder than generation benchmarks
- Coherence breaks on long conversations but not short ones

**Phase to address:**
Phase 1 (kvtc compression) — must be part of the initial implementation, not added later as a quality fix.

---

### Pitfall 5: Per-prompt PCA instead of one-time PCA

**What goes wrong:**
Computing a PCA basis per-prompt (instead of calibrating once per model) reduces the actual achieved compression ratio to 1.3–12.4× at nominal 64× settings, because storing the per-prompt basis consumes most of the gained space. The method also generalizes worse to conversation continuation.

**Why it happens:**
Per-prompt PCA seems appealing as a quality improvement — the basis is perfectly tuned to the content. This is a classic over-engineering trap. The kvtc paper's ablation (B.11) shows it is strictly worse on all dimensions.

**How to avoid:**
The PCA calibration CLI (`omlx calibrate-kv <model>`) must be the only code path that computes the basis matrix. The compression path takes a precomputed basis as a mandatory input — there is no fallback that computes it on the fly.

**Warning signs:**
- Compression ratio unexpectedly low at runtime
- Calibration being triggered per-request rather than per-model

**Phase to address:**
Phase 2 (calibration CLI and storage) — the CLI must be built before the compression path is wired into the cache system.

---

### Pitfall 6: Skipping DP quantization and relying on PCA truncation alone

**What goes wrong:**
Dropping the DP bit-allocation step and instead just truncating low-variance PCA components collapses accuracy dramatically. The ablation shows: at kvtc 64× with window=128, LITM drops from 99.4 to 13.1 without DP quantization.

**Why it happens:**
The DP step feels optional — PCA truncation already reduces dimensionality, and quantization seems like a separate optimization pass. In fact DP quantization is load-bearing: it also makes the byte stream far more compressible for the lossless codec (zstd needs discrete values, not truncated floats).

**How to avoid:**
Implement DP quantization as part of the core compression path, not as an enhancement. The bit-allocation table must be computed during calibration and stored with the PCA basis. There is no "lite mode" that omits it.

**Warning signs:**
- Lower-than-expected compression ratio from zstd
- Quality degrading faster than the paper's numbers at the same nominal CR

**Phase to address:**
Phase 1 (kvtc core) — DP quantization must be implemented in the first pass, not deferred.

---

### Pitfall 7: Treating cosine similarity between AM output and original as the quality gate

**What goes wrong:**
The combined AM + kvtc pipeline showed end-to-end cosine similarity of 0.72 in the spike. If this metric is used as the primary quality gate, the pipeline will appear to fail when the quality is actually acceptable. The papers use downstream task accuracy (perplexity, MMLU, LITM, AIME) as the real quality measure.

**Why it happens:**
Cosine similarity is easy to compute and familiar from embedding similarity tasks. But it compounds errors non-linearly when chained through attention layers. A 0.72 cosine on the full pipeline is consistent with the paper's claim of within-1-point of vanilla on benchmarks.

**How to avoid:**
Use cosine similarity only as a fast sanity check during development. Define acceptance criteria in terms of task accuracy on held-out prompts (perplexity on a fixed test set, factual recall, math reasoning). The benchmark suite must be built before any quality sign-off.

**Warning signs:**
- Build criteria defined solely as cosine > threshold
- No perplexity or task-accuracy measurement in the test suite

**Phase to address:**
Phase 3 (benchmark suite) — must be built before integration and used as the acceptance gate for production readiness.

---

### Pitfall 8: Incorrect GQA head handling in AM

**What goes wrong:**
GQA models (Qwen 2.5, Llama 3.x) have many fewer KV heads than query heads. AM operates on KV heads. If the implementation iterates over query heads (28 for Qwen 2.5 7B) instead of KV heads (4), it runs 7× the necessary work — or worse, shapes mismatch and the wrong query-to-KV mapping is used.

**Why it happens:**
Attention code usually loops over all heads. The asymmetry between Q and KV head counts in GQA is a relatively recent addition. A naive loop `for h in range(num_heads)` uses the query head count.

**How to avoid:**
Explicitly extract `num_kv_heads` from the model config at AM construction time. The inner AM loop must iterate over `num_kv_heads`. For the reference-query gathering step, apply the GQA fan-out: each KV head corresponds to `num_q_heads // num_kv_heads` query heads, and all of them contribute reference queries for that KV head's fitting.

**Warning signs:**
- AM runtime scales with query heads rather than KV heads
- Shape errors in NNLS matrix construction
- Nonuniform head budgets computed on wrong head count

**Phase to address:**
Phase 1 (AM core) — must be validated on GQA models (Qwen 2.5 as the primary test model) before completion.

---

### Pitfall 9: Applying AM to sliding-window attention layers

**What goes wrong:**
Models with Sliding Window Attention (Gemma 3, Mistral) have layers that only attend over a local window. Compacting those layers' KV caches provides no benefit (the tokens outside the window are already effectively excluded) and may harm quality by disrupting the local window structure.

**Why it happens:**
The AM compaction loop iterates over all layers uniformly. Sliding-window layers are architecturally different but look the same at the Python API level unless the model config is checked.

**How to avoid:**
Read the model's attention config to classify layers as global or local (sliding-window). Only apply AM to global-attention layers. Derive this from the model config, not a hardcoded list. The paper validates this approach on Gemma3-12B.

**Warning signs:**
- Quality degrades specifically on Gemma models but not Llama/Qwen
- Compaction attempts on short effective context windows

**Phase to address:**
Phase 4 (multi-model validation) — must be implemented before Gemma validation, but can be deferred past the initial Qwen/Llama phases.

---

### Pitfall 10: Compressing during active decode rather than between inference phases

**What goes wrong:**
Running AM compaction or kvtc compression synchronously during the decode loop adds unbounded latency per token. The spike measured 0.26s per layer for the combined pipeline — on a 28-layer model that is 7+ seconds of blocking work inserted into the decode path.

**Why it happens:**
The integration point is ambiguous. The cache manager's `evict` and `store` paths are called during inference. Wiring compression into those callbacks without an async boundary adds latency to every request.

**How to avoid:**
Compression must run between inference phases, never during active decoding. The tiered manager's eviction path should enqueue a compression job and return immediately with the uncompressed cache. Compression completes asynchronously and replaces the cache entry. AM compaction specifically should be triggered on memory pressure events, not per-token events.

**Warning signs:**
- TTFT or inter-token latency increasing after integration
- Compression time appearing in decode-loop profiling traces
- Cache operations blocking the inference thread

**Phase to address:**
Phase 3 (omlx cache integration) — the async boundary must be part of the integration design, not retrofitted after latency regressions appear.

---

### Pitfall 11: RoPE embeddings left in calibration data for kvtc PCA

**What goes wrong:**
If RoPE positional embeddings are not removed before computing the cross-layer PCA, the position-dependent variation dominates the principal components. The PCA basis captures positional patterns instead of semantic structure. Reconstruction quality degrades sharply, especially for tokens far from position 0.

**Why it happens:**
The KV cache tensors include RoPE-rotated keys. It is easy to feed raw cache tensors into the PCA calibration without stripping RoPE first. The kvtc paper explicitly removes RoPE before calibration (§3.1) but this is easy to miss as an implementation detail.

**How to avoid:**
In the calibration path, apply the inverse RoPE rotation to keys before building the calibration matrix. Store un-rotated keys in the calibration buffer. At compression time, strip RoPE before projecting into PCA space and re-apply RoPE after reconstruction.

**Warning signs:**
- PCA components that look like sinusoidal patterns rather than semantic features
- Reconstruction quality deteriorating for longer sequences
- Cosine similarity of reconstructed keys dropping with token position

**Phase to address:**
Phase 1 (kvtc calibration) — must be part of the initial calibration implementation.

---

### Pitfall 12: Breaking existing cache behavior when compression is disabled

**What goes wrong:**
The compression integration modifies core cache paths in `omlx/cache/` in ways that affect behavior even when compression is disabled. This breaks the existing system for users who do not use the new feature, and makes it impossible to A/B test compression against the baseline.

**Why it happens:**
Compression is added as a feature flag, but implementation takes shortcuts: modifying `CacheManager.evict()` in-place, adding compression-specific fields to base cache entries, or changing serialization format for all caches.

**How to avoid:**
The compression layer must be a pure decorator around existing cache managers. The base `CacheManager` ABC must not change. Compression is activated by wrapping: `CompressingCacheManager(base_manager, compressor)`. All existing tests must pass unchanged when compression is disabled.

**Warning signs:**
- Existing cache tests requiring modification after the integration PR
- The base `CacheManager` interface gaining compression-specific methods
- Serialization format changes that affect uncompressed caches

**Phase to address:**
Phase 3 (omlx cache integration) — integration design review must confirm the decorator pattern before any code is written.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Hardcode PCA dimensions (64/128) instead of DP-optimized bit allocation | Simpler first implementation | Sub-optimal bit allocation; harder to tune per model | Never — DP is load-bearing per ablation B.9 |
| Uniform head budgets instead of precomputed nonuniform | Skip head sensitivity calibration step | Significantly worse quality; largest single ablation gain in AM paper | Prototype only — must be implemented before production |
| Skip sink/window exemptions in kvtc | Cleaner loop code | Model collapse at high CRs per ablation B.3 | Never |
| scipy NNLS via numpy interop instead of pure MLX | Unblocked immediately | Extra data transfer GPU to CPU per call | Acceptable in v1; revisit if NNLS is a bottleneck |
| One combined compress/decompress test instead of per-layer tests | Faster test writing | Hard to isolate which layer failed quality gate | Never for production paths |
| Store PCA matrix alongside model weights on disk with no versioning | Simple discovery | PCA matrix may silently mismatch model checkpoint | Acceptable for v1 CLI; needs versioning before distribution |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| omlx tiered_manager | Wire compression into the synchronous evict() path | Enqueue async compression job; return immediately with uncompressed entry |
| MLX linalg (svd, pinv) | Call without `stream=mx.cpu` | Always pass `stream=mx.cpu`; wrap in helpers at a single call site |
| numpy/scipy interop | Call `.numpy()` on a tensor that has not been materialized | Call `mx.eval(tensor)` first to synchronize, then `.numpy()` |
| AM + omlx prefix cache | Compact a cache shared across multiple prefix-sharing requests | Check reference count before compacting; only compact caches with refcount=1 |
| kvtc + paged_ssd_cache | Write compressed bytes into fixed-size page slots designed for uncompressed data | Reserve max-size slots or implement variable-length page entries for compressed data |
| Memory pressure trigger | Trigger AM on every allocation after threshold | Debounce: trigger at most once per N seconds or per M new tokens to avoid compaction storms |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Reference queries generated via self-study at runtime | AM takes minutes per context instead of seconds | Default to repeat-prefill; self-study is optional and offline-only | As soon as context > 8K tokens |
| Synchronous PCA calibration on first request | First request hangs for 4–10 seconds | Calibration must be a one-time CLI step, never lazy-initialized | On the first request after model load if calibration was skipped |
| Recomputing PCA basis at each kvtc call | Calibration cost paid per compression | Load PCA matrix once at startup, cache in memory | Immediately — every call would take 4+ seconds |
| AM over all layers including SWA layers | 7× wasted work on Gemma models | Filter to global-attention layers only | First Gemma model tested |
| zstd over float tensors before quantization | Poor compression ratio from zstd (1.0–1.1×) | Quantize to integers before zstd; discrete values compress far better | Every run — quantization and entropy coding are interdependent |
| Large calibration matrices held entirely in GPU memory | OOM during calibration on models with large hidden dim | Stream calibration data in batches; compute incremental SVD | Models with >4 KV heads and large head_dim |

---

## "Looks Done But Isn't" Checklist

- [ ] **kvtc calibration:** PCA matrix built with RoPE stripped — verify by checking that reconstruction cosine does not drop with token position
- [ ] **kvtc compression:** Sink tokens (first 4) and sliding window (last 128) are always excluded — verify with an ablation that shows collapse when removed
- [ ] **AM head budgets:** Nonuniform budget precomputed per model, not uniform — verify that head entropy varies across heads (spike shows 0.34–2.47 range)
- [ ] **AM GQA handling:** Inner loop iterates over `num_kv_heads` not `num_heads` — verify on Qwen 2.5 7B (4 KV heads, 28 Q heads)
- [ ] **linalg helpers:** Every `mx.linalg.*` call goes through float32-casting helpers with `stream=mx.cpu` — verify by grepping call sites
- [ ] **Compression disabled path:** All existing omlx cache tests pass without modification when compression is off — verify in CI
- [ ] **Async integration:** Compression never runs on the decode thread — verify by profiling TTFT with and without cache integration
- [ ] **DP quantization:** Bit-allocation table stored with PCA matrix and loaded at startup — verify that achieved CR matches calibration target
- [ ] **Quality gate:** Acceptance criteria are task-accuracy metrics (perplexity or benchmark), not cosine similarity alone — verify benchmark suite exists before sign-off

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| float16 linalg producing NaN | LOW | Add float32 cast in linalg helpers; re-run calibration to regenerate PCA matrix |
| RoPE not stripped in calibration | MEDIUM | Recalibrate with RoPE-stripped keys; existing compressed caches are invalid and must be discarded |
| Sink/window exemptions missing | LOW | Add exemption check in compression path; caches compressed at high CR need recompression |
| Synchronous compression on decode thread | MEDIUM | Introduce async job queue; requires integration refactor but does not touch math layer |
| Per-prompt PCA discovered in production | HIGH | Full rearchitecture of calibration path; compressed caches in wrong format require migration |
| Compression breaks existing cache tests | MEDIUM | Refactor to decorator pattern; requires interface review and test updates |
| DP quantization missing from pipeline | HIGH | Core compression path must be rewritten; all stored compressed caches are in wrong format |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| float16 linalg ops | Phase 1: AM + kvtc math layer | Unit tests for linalg helpers with float16 input; assert output dtype and no NaN |
| float16 softmax overflow | Phase 1: AM beta and value fitting | Test with sequences > 512 tokens; check for NaN in attention outputs |
| Missing CPU stream on linalg | Phase 1: linalg helper layer | Code review gate: no bare `mx.linalg.*` call sites outside helpers |
| No sink/window exemption | Phase 1: kvtc compression | Ablation: compress with window=1 and verify collapse; then restore default |
| Per-prompt PCA | Phase 2: calibration CLI | Verify CR matches target; reject if calibration triggered at inference time |
| Missing DP quantization | Phase 1: kvtc compression | Compare achieved CR with and without DP; must match paper numbers |
| Cosine as sole quality gate | Phase 3: benchmark suite | Benchmark suite with perplexity + task accuracy must exist before integration sign-off |
| GQA head count error | Phase 1: AM core | Test on Qwen 2.5 7B; assert `num_kv_heads=4` is used in inner loop |
| SWA layer compaction | Phase 4: Gemma validation | Gemma 3 model test; assert SWA layers are classified and skipped |
| Synchronous compression | Phase 3: omlx integration | Profile TTFT with cache hit; compression latency must not appear in decode trace |
| RoPE in calibration | Phase 2: calibration CLI | Plot reconstruction cosine by token position; must be flat, not declining with position |
| Breaking existing cache paths | Phase 3: omlx integration | Existing cache test suite must pass unchanged with compression disabled |

---

## Sources

- Spike results: `/Users/tonysina/projects/omlx/docs/research/kv-cache-compression/SPIKE-RESULTS.md` (HIGH confidence — direct observation on M3 Max, MLX 0.31.0)
- AM paper: Zweiger et al., "Fast KV Compaction via Attention Matching", arXiv:2602.16284 (Feb 2026) — ablations §4 and appendix: nonuniform budgets (largest gain), OMP vs HighestAttn, beta stability, solver comparison (pinv is slowest/worst, lstsq preferred)
- kvtc paper: Staniszewski & Łańcucki, "KV Cache Transform Coding", ICLR 2026 — ablations B.3 (sink tokens), B.7 (window size), B.9 (DP quantization load-bearing), B.10 (cross-layer PCA load-bearing), B.11 (per-prompt PCA strictly worse), B.8 (zstd beats all others at 32×)
- MLX platform constraints: SVD and pinv require float32 and CPU stream — confirmed in spike; float16 softmax overflow confirmed in spike
- omlx codebase: `omlx/cache/` — CacheManager ABC, tiered_manager, paged_ssd_cache, existing integration points

---
*Pitfalls research for: KV cache compression pipeline on Apple Silicon (MLX)*
*Researched: 2026-03-18*
