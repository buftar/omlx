# Feature Research

**Domain:** KV cache compression pipeline (token compaction + storage compression) for Apple Silicon LLM inference
**Researched:** 2026-03-18
**Confidence:** HIGH — derived from two primary papers (AM arXiv:2602.16284, kvtc ICLR 2026), a validated spike, and existing omlx architecture review

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features the compression pipeline must have or it is not useful as an omlx feature.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| AM token compaction (closed-form) | Core value: reduce hot cache size without gradient descent. Without this the pipeline has no token-count reduction. | HIGH | NNLS β-fitting + OLS value-fitting + key selection. MLX CPU stream required for pinv; scipy NNLS via numpy interop for beta fitting. Spike validated at 4× (cos=0.9987). |
| kvtc storage compression (PCA + DP quantization + entropy coding) | Core value: byte-level compression for cold SSD storage. Without this the pipeline has no storage compression. | HIGH | Cross-layer PCA (float32 cast required for SVD), DP bit allocation per component, zstd entropy coding (replaces nvCOMP). Spike validated at 6.8×. |
| End-to-end compress/decompress round-trip | Users need to store a cache and restore it losslessly enough to continue inference. If decompression doesn't restore a usable cache the feature is broken. | MEDIUM | AM→kvtc pipeline order is load-bearing: AM first reduces data that kvtc processes, giving multiplicative gains. Spike shows 16× combined. |
| Decompression on cache miss | When a cold cache is fetched from SSD the model must be able to resume inference. No decompression = cold cache is unusable. | MEDIUM | Must integrate with omlx tiered_manager's demotion/promotion path. Spike shows 1.5ms/layer decompression — well under 10ms target. |
| Correct cache interface integration | Must slot into the existing `CacheManager` ABC (fetch/store/evict/clear). Breaking existing cache behavior when compression is disabled violates the compatibility constraint. | MEDIUM | Integration point TBD; memory monitor and tiered_manager already exist. Disable path must be a no-op. |
| Memory-pressure trigger for AM | AM is a hot-cache operation — it needs to fire when GPU memory is under pressure, not on every token. Without a trigger it either never runs or runs wastefully. | LOW | omlx memory monitor and process memory enforcer already exist; hook into existing pressure signals. |
| Eviction-path trigger for kvtc | kvtc is a cold-storage operation — it must compress before writing to SSD. Without this the SSD eviction path writes uncompressed caches, wasting storage. | LOW | Hook into tiered_manager's eviction path. |
| Attention sink exemption | Compressing the first 4 tokens at high CR causes accuracy to collapse (kvtc paper: LITM 90.2 → 0.0 at 64× if sinks are compressed). Non-negotiable for correctness. | LOW | Exclude first s=4 tokens from PCA/quantization. Already specified in PROJECT.md. |
| Sliding window exemption | Last w=128 tokens have high attention mass and are actively queried. Compressing them degrades quality; at high CRs accuracy collapses. | LOW | Exclude last w=128 tokens from compression. Already specified in PROJECT.md. kvtc paper: critical threshold is window ≤16 vs ≥64. |
| Enable/disable config flags | Any operator must be able to disable compression entirely for debugging or compatibility testing. | LOW | Boolean flags for AM enable, kvtc enable, compression ratio targets. |

### Differentiators (Competitive Advantage)

Features that distinguish this pipeline from naive or first-pass approaches.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Non-uniform head budgets for AM | Largest quality gain in AM ablations — more important than query strategy or self-study. Allows assigning budget proportional to each head's information entropy. | MEDIUM | Head sensitivity curves are input-invariant and stable across contexts; precompute once per model. Spike measured per-head entropy 0.34–2.47 (significant variation). Budget schedule assigns more to later layers, contradicting PyramidKV — counter-intuitive but validated. |
| PCA calibration CLI (`omlx calibrate-kv <model>`) | Enables the user to calibrate for any new model without code changes. Without it, adding a new model requires developer intervention. | MEDIUM | One-time cost per model (~4s for 28 layers on Qwen 2.5 7B in spike; ~1.5 min for 160K tokens on H100 in paper). Stores V^T, µ, DP bit allocation table alongside model weights. PCA overhead: 2.4–8.7% of model params — negligible. |
| GQA-aware compression | GQA models (Qwen, Llama) have far fewer KV heads than query heads (e.g., 4 KV heads vs 28 query heads). Compression that ignores this is slower and less efficient. | LOW | Cross-head PCA multiplier is 4× not 28× for GQA; AM operates on KV heads only. Spike confirmed: 0.01s/layer for AM with 4 heads. |
| Asymmetric key/value compression for kvtc | Keys need higher precision for long-context retrieval (attention scores depend on key accuracy); values can be compressed more aggressively on retrieval tasks. Task-specific tuning can yield further gains. | MEDIUM | kvtc paper appendix B.4: keys at 32× + values at 256× gives LITM 71.9 vs vanilla 99.4. Not default behavior but useful knob. Defer from v1 unless benchmarks demand it. |
| Benchmark suite (compression ratio, quality, latency) | Without measurement there is no way to know if the pipeline is working correctly or regressing. Lets operators tune compression targets against quality budgets. | MEDIUM | Target models: Qwen 2.5 7B, Llama 3.x 8B, Gemma 3 variants, DeepSeek R1. Quality metric: cosine similarity + downstream task accuracy (perplexity is insufficient — generative eval required per kvtc paper appendix A). |
| Stats and metrics in admin UI | Operators need visibility into compression ratios, cache hit rates, and latency to tune and debug in production. | LOW | Expose: compression ratio achieved, compaction ratio, decompression latency, cache miss rate post-compression. Hook into existing omlx admin UI. |
| Repeat-prefill reference query strategy | The AM paper shows repeat-prefill ("{C} Repeat the previous context. {C}") is nearly as good as self-study (which takes 139s for 60k tokens) at a fraction of the cost. Default to this. | LOW | Spike uses context-prefill (cheapest). Production should use repeat-prefill. Self-study optional for highest-quality offline compaction. |
| zstd over DEFLATE for entropy coding | kvtc paper appendix B.8: Zstandard achieves the best compression ratio of any codec tested (34.6–46.3× at 32× target vs DEFLATE 34.7–42.9×) and is cross-platform. nvCOMP/GDeflate are NVIDIA-only. | LOW | zstd is a drop-in replacement. Python bindings via `zstd` or `zstandard` packages. Spike validated 6.8× total including zstd. |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Pre-bundled PCA matrices for popular models | Avoids calibration step for common models; convenient for users. | Creates a distribution and versioning problem — PCA matrix must match exact model weights and quantization. A mismatch silently degrades quality with no error. kvtc paper: per-prompt PCA gives dramatically lower real compression ratios (1.3–12.4× vs 60–88× one-time at 64× target). | Ship the calibration CLI first (v1). Pre-bundled matrices are a follow-up after the CLI is stable and the distribution story is clear. |
| Online/streaming compaction during generation | Appealing for agentic/long-generation use cases; AM paper has a preliminary AIME result. | AM paper is explicit that streaming compaction is only a preliminary result. Correct RoPE phase handling for streaming is complex; bugs cause silent quality degradation. | Offline compaction between inference phases (after prefill, before decode; or at the end of a turn). This is the validated, correct path. |
| Direct key optimization (Cₖ not restricted to original keys) | Would push AM to Cartridges-level quality at 100× ratios; broader search space. | Requires gradient descent; eliminates the closed-form property that makes AM fast. GPU-hours per context, not seconds. | Use the closed-form subset-selection approach (HighestAttnKeys or OMP). The quality gap only appears at extreme ratios (100×+) that are not the omlx target. |
| Inference in the compressed PCA domain | Would eliminate decompression overhead entirely; always-compressed KV cache. | kvtc paper explicitly defers this as future work — it requires model weight modifications (attention layers must operate on PCA-projected inputs). Not deployment-friendly and breaks the constraint that the model operates unchanged. | Decompress before attention. At 1.5ms/layer the decompression latency is not a bottleneck (well under 10ms target). |
| MLA (Multi-head Latent Attention) support | DeepSeek-V2 and newer models use MLA; coverage is appealing. | MLA's latent KV representation fundamentally changes the attention pattern. The AM paper's closed-form derivation assumes standard MHA/GQA attention; the kvtc cross-layer PCA structure assumes independent K and V projections. Both are explicitly out of scope in papers. | Defer until AM and kvtc publish MLA-specific derivations. Flag DeepSeek R1 testing as validating DeepSeek R1 (which uses standard GQA), not DeepSeek-V2. |
| L2 regularization on AM β or Cᵥ | Seems like standard ML practice; regularization usually helps prevent overfitting. | AM paper appendix: L2 regularization on β/Cᵥ *degrades* performance at all λ > 0. The closed-form NNLS/OLS solution is already the right answer for this objective. | Use unregularized NNLS for β and unregularized OLS (pinv) for Cᵥ. Box-constrain β ∈ [−3, 3] for HighestAttnKeys; prune keys with β < −7 post-selection for OMP. |
| Per-prompt PCA calibration | Would tailor the PCA basis to each specific conversation; theoretically more accurate. | kvtc paper appendix B.11: storing V^T per prompt eats the compression gains (effective CR drops to 1.3–12.4× at 64× target). One-time per-model PCA is the only viable approach for claimed compression ratios. | One-time calibration per model. The method generalizes well across domains (even code-only calibration data retains long-context retrieval ability at 16×). |
| Training-aware compaction or weight modifications | Could make models natively compression-friendly; better quality ceiling. | Requires retraining — not deployment-friendly, violates the constraint that omlx runs off-the-shelf models without modification. | Use the existing closed-form post-hoc approach. Quality at 4–16× is already within 1 point of vanilla on standard benchmarks. |

---

## Feature Dependencies

```
[PCA Calibration CLI]
    └──required-by──> [kvtc storage compression]
                          └──required-by──> [End-to-end compress/decompress]
                                                └──required-by──> [Decompression on cache miss]

[AM token compaction]
    └──required-by──> [End-to-end compress/decompress]

[Attention sink exemption]
    └──required-by──> [kvtc storage compression]  (correctness, not feature-gated)

[Sliding window exemption]
    └──required-by──> [kvtc storage compression]  (correctness, not feature-gated)

[Non-uniform head budgets]
    └──enhances──> [AM token compaction]  (single biggest quality gain in ablations)

[Memory-pressure trigger]
    └──activates──> [AM token compaction]

[Eviction-path trigger]
    └──activates──> [kvtc storage compression]

[Benchmark suite]
    └──validates──> [AM token compaction]
    └──validates──> [kvtc storage compression]
    └──validates──> [End-to-end compress/decompress]

[Stats + metrics]
    └──observes──> [End-to-end compress/decompress]
    └──observes──> [Cache interface integration]

[Asymmetric key/value compression]
    └──enhances──> [kvtc storage compression]  (optional knob, not required)
```

### Dependency Notes

- **kvtc requires PCA calibration:** The PCA basis V^T and DP bit allocation table must exist before any compression can run. Calibration is the unblocking step for kvtc.
- **AM must precede kvtc in pipeline:** AM reduces token count first, then kvtc compresses the smaller cache. Reversing this order loses the multiplicative gain.
- **Sink and window exemptions are correctness requirements, not optional features:** The kvtc paper shows accuracy collapses without them at high compression ratios. They are not "features to add" — they are correctness constraints to build in from the start.
- **Non-uniform head budgets enhance but do not gate AM:** AM works without them (uniform budget), but non-uniform is the single largest quality improvement in ablations. Build uniform first, add non-uniform in the same milestone.
- **Triggers depend on existing omlx infrastructure:** Memory pressure trigger hooks into the existing omlx memory monitor. Eviction trigger hooks into tiered_manager. These are integration points, not new infrastructure.
- **Benchmark suite unblocks production readiness:** Without benchmarks, there is no way to know if quality constraints (cosine similarity >0.998, within 1 point on task benchmarks) are being met.

---

## MVP Definition

### Launch With (v1 — the compression pipeline milestone)

- [ ] AM closed-form compaction (NNLS β-fitting, OLS value-fitting, HighestAttnKeys selection) — without this there is no token compaction
- [ ] Non-uniform head budgets precomputed per model — included with AM because it is the dominant quality lever and adds negligible complexity given entropy is already being computed
- [ ] kvtc PCA-based storage compression (cross-layer PCA, DP quantization, zstd) — without this there is no storage compression
- [ ] Attention sink + sliding window exemptions — correctness requirements, not optional
- [ ] PCA calibration CLI (`omlx calibrate-kv <model>`) — kvtc cannot run without it; CLI is the user-facing entry point
- [ ] AM→kvtc combined pipeline with end-to-end compress/decompress — the core integration
- [ ] Memory-pressure trigger for AM — without a trigger AM is never called in production
- [ ] Eviction-path trigger for kvtc — without a trigger kvtc is never called in production
- [ ] Decompression on cache miss — without this SSD caches are write-only
- [ ] Correct omlx cache interface integration with disable path as no-op — compatibility constraint
- [ ] Enable/disable config flags — required for safe rollout and debugging
- [ ] Benchmark suite covering compression ratio, cosine similarity, downstream task accuracy, decompression latency — required to validate quality constraints are met

### Add After Validation (v1.x)

- [ ] Asymmetric key/value compression ratios — add if benchmark results show keys are the quality bottleneck at target compression ratios
- [ ] Stats and metrics in admin UI — add once compression is stable and operators need visibility into production behavior
- [ ] Repeat-prefill reference query strategy for AM — spike used context-prefill; upgrade to repeat-prefill if benchmarks show quality benefit justifies the prefill cost
- [ ] Validation against Gemma 3 (SWA-heavy) and DeepSeek R1 (reasoning model) — Qwen is validated in spike; add other model families after core pipeline is stable

### Future Consideration (v2+)

- [ ] Self-study query generation for AM — 139s for 60k tokens on H100; may be viable for offline batch compaction but not interactive use; only meaningful if quality gap over repeat-prefill is confirmed
- [ ] OMP key selection for AM — best quality, 104–565s per compaction; viable only for offline/batch use cases; defer until there is a concrete use case that justifies the latency
- [ ] Pre-bundled PCA matrices for popular models — follow-up after CLI calibration is stable and distribution/versioning story is clear
- [ ] Chunked compaction for very long contexts — not needed for omlx's current context length targets; add if users report quality issues on very long inputs

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| AM token compaction (HighestAttnKeys) | HIGH | HIGH | P1 |
| Non-uniform head budgets | HIGH | MEDIUM | P1 |
| kvtc PCA + DP quantization + zstd | HIGH | HIGH | P1 |
| Sink + window exemptions | HIGH (correctness) | LOW | P1 |
| PCA calibration CLI | HIGH | MEDIUM | P1 |
| AM→kvtc combined pipeline | HIGH | MEDIUM | P1 |
| Memory-pressure trigger | HIGH | LOW | P1 |
| Eviction-path trigger | HIGH | LOW | P1 |
| Decompression on cache miss | HIGH | MEDIUM | P1 |
| omlx cache interface integration | HIGH (compatibility) | MEDIUM | P1 |
| Enable/disable config flags | HIGH (safety) | LOW | P1 |
| Benchmark suite | HIGH (validation) | MEDIUM | P1 |
| Stats and metrics in admin UI | MEDIUM | LOW | P2 |
| Asymmetric key/value compression | MEDIUM | MEDIUM | P2 |
| Repeat-prefill reference queries | MEDIUM | LOW | P2 |
| Gemma 3 + DeepSeek R1 validation | MEDIUM | LOW | P2 |
| OMP key selection | LOW | HIGH | P3 |
| Self-study query generation | LOW | MEDIUM | P3 |
| Pre-bundled PCA matrices | MEDIUM | HIGH | P3 |
| Chunked compaction | LOW | HIGH | P3 |

**Priority key:**
- P1: Must have for milestone to be considered complete
- P2: Should have, add when core is stable
- P3: Nice to have, future milestone

---

## Competitor Feature Analysis

This is an internal omlx feature, not a product competing in a market. The relevant comparison is against alternative KV cache management strategies that omlx could implement instead.

| Feature | H2O / SnapKV (token eviction) | Cartridges (gradient-based) | kvtc alone (storage only) | AM + kvtc (this pipeline) |
|---------|-------------------------------|----------------------------|--------------------------|--------------------------|
| Token count reduction | Yes — eviction | Yes — compaction | No | Yes — closed-form compaction |
| Storage compression | No | No | Yes — PCA + quant | Yes — PCA + quant |
| Quality at 8–16× | Poor (H2O: LITM 20.2 at 8×) | Matches original at 50×+ | Near-lossless at 16× | Near-lossless combined |
| Calibration required | No | No | Yes — one-time per model | Yes — one-time per model |
| Compaction time | <1s | GPU-hours | N/A | Seconds (HighestAttnKeys) |
| Apple Silicon compatible | Yes | Yes | Needs nvCOMP→zstd swap | Yes (zstd, MLX linalg) |
| Preserves model unchanged | Yes | No (gradient search) | Yes | Yes |

---

## Sources

- Fast KV Compaction via Attention Matching — Zweiger, Fu, Guo, Kim (MIT), arXiv:2602.16284, Feb 2026
- KV Cache Transform Coding — Staniszewski & Łańcucki (NVIDIA/University of Warsaw), ICLR 2026, arXiv:2511.01815v2
- omlx spike results — `docs/research/kv-cache-compression/SPIKE-RESULTS.md`, Qwen2.5-7B-Instruct-4bit on M3 Max, 2026-03-17
- omlx PROJECT.md — milestone requirements and out-of-scope decisions

---
*Feature research for: KV cache compression pipeline (AM + kvtc) for omlx*
*Researched: 2026-03-18*
