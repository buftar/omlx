# Notes: arXiv 2602.16284 — "Fast KV Compaction via Attention Matching"
**Authors**: Adam Zweiger, Xinghong Fu, Han Guo, Yoon Kim (MIT)
**Date**: Feb 18, 2026

---

## 1. Research Question

How to perform fast, high-quality KV cache compaction in latent space?
- Token-space approaches (summarization, eviction) degrade at high compaction ratios (20×–100×)
- Cartridges (Eyuboglu 2025) achieves 50× compaction via end-to-end gradient optimization but takes GPU-hours per context
- This work: match or exceed Cartridges quality in seconds/minutes via closed-form solutions

## 2. Mathematical Formulation

**Objective**: Replace (K, V) ∈ R^{T×d} with (Ck, β, Cv) where Ck, Cv ∈ R^{t×d}, β ∈ R^t, t << T

Two matching conditions for all reference queries q:

**(1) Attention output matching:**
exp(qK^T)V / Σ_j exp(qK_j^T)  ≈  [exp(qCk^T + β) / Σ_j exp(q(Ck)_j^T + βj)] Cv

**(2) Attention mass matching:**
Σ_j^T exp(qK_j^T)  ≈  Σ_j^t exp(q(Ck)_j^T + βj)

Why both conditions? They preserve the block's contribution under concatenation with future tokens (via attention decomposition used in FlashAttention / Cascade Inference). Without β, exact mass matching is impossible since t < T. β acts as per-key multiplicative weight (wj = exp(βj) > 0).

**Fitting β (NNLS)**: Given Ck and reference queries, solve:
- min_{w≥0} ||Aw - m||^2  where Aij = exp(qi(Ck)_j^T), m_i = Σ_k exp(qi K_k^T)
- β_j = log(w_j)

**Fitting Cv (OLS)**: Given Ck, β, and reference queries, solve:
- Cv* = argmin_{Cv} ||XCv - Y||^2_F = (X^TX)^-1 X^T Y
- y_i = softmax(qi K^T) V  [target attention output]
- x_i = softmax(qi Ck^T + β)  [predicted attention weights from compact cache]

**Selecting Ck**: Restrict to subset of original keys: Ck = K_{S,:} for index set S ⊂ {1,...,T}

## 3. Algorithm Details

### OMP Key Selection (Algorithm 1)
Greedy pursuit to match attention mass:
1. Φij = exp(qi K_j^T / sqrt(d))  [mass feature matrix, n×T]
2. m_i = Σ_j Φij  [target mass]
3. r = m, S = ∅
4. for k=1 to t:
   - j* = argmax_{j∉S}(r^T Φ_{:,j})  [most correlated column]
   - S = S ∪ {j*}
   - w = argmin_{w≥0} ||Φ_{:,S}w - m||^2  [NNLS refit]
   - r = m - Φ_{:,S}w
5. β = log w

**OMP speedups** (AM-OMP-fast):
- k_choice=4: select top 4 keys per iteration instead of 1 (4–8× speedup, little degradation)
- nnls_interval=2: refit NNLS every 2 iterations instead of every iteration
- progressive_schedule: [(300, 1, 1), (1500, 2, 2), (None, 4, 2)] — varies k_choice and nnls_interval based on number of keys selected so far

### Highest Attention Keys
- For each query qi, compute ai = softmax(qi K^T)
- Aggregate per-key importance via RMS: sj = RMS(a1j, ..., anj)
- Select top-t keys (faster but slightly worse than OMP)

### Optimization-Based Variants (OptimC1BetaCompaction, OptimJointCompaction)
- **OptimC1BetaCompaction**: Optimize C1, β via gradient descent to match logsumexp; then solve C2 via OLS
  - Loss: (logsumexp(qK^T) - logsumexp(qCk^T + β))^2
  - Optimizers: Adam, LBFGS, or Adam→LBFGS
- **OptimJointCompaction**: Jointly optimize C1, β, C2
  - Loss: output reconstruction + λ * partition function matching + λ_l2 * L2 reg
  - Warm-starts C2 from OLS solution

## 4. Query Generation Strategies

Four strategies (ordered by performance):
1. **self-study** (best): Generate synthetic interactions conditioned on context C with 4 fixed prompts (e.g., "Aggregate all key facts"), sample responses, extract query vectors. ~5k tokens on QuALITY.
2. **repeat-prefill** (nearly as good): "{C} Repeat the previous context. {C}" — prefill, extract queries during reconstruction. ~7k tokens on QuALITY.
3. **context-prefill** (fast, slightly worse than repeat-prefill): Prefill on C alone, extract queries.
4. **random-vectors**: qi ~ N(0, I_d). Works but lags others.

**On-policy queries**: Compact layers sequentially; for each layer ℓ, extract Qref by running model with layers < ℓ already compacted. Reduces distribution shift. Slight but consistent improvement.

## 5. Compaction Method Variants (ordered by quality)

| Method | Key Selection | Query Gen | Notes |
|--------|--------------|-----------|-------|
| AM-OMP | OMP | self-study + repeat-prefill + on-policy | Best quality |
| AM-OMP-fast | OMP (k=4, interval=2) | self-study + repeat-prefill + on-policy | 4–8× faster |
| AM-HighestAttnKeys | Top-attention | self-study + repeat-prefill + on-policy | Fast heuristic |
| AM-HighestAttnKeys-fast | Top-attention | repeat-prefill only | Fastest |

## 6. Nonuniform Compaction

Different heads have different sensitivity to KV budget:
- Measure head influence curves: fix all heads at baseline ratio, vary one head's ratio, measure Δlog(perplexity)
- Sensitivity ranking is stable across inputs/datasets → precompute once per model
- Budget allocation: greedy exchange algorithm (Algorithm 4), iteratively swap budget units between heads to minimize predicted loss
- Requires FlashAttention varlen packing to avoid memory overhead from variable-length heads

## 7. Chunked Compaction (long context)

For long contexts (e.g., 60k tokens), apply independently to chunks:
- **KV-based chunking** (default): Prefill full context → slice out per-chunk KV states → compact independently → concatenate
- **Text-based chunking**: Prefill each chunk independently → RoPE phase shift to align positions → merge (approximation; less faithful)

## 8. Benchmarks and Results

**Datasets**:
- QuALITY: long-document comprehension, 5-8k tokens, 15-20 questions/context, 50 contexts
- LongHealth: patient-records QA, 60k tokens/context, 4 contexts, 100 questions, very information-dense

**Models**: Qwen3-4B, Llama3.1-8B, Gemma3-12B

**Key results (50× compaction, Qwen3-4B, QuALITY)**:
- Original cache: ~71.5% accuracy
- AM-OMP: ~62-65% accuracy (matches Cartridges, 100× faster)
- AM-OMP-fast: seconds vs. minutes for OMP
- Summarization: ~55%
- H2O+, KVzip, SnapKV: ~42-48%
- No context: ~42%

**Wall-clock breakdown (60k LongHealth, Gemma-3-12B, H200 GPU)**:
- context-prefill: 7s; repeat-prefill: 8s; self-study: 139s
- Highest attention key selection: 3s
- OMP: 565s; OMP-fast: 104s
- β fitting (NNLS): 2.2s
- Cv fitting (least squares): 1.8s

**Summarization + AM-OMP**: AM-OMP on summarized text achieves ~200× compaction (6340→31 tokens avg) with accuracy comparable to summarization alone (~20× compaction).

## 9. Ablations

Leave-one-out on AM-OMP (log-perplexity metric, lower=better):
- **Nonuniform head budget** → most important component
- **Learned values (Cv)** → second most important
- **Attention biases (β)** → important
- **On-policy queries** → small consistent improvement
- **Self-study** → least important (repeat-prefill alone nearly as good)

## 10. Limitations and Future Work

**Limitations**:
- OMP + self-study still takes several minutes for long contexts
- At 100× compaction on LongHealth, Cartridges outperforms (gradient-based can search wider space; not restricted to selecting from original keys)

**Future work**:
- Move beyond subset selection for Ck (directly optimize compact keys)
- Architectures that support compaction as a primitive
- Integrate into inference engines (RadixAttention, varlen KV packing, disaggregated compaction)
- Online compaction for long-horizon agentic settings
- Appendix F.3 shows repeated compaction preserves AIME reasoning performance

---
## Batch Status: Batch 1 (pp. 1-12) complete. Batch 2 (pp. 13-24) = appendices = pending.
