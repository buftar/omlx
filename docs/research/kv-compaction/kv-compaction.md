# Fast KV Cache Compaction via Attention Matching

**Paper**: arXiv:2602.16284 — "Fast KV Compaction via Attention Matching"
**Authors**: Adam Zweiger, Xinghong Fu, Han Guo, Yoon Kim (MIT, Feb 2026)
**Repo**: https://github.com/adamzweiger/compaction

---

## Overview

**Problem**: KV cache memory is the primary bottleneck for long-context LLM inference. Existing approaches—token eviction (H2O, SnapKV), token merging, and summarization—degrade rapidly at high compaction ratios (20×–100×). Cartridges achieves high-quality latent-space compaction but requires several GPU-hours per context.

**Approach**: Attention Matching (AM) — construct compact keys C1, bias β, and values C2 that reproduce the original attention outputs and preserve attention mass, per KV-head per layer. The formulation decomposes into simple closed-form subproblems, avoiding gradient-based optimization entirely at compaction time.

**Key result**: Up to 50× compaction in seconds to minutes, matching Cartridges quality at 100× lower cost. AM methods form the Pareto frontier of compaction time vs. quality on QuALITY and LongHealth benchmarks across Qwen3-4B, Llama3.1-8B, and Gemma3-12B.

---

## Mathematical Formulation

### Setup

For a single KV-head, let:
- **K, V** ∈ ℝ^{T×d} — original keys and values for T tokens, head dim d
- **(C1, β, C2)** — compact representation: C1, C2 ∈ ℝ^{t×d}, β ∈ ℝ^t, with t << T

Define scaled logits and attention operators:
```
ℓ(q; K)     = (1/√d) q K^T
Mass(q; K)  = Σ_j exp(ℓ(q; K)_j)
Attn(q;K,V) = exp(ℓ(q;K)) V / Mass(q;K)

ℓ(q; C1, β)       = (1/√d) q C1^T + β
Mass(q; C1, β)     = Σ_j exp(ℓ(q; C1, β)_j)
Attn(q; C1, β, C2) = exp(ℓ(q; C1, β)) C2 / Mass(q; C1, β)
```

### Two Matching Objectives

Over a set of reference queries q₁…qₙ:

**(1) Attention output matching** — reproduce the attention value:
```
exp(qK^T) V / Σ_j exp(qK_j^T)  ≈  [exp(qC1^T + β) / Σ_j exp(q(C1)_j^T + βj)] C2
```

**(2) Attention mass matching** — preserve contribution under concatenation:
```
Σ_j^T exp(q K_j^T)  ≈  Σ_j^t exp(q (C1)_j^T + βj)
```

### Why Both Objectives?

Attention over concatenated blocks decomposes as a mixture weighted by unnormalized mass (the same decomposition exploited by FlashAttention and Cascade Inference):

```
Attn(q; [K; Kfixed], [V; Vfixed]) =
    Mass(q;K)/(Mass(q;K)+Mass(q;Kfixed)) · Attn(q;K,V)
  + Mass(q;Kfixed)/(Mass(q;K)+Mass(q;Kfixed)) · Attn(q;Kfixed,Vfixed)
```

Matching both the local attention output **and** the mass means the compacted block's contribution is preserved regardless of what future tokens are appended — enabling one-shot compaction without knowing future queries.

### Why β is Necessary

If C1 is a subset of K with t < T, then `Mass(q; C1) ≤ Mass(q; K)` for all q — the compacted block receives systematically too little global weight. The bias β introduces per-key multiplicative weights `wj = exp(βj) > 0`, so each retained key can represent the mass of many removed keys. For example, at q=0: `Mass(0; K) = T` but `Mass(0; C1) = t` without bias.

### Fitting β (NNLS)

Given Ck and reference queries, parameterize `w = exp(β)` and solve:

```
min_{w≥0} ||A w - m||²
where  A_ij = exp(q_i (C1)_j^T / √d)   ← mass feature matrix Φ ∈ ℝ^{n×t}
       m_i  = Σ_k exp(q_i K_k^T / √d)  ← target mass vector
```

Then `β_j = log(w_j)`, clamping `w_j` to a small positive value if needed.

### Fitting C2 (OLS)

Given C1 and β, solve the least-squares problem:

```
C2* = argmin_{C2} ||X C2 - Y||²_F  =  (X^T X)^{-1} X^T Y

where  y_i = softmax(q_i K^T / √d) V    ∈ ℝ^{1×d}   ← target attention output
       x_i = softmax(q_i C1^T / √d + β) ∈ ℝ^{1×t}   ← predicted attn weights
       Y = [y_1; …; y_n] ∈ ℝ^{n×d},   X = [x_1; …; x_n] ∈ ℝ^{n×t}
```

### Numerical Precision

All computations of C1, β, C2 are done in **FP32**, then cast to **BF16** for storage. Attention computations use per-query max-shift for numerical stability:
```python
# QK matmul in original dtype; upcast only for softmax
scores_raw = queries @ K.T          # (n, T) original dtype
scores32   = scores_raw.to(fp32) * inv_sqrt_d
m          = scores32.max(dim=1, keepdim=True)[0]
exp_scores = torch.exp(scores32 - m)
```

---

## Architecture

### Package Layout

```
compaction/
├── algorithms/
│   ├── base.py           ← CompactionAlgorithm ABC + _compute_C2 + evaluate_compaction
│   ├── omp.py            ← OMPCompaction, SimpleOMPCompaction
│   ├── optim.py          ← OptimC1BetaCompaction, OptimJointCompaction
│   └── (attention_score, random, etc.)
├── compaction_methods/
│   ├── base.py           ← FullCacheCompactionAlgorithm ABC
│   └── per_layer_head.py ← PerLayerHeadCompaction
├── query_generation/
│   └── query_gen.py      ← QueryConfig + generation strategies
└── models/
    └── cache.py          ← CompactedPrefixCache, CompactedPrefixLayer
```

### Core Abstraction: `CompactionAlgorithm`

`compaction/algorithms/base.py`

The base class for all single-(layer, head) algorithms:

```python
class CompactionAlgorithm(ABC):

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def compute_compacted_cache(
        self,
        K: Tensor,            # (T, d) original keys
        V: Tensor,            # (T, d) original values
        queries: Tensor,      # (n, d) reference queries
        t: int,               # target compacted size
        attention_bias: Tensor = None,  # (T,) or (n, T) additive bias
    ) -> Tuple[C1, beta, C2, indices]:
        # Returns:
        #   C1      : (t, d) compacted keys
        #   beta    : (t,)   bias terms
        #   C2      : (t, d) compacted values
        #   indices : list[int] selected key indices (if applicable)
        ...
```

The base class provides shared helpers:
- `_compute_C2(C1, beta, K, V, queries, ...)` — OLS solver for C2 (three backends: `lstsq`, `cholesky`, `pinv`; default `lstsq`)
- `_direct_C2(C1, K, V, indices)` — select C2 directly from V via nearest-neighbor matching
- `_compute_C2_with_method(...)` — dispatches to `lsq` or `direct`
- `_nnls_pg(M, y, iters, ...)` — NNLS via projected gradient descent (Algorithm 3)
- `evaluate_compaction(K, V, C1, beta, C2, test_queries)` — returns MSE, relative L2 error, cosine sim, sumexp error, and compaction ratio

### Algorithm Implementations

#### OMP — `OMPCompaction` (`algorithms/omp.py`)

The recommended algorithm. Greedily selects keys that best match the attention mass via Orthogonal Matching Pursuit (Algorithm 1/2).

```python
from compaction.algorithms.omp import OMPCompaction

alg = OMPCompaction(
    nnls_iters=0,                 # NNLS solver: 0=lstsq+clamp, >0=projected gradient
    nnls_lower_bound=None,        # Lower bound for w (default: 1e-12)
    nnls_upper_bound=None,        # Upper bound for w (default: e^7)
    c2_method='lsq',              # 'lsq' or 'direct' for fitting C2
    k_choice=1,                   # Keys selected per OMP iteration (1=standard, 4=fast)
    c2_ridge_lambda=0,            # Ridge regularization for C2 (default off)
    c2_solver='lstsq',            # 'lstsq', 'cholesky', or 'pinv'
    c2_ridge_scale='spectral',    # 'spectral', 'frobenius', or 'fixed'
    nnls_interval=1,              # Refit NNLS every N iters (1=every iter, 2=fast)
    use_abs_corr=False,           # Use |correlation| vs raw correlation for selection
    drop_key_beta_cutoff=None,    # Refinement: drop keys with β < cutoff, re-select
    progressive_schedule=None,    # Override k_choice/nnls_interval by key count
    zerobeta=False,               # Zero out β before C2 fitting (ablation)
)
C1, beta, C2, indices = alg.compute_compacted_cache(K, V, queries, t)
```

**OMP Algorithm** (from paper Algorithm 1):
1. Build mass feature matrix: `Φ_ij = exp(q_i K_j^T / √d)`
2. Target: `m_i = Σ_j Φ_ij`
3. Greedy loop: select key `j* = argmax_{j∉S}(r^T Φ_{:,j})`, update S, refit w via NNLS, update residual `r = m - Φ_{:,S}w`
4. `β = log(w)`

**Speed-quality knobs**:

| Config | `k_choice` | `nnls_interval` | Speedup |
|--------|-----------|----------------|---------|
| Standard (AM-OMP) | 1 | 1 | baseline |
| Fast (AM-OMP-fast) | 4 | 2 | ~4–8× |
| Progressive | schedule-based | schedule-based | adaptive |

**Default progressive schedule** (`DEFAULT_PROGRESSIVE_SCHEDULE`):
```python
[(300, 1, 1),    # first 300 keys: standard
 (1500, 2, 2),   # keys 301–1500: k=2, interval=2
 (None, 4, 2)]   # keys 1501+:   k=4, interval=2
```

**β Stabilization for OMP**: After greedy selection, any key with `β < -7` is dropped and replaced. Weights are capped at `exp(7)`. In practice these rarely trigger.

#### Optimization-Based — `OptimC1BetaCompaction` (`algorithms/optim.py`)

Optimizes C1 and β via gradient descent to match the log-sum-exp (partition function), then solves C2 via OLS:

```python
from compaction.algorithms.optim import OptimC1BetaCompaction

alg = OptimC1BetaCompaction(
    lr=0.01,
    num_steps=1000,
    patience=100,
    optimizer='lbfgs',   # 'adam', 'lbfgs', or 'adam_lbfgs'
    lam=0.0,             # L2 regularization on C1 and β
)
```

**Loss**: `L = mean((logsumexp(qK^T/√d) - logsumexp(qC1^T/√d + β))²) + λ(||C1||² + ||β||²)`

Initializes C1 by randomly sampling t keys from K; β initialized to zeros.

#### Joint Optimization — `OptimJointCompaction` (`algorithms/optim.py`)

Jointly optimizes C1, β, and C2:

```python
from compaction.algorithms.optim import OptimJointCompaction

alg = OptimJointCompaction(
    lr=0.01,
    num_steps=5000,
    lam=1.0,             # Weight for partition function matching loss
    patience=200,
    optimizer='adam_lbfgs',  # Two-stage: Adam then LBFGS
    lam_l2=0.0,
    use_lr_decay=True,
    eta_min=0.0,
    adam_steps=5000,
    lbfgs_steps=5000,
)
```

**Loss**: `L = ||softmax(qK^T)V - softmax(qC1^T+β)C2||² + λ·(logsumexp match)² + λ_l2·||params||²`

Warm-starts C2 from OLS solution. Optimizers: `'adam'`, `'lbfgs'`, `'adam_lbfgs'` (Adam → LBFGS two-stage).

#### Attention-Score Selection

`AM-HighestAttnKeys`: Select the t keys with highest RMS attention score under reference queries, then fit β and C2:
```
s_j^{RMS} = sqrt( (1/n) Σ_i softmax(q_i K^T)_j² )
```
RMS is more robust than mean or max aggregation (per ablations).

**β Stabilization for HighestAttnKeys**: Uses bounded NNLS with `exp(-3) ≤ w_j ≤ exp(3)` (i.e., `β_j ∈ [-3, 3]`), using `iters=2` projected gradient steps.

#### Random Baselines

`RandomSubsetKeysCompaction`: Select random t keys from K; optionally fit β, C2 via OLS. Used as ablation baseline.

### Compaction Methods: `PerLayerHeadCompaction`

`compaction/compaction_methods/per_layer_head.py`

The top-level wrapper that applies any single-(layer, head) algorithm across the entire model:

```python
from compaction.compaction_methods.per_layer_head import PerLayerHeadCompaction
from compaction.algorithms.omp import OMPCompaction

compactor = PerLayerHeadCompaction(
    algorithm_class=OMPCompaction,
    algorithm_kwargs={'k_choice': 4, 'nnls_interval': 2},
    use_batched=False,                   # batched mode (experimental)
    config_name='AM-OMP-fast',           # display name
    precomputed_budget_path=None,        # path to JSON with per-head budgets
    max_ratio_per_head=1.0,
)

compacted_cache, stats = compactor.compact_kv_cache(
    past_key_values=past_kv,          # tuple of (K,V) or (K,bias,V) per layer
    target_size=t,                     # total compacted length
    indices=None,                      # None = compact entire sequence
    query_config=query_cfg,
    model=model,
    tokenizer=tokenizer,
    formatted_context=context_str,
    sliding_layer_indices={...},       # set of sliding-window layer indices
)
# Returns: ((C1_L0, beta_L0, C2_L0), (C1_L1, beta_L1, C2_L1), ...)
```

**KV cache format**:
- Input: `past_key_values` as `tuple[tuple[Tensor]]` — shape per tensor: `(batch, num_heads, seq_len, head_dim)`
- Input supports `(K, V)` or `(K, attention_bias, V)` per layer, or `CompactedPrefixCache`
- Output: `tuple[tuple[Tensor]]` — `((C1, beta, C2), ...)` per layer

**Nonuniform head budgets**: Pass a JSON file path to `precomputed_budget_path`. The JSON maps `(layer_idx, head_idx)` → fraction of total budget. This is precomputed once per model using per-head sensitivity curves.

### Query Generation

Four strategies (from `compaction/query_generation/`):

| Strategy | Description | Speed | Quality |
|----------|-------------|-------|---------|
| `context_prefill` | Prefill on context C, extract query vectors | Fastest (7s/60k) | Good |
| `repeat_prefill` | Prefill on `"{C} Repeat the previous context. {C}"`, extract reconstruction queries | Fast (8s/60k) | Better |
| `self_study` | Generate 4 synthetic conversations, extract query vectors during response decoding | Slow (139s/60k) | Best |
| `random_vectors` | Sample q_i ~ N(0, I_d) with q-norm scaling | Instant | Weakest |

**On-policy queries**: Compact layers sequentially; for each layer ℓ, extract Qref by running the model with layers 0…ℓ-1 already compacted. Reduces distribution shift from early-layer compaction propagating to later layers. Provides slight but consistent improvement.

**Self-study prompts** (4 fixed templates):

| Name | Prompt |
|------|--------|
| `3-question` | "Write 3 questions that test understanding of different parts of the context." |
| `summarize` | "Summarize the main points of the context." |
| `structure_json` | "Structure the information in JSON form and include all important details…" |
| `aggregate` | "Aggregate all the key facts mentioned in the context." |

**Max queries per head**: 50,000 (reservoir sampling). In practice ~16,000 per head for QuALITY articles.

**QueryConfig fields** (from `query_generation/query_gen.py`):
```python
@dataclass
class QueryConfig:
    query_type: str           # 'repeat_prefill', 'context_prefill', 'self_study', 'random'
    num_queries: int          # max reference queries per head
    on_policy: bool = False   # use on-policy query extraction
    vllm_model: Any = None    # pre-initialized vLLM model for self-study
```

### Model Cache: `CompactedPrefixCache`

`compaction/models/cache.py`

Stores the compacted cache with logical vs. physical length separation:

```python
@dataclass
class CompactedPrefixLayer:
    keys:   Tensor   # (batch, num_heads, t, head_dim)
    beta:   Tensor   # (batch, num_heads, t)
    values: Tensor   # (batch, num_heads, t, head_dim)

@dataclass
class CompactedPrefixCache:
    layers: list[CompactedPrefixLayer | DynamicSlidingWindowLayer]
    logical_length: int    # original T (used for RoPE position IDs)
    # physical size = t (actual stored entries)
```

The compacted cache retains **logical length T** so newly appended tokens receive correct RoPE position IDs, decoupling physical cache size from sequence length.

**β integration with attention**: β is passed as an additive `attn_mask` to `scaled_dot_product_attention`:
```python
# β shape: (1, 1, 1, t) → broadcast to (batch, heads, n_queries, t)
beta_mask = beta.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(...)
out = F.scaled_dot_product_attention(q, C1, C2, attn_mask=beta_mask)
```
Supported by PyTorch SDPA and FlexAttention with zero runtime overhead.

---

## Setup & Installation

### Dependencies

The repo has no `requirements.txt` or PyPI package. Install manually:

```bash
# Core
pip install torch transformers

# For self-study query generation
pip install vllm

# For PDF splitting (docs only)
pip install PyPDF2
```

**Tested models**: `Qwen3-4B`, `Qwen3-4B-Instruct-2507` (long-context), `Llama-3.1-8B-Instruct`, `Gemma-3-12b-it`

### Environment

```bash
git clone https://github.com/adamzweiger/compaction
cd compaction
pip install -e .
```

The repo uses relative imports (`from models.cache import ...`), so running examples from the repo root is required.

### Quick Start

The `qa_demo.py` example demonstrates end-to-end compaction:

```bash
# Basic run (Qwen3-4B, 10% target size, CUDA)
python examples/qa_demo.py

# Custom model and ratio
python examples/qa_demo.py \
    --model Qwen/Qwen3-4B \
    --target-size 0.1 \
    --device cuda
```

The demo:
1. Loads the model and tokenizer
2. Prefills a ~1200-char mock article (fictional "Verandia" nation) into the KV cache
3. Answers 3 multiple-choice questions from the **full** cache
4. Compacts the cache to `--target-size` fraction using `AM-HighestAttnKeys`
5. Answers identical questions from the **compacted** cache
6. Prints accuracy comparison and compaction timing

**Minimal programmatic usage**:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from compaction.algorithms.omp import OMPCompaction
from compaction.compaction_methods.per_layer_head import PerLayerHeadCompaction
from compaction.query_generation.query_gen import QueryConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
model.eval().cuda()

# 1. Prefill context
inputs = tokenizer(context_text, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model(**inputs, use_cache=True)
past_kv = out.past_key_values   # tuple of (K, V) per layer

# 2. Configure compactor
compactor = PerLayerHeadCompaction(
    algorithm_class=OMPCompaction,
    algorithm_kwargs={'k_choice': 4, 'nnls_interval': 2},
    config_name='AM-OMP-fast',
)
query_cfg = QueryConfig(
    query_type='repeat_prefill',
    num_queries=50000,
    on_policy=False,
)

# 3. Compact (target: 10% of original tokens)
T = past_kv[0][0].shape[2]   # sequence length from first layer keys
t = int(T * 0.10)
compacted_kv, stats = compactor.compact_kv_cache(
    past_key_values=past_kv,
    target_size=t,
    indices=None,
    query_config=query_cfg,
    model=model,
    tokenizer=tokenizer,
    formatted_context=context_text,
)

# 4. Use compacted cache for generation
# compacted_kv: ((C1_L0, beta_L0, C2_L0), ...) per layer
```

---

## Configuration Reference

### Key Hyperparameters

| Parameter | Scope | Default | Notes |
|-----------|-------|---------|-------|
| `k_choice` | OMP | 1 | Keys selected per iteration. 4 = fast mode (~4–8× speedup) |
| `nnls_interval` | OMP | 1 | NNLS refit every N iterations. 2 = fast mode |
| `progressive_schedule` | OMP | None | `[(max_keys, k, τ), ...]` to adaptively change k and τ |
| `drop_key_beta_cutoff` | OMP | None | Drop keys with β < cutoff after selection; re-select replacements |
| `c2_method` | Both | `'lsq'` | `'lsq'` (OLS) or `'direct'` (nearest-neighbor from V) |
| `c2_ridge_lambda` | Both | 0 | Ridge regularization for C2 (tested; degraded performance at all λ>0) |
| `nnls_lower_bound` | Both | 1e-12 | Minimum value of w = exp(β) |
| `nnls_upper_bound` | OMP | exp(7) | Maximum value of w = exp(β) |
| `nnls_upper_bound` | HighestAttnKeys | exp(3) | Tighter bound: β ∈ [-3, 3] |
| `zerobeta` | OMP | False | Ablation: zero out β before C2 fitting |
| `max_ratio_per_head` | PerLayerHead | 1.0 | Cap per-head budget allocation |
| `precomputed_budget_path` | PerLayerHead | None | JSON with per-(layer,head) allocation fractions |

### Chunking Strategies

For long contexts (LongHealth: 60k tokens), compaction is applied chunk-by-chunk:

**KV-based chunking** (default, more faithful):
```
[prefix] + [compacted_chunk_1] + ... + [compacted_chunk_N] + [suffix]
```
- Prefill full context once
- Slice KV tensors per chunk; compact independently
- Concatenate results
- Preserves global RoPE positions throughout

**Text-based chunking** (approximation):
- Prefill each chunk in isolation with local positions
- Apply RoPE phase shift `Δ = p_global - p_local` to align compacted keys to global positions
- Faster but less faithful for inter-chunk interactions

**Self-study for chunks**: Queries are extracted from `[prefix] + [chunk_i] + [suffix]` — the actual tokens that will surround the chunk in the final cache.

**Recommended chunk count**: 5 chunks for 60k-token contexts (LongHealth configuration).

### Linear Algebra Backends

C2 fitting uses `torch.linalg.lstsq` (QR decomposition via `gels` driver on CUDA) by default:

```python
C2 = torch.linalg.lstsq(X, Y, driver='gels').solution  # (t, d)
```

Fallback chain on NaN: lstsq → Cholesky with λ=1e-6.

NNLS uses `iters=0` (lstsq + clamp) for OMP and `iters=2` (projected gradient) for HighestAttnKeys.

---

## Integration with omlx

The compaction library operates on standard HuggingFace `past_key_values` format, making it framework-agnostic. Key integration points:

1. **After prefill**: Intercept `out.past_key_values` and pass to `PerLayerHeadCompaction.compact_kv_cache()`
2. **Cache format**: The compacted output `((C1, beta, C2), ...)` must be passed to the attention layer with `beta` as an additive `attn_mask` — supported by PyTorch SDPA natively
3. **Logical length**: The `CompactedPrefixCache` stores `logical_length=T` so position IDs for new tokens are computed correctly despite the physical cache being smaller
4. **Sliding window models** (Gemma-3): Pass `sliding_layer_indices` to skip compaction on local-attention layers; only global-attention layers are compacted
5. **GQA support**: Aggregate queries from all query heads that attend to each KV head into a shared query set

**Nonuniform budget files**: Per-model JSON files with per-head budget fractions (precomputed once via sensitivity curves). These generalize across contexts and compaction ratios for a given model.

---

## Evaluation

### Metrics

`evaluate_compaction(K, V, C1, beta, C2, test_queries)` returns:

| Metric | Description |
|--------|-------------|
| `mean_output_mse` | Mean MSE of attention outputs across queries |
| `mean_output_relative_l2_error` | `||orig - compact|| / ||orig||` |
| `mean_output_cosine_sim` | Cosine similarity of attention outputs |
| `mean_sumexp_relative_error` | `|exp(lse_compact - lse_orig) - 1|` (mass matching quality) |
| `compaction_ratio` | T/t |

### Benchmarks

| Benchmark | Context length | Questions | Task |
|-----------|---------------|-----------|------|
| QuALITY | 5–8k tokens | 15–20 per context, 50 contexts | Multi-choice comprehension |
| LongHealth | 60k tokens (5 patients) | 100 per context, 4 contexts | Medical QA |
| AIME 2025 | varies | 30 problems | Mathematical reasoning |

### Key Results (50× compaction, Qwen3-4B, QuALITY)

| Method | Accuracy | Compaction Time | Notes |
|--------|----------|----------------|-------|
| Original cache | ~71.5% | — | Baseline |
| **AM-OMP** | ~62–65% | ~10 min | Best quality |
| **AM-OMP-fast** | ~61–64% | ~2 min | k=4, interval=2 |
| **AM-HighestAttnKeys** | ~60–62% | ~30s | Fast heuristic |
| Cartridges | ~62% | ~5 GPU-hours | End-to-end optimization |
| Summarization | ~55% | fast | Token-space |
| H2O+, KVzip, SnapKV | ~42–48% | fast | Token eviction |
| No context | ~42% | — | Lower bound |

### Compaction Stage Timings (60k LongHealth, Gemma-3-12B, H200 GPU)

| Stage | Method | Time (s) |
|-------|--------|----------|
| Query gen | context-prefill | 7 |
| Query gen | repeat-prefill | 8 |
| Query gen | self-study | 139 |
| Key selection | Highest attention | 3 |
| Key selection | OMP | 565 |
| Key selection | OMP-fast (k=4, τ=2) | 104 |
| β fitting | NNLS | 2.2 |
| C2 fitting | Least squares | 1.8 |

### Online Compaction (AIME 2025, Qwen3-4B)

Mid-trajectory compaction with 50% reduction on hitting physical budget:

| Phys Len | Eff Len | AIME/30 | Compactions |
|----------|---------|---------|-------------|
| 2048 | 2048 | 1.25 | none |
| 4096 | 4096 | 7.75 | none |
| 8192 | 8192 | 13 | none |
| **2048** | **4096** | **8** | ≤2 |
| **2048** | **8192** | **13** | ≤6 |

Repeated mid-trajectory compaction matches standard decoding at the same effective length — strong evidence that essential reasoning state survives compaction.

---

## References

- **Paper**: Zweiger et al. (2026). "Fast KV Compaction via Attention Matching." arXiv:2602.16284
- **Repo**: https://github.com/adamzweiger/compaction
- **Cartridges** (predecessor): Eyuboglu et al. (2025). arXiv:2506.06266
- **KVzip** (strong baseline): Kim et al. (2025). NeurIPS 2025
- **OMP**: Tropp & Gilbert (2007). IEEE Trans. Information Theory 53(12):4655–4666
- **FlashAttention**: Dao et al. (2022/2024). NeurIPS / ICLR
- **QuALITY**: Pang et al. (2022). NAACL
- **LongHealth**: Adams et al. (2024). arXiv:2401.14490
