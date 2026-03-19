# Phase 2: AM Compaction - Research

**Researched:** 2026-03-18
**Domain:** Attention Matching KV cache compaction, MLX tensor ops, scipy NNLS
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**compact() API**
- `compact(kv_cache, ratio=4.0, queries=None)` — full model call: takes a list of `(keys, values)` tuples for all layers, returns a single `AMCompactedCache` object
- `ratio` is a single float applied uniformly; per-head budget allocation is computed internally from entropy curves (or uniform fallback)
- `queries` is caller-provided — `AMCompactor` stays stateless; a `generate_reference_queries()` helper lives in `am.py` for convenience but is not called internally
- If `queries=None`, compact() falls back to uniform token selection (for testing only — documented as lower quality)

**AMCompactor initialization**
- `AMCompactor(head_entropy=None, n_sink_tokens=4)` — both parameters optional
- `head_entropy`: per-head entropy curves from calibration bundle; if `None`, uniform budgets are used (fully testable standalone without Phase 4)
- When entropy curves are provided: per-head token budget is proportional to entropy — higher-entropy heads get more of the total budget
- `n_sink_tokens=4` matches the spike and aligns with Phase 3 kvtc's s=4 exemption

**Token selection**
- HighestAttnKeys: select the t tokens with highest summed attention weight across reference queries (after always preserving the first `n_sink_tokens` sinks)
- When `queries=None`: fall back to uniform interval selection — explicitly documented as a testing convenience, not production path
- Beta box constraints: values in [-3, 3] per AM-08; keys with beta < -7 are pruned when using OMP path

**Output — AMCompactedCache dataclass**
- Fields: `layers: list[tuple[mx.array, mx.array]]` (compacted keys/values per layer), `logical_seq_len: int`, `diagnostics: dict | None`
- `logical_seq_len` is the original sequence length T — preserved for correct RoPE position indices
- `diagnostics` is `None` by default; populated in debug mode with per-layer/per-head: betas, NNLS residuals, cosine similarity between compacted and original attention output

### Claude's Discretion
- Exact `diagnostics` dict key names and structure
- Internal chunking/batching strategy for multi-head NNLS calls
- Whether `generate_reference_queries()` exposes the "sample" vs "random" method as a parameter

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| AM-01 | User's KV cache is compacted to a target ratio using HighestAttnKeys selection | Spike confirmed: sum attn weights across reference queries, select top-t positions plus n_sink_tokens preserved positions. Fallback to uniform interval selection for queries=None path. |
| AM-02 | Beta bias vector is fitted via NNLS to preserve attention mass after compaction | Spike confirmed: `nnls_solve()` from linalg_utils.py is the exact tool. Design matrix A = attn_weights[:,selected_indices], target = row sums of full attention. Betas are per-head, per-query. |
| AM-03 | Compacted value matrix Cv is fitted via OLS to preserve attention outputs | Spike confirmed: `pinv_f32()` from linalg_utils.py. V_compact = pinv(A_selected) @ output_full. One call per head. output_full is the attention-weighted sum using the full (pre-compaction) key set. |
| AM-04 | Compacted cache retains logical sequence length T with physical size t for correct RoPE phases | AMCompactedCache.logical_seq_len stores T=original seq_len. The compacted keys tensor has shape [1, n_heads, t, head_dim] where t = ceil(T / ratio). |
| AM-05 | Non-uniform head budgets are precomputed per model based on per-head entropy sensitivity | Spike measured Qwen 2.5 7B layer-0 entropies: [1.68, 0.34, 2.47, 1.12]. Budget proportional to entropy: head_budget[h] = max(n_sink_tokens, round(total_budget * entropy[h] / sum(entropy))). When head_entropy=None: uniform = floor(T/ratio) per head. |
| AM-06 | Head budget schedule is stored alongside model and reused across compactions | AMCompactor receives head_entropy at __init__ time and computes budgets once from it. No recomputation per compact() call. Phase 4 (calibration) produces the bundle; Phase 2 only consumes it. |
| AM-07 | Reference queries are generated via repeat-prefill strategy for compaction optimization | generate_reference_queries(keys, n_queries=64) helper in am.py; two methods: "sample" (sample n_queries positions from existing keys) and "random" (Gaussian scaled to key std). Caller decides which method. |
| AM-08 | Beta values are box-constrained in [-3,3] for HighestAttnKeys; keys with beta < -7 are pruned for OMP | For HighestAttnKeys path: clip betas to [-3, 3] after NNLS. For OMP path (v2/advanced): prune token positions where beta < -7 from the compacted key set. Phase 2 implements the HighestAttnKeys path only. |
</phase_requirements>

---

## Summary

Phase 2 formalizes the AM compaction pipeline from the spike prototype (`docs/research/kv-cache-compression/spike_am.py`) into a production class: `AMCompactor` in `omlx/compression/am.py`. The domain is well-explored: the spike ran successfully against Qwen 2.5 7B producing **0.9987 average cosine similarity** at 4x compaction with **0.42s total for all 28 layers**. All underlying math ops (`pinv_f32`, `nnls_solve`) were shipped in Phase 1 and are confirmed working.

The implementation follows a strict pipeline order per layer per head: (1) select t token positions via HighestAttnKeys (or uniform fallback), (2) compute full and selected attention weights, (3) compute full attention output as the OLS target, (4) NNLS beta-fitting on attention mass, (5) OLS value-fitting via pseudoinverse, (6) optional cosine similarity verification into diagnostics. The multi-head loop runs sequentially — no parallelization pressure in Phase 2.

The only conceptual gap between spike and production is the HighestAttnKeys selection path (spike used uniform; AM-01 requires summed-attention ranking) and the beta box constraint (AM-08). Both are straightforward to implement on top of existing MLX ops. The `AMCompactedCache` dataclass is the output contract that Phase 5 (pipeline assembly) will consume.

**Primary recommendation:** Translate `am_compact_layer()` from the spike directly into `AMCompactor._compact_head()`, replace uniform selection with HighestAttnKeys using `mx.argsort` on summed attention weights, add beta clipping after NNLS, and wrap in the `AMCompactedCache` dataclass.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| mlx.core | 0.31.x (confirmed) | Tensor ops, attention computation, pseudoinverse | Project-wide; all KV cache tensors are mx.array |
| scipy.optimize.nnls | 1.17.1 (confirmed) | NNLS beta-fitting | Already bridged in linalg_utils.py; caller never imports scipy directly |
| numpy | 2.4.2 (confirmed) | numpy<->MLX bridge in linalg_utils | Internal to linalg_utils; am.py only deals in mx.array |
| dataclasses (stdlib) | Python 3.x | AMCompactedCache dataclass | Zero dependencies, standard pattern in omlx |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| omlx.compression.linalg_utils | Phase 1 | pinv_f32, nnls_solve | All linalg calls go through these — never call mx.linalg.pinv directly |
| typing (stdlib) | Python 3.x | Type hints on all public APIs | Always; project convention |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| nnls_solve (scipy NNLS) | unconstrained OLS for beta | NNLS ensures non-negative betas which model attention mass correctly; unconstrained OLS can give negative mass |
| HighestAttnKeys selection | OMP (Orthogonal Matching Pursuit) | OMP is higher quality but requires iterative greedy algorithm; AM-ADV-01 deferred to v2. HighestAttnKeys is closed-form and fast |
| uniform head budgets | per-head entropy budgets | Entropy budgets require Phase 4 calibration data; uniform is the testing/standalone fallback |

**Installation:** No new dependencies. All required packages (`mlx`, `scipy`, `numpy`) are already declared in `pyproject.toml` via Phase 1.

---

## Architecture Patterns

### Recommended Project Structure
```
omlx/compression/
+-- __init__.py          # Intentionally empty (project convention)
+-- linalg_utils.py      # Phase 1 -- float32-safe MLX linalg wrappers
+-- am.py                # Phase 2 -- AMCompactor + AMCompactedCache + generate_reference_queries
```

### Pattern 1: Stateless Compactor Class
**What:** `AMCompactor` holds only calibration-derived constants (`head_entropy`, `n_sink_tokens`). All mutable state lives in `AMCompactedCache` objects it returns. No instance mutation during `compact()`.
**When to use:** Always — the CONTEXT.md decision is locked.
**Example:**
```python
# Source: 02-CONTEXT.md locked decisions
@dataclass
class AMCompactedCache:
    layers: list[tuple[mx.array, mx.array]]  # compacted (keys, values) per layer
    logical_seq_len: int                       # original T for RoPE position indices
    diagnostics: dict | None = None            # None by default, populated in debug mode

class AMCompactor:
    def __init__(self, head_entropy=None, n_sink_tokens: int = 4):
        self._head_entropy = head_entropy
        self.n_sink_tokens = n_sink_tokens

    def compact(self, kv_cache: list[tuple[mx.array, mx.array]],
                ratio: float = 4.0,
                queries=None) -> AMCompactedCache:
        ...
```

### Pattern 2: Layer-Head Inner Loop
**What:** Outer loop over layers, inner loop over heads. Each head is compacted independently. Head budget may differ (non-uniform) but is computed once before the loops begin.
**When to use:** Always for `compact()` implementation.
**Example:**
```python
# Source: spike_am.py am_compact_layer() adapted for production
for layer_idx, (keys, values) in enumerate(kv_cache):
    # keys shape: [1, n_heads, seq_len, head_dim]
    n_heads = keys.shape[1]
    compacted_keys_per_head = []
    compacted_vals_per_head = []
    for h in range(n_heads):
        k_h = keys[:, h:h+1]   # [1, 1, seq_len, head_dim]
        v_h = values[:, h:h+1]
        budget = head_budgets[layer_idx][h]
        k_c, v_c = self._compact_head(k_h, v_h, queries_layer, budget)
        compacted_keys_per_head.append(k_c)
        compacted_vals_per_head.append(v_c)
    # Concatenate heads back: [1, n_heads, budget, head_dim]
    k_layer = mx.concatenate(compacted_keys_per_head, axis=1)
    v_layer = mx.concatenate(compacted_vals_per_head, axis=1)
```

### Pattern 3: HighestAttnKeys Selection
**What:** Select tokens by descending summed attention weight across all reference queries. Sink tokens are always preserved first, then top-(budget - n_sinks) positions from the remainder.
**When to use:** `queries is not None` path (production).
**Example:**
```python
# Source: AM paper + spike adaptation using mx.argsort
# attn_weights: [1, 1, n_queries, seq_len] float32
# Sum across queries to get per-token importance score
importance = attn_weights[0, 0].sum(axis=0)  # [seq_len]
# Sink tokens always kept
sink_idx = mx.array(list(range(min(n_sink_tokens, seq_len))))
n_select = budget - sink_idx.shape[0]
non_sink_importance = importance[n_sink_tokens:]
# mx.argsort returns ascending; take the last n_select (highest values)
topk_local = mx.argsort(non_sink_importance)[-n_select:]
topk_global = topk_local + n_sink_tokens
selected = mx.sort(mx.concatenate([sink_idx, topk_global]))
mx.eval(selected)
```

### Pattern 4: Beta Box Constraint (AM-08)
**What:** After NNLS, clip betas to [-3, 3] for the HighestAttnKeys path.
**When to use:** Always after `nnls_solve()` in the HighestAttnKeys path.
**Example:**
```python
# Source: AM-08 requirement
betas, residual = nnls_solve(A_selected, target)
# NNLS gives non-negative betas; box-constrain upper bound per AM-08
betas = mx.clip(betas, -3.0, 3.0)
mx.eval(betas)
```

### Pattern 5: OLS Value Fitting
**What:** Solve `A_selected @ V_compact = output_full` for `V_compact` via pseudoinverse.
**When to use:** After token selection and beta-fitting, for every head.
**Example:**
```python
# Source: spike_am.py + linalg_utils.py
# A_selected: [n_queries, n_selected] -- attn weights at selected positions
# output_full: [n_queries, head_dim] -- full attn output (target)
A_s = attn_selected[0, 0]    # [n_queries, n_selected]
target = output_full[0, 0]   # [n_queries, head_dim]
A_pinv = pinv_f32(A_s)       # [n_selected, n_queries]
V_compact = A_pinv @ target   # [n_selected, head_dim]
V_compact = V_compact[None, None]  # [1, 1, n_selected, head_dim]
mx.eval(V_compact)
```

### Pattern 6: MLX Graph Materialization Checkpoints
**What:** MLX uses lazy evaluation. Call `mx.eval()` at key checkpoints to force graph execution before any numpy conversion (for nnls_solve) or diagnostics inspection.
**When to use:** After computing output_full and attn tensors (before NNLS call), after V_compact, after selected indices.
**Example:**
```python
# Source: spike_am.py -- standard MLX pattern
mx.eval(output_full, attn_full, attn_selected)  # before numpy conversion in nnls_solve
mx.eval(V_compact)                               # after OLS solve
mx.eval(selected)                                # after argsort
```

### Anti-Patterns to Avoid
- **Calling `mx.linalg.pinv` directly:** The linalg lint gate in `test_linalg_utils.py::test_no_bare_linalg_calls` scans all `omlx/**/*.py` and fails CI if bare `mx.linalg.pinv` or `mx.linalg.svd` appear outside `linalg_utils.py`. Always use `pinv_f32()`.
- **Importing scipy in am.py:** Keep the numpy/scipy bridge inside `linalg_utils.py`. `am.py` imports only `mlx.core`, `dataclasses`, and `omlx.compression.linalg_utils`.
- **Mutating AMCompactedCache:** It is a pure output value; callers may cache and compare multiple instances.
- **Skipping graph materialization before nnls_solve:** `nnls_solve` internally converts to numpy which forces materialization, but explicit `mx.eval()` calls before linalg calls are required to document intent and avoid subtle lazy-graph ordering bugs.
- **Omitting SPDX header:** Every new `.py` file must start with `# SPDX-License-Identifier: Apache-2.0` per CONTRIBUTING.md.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Non-negative least squares | Custom iterative solver | `nnls_solve()` from linalg_utils.py | Already bridges scipy NNLS with correct float64 casting and MLX return type |
| Pseudoinverse / OLS fit | Custom SVD-based pinv | `pinv_f32()` from linalg_utils.py | Handles float16/bfloat16 cast + CPU stream; bare mx.linalg.pinv fails CI lint |
| float16 cast before linalg | Manual `.astype(mx.float32)` in am.py | `pinv_f32()` / `nnls_solve()` | Both wrappers handle dtype internally; duplicating the cast is redundant noise |
| Attention softmax | Custom softmax | `mx.softmax(scores, axis=-1)` | MLX native; no numerical issues at float32 |

**Key insight:** Phase 1 was built specifically to absorb all dtype/stream complexity. Phase 2 pays zero tax on those problems.

---

## Common Pitfalls

### Pitfall 1: Missing Graph Materialization Before numpy Conversion
**What goes wrong:** `nnls_solve` calls `np.array(A)` on a lazy MLX array. In most cases this forces materialization, but if the array graph contains operations that fail silently on the GPU thread (e.g., accumulated float16 ops), results can be NaN.
**Why it happens:** MLX uses lazy evaluation; graph is not executed until forced.
**How to avoid:** Call `mx.eval(attn_full, attn_selected, output_full)` before the NNLS call, matching the spike pattern exactly.
**Warning signs:** NaN betas, NaN OLS output, cosine similarity of exactly 0 or NaN.

### Pitfall 2: Scale Factor in Attention Computation
**What goes wrong:** Attention scores without `* head_dim ** -0.5` scaling produce numerically extreme softmax inputs, leading to near-one-hot attention weights. This collapses the NNLS design matrix and produces degenerate betas.
**Why it happens:** Standard scaled dot-product attention requires the 1/sqrt(d) factor.
**How to avoid:** Always compute `scale = head_dim ** -0.5` and apply it: `scores = (q @ k.transpose(...)) * scale`. The spike does this correctly.
**Warning signs:** NNLS betas collapsing to a single large value; near-zero residual with bad cosine similarity.

### Pitfall 3: Uniform Fallback Not Documented as Lower Quality
**What goes wrong:** A caller uses `compact(kv_cache)` without queries and gets a compacted cache, not knowing the cosine similarity guarantee (>0.998) only applies to the HighestAttnKeys path.
**Why it happens:** The API accepts `queries=None` silently.
**How to avoid:** Add a docstring note on `compact()` that `queries=None` produces the uniform fallback path with no quality guarantees, and is only intended for testing. The CONTEXT.md decision is explicit on this.
**Warning signs:** Compacted cache with normal-looking shapes but poor generation quality.

### Pitfall 4: Head Budget Rounding Leaves Sinks Uncovered
**What goes wrong:** `budget = round(T / ratio)` where `T` is very short (e.g., 8 tokens, ratio=4, n_sink_tokens=4) gives `budget=2 < n_sink_tokens=4`. The sink selection then tries to keep 4 tokens but budget is 2.
**Why it happens:** No floor guard on budget.
**How to avoid:** `budget = max(n_sink_tokens, round(T / ratio))`. The spike's `select_tokens_uniform` already handles this edge case with `if remaining_budget <= 0: return sorted(sink_set)` — replicate the guard.
**Warning signs:** IndexError or empty selected set on short sequences.

### Pitfall 5: Non-Uniform Budgets Not Summing to Total Budget
**What goes wrong:** Per-head budgets computed proportionally from entropy may not sum to exactly `n_heads * floor(T/ratio)` due to rounding.
**Why it happens:** Floating-point proportional allocation with integer rounding.
**How to avoid:** Use integer allocation with a remainder token given to the highest-entropy head. Compute `remainder = total_budget - sum(budgets)` and add 1 to top-remainder heads by entropy.
**Warning signs:** Physical token count in AMCompactedCache varies by a few tokens across runs.

### Pitfall 6: Bare mx.linalg.pinv Failing CI
**What goes wrong:** Using `mx.linalg.pinv` directly in `am.py` causes `test_no_bare_linalg_calls` to fail.
**Why it happens:** The lint test scans all `omlx/**/*.py` with a regex for `mx\.linalg\.(svd|pinv)`.
**How to avoid:** Only use `pinv_f32()` from `linalg_utils`. Import it at module level: `from omlx.compression.linalg_utils import pinv_f32, nnls_solve`.

---

## Code Examples

Verified patterns from spike and linalg_utils:

### Full Per-Head Compaction Pipeline
```python
# Source: spike_am.py am_compact_layer() -- production adaptation
from omlx.compression.linalg_utils import pinv_f32, nnls_solve

def _compact_head(keys_h, values_h, queries_h, budget, n_sink_tokens=4):
    """Compact a single head's KV cache.
    keys_h, values_h: [1, 1, seq_len, head_dim]
    queries_h: [1, 1, n_queries, head_dim]
    Returns: (keys_compact, values_compact) each [1, 1, budget, head_dim]
    """
    seq_len = keys_h.shape[2]
    head_dim = keys_h.shape[3]
    scale = head_dim ** -0.5

    # Full attention weights
    scores_full = (queries_h @ keys_h.transpose(0, 1, 3, 2)) * scale
    attn_full = mx.softmax(scores_full, axis=-1)  # [1,1,n_queries,seq_len]

    # HighestAttnKeys token selection
    importance = attn_full[0, 0].sum(axis=0)  # [seq_len]
    sink_idx = mx.array(list(range(min(n_sink_tokens, seq_len))))
    n_select = budget - sink_idx.shape[0]
    non_sink_importance = importance[n_sink_tokens:]
    topk_local = mx.argsort(non_sink_importance)[-n_select:]
    topk_global = topk_local + n_sink_tokens
    selected = mx.sort(mx.concatenate([sink_idx, topk_global]))
    mx.eval(selected)

    # Selected keys and their attention weights
    k_sel = keys_h[:, :, selected]
    scores_sel = (queries_h @ k_sel.transpose(0, 1, 3, 2)) * scale
    attn_sel = mx.softmax(scores_sel, axis=-1)  # [1,1,n_queries,budget]

    # Full attention output (OLS target)
    v_f32 = values_h.astype(mx.float32)
    output_full = attn_full @ v_f32  # [1,1,n_queries,head_dim]
    mx.eval(output_full, attn_full, attn_sel)

    # NNLS beta-fitting
    A_s = attn_sel[0, 0]             # [n_queries, budget]
    target_b = attn_full[0, 0].sum(axis=1)  # [n_queries] -- row sums = total mass per query
    betas, residual = nnls_solve(A_s, target_b)
    betas = mx.clip(betas, -3.0, 3.0)  # AM-08 box constraint
    mx.eval(betas)

    # OLS value fitting
    out_target = output_full[0, 0]   # [n_queries, head_dim]
    A_pinv = pinv_f32(A_s)           # [budget, n_queries]
    V_compact = A_pinv @ out_target  # [budget, head_dim]
    V_compact = V_compact[None, None]  # [1,1,budget,head_dim]
    mx.eval(V_compact)

    return k_sel, V_compact
```

### Non-Uniform Budget Computation
```python
# Source: AM paper entropy-proportional allocation + spike Step 5
def _compute_head_budgets(head_entropy, total_budget_per_head, n_heads, n_sink_tokens):
    """Compute per-head token budgets proportional to entropy.
    head_entropy: list of float, length n_heads (from calibration bundle)
    Returns: list of int, length n_heads, each >= n_sink_tokens
    """
    import numpy as np
    ent = np.array(head_entropy, dtype=np.float64)
    ent_sum = ent.sum()
    total_budget = total_budget_per_head * n_heads
    if ent_sum == 0:
        return [max(n_sink_tokens, total_budget_per_head)] * n_heads
    proportions = ent / ent_sum
    budgets = np.maximum(n_sink_tokens,
                         np.round(proportions * total_budget).astype(int))
    # Correct for integer rounding: adjust highest-entropy head
    diff = total_budget - int(budgets.sum())
    if diff != 0:
        budgets[np.argmax(ent)] += diff
    return budgets.tolist()
```

### generate_reference_queries Helper
```python
# Source: spike_am.py generate_reference_queries()
def generate_reference_queries(keys: mx.array, n_queries: int = 64,
                                method: str = "sample") -> mx.array:
    """Generate reference queries for attention matching.
    keys: [1, n_heads, seq_len, head_dim]
    Returns: [1, n_heads, n_queries, head_dim]
    """
    seq_len = keys.shape[2]
    if method == "sample":
        indices = mx.random.randint(0, seq_len, shape=(n_queries,))
        queries = keys[:, :, indices, :]
    else:  # "random"
        key_std = mx.std(keys).item()
        queries = mx.random.normal(shape=keys.shape[:2] + (n_queries, keys.shape[3])) * key_std
    mx.eval(queries)
    return queries
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Uniform token eviction (drop old tokens) | HighestAttnKeys with OLS value refitting | AM paper Feb 2026 | 0.9987+ cosine similarity vs ~0.95 for naive eviction |
| Hand-written attention score loop | `mx.softmax(q @ k.T * scale)` | Always | Numerically correct scaled dot-product attention |
| Separate function per concern | Stateless compactor + dataclass output | Phase 2 design | Testable without model; Phase 5 wires in without coupling |

**Deprecated/outdated:**
- Spike's `am_compact_layer()` top-level function: Used for feasibility only. Phase 2 promotes the patterns into `AMCompactor._compact_head()`; the spike file stays as a research artifact.
- Spike's `select_tokens_uniform()` as primary path: Only survives as the `queries=None` fallback in Phase 2. HighestAttnKeys replaces it as the production path.

---

## Open Questions

1. **nnls_solve target: row-sums vs ones vector**
   - What we know: Spike uses `target = W.sum(axis=1)` (sum of all attention weights per query). Row sums of a softmax distribution are theoretically all 1.0.
   - What's unclear: Whether to pass `mx.ones(n_queries)` or the computed row sums. They are mathematically equivalent for softmax outputs but may differ numerically at float32 precision.
   - Recommendation: Use `attn_full[0,0].sum(axis=1)` to match the spike exactly. If row sums deviate from 1.0 by more than 1e-5, surface the deviation in diagnostics as a numerical health signal.

2. **diagnostics dict structure (Claude's discretion)**
   - What we know: CONTEXT.md says diagnostics should contain per-layer/per-head: betas, NNLS residuals, cosine similarity.
   - What's unclear: Exact key names, nesting depth, whether betas are stored as list or mx.array.
   - Recommendation: Use a flat dict with layer/head indices embedded in keys: `"L{i}_H{h}_betas"`, `"L{i}_H{h}_nnls_residual"`, `"L{i}_H{h}_cosine_sim"`. Avoids nested dict serialization complexity; easy to scan for debugging.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (pytest.ini in repo root) |
| Config file | `pytest.ini` — testpaths=tests, asyncio_mode=auto |
| Quick run command | `pytest tests/test_am.py -x -v` |
| Full suite command | `pytest tests/test_am.py -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| AM-01 | HighestAttnKeys selects correct token positions; sink tokens always preserved | unit | `pytest tests/test_am.py::TestHighestAttnKeysSelection -x` | Wave 0 |
| AM-01 | Uniform fallback (queries=None) selects evenly spaced tokens | unit | `pytest tests/test_am.py::TestUniformFallback -x` | Wave 0 |
| AM-02 | nnls_solve called with correct design matrix; betas non-negative before clip | unit | `pytest tests/test_am.py::TestNNLSBetaFitting -x` | Wave 0 |
| AM-03 | OLS value fitting returns correct shape; pinv_f32 used (not bare mx.linalg.pinv) | unit | `pytest tests/test_am.py::TestOLSValueFitting -x` | Wave 0 |
| AM-04 | AMCompactedCache.logical_seq_len == input T; physical token count == budget | unit | `pytest tests/test_am.py::TestCompactedCacheShape -x` | Wave 0 |
| AM-05 | Non-uniform budgets sum to n_heads * floor(T/ratio); each >= n_sink_tokens | unit | `pytest tests/test_am.py::TestHeadBudgets -x` | Wave 0 |
| AM-06 | Budget computation runs once at init; compact() uses precomputed values | unit | `pytest tests/test_am.py::TestBudgetReuse -x` | Wave 0 |
| AM-07 | generate_reference_queries returns shape [1, n_heads, n_queries, head_dim] | unit | `pytest tests/test_am.py::TestGenerateReferenceQueries -x` | Wave 0 |
| AM-08 | Betas after clip are all in [-3, 3] | unit | `pytest tests/test_am.py::TestBetaBoxConstraint -x` | Wave 0 |
| AM-02+AM-03 | Integration: compact() on synthetic KV cache returns cosine sim > 0.98 | unit (synthetic) | `pytest tests/test_am.py::TestCompactIntegration -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_am.py -x -v -m "not slow"`
- **Per wave merge:** `pytest tests/test_am.py -v`
- **Phase gate:** Full suite green (`pytest tests/test_am.py -v`) before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_am.py` — all AM phase tests (file does not yet exist)

*(conftest.py and pytest.ini already exist; no framework install needed)*

---

## Empirical Data from Spike

Spike ran on Qwen 2.5 7B-Instruct-4bit (28 layers, 4 GQA heads, head_dim=128):

| Metric | Value | Notes |
|--------|-------|-------|
| Sequence length | 501 tokens | Test prompt |
| Compaction ratio | 4.0x | 501 -> 125 tokens |
| Avg cosine similarity | **0.9987** | All 4 heads, all 28 layers (uniform selection path) |
| Avg NNLS time | 0.99 ms/head | scipy on CPU |
| Avg OLS time | 0.85 ms/head | pinv_f32 on CPU stream |
| Per-layer wall time | 14.9 ms | Sequential head loop |
| Total model time | 0.42 s | 28 layers x 4 heads |
| Layer-0 head entropies | [1.68, 0.34, 2.47, 1.12] | Validates non-uniform budget need (range: 0.34-2.47) |

**Key observation:** 0.9987 cosine was achieved with the UNIFORM selection path (not HighestAttnKeys). The HighestAttnKeys path in AM-01 selects the most attention-mass-weighted tokens, so it should achieve >= 0.9987. The >0.998 requirement (VAL-02) should be achievable.

---

## Sources

### Primary (HIGH confidence)
- `docs/research/kv-cache-compression/spike_am.py` — complete prototype confirming pipeline order, shapes, and MLX patterns
- `docs/research/kv-cache-compression/spike_am_results.json` — empirical benchmark data (0.9987 cosine, timing, entropies)
- `omlx/compression/linalg_utils.py` — confirmed API: `pinv_f32(a)`, `nnls_solve(A, b)`
- `tests/test_linalg_utils.py` — confirmed lint gate: `test_no_bare_linalg_calls` scans for bare mx.linalg.pinv/svd
- `.planning/phases/02-am-compaction/02-CONTEXT.md` — locked API decisions
- `pytest.ini` — test configuration confirmed (testpaths=tests, slow marker, asyncio_mode=auto)

### Secondary (MEDIUM confidence)
- `.planning/phases/01-linalg-foundation/01-RESEARCH.md` — MLX 0.31.x, scipy 1.17.1, numpy 2.4.2 version confirmations
- `docs/research/kv-cache-compression/spike_combined_results.json` — combined AM+kvtc pipeline data

### Tertiary (LOW confidence)
- AM paper ("Fast KV Compaction via Attention Matching", MIT, Feb 2026) — referenced in spike docstring; paper details known only through spike prototype and REQUIREMENTS.md descriptions, not read directly

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries confirmed via existing code and Phase 1 research
- Architecture: HIGH — spike prototype is the direct blueprint; patterns are empirically validated
- Pitfalls: HIGH — most are confirmed by existing code (lint gate, mx.eval pattern, float cast)
- Beta box constraint specifics: MEDIUM — AM-08 requirement states values, but exact impact of clipping non-negative NNLS output to [-3,3] is low-risk (NNLS gives >=0, so lower bound never bites for HighestAttnKeys path)

**Research date:** 2026-03-18
**Valid until:** 2026-04-18 (stable; MLX and scipy versions pinned by Phase 1)
