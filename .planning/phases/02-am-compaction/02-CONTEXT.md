# Phase 2: AM Compaction - Context

**Gathered:** 2026-03-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement `omlx/compression/am.py` — a stateless `AMCompactor` class that takes a full model's KV cache (all layers, all heads) and returns a compacted cache with fewer physical tokens while preserving logical sequence length T for correct RoPE position indices. This is pure in-memory tensor math. No cache system integration, no file I/O, no pipeline wiring. Phase 5 (pipeline assembly) will wire the compactor into triggers.

</domain>

<decisions>
## Implementation Decisions

### compact() API

- `compact(kv_cache, ratio=4.0, queries=None)` — full model call: takes a list of `(keys, values)` tuples for all layers, returns a single `AMCompactedCache` object
- `ratio` is a single float applied uniformly; per-head budget allocation is computed internally from entropy curves (or uniform fallback)
- `queries` is caller-provided — `AMCompactor` stays stateless; a `generate_reference_queries()` helper lives in `am.py` for convenience but is not called internally
- If `queries=None`, compact() falls back to uniform token selection (for testing only — documented as lower quality)

### AMCompactor initialization

- `AMCompactor(head_entropy=None, n_sink_tokens=4)` — both parameters optional
- `head_entropy`: per-head entropy curves from calibration bundle; if `None`, uniform budgets are used (fully testable standalone without Phase 4)
- When entropy curves are provided: per-head token budget is proportional to entropy — higher-entropy heads get more of the total budget
- `n_sink_tokens=4` matches the spike and aligns with Phase 3 kvtc's s=4 exemption

### Token selection

- HighestAttnKeys: select the t tokens with highest summed attention weight across reference queries (after always preserving the first `n_sink_tokens` sinks)
- When `queries=None`: fall back to uniform interval selection — explicitly documented as a testing convenience, not production path
- Beta box constraints: values in [-3, 3] per AM-08; keys with beta < -7 are pruned when using OMP path

### Output — AMCompactedCache dataclass

- Fields: `layers: list[tuple[mx.array, mx.array]]` (compacted keys/values per layer), `logical_seq_len: int`, `diagnostics: dict | None`
- `logical_seq_len` is the original sequence length T — preserved for correct RoPE position indices
- `diagnostics` is `None` by default; populated in debug mode with per-layer/per-head: betas, NNLS residuals, cosine similarity between compacted and original attention output

### Claude's Discretion

- Exact `diagnostics` dict key names and structure
- Internal chunking/batching strategy for multi-head NNLS calls
- Whether `generate_reference_queries()` exposes the "sample" vs "random" method as a parameter

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets

- `omlx/compression/linalg_utils.py`: `pinv_f32` for OLS value-fitting (`A_pinv = pinv_f32(A_selected); V_compact = A_pinv @ output_full`), `nnls_solve` for beta-fitting — both ready to use
- `docs/research/kv-cache-compression/spike_am.py`: complete prototype for `am_compact_layer()`, `nnls_beta_fitting()`, `compute_attention_weights()`, `compute_attention_output()` — Phase 2 formalizes these patterns

### Established Patterns

- MLX KV cache: `layer_cache.state = (keys, values)` where shapes are `[1, n_heads, seq_len, head_dim]`
- All linalg ops (`pinv_f32`, `nnls_solve`) handle float32 cast + CPU stream internally — callers never pass a stream
- MLX lazy graph materialization: `mx.eval()` calls are needed to force computation; documented in spike as standard MLX pattern
- `__init__.py` stays empty — callers will import: `from omlx.compression.am import AMCompactor`

### Integration Points

- `am.py` is a sibling of `linalg_utils.py` in `omlx/compression/`
- Phase 4 (calibration CLI) will produce a bundle with `head_entropy` data that feeds `AMCompactor(head_entropy=...)`
- Phase 5 (pipeline assembly) calls `compact(kv_cache, ratio=..., queries=...)` and passes the resulting `AMCompactedCache` to kvtc

</code_context>

<specifics>
## Specific Ideas

- The spike's `am_compact_layer()` structure (select → compute attn → NNLS beta-fit → OLS value-fit → verify) is the right internal pipeline order; Phase 2 should follow it closely
- Spike's multi-head loop runs sequentially per head; no pressure to parallelize in Phase 2

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 02-am-compaction*
*Context gathered: 2026-03-18*
