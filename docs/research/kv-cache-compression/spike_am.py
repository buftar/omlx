#!/usr/bin/env python3
"""
Research Spike: Attention Matching (AM) KV cache compaction on MLX.

Tests the AM compaction pipeline from "Fast KV Compaction via Attention Matching"
(MIT, Feb 2026) on Apple Silicon:
1. Extract KV cache from a real MLX model prefill
2. Generate reference queries for attention matching
3. NNLS beta-fitting (attention mass matching)
4. OLS value fitting via pseudoinverse
5. Measure attention output error before/after compaction

Target model: Qwen2.5-7B-Instruct-4bit

Note: mx.eval() calls in this file force materialization of lazy MLX computation
graphs. This is the standard MLX pattern — not arbitrary code evaluation.
"""

import time
import sys
import os
import json

import numpy as np
import mlx.core as mx
import mlx.nn as nn


# ── 1. Extract KV cache + attention weights ──────────────────────────────────

def extract_kv_and_attention(model_path: str, prompt: str = "The quick brown fox jumps over the lazy dog. " * 50):
    """Run prefill and extract KV cache tensors."""
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    print(f"Loading model from {model_path}...")
    t0 = time.perf_counter()
    model, tokenizer = load(model_path)
    print(f"  Model loaded in {time.perf_counter() - t0:.1f}s")

    tokens = mx.array(tokenizer.encode(prompt))
    seq_len = tokens.shape[0]
    print(f"  Prompt tokens: {seq_len}")

    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    model(tokens[None], cache=cache)
    # Force lazy MLX graph materialization
    mx.eval(cache[0].state[0])
    print(f"  Prefill completed in {time.perf_counter() - t0:.2f}s")

    kv_pairs = []
    for layer_cache in cache:
        keys, values = layer_cache.state
        kv_pairs.append((keys, values))

    k0, v0 = kv_pairs[0]
    print(f"  Layers: {len(kv_pairs)}")
    print(f"  Layer 0 keys shape: {k0.shape}")
    print(f"  dtype: {k0.dtype}")

    return kv_pairs, seq_len, model, tokenizer


# ── 2. Reference Query Generation ────────────────────────────────────────────

def generate_reference_queries(keys, n_queries=64, method="random"):
    """
    Generate reference queries for attention matching.

    AM paper uses calibration data or random projections.
    We use random queries sampled from the key distribution.

    Args:
        keys: [1, n_heads, seq_len, head_dim]
        n_queries: number of reference queries
        method: "random" or "sample"

    Returns:
        queries: [1, n_heads, n_queries, head_dim]
    """
    n_heads = keys.shape[1]
    seq_len = keys.shape[2]
    head_dim = keys.shape[3]

    if method == "sample":
        # Sample from existing key positions
        indices = mx.random.randint(0, seq_len, shape=(n_queries,))
        queries = keys[:, :, indices, :]
    else:
        # Random Gaussian queries scaled to match key statistics
        key_std = mx.std(keys).item()
        queries = mx.random.normal(shape=(1, n_heads, n_queries, head_dim)) * key_std

    return queries


# ── 3. Attention Computation ─────────────────────────────────────────────────

def compute_attention_weights(queries, keys, scale=None):
    """
    Compute softmax attention weights.

    Args:
        queries: [1, n_heads, n_queries, head_dim]
        keys: [1, n_heads, seq_len, head_dim]

    Returns:
        weights: [1, n_heads, n_queries, seq_len]
    """
    # Cast to float32 to avoid overflow in attention score computation
    q = queries.astype(mx.float32) if queries.dtype == mx.float16 else queries
    k = keys.astype(mx.float32) if keys.dtype == mx.float16 else keys
    head_dim = q.shape[-1]
    if scale is None:
        scale = head_dim ** -0.5

    scores = (q @ k.transpose(0, 1, 3, 2)) * scale
    weights = mx.softmax(scores, axis=-1)
    return weights


def compute_attention_output(weights, values):
    """
    Compute attention output: weights @ values.

    Args:
        weights: [1, n_heads, n_queries, seq_len]
        values: [1, n_heads, seq_len, head_dim]

    Returns:
        output: [1, n_heads, n_queries, head_dim]
    """
    v = values.astype(mx.float32) if values.dtype == mx.float16 else values
    return weights @ v


# ── 4. NNLS Beta-Fitting (Attention Mass Matching) ───────────────────────────

def nnls_beta_fitting(attn_weights, selected_indices, head_idx=0):
    """
    Fit non-negative betas so that sum(beta_j * delta(selected_j)) approximates
    the attention mass distribution.

    AM paper: for each reference query q, find beta >= 0 such that
    A_full @ 1 approx A_selected @ beta (mass matching)

    Args:
        attn_weights: [1, n_heads, n_queries, seq_len] full attention weights
        selected_indices: indices of selected tokens
        head_idx: which head to fit

    Returns:
        betas: [len(selected_indices)] non-negative weights
        residual: fitting residual
    """
    from scipy.optimize import nnls

    # Extract per-head: [n_queries, seq_len]
    W = np.array(attn_weights[0, head_idx], dtype=np.float64)
    n_queries, seq_len = W.shape

    # Target: sum of attention mass per query across all tokens
    target = W.sum(axis=1)  # [n_queries]

    # Design matrix: attention weights at selected positions
    A = W[:, selected_indices]

    # Solve: A @ beta approx target, beta >= 0
    betas, residual = nnls(A, target)

    return betas, residual


# ── 5. Full AM Compaction Pipeline ───────────────────────────────────────────

def select_tokens_uniform(seq_len, budget, sink_tokens=4):
    """Select tokens: keep sink tokens + uniformly sample the rest."""
    sink_set = set(range(min(sink_tokens, seq_len)))

    remaining_budget = budget - len(sink_set)
    if remaining_budget <= 0:
        return sorted(sink_set)

    candidates = [i for i in range(sink_tokens, seq_len)]
    if remaining_budget >= len(candidates):
        return list(range(seq_len))

    step = len(candidates) / remaining_budget
    selected = [candidates[int(i * step)] for i in range(remaining_budget)]

    return sorted(sink_set | set(selected))


def am_compact_layer(keys, values, queries, budget, head_idx=0, sink_tokens=4):
    """
    Run AM compaction on a single layer+head.

    Pipeline:
    1. Select token positions (uniform + sinks)
    2. NNLS beta-fitting on attention masses
    3. OLS value fitting via pseudoinverse

    Args:
        keys: [1, n_heads, seq_len, head_dim]
        values: [1, n_heads, seq_len, head_dim]
        queries: [1, n_heads, n_queries, head_dim]
        budget: target number of tokens after compaction
        head_idx: which head to compact

    Returns:
        keys_compact: [1, 1, budget, head_dim]
        values_compact: [1, 1, budget, head_dim]
        info: dict with diagnostics
    """
    seq_len = keys.shape[2]
    head_dim = keys.shape[3]

    # Step 1: Select tokens
    selected = select_tokens_uniform(seq_len, budget, sink_tokens)
    n_selected = len(selected)
    sel_idx = mx.array(selected)

    # Extract single head
    k_head = keys[:, head_idx:head_idx+1]
    v_head = values[:, head_idx:head_idx+1]

    # Selected KV
    k_selected = k_head[:, :, sel_idx]
    v_selected = v_head[:, :, sel_idx]

    # Step 2: Compute full and selected attention
    q_head = queries[:, head_idx:head_idx+1]

    attn_full = compute_attention_weights(q_head, k_head)
    attn_selected = compute_attention_weights(q_head, k_selected)

    # Full attention output (target)
    output_full = compute_attention_output(attn_full, v_head)
    # Force lazy MLX graph materialization
    mx.eval(output_full, attn_full, attn_selected)

    # Step 3: NNLS beta-fitting
    t0 = time.perf_counter()
    attn_full_np = np.array(attn_full, dtype=np.float64)
    betas, nnls_residual = nnls_beta_fitting(attn_full_np, selected, head_idx=0)
    nnls_time = time.perf_counter() - t0

    # Step 4: OLS value fitting via pseudoinverse
    # Target: attn_selected @ V_compact approx output_full
    # V_compact = pinv(attn_selected) @ output_full
    t0 = time.perf_counter()
    A_s = attn_selected[0, 0]  # [n_queries, n_selected]
    target = output_full[0, 0]  # [n_queries, head_dim]

    # Use MLX pseudoinverse
    A_pinv = mx.linalg.pinv(A_s, stream=mx.cpu)  # [n_selected, n_queries]
    V_compact = A_pinv @ target  # [n_selected, head_dim]
    V_compact = V_compact[None, None]  # [1, 1, n_selected, head_dim]
    # Force lazy MLX graph materialization
    mx.eval(V_compact)
    ols_time = time.perf_counter() - t0

    # Step 5: Verify — compute compacted attention output
    output_compact = compute_attention_output(attn_selected, V_compact)
    # Force lazy MLX graph materialization
    mx.eval(output_compact)

    # Measure error
    output_err = mx.mean((output_full - output_compact) ** 2).item()
    output_cos = mx.mean(
        mx.sum(output_full.reshape(-1, head_dim) * output_compact.reshape(-1, head_dim), axis=-1) /
        (mx.sqrt(mx.sum(output_full.reshape(-1, head_dim) ** 2, axis=-1)) *
         mx.sqrt(mx.sum(output_compact.reshape(-1, head_dim) ** 2, axis=-1)) + 1e-8)
    ).item()

    return k_selected, V_compact, {
        "n_selected": n_selected,
        "nnls_time_ms": nnls_time * 1000,
        "ols_time_ms": ols_time * 1000,
        "nnls_residual": float(nnls_residual),
        "output_mse": output_err,
        "output_cosine": output_cos,
        "betas_nonzero": int(np.sum(betas > 1e-6)),
        "betas_mean": float(np.mean(betas[betas > 1e-6])) if np.any(betas > 1e-6) else 0,
    }


def run_am_spike(model_path: str):
    """Run the full AM compaction feasibility spike."""
    print("=" * 70)
    print("AM Feasibility Spike — Attention Matching KV Compaction on MLX")
    print("=" * 70)

    # Step 1: Extract KV cache
    print("\n── Step 1: Extract KV cache ──")
    kv_pairs, seq_len, model, tokenizer = extract_kv_and_attention(model_path)
    k0, v0 = kv_pairs[0]
    n_heads = k0.shape[1]
    head_dim = k0.shape[3]
    n_layers = len(kv_pairs)

    # Step 2: Generate reference queries
    print("\n── Step 2: Generate reference queries ──")
    n_queries = 64
    queries = generate_reference_queries(k0, n_queries=n_queries, method="sample")
    # Force lazy MLX graph materialization
    mx.eval(queries)
    print(f"  Generated {n_queries} reference queries from key distribution")
    print(f"  Query shape: {queries.shape}")

    # Step 3: Test compaction at different budgets
    print("\n── Step 3: Compaction at different budgets (layer 0, head 0) ──")
    budgets = [seq_len // 2, seq_len // 4, seq_len // 8, seq_len // 16]

    for budget in budgets:
        k_compact, v_compact, info = am_compact_layer(
            k0, v0, queries, budget=budget, head_idx=0
        )
        ratio = seq_len / info["n_selected"]
        print(f"  Budget={budget} ({ratio:.1f}x): "
              f"output_MSE={info['output_mse']:.8f} "
              f"output_cos={info['output_cosine']:.6f} "
              f"NNLS={info['nnls_time_ms']:.1f}ms "
              f"OLS={info['ols_time_ms']:.1f}ms "
              f"betas_nz={info['betas_nonzero']}/{info['n_selected']}")

    # Step 4: Multi-head compaction (all heads, layer 0)
    print(f"\n── Step 4: Multi-head compaction (all {n_heads} heads, layer 0) ──")
    budget = seq_len // 4
    head_results = []
    t0_all_heads = time.perf_counter()

    for h in range(n_heads):
        _, _, info = am_compact_layer(k0, v0, queries, budget=budget, head_idx=h)
        head_results.append(info)

    all_heads_time = time.perf_counter() - t0_all_heads
    avg_mse = np.mean([r["output_mse"] for r in head_results])
    avg_cos = np.mean([r["output_cosine"] for r in head_results])
    avg_nnls = np.mean([r["nnls_time_ms"] for r in head_results])
    avg_ols = np.mean([r["ols_time_ms"] for r in head_results])

    print(f"  Budget={budget} ({seq_len/budget:.0f}x compaction)")
    print(f"  Time for all heads: {all_heads_time:.2f}s")
    print(f"  Avg output MSE: {avg_mse:.8f}")
    print(f"  Avg output cosine: {avg_cos:.6f}")
    print(f"  Avg NNLS time: {avg_nnls:.1f}ms")
    print(f"  Avg OLS time: {avg_ols:.1f}ms")
    print(f"  Estimated per-layer time: {all_heads_time:.2f}s")
    print(f"  Estimated full model ({n_layers} layers): {all_heads_time * n_layers:.1f}s")

    # Step 5: Vary head budgets (non-uniform, as in paper)
    print(f"\n── Step 5: Non-uniform head budgets (layer 0) ──")
    entropies = []
    for h in range(min(n_heads, 8)):
        q_h = queries[:, h:h+1]
        k_h = k0[:, h:h+1]
        attn = compute_attention_weights(q_h, k_h)
        # Force lazy MLX graph materialization
        mx.eval(attn)
        attn_np = np.array(attn[0, 0], dtype=np.float64)
        entropy = -np.sum(attn_np * np.log(attn_np + 1e-12), axis=-1).mean()
        entropies.append(entropy)

    print(f"  Attention entropy (first 8 heads): "
          f"{[f'{e:.2f}' for e in entropies]}")
    print(f"  High-entropy heads need more budget (more spread attention)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Model: {model_path}")
    print(f"  Sequence length: {seq_len} tokens")
    print(f"  Cache shape: {n_layers}L x {n_heads}H x {head_dim}D")
    print(f"  4x compaction ({seq_len} -> {budget} tokens):")
    print(f"    Avg output cosine similarity: {avg_cos:.6f}")
    print(f"    Per-layer time: {all_heads_time:.2f}s")
    print(f"    Estimated full model: {all_heads_time * n_layers:.1f}s")
    print(f"  MLX ops used: svd=no, pinv=yes, nnls=scipy")
    blockers = "No MLX blockers detected" if avg_cos > 0.99 else "WARNING: low cosine similarity"
    print(f"  {blockers}")

    return {
        "seq_len": seq_len,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "budget": budget,
        "compaction_ratio": seq_len / budget,
        "avg_output_mse": float(avg_mse),
        "avg_output_cosine": float(avg_cos),
        "per_layer_time_s": all_heads_time,
        "estimated_full_model_time_s": all_heads_time * n_layers,
        "avg_nnls_time_ms": float(avg_nnls),
        "avg_ols_time_ms": float(avg_ols),
        "head_entropies": entropies,
    }


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/tonysina/models/Qwen2.5-7B-Instruct-4bit"
    results = run_am_spike(model_path)

    output_path = os.path.join(os.path.dirname(__file__), "spike_am_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
