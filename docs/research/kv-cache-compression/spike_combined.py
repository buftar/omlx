#!/usr/bin/env python3
"""
Research Spike: Combined AM + kvtc Pipeline.

Tests the full two-stage compression pipeline:
  Stage 1: AM compaction (reduce token count)
  Stage 2: kvtc compression (reduce byte size of compacted cache)

Measures the combined compression ratio and reconstruction quality.

Note: All mx.eval() calls in this file serve to force materialization of lazy
MLX computation graphs — this is the standard MLX API pattern, not code eval.
"""

import time
import sys
import os
import json

import numpy as np
import mlx.core as mx

# Import from sibling spike scripts
sys.path.insert(0, os.path.dirname(__file__))
from spike_kvtc import (
    pca_calibrate, pca_project, pca_reconstruct,
    dp_quantize, compress_zstd, decompress_zstd,
    measure_reconstruction_error,
)
from spike_am import (
    extract_kv_and_attention, generate_reference_queries,
    compute_attention_weights, compute_attention_output,
    am_compact_layer,
)


def _materialize(*arrays):
    """Force materialization of lazy MLX computation graphs."""
    mx.eval(*arrays)


def run_combined_spike(model_path: str):
    """Run AM compaction followed by kvtc compression."""
    print("=" * 70)
    print("Combined Pipeline Spike — AM Compaction + kvtc Compression")
    print("=" * 70)

    # Step 1: Extract KV cache
    print("\n── Step 1: Extract KV cache ──")
    kv_pairs, seq_len, model, tokenizer = extract_kv_and_attention(model_path)
    k0, v0 = kv_pairs[0]
    n_heads = k0.shape[1]
    head_dim = k0.shape[3]
    n_layers = len(kv_pairs)

    bytes_per_element = 2
    original_bytes = n_layers * 2 * n_heads * seq_len * head_dim * bytes_per_element
    print(f"  Original KV cache: {original_bytes / 1024 / 1024:.2f} MB")

    # Step 2: AM compaction (layer 0 as representative)
    print("\n── Step 2: AM Compaction (4x token reduction) ──")
    budget = seq_len // 4
    queries = generate_reference_queries(k0, n_queries=64, method="sample")
    _materialize(queries)

    # Compact all heads for layer 0
    compacted_keys_list = []
    compacted_values_list = []

    t0 = time.perf_counter()
    for h in range(n_heads):
        k_compact, v_compact, info = am_compact_layer(
            k0, v0, queries, budget=budget, head_idx=h
        )
        compacted_keys_list.append(k_compact)
        compacted_values_list.append(v_compact)
    am_time = time.perf_counter() - t0

    # Stack heads back: [1, n_heads, budget, head_dim]
    compacted_keys = mx.concatenate(compacted_keys_list, axis=1)
    compacted_values = mx.concatenate(compacted_values_list, axis=1)
    _materialize(compacted_keys, compacted_values)

    compacted_bytes = 2 * n_heads * budget * head_dim * bytes_per_element  # Per layer
    am_ratio = (n_heads * seq_len) / (n_heads * budget)
    print(f"  Compacted: {seq_len} -> {budget} tokens ({am_ratio:.1f}x token reduction)")
    print(f"  Compacted layer size: {compacted_bytes / 1024:.1f} KB (from {original_bytes / n_layers / 1024:.1f} KB)")
    print(f"  AM time (1 layer): {am_time:.2f}s")

    # Step 3: kvtc compression on compacted cache
    print("\n── Step 3: kvtc Compression of compacted cache ──")
    compacted_kv = [(compacted_keys, compacted_values)]
    n_comp = head_dim // 2

    t0 = time.perf_counter()
    pca_bases = pca_calibrate(compacted_kv)
    pca_time = time.perf_counter() - t0

    basis_k = pca_bases[0]["key_basis"][:, :n_comp]
    mean_k = pca_bases[0]["key_mean"]
    basis_v = pca_bases[0]["value_basis"][:, :n_comp]
    mean_v = pca_bases[0]["value_mean"]

    # Compress keys and values
    t0 = time.perf_counter()
    total_kvtc_compressed = 0

    for name, tensor, basis, mean in [
        ("key", compacted_keys, basis_k, mean_k),
        ("value", compacted_values, basis_v, mean_v),
    ]:
        coeffs = pca_project(tensor, basis, mean)
        _materialize(coeffs)
        coeffs_np = np.array(coeffs.reshape(-1, n_comp), dtype=np.float32)

        quantized_np, codebooks, indices = dp_quantize(coeffs_np, n_bits=4)

        indices_flat = indices.astype(np.uint8).flatten()
        if len(indices_flat) % 2 != 0:
            indices_flat = np.append(indices_flat, 0)
        packed = (indices_flat[0::2] << 4) | indices_flat[1::2]

        packed_bytes = packed.tobytes()
        codebook_bytes = np.array(codebooks, dtype=np.float16).tobytes()
        compressed = compress_zstd(packed_bytes + codebook_bytes, level=3)
        total_kvtc_compressed += len(compressed)

    kvtc_time = time.perf_counter() - t0

    # Add metadata overhead
    basis_bytes = 2 * head_dim * n_comp * 2  # K and V bases, float16
    mean_bytes = 2 * head_dim * 2
    total_compressed = total_kvtc_compressed + basis_bytes + mean_bytes

    kvtc_ratio = compacted_bytes / total_compressed
    combined_ratio = (original_bytes / n_layers) / total_compressed

    print(f"  PCA calibration: {pca_time:.2f}s")
    print(f"  kvtc compression: {kvtc_time:.2f}s")
    print(f"  Compressed size: {total_compressed / 1024:.1f} KB")
    print(f"  kvtc ratio (on compacted): {kvtc_ratio:.1f}x")
    print(f"  Combined ratio (vs original): {combined_ratio:.1f}x")

    # Step 4: Round-trip quality check
    print("\n── Step 4: End-to-end quality verification ──")
    # Compute original attention output
    q_test = queries[:, 0:1]
    k_orig = k0[:, 0:1]
    v_orig = v0[:, 0:1]

    attn_orig = compute_attention_weights(q_test, k_orig)
    output_orig = compute_attention_output(attn_orig, v_orig)
    _materialize(output_orig)

    # Reconstruct from kvtc
    coeffs_k = pca_project(compacted_keys[:, 0:1], basis_k, mean_k)
    _materialize(coeffs_k)
    coeffs_k_np = np.array(coeffs_k.reshape(-1, n_comp), dtype=np.float32)
    quant_k_np, cb_k, idx_k = dp_quantize(coeffs_k_np, n_bits=4)
    recon_coeffs_k = mx.array(quant_k_np.reshape(1, 1, budget, n_comp))
    recon_keys = pca_reconstruct(recon_coeffs_k, basis_k, mean_k)

    coeffs_v = pca_project(compacted_values[:, 0:1], basis_v, mean_v)
    _materialize(coeffs_v)
    coeffs_v_np = np.array(coeffs_v.reshape(-1, n_comp), dtype=np.float32)
    quant_v_np, cb_v, idx_v = dp_quantize(coeffs_v_np, n_bits=4)
    recon_coeffs_v = mx.array(quant_v_np.reshape(1, 1, budget, n_comp))
    recon_values = pca_reconstruct(recon_coeffs_v, basis_v, mean_v)

    _materialize(recon_keys, recon_values)

    # Compute attention with reconstructed compacted cache
    attn_recon = compute_attention_weights(q_test, recon_keys)
    output_recon = compute_attention_output(attn_recon, recon_values)
    _materialize(output_recon)

    # Compare outputs
    output_mse = mx.mean((output_orig - output_recon) ** 2).item()
    dot = mx.sum(output_orig.reshape(-1, head_dim) * output_recon.reshape(-1, head_dim), axis=-1)
    norm_o = mx.sqrt(mx.sum(output_orig.reshape(-1, head_dim) ** 2, axis=-1))
    norm_r = mx.sqrt(mx.sum(output_recon.reshape(-1, head_dim) ** 2, axis=-1))
    output_cos = mx.mean(dot / (norm_o * norm_r + 1e-8)).item()

    print(f"  End-to-end attention output MSE: {output_mse:.8f}")
    print(f"  End-to-end attention output cosine: {output_cos:.6f}")

    # Summary
    print("\n" + "=" * 70)
    print("COMBINED PIPELINE SUMMARY")
    print("=" * 70)
    print(f"  Model: {model_path}")
    print(f"  Original: {seq_len} tokens, {original_bytes / n_layers / 1024:.1f} KB/layer")
    print(f"  After AM:   {budget} tokens ({am_ratio:.0f}x token reduction)")
    print(f"  After kvtc: {total_compressed / 1024:.1f} KB/layer ({kvtc_ratio:.1f}x byte reduction)")
    print(f"  COMBINED:   {combined_ratio:.0f}x total compression")
    print(f"  Extrapolated full model ({n_layers} layers):")
    print(f"    Original:   {original_bytes / 1024 / 1024:.1f} MB")
    print(f"    Compressed: {total_compressed * n_layers / 1024 / 1024:.2f} MB")
    print(f"  Latency:")
    print(f"    AM compaction: {am_time:.2f}s/layer")
    print(f"    kvtc compress: {kvtc_time:.2f}s/layer")
    print(f"    Total: {am_time + kvtc_time:.2f}s/layer")
    print(f"  Quality:")
    print(f"    Output cosine similarity: {output_cos:.6f}")
    print(f"    Output MSE: {output_mse:.8f}")

    return {
        "seq_len": seq_len,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "am_budget": budget,
        "am_ratio": am_ratio,
        "kvtc_ratio": kvtc_ratio,
        "combined_ratio": combined_ratio,
        "original_bytes_per_layer": original_bytes // n_layers,
        "compressed_bytes_per_layer": total_compressed,
        "am_time_s": am_time,
        "kvtc_time_s": kvtc_time,
        "output_mse": output_mse,
        "output_cosine": output_cos,
    }


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/tonysina/models/Qwen2.5-7B-Instruct-4bit"
    results = run_combined_spike(model_path)

    output_path = os.path.join(os.path.dirname(__file__), "spike_combined_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
