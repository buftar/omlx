#!/usr/bin/env python3
"""
Research Spike: KV Cache Transform Coding (kvtc) feasibility on MLX.

Tests the kvtc pipeline from the ICLR 2026 paper on Apple Silicon:
1. Extract KV cache from a real MLX model prefill
2. PCA calibration (randomized SVD) on KV tensors
3. PCA projection + reconstruction error measurement
4. DP-optimized quantization of PCA coefficients
5. Lossless compression (zstd) of quantized coefficients
6. Full round-trip: compress → decompress → measure error

Target model: Qwen2.5-7B-Instruct-4bit

Note: mx.eval() calls in this script are used to force materialization of
lazy MLX tensors — this is standard MLX usage, not arbitrary code evaluation.
"""

import time
import sys
import os
import json

import numpy as np
import mlx.core as mx
import mlx.nn as nn


# ── 1. Extract KV cache from model prefill ──────────────────────────────────

def extract_kv_cache(model_path: str, prompt: str = "The quick brown fox jumps over the lazy dog. " * 50):
    """Run prefill on a model and extract KV cache tensors."""
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    print(f"Loading model from {model_path}...")
    t0 = time.perf_counter()
    model, tokenizer = load(model_path)
    print(f"  Model loaded in {time.perf_counter() - t0:.1f}s")

    # Tokenize
    tokens = mx.array(tokenizer.encode(prompt))
    seq_len = tokens.shape[0]
    print(f"  Prompt tokens: {seq_len}")

    # Create cache and run prefill
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    model(tokens[None], cache=cache)
    # Force materialization of lazy MLX tensors
    mx.eval(cache[0].state[0])
    print(f"  Prefill completed in {time.perf_counter() - t0:.2f}s")

    # Extract per-layer KV tensors
    kv_pairs = []
    for layer_idx, layer_cache in enumerate(cache):
        keys, values = layer_cache.state
        kv_pairs.append((keys, values))

    # Report shape info
    k0, v0 = kv_pairs[0]
    print(f"  Layers: {len(kv_pairs)}")
    print(f"  Layer 0 keys shape: {k0.shape}")  # [1, n_heads, seq_len, head_dim]
    print(f"  Layer 0 values shape: {v0.shape}")
    print(f"  dtype: {k0.dtype}")

    return kv_pairs, seq_len


# ── 2. PCA Calibration via Randomized SVD ────────────────────────────────────

def pca_calibrate(kv_pairs, n_components=None, fraction=0.5):
    """
    Compute PCA basis per layer per type (K/V) using randomized SVD.

    kvtc paper: cross-layer PCA with Procrustes alignment.
    Simplified here: per-layer PCA (cross-head, since heads share subspace).

    Args:
        kv_pairs: list of (keys, values) tuples, each [1, n_heads, seq_len, head_dim]
        n_components: number of PCA components to keep (default: head_dim)
        fraction: fraction of variance to target if n_components not specified

    Returns:
        pca_bases: list of dicts with 'key_basis', 'value_basis', 'key_mean', 'value_mean'
                   each basis is [head_dim, n_components]
    """
    pca_bases = []
    n_layers = len(kv_pairs)

    for layer_idx in range(n_layers):
        keys, values = kv_pairs[layer_idx]
        # keys shape: [1, n_heads, seq_len, head_dim]
        n_heads = keys.shape[1]
        seq_len = keys.shape[2]
        head_dim = keys.shape[3]

        if n_components is None:
            n_components = head_dim  # Keep all for now, truncate later

        layer_bases = {}
        for name, tensor in [("key", keys), ("value", values)]:
            # Reshape to [n_heads * seq_len, head_dim] for cross-head PCA
            data = tensor.reshape(-1, head_dim)  # [n_heads * seq_len, head_dim]

            # Cast to float32 for SVD (MLX SVD requires float32+)
            data = data.astype(mx.float32)

            # Center the data
            mean = mx.mean(data, axis=0, keepdims=True)  # [1, head_dim]
            centered = data - mean

            # SVD: compute on centered data
            t0 = time.perf_counter()
            U, S, Vt = mx.linalg.svd(centered, stream=mx.cpu)
            # Force materialization of lazy MLX tensors
            mx.eval(U, S, Vt)
            svd_time = time.perf_counter() - t0

            # Vt: [min(N, head_dim), head_dim] — rows are principal components
            # Basis: [head_dim, n_components]
            basis = Vt[:min(n_components, Vt.shape[0])].T

            # Compute explained variance ratio
            total_var = mx.sum(S ** 2)
            explained_var = mx.sum(S[:min(n_components, S.shape[0])] ** 2) / total_var

            layer_bases[f"{name}_basis"] = basis
            layer_bases[f"{name}_mean"] = mean.squeeze(0)
            layer_bases[f"{name}_singular_values"] = S

            if layer_idx == 0:
                print(f"  Layer {layer_idx} {name}: SVD in {svd_time:.3f}s, "
                      f"basis shape {basis.shape}, "
                      f"explained variance (all components): {explained_var.item():.4f}")

        pca_bases.append(layer_bases)

    return pca_bases


def pca_project(tensor, basis, mean):
    """Project tensor to PCA coefficients.

    Args:
        tensor: [1, n_heads, seq_len, head_dim]
        basis: [head_dim, n_components]
        mean: [head_dim]

    Returns:
        coefficients: [1, n_heads, seq_len, n_components]
    """
    t = tensor.astype(mx.float32) if tensor.dtype == mx.float16 else tensor
    centered = t - mean
    coeffs = centered @ basis
    return coeffs


def pca_reconstruct(coeffs, basis, mean):
    """Reconstruct from PCA coefficients.

    Args:
        coeffs: [1, n_heads, seq_len, n_components]
        basis: [head_dim, n_components]
        mean: [head_dim]

    Returns:
        reconstructed: [1, n_heads, seq_len, head_dim]
    """
    reconstructed = coeffs @ basis.T + mean
    return reconstructed


# ── 3. DP-Optimized Quantization ─────────────────────────────────────────────

def dp_quantize(coeffs_np, n_bits=4):
    """
    DP-optimized non-uniform quantization of PCA coefficients.

    Simplified version of kvtc's DP quantization:
    - Find optimal quantization levels via Lloyd's algorithm
    - Minimize MSE distortion for given bit budget

    Args:
        coeffs_np: numpy array of PCA coefficients [N, n_components]
        n_bits: bits per coefficient

    Returns:
        quantized: quantized coefficients (same shape)
        codebook: quantization levels per component
        indices: quantization indices per component
    """
    n_levels = 2 ** n_bits
    n_samples, n_components = coeffs_np.shape

    codebooks = []
    all_indices = np.zeros_like(coeffs_np, dtype=np.int32)

    for comp_idx in range(n_components):
        col = coeffs_np[:, comp_idx]
        sorted_vals = np.sort(col)

        # Percentile-based initialization + Lloyd's algorithm
        percentiles = np.linspace(0, 100, n_levels + 1)
        boundaries = np.percentile(sorted_vals, percentiles)

        # Lloyd's algorithm (3 iterations is enough for convergence)
        centroids = (boundaries[:-1] + boundaries[1:]) / 2
        for _ in range(3):
            dists = np.abs(col[:, None] - centroids[None, :])
            indices = np.argmin(dists, axis=1)
            for k in range(n_levels):
                mask = indices == k
                if np.any(mask):
                    centroids[k] = np.mean(col[mask])

        # Final assignment
        dists = np.abs(col[:, None] - centroids[None, :])
        indices = np.argmin(dists, axis=1)

        codebooks.append(centroids)
        all_indices[:, comp_idx] = indices

    # Reconstruct quantized values
    quantized = np.zeros_like(coeffs_np)
    for comp_idx in range(n_components):
        quantized[:, comp_idx] = codebooks[comp_idx][all_indices[:, comp_idx]]

    return quantized, codebooks, all_indices


# ── 4. Lossless Compression (zstd) ───────────────────────────────────────────

def compress_zstd(data_bytes, level=3):
    """Compress bytes with zstandard."""
    import zstandard as zstd
    compressor = zstd.ZstdCompressor(level=level)
    return compressor.compress(data_bytes)


def decompress_zstd(compressed_bytes):
    """Decompress zstd bytes."""
    import zstandard as zstd
    decompressor = zstd.ZstdDecompressor()
    return decompressor.decompress(compressed_bytes)


# ── 5. Full Pipeline Test ────────────────────────────────────────────────────

def measure_reconstruction_error(original, reconstructed):
    """Compute MSE and cosine similarity between original and reconstructed."""
    original = original.astype(mx.float32) if original.dtype == mx.float16 else original
    reconstructed = reconstructed.astype(mx.float32) if reconstructed.dtype == mx.float16 else reconstructed
    orig_flat = original.reshape(-1)
    recon_flat = reconstructed.reshape(-1)

    mse = mx.mean((orig_flat - recon_flat) ** 2).item()

    # Cosine similarity per token
    orig_tokens = original.reshape(-1, original.shape[-1])
    recon_tokens = reconstructed.reshape(-1, reconstructed.shape[-1])

    dot = mx.sum(orig_tokens * recon_tokens, axis=-1)
    norm_orig = mx.sqrt(mx.sum(orig_tokens ** 2, axis=-1))
    norm_recon = mx.sqrt(mx.sum(recon_tokens ** 2, axis=-1))
    cos_sim = dot / (norm_orig * norm_recon + 1e-8)
    avg_cos_sim = mx.mean(cos_sim).item()

    return {"mse": mse, "cosine_similarity": avg_cos_sim}


def run_kvtc_spike(model_path: str):
    """Run the full kvtc feasibility spike."""
    print("=" * 70)
    print("kvtc Feasibility Spike — KV Cache Transform Coding on MLX")
    print("=" * 70)

    # Step 1: Extract KV cache
    print("\n── Step 1: Extract KV cache from model prefill ──")
    kv_pairs, seq_len = extract_kv_cache(model_path)
    k0, v0 = kv_pairs[0]
    n_heads = k0.shape[1]
    head_dim = k0.shape[3]
    n_layers = len(kv_pairs)

    # Compute original size
    bytes_per_element = 2  # float16
    original_bytes = n_layers * 2 * n_heads * seq_len * head_dim * bytes_per_element
    print(f"\n  Original KV cache size: {original_bytes / 1024 / 1024:.2f} MB")
    print(f"  ({n_layers} layers x 2 x {n_heads} heads x {seq_len} tokens x {head_dim} dim x 2 bytes)")

    # Step 2: PCA calibration
    print("\n── Step 2: PCA calibration (randomized SVD) ──")
    t0 = time.perf_counter()
    pca_bases = pca_calibrate(kv_pairs)
    pca_time = time.perf_counter() - t0
    print(f"  Total PCA calibration time: {pca_time:.2f}s for {n_layers} layers")

    # Step 3: Test different PCA component counts
    print("\n── Step 3: PCA reconstruction error vs. component count ──")
    test_components = [head_dim // 4, head_dim // 2, 3 * head_dim // 4, head_dim]

    for n_comp in test_components:
        basis_k = pca_bases[0]["key_basis"][:, :n_comp]
        mean_k = pca_bases[0]["key_mean"]
        basis_v = pca_bases[0]["value_basis"][:, :n_comp]
        mean_v = pca_bases[0]["value_mean"]

        coeffs_k = pca_project(k0, basis_k, mean_k)
        recon_k = pca_reconstruct(coeffs_k, basis_k, mean_k)
        mx.eval(recon_k)

        coeffs_v = pca_project(v0, basis_v, mean_v)
        recon_v = pca_reconstruct(coeffs_v, basis_v, mean_v)
        mx.eval(recon_v)

        err_k = measure_reconstruction_error(k0, recon_k)
        err_v = measure_reconstruction_error(v0, recon_v)

        print(f"  Components={n_comp}/{head_dim}: "
              f"Key MSE={err_k['mse']:.6f} cos={err_k['cosine_similarity']:.6f} | "
              f"Val MSE={err_v['mse']:.6f} cos={err_v['cosine_similarity']:.6f}")

    # Step 4: Quantization + compression pipeline
    print("\n── Step 4: DP quantization + zstd compression ──")
    n_comp = head_dim // 2
    total_compressed_bytes = 0
    total_quant_error = {"key_mse": 0, "key_cos": 0, "val_mse": 0, "val_cos": 0}

    t0_pipeline = time.perf_counter()

    for layer_idx in range(n_layers):
        keys, values = kv_pairs[layer_idx]
        basis_k = pca_bases[layer_idx]["key_basis"][:, :n_comp]
        mean_k = pca_bases[layer_idx]["key_mean"]
        basis_v = pca_bases[layer_idx]["value_basis"][:, :n_comp]
        mean_v = pca_bases[layer_idx]["value_mean"]

        layer_compressed = 0
        for name, tensor, basis, mean in [
            ("key", keys, basis_k, mean_k),
            ("value", values, basis_v, mean_v),
        ]:
            # PCA project
            coeffs = pca_project(tensor, basis, mean)
            mx.eval(coeffs)

            # Convert to numpy for quantization
            coeffs_np = np.array(coeffs.reshape(-1, n_comp), dtype=np.float32)

            # DP quantize (4-bit)
            quantized_np, codebooks, indices = dp_quantize(coeffs_np, n_bits=4)

            # Pack indices as uint8 (4-bit: 2 indices per byte)
            indices_flat = indices.astype(np.uint8).flatten()
            if len(indices_flat) % 2 != 0:
                indices_flat = np.append(indices_flat, 0)
            packed = (indices_flat[0::2] << 4) | indices_flat[1::2]

            # Compress with zstd
            packed_bytes = packed.tobytes()
            codebook_bytes = np.array(codebooks, dtype=np.float16).tobytes()
            compressed = compress_zstd(packed_bytes + codebook_bytes, level=3)
            layer_compressed += len(compressed)

            # Measure quantization error
            recon_coeffs = mx.array(quantized_np.reshape(tensor.shape[:-1] + (n_comp,)))
            recon_tensor = pca_reconstruct(recon_coeffs, basis, mean)
            mx.eval(recon_tensor)
            err = measure_reconstruction_error(tensor, recon_tensor)

            if name == "key":
                total_quant_error["key_mse"] += err["mse"]
                total_quant_error["key_cos"] += err["cosine_similarity"]
            else:
                total_quant_error["val_mse"] += err["mse"]
                total_quant_error["val_cos"] += err["cosine_similarity"]

        total_compressed_bytes += layer_compressed

    pipeline_time = time.perf_counter() - t0_pipeline

    # Add PCA basis storage cost
    basis_bytes = n_layers * 2 * head_dim * n_comp * 2  # float16
    mean_bytes = n_layers * 2 * head_dim * 2  # float16
    total_with_metadata = total_compressed_bytes + basis_bytes + mean_bytes

    print(f"\n  Pipeline time: {pipeline_time:.2f}s for {n_layers} layers")
    print(f"  Compressed data: {total_compressed_bytes / 1024:.1f} KB")
    print(f"  + PCA bases: {basis_bytes / 1024:.1f} KB")
    print(f"  + means: {mean_bytes / 1024:.1f} KB")
    print(f"  Total compressed: {total_with_metadata / 1024:.1f} KB")
    print(f"  Original: {original_bytes / 1024:.1f} KB")
    print(f"  Compression ratio: {original_bytes / total_with_metadata:.1f}x")
    print(f"\n  Avg quantization error (PCA {n_comp}/{head_dim} + 4-bit quant):")
    print(f"    Key MSE={total_quant_error['key_mse']/n_layers:.6f} "
          f"cos={total_quant_error['key_cos']/n_layers:.6f}")
    print(f"    Val MSE={total_quant_error['val_mse']/n_layers:.6f} "
          f"cos={total_quant_error['val_cos']/n_layers:.6f}")

    # Step 5: Round-trip verification
    print("\n── Step 5: Full round-trip verification (layer 0) ──")
    keys, values = kv_pairs[0]
    basis_k = pca_bases[0]["key_basis"][:, :n_comp]
    mean_k = pca_bases[0]["key_mean"]

    # Forward: project -> quantize -> compress
    coeffs = pca_project(keys, basis_k, mean_k)
    mx.eval(coeffs)
    coeffs_np = np.array(coeffs.reshape(-1, n_comp), dtype=np.float32)
    quantized_np, codebooks, indices = dp_quantize(coeffs_np, n_bits=4)

    # Pack and compress
    indices_flat = indices.astype(np.uint8).flatten()
    if len(indices_flat) % 2 != 0:
        indices_flat = np.append(indices_flat, 0)
    packed = (indices_flat[0::2] << 4) | indices_flat[1::2]
    packed_bytes = packed.tobytes()
    codebook_bytes_data = np.array(codebooks, dtype=np.float16).tobytes()
    compressed = compress_zstd(packed_bytes + codebook_bytes_data)

    # Reverse: decompress -> dequantize -> reconstruct
    t0_decomp = time.perf_counter()
    decompressed = decompress_zstd(compressed)

    # Unpack
    n_codebook_bytes = n_comp * (2 ** 4) * 2  # n_comp codebooks x 16 levels x float16
    packed_data = decompressed[:len(decompressed) - n_codebook_bytes]
    codebook_data = decompressed[len(decompressed) - n_codebook_bytes:]

    unpacked = np.frombuffer(packed_data, dtype=np.uint8)
    indices_recovered = np.zeros(len(unpacked) * 2, dtype=np.uint8)
    indices_recovered[0::2] = unpacked >> 4
    indices_recovered[1::2] = unpacked & 0x0F

    codebooks_recovered = np.frombuffer(codebook_data, dtype=np.float16).reshape(n_comp, -1)

    # Dequantize
    n_total = coeffs_np.shape[0]
    indices_recovered = indices_recovered[:n_total * n_comp].reshape(n_total, n_comp)
    dequantized = np.zeros((n_total, n_comp), dtype=np.float32)
    for c in range(n_comp):
        dequantized[:, c] = codebooks_recovered[c][indices_recovered[:, c]].astype(np.float32)

    # Reconstruct
    recon_coeffs = mx.array(dequantized.reshape(keys.shape[:-1] + (n_comp,)))
    recon_keys = pca_reconstruct(recon_coeffs, basis_k, mean_k)
    mx.eval(recon_keys)
    decomp_time = time.perf_counter() - t0_decomp

    err = measure_reconstruction_error(keys, recon_keys)
    print(f"  Decompression time: {decomp_time * 1000:.1f}ms")
    print(f"  Round-trip Key MSE: {err['mse']:.6f}")
    print(f"  Round-trip Key cosine similarity: {err['cosine_similarity']:.6f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Model: {model_path}")
    print(f"  Sequence length: {seq_len} tokens")
    print(f"  Cache shape: {n_layers}L x {n_heads}H x {head_dim}D")
    print(f"  PCA calibration: {pca_time:.2f}s")
    print(f"  Pipeline (all layers): {pipeline_time:.2f}s")
    print(f"  Compression ratio: {original_bytes / total_with_metadata:.1f}x")
    print(f"  Key reconstruction cos sim: {total_quant_error['key_cos']/n_layers:.6f}")
    print(f"  Val reconstruction cos sim: {total_quant_error['val_cos']/n_layers:.6f}")
    print(f"  Decompression latency (1 layer): {decomp_time * 1000:.1f}ms")

    return {
        "seq_len": seq_len,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "original_bytes": original_bytes,
        "compressed_bytes": total_with_metadata,
        "compression_ratio": original_bytes / total_with_metadata,
        "pca_calibration_time_s": pca_time,
        "pipeline_time_s": pipeline_time,
        "avg_key_cos_sim": total_quant_error["key_cos"] / n_layers,
        "avg_val_cos_sim": total_quant_error["val_cos"] / n_layers,
        "decompression_latency_ms": decomp_time * 1000,
    }


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/tonysina/models/Qwen2.5-7B-Instruct-4bit"
    results = run_kvtc_spike(model_path)

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "spike_kvtc_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
