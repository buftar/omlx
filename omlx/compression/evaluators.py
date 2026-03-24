# SPDX-License-Identifier: Apache-2.0
"""Evaluator functions for compression benchmarking."""

import json
import os
from typing import Optional

import numpy as np


def run_gsm8k(
    model, tokenizer, pipeline, n_samples: int = 200, seed: int = 42
) -> dict:
    """Run GSM8K benchmark.

    Args:
        model: The language model to evaluate.
        tokenizer: Tokenizer for the model.
        pipeline: Inference pipeline.
        n_samples: Number of samples to evaluate.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with vanilla and compressed accuracy.
    """
    raise NotImplementedError("run_gsm8k not yet implemented")


def run_mmlu(
    model, tokenizer, pipeline, n_samples: int = 200, seed: int = 42
) -> dict:
    """Run MMLU benchmark.

    Args:
        model: The language model to evaluate.
        tokenizer: Tokenizer for the model.
        pipeline: Inference pipeline.
        n_samples: Number of samples to evaluate.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with vanilla and compressed accuracy.
    """
    raise NotImplementedError("run_mmlu not yet implemented")


def run_litm(
    model, tokenizer, pipeline, n_problems: int = 100, seed: int = 42
) -> dict:
    """Run LiT-M (Long Text Memory) benchmark.

    Args:
        model: The language model to evaluate.
        tokenizer: Tokenizer for the model.
        pipeline: Inference pipeline.
        n_problems: Number of problems to evaluate.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with vanilla and compressed recall.
    """
    raise NotImplementedError("run_litm not yet implemented")


def cosine_sim_kv(layers_a, layers_b) -> float:
    """Compute cosine similarity between KV cache layers.

    Args:
        layers_a: First set of layers, each element is a tuple (keys, values).
        layers_b: Second set of layers, same structure as layers_a.

    Returns:
        Cosine similarity score between 0.0 and 1.0.
    """
    if not layers_a or not layers_b:
        return 0.0

    sims = []
    for (ka, va), (kb, vb) in zip(layers_a, layers_b):
        # Flatten both tensors to 1D float32 arrays
        a = np.array(ka, dtype=np.float32).ravel()
        b = np.array(kb, dtype=np.float32).ravel()
        va_flat = np.array(va, dtype=np.float32).ravel()
        vb_flat = np.array(vb, dtype=np.float32).ravel()

        # Compute cosine similarity for keys
        denom_a = np.linalg.norm(a)
        denom_b = np.linalg.norm(b)
        denom = denom_a * denom_b + 1e-8
        if denom > 0:
            sim_k = np.dot(a, b) / denom
            sims.append(sim_k)

        # Compute cosine similarity for values
        denom_a = np.linalg.norm(va_flat)
        denom_b = np.linalg.norm(vb_flat)
        denom = denom_a * denom_b + 1e-8
        if denom > 0:
            sim_v = np.dot(va_flat, vb_flat) / denom
            sims.append(sim_v)

    return float(np.mean(sims)) if sims else 0.0


def measure_decompression_latency(
    pipeline, blob, n_layers: int, n_warmup: int = 1
) -> float:
    """Measure decompression latency.

    Args:
        pipeline: Inference pipeline.
        blob: Compressed blob to decompress.
        n_layers: Number of layers in the blob.
        n_warmup: Number of warmup runs.

    Returns:
        Latency in milliseconds per layer.
    """
    raise NotImplementedError("measure_decompression_latency not yet implemented")


def get_compressible_layer_indices(prompt_cache) -> list[int]:
    """Get indices of layers that can be compressed.

    Args:
        prompt_cache: The prompt cache to analyze.

    Returns:
        List of layer indices that are compressible (non-SWA layers).
    """
    from mlx_lm.models.cache import RotatingKVCache

    return [i for i, c in enumerate(prompt_cache) if not isinstance(c, RotatingKVCache)]


def detect_swa_layers(model_path: str, num_hidden_layers: int) -> set[int]:
    """Detect SWA (Sliding Window Attention) layers in a model.

    Args:
        model_path: Path to the model configuration directory.
        num_hidden_layers: Number of hidden layers in the model.

    Returns:
        Set of layer indices that use SWA.
    """
    config_path = os.path.join(model_path, "config.json")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return set()

    model_type = config.get("model_type", "")
    if model_type not in ("gemma3", "gemma3_text"):
        return set()

    pattern = config.get("sliding_window_pattern", 6)
    # SWA layers are those where i % pattern == pattern - 1
    return {i for i in range(num_hidden_layers) if i % pattern == pattern - 1}
