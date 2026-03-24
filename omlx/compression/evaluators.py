# SPDX-License-Identifier: Apache-2.0
"""Evaluator functions for compression benchmarking."""

from typing import Optional


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
        layers_a: First set of layers.
        layers_b: Second set of layers.

    Returns:
        Cosine similarity score.
    """
    raise NotImplementedError("cosine_sim_kv not yet implemented")


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
    raise NotImplementedError("get_compressible_layer_indices not yet implemented")


def detect_swa_layers(model_path: str, num_hidden_layers: int) -> set[int]:
    """Detect SWA (Sliding Window Attention) layers in a model.

    Args:
        model_path: Path to the model configuration.
        num_hidden_layers: Number of hidden layers in the model.

    Returns:
        Set of layer indices that use SWA.
    """
    raise NotImplementedError("detect_swa_layers not yet implemented")
