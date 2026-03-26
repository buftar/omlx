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
    import re

    import mlx.core as mx
    from datasets import load_dataset
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import RotatingKVCache, make_prompt_cache

    # Load dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")

    # Sample with seed
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), min(n_samples, len(ds)), replace=False)

    vanilla_correct = 0
    compressed_correct = 0

    for idx in indices:
        item = ds[idx]
        question = item["question"]
        gold_answer = _extract_gsm8k_answer(item["answer"])

        # Vanilla path
        vanilla_correct += _run_single_gsm8k_sample(
            model, tokenizer, pipeline, question, gold_answer, vanilla=True
        )

        # Compressed path
        compressed_correct += _run_single_gsm8k_sample(
            model, tokenizer, pipeline, question, gold_answer, vanilla=False
        )

    return {
        "gsm8k_vanilla": vanilla_correct / len(indices) * 100,
        "gsm8k_compressed": compressed_correct / len(indices) * 100,
    }


def _run_single_gsm8k_sample(model, tokenizer, pipeline, question, gold_answer, vanilla):
    """Run a single GSM8K sample (vanilla or compressed path)."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import RotatingKVCache, make_prompt_cache

    prompt = f"Solve step by step. End with: #### <number>\n\nProblem: {question}\nSolution:"
    prompt_ids = tokenizer.encode(prompt)

    if vanilla:
        # Vanilla path: build fresh cache
        cache = make_prompt_cache(model)
        tokens = mx.array(prompt_ids)[None]
        _ = model(tokens, cache=cache)
        mx.eval(*[t for c in cache for t in [c.keys, c.values] if t is not None])

        # Generate
        tokens_out = []
        for token, _ in generate_step(tokens.flatten(), model, prompt_cache=cache, max_tokens=256):
            if token in tokenizer.eos_token_ids:
                break
            tokens_out.append(token)

        decoded = tokenizer.decode(tokens_out)
        pred_answer = _extract_gsm8k_answer(decoded)
        return 1 if pred_answer == gold_answer else 0

    else:
        # Compressed path
        cache = make_prompt_cache(model)
        tokens = mx.array(prompt_ids)[None]
        _ = model(tokens, cache=cache)
        mx.eval(*[t for c in cache for t in [c.keys, c.values] if t is not None])

        # Extract kv_cache
        offset = cache[0].offset
        kv_cache = [(c.keys[:, :, :offset, :], c.values[:, :, :offset, :]) for c in cache]

        # Compress and decompress
        blob = pipeline.compress(kv_cache)
        decompressed_layers, _ = pipeline.decompress(blob)

        # Inject into fresh cache
        new_cache = make_prompt_cache(model)
        for cache_obj, (keys, values) in zip(new_cache, decompressed_layers):
            if isinstance(cache_obj, RotatingKVCache):
                continue
            cache_obj.state = (keys, values)

        # Generate
        tokens_out = []
        for token, _ in generate_step(tokens.flatten(), model, prompt_cache=new_cache, max_tokens=256):
            if token in tokenizer.eos_token_ids:
                break
            tokens_out.append(token)

        decoded = tokenizer.decode(tokens_out)
        pred_answer = _extract_gsm8k_answer(decoded)
        return 1 if pred_answer == gold_answer else 0


def _extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract answer from GSM8K output using #### marker."""
    import re

    m = re.search(r"####\s*([\d,.-]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    return None


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
    import mlx.core as mx
    from datasets import load_dataset
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import RotatingKVCache, make_prompt_cache

    subjects = [
        "elementary_mathematics",
        "high_school_physics",
        "moral_scenarios",
        "professional_medicine",
        "computer_security",
    ]

    samples_per_subject = max(1, n_samples // len(subjects))
    all_results = []

    for subject in subjects:
        ds = load_dataset("cais/mmlu", subject, split="test")
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(ds), min(samples_per_subject, len(ds)), replace=False)

        for idx in indices:
            item = ds[idx]
            question = item["question"]
            choices = item["choices"]
            correct_idx = item["answer"]

            # Run vanilla path
            vanilla_correct = _run_single_mmlu_sample(
                model, tokenizer, question, choices, correct_idx, vanilla=True
            )
            all_results.append(("vanilla", vanilla_correct))

            # Run compressed path
            compressed_correct = _run_single_mmlu_sample(
                model, tokenizer, question, choices, correct_idx, vanilla=False
            )
            all_results.append(("compressed", compressed_correct))

    vanilla_total = sum(1 for t, c in all_results if t == "vanilla" and c)
    compressed_total = sum(1 for t, c in all_results if t == "compressed" and c)
    total = len(all_results) // 2

    return {
        "mmlu_vanilla": vanilla_total / total * 100,
        "mmlu_compressed": compressed_total / total * 100,
    }


def _run_single_mmlu_sample(model, tokenizer, question, choices, correct_idx, vanilla):
    """Run a single MMLU sample (vanilla or compressed path)."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import RotatingKVCache, make_prompt_cache

    # Build prompt with choices
    prompt = f"Question: {question}\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\nAnswer:"
    prompt_ids = tokenizer.encode(prompt)

    if vanilla:
        # Vanilla path
        cache = make_prompt_cache(model)
        tokens = mx.array(prompt_ids)[None]
        _ = model(tokens, cache=cache)
        mx.eval(*[t for c in cache for t in [c.keys, c.values] if t is not None])

        # Generate 1 token
        for token, _ in generate_step(tokens.flatten(), model, prompt_cache=cache, max_tokens=1):
            predicted_token = token
            break

        # Get first non-whitespace character
        decoded = tokenizer.decode([predicted_token])
        pred_label = decoded.strip()[0] if decoded.strip() else ""
        expected_label = ["A", "B", "C", "D"][correct_idx]
        return 1 if pred_label == expected_label else 0

    else:
        # Compressed path
        cache = make_prompt_cache(model)
        tokens = mx.array(prompt_ids)[None]
        _ = model(tokens, cache=cache)
        mx.eval(*[t for c in cache for t in [c.keys, c.values] if t is not None])

        # Extract kv_cache
        offset = cache[0].offset
        kv_cache = [(c.keys[:, :, :offset, :], c.values[:, :, :offset, :]) for c in cache]

        # Compress and decompress
        blob = pipeline.compress(kv_cache)
        decompressed_layers, _ = pipeline.decompress(blob)

        # Inject into fresh cache
        new_cache = make_prompt_cache(model)
        for cache_obj, (keys, values) in zip(new_cache, decompressed_layers):
            if isinstance(cache_obj, RotatingKVCache):
                continue
            cache_obj.state = (keys, values)

        # Generate 1 token
        for token, _ in generate_step(tokens.flatten(), model, prompt_cache=new_cache, max_tokens=1):
            predicted_token = token
            break

        # Get first non-whitespace character
        decoded = tokenizer.decode([predicted_token])
        pred_label = decoded.strip()[0] if decoded.strip() else ""
        expected_label = ["A", "B", "C", "D"][correct_idx]
        return 1 if pred_label == expected_label else 0


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
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import RotatingKVCache, make_prompt_cache

    rng = np.random.default_rng(seed)

    vanilla_correct = 0
    compressed_correct = 0

    for _ in range(n_problems):
        # Generate code
        code = rng.integers(1000, 9999)

        # Generate 20 passages
        passages = []
        for i in range(20):
            if i == 10:
                # Target passage at index 10
                passage = f"Document 10: The secret code is {code}."
            else:
                # Filler passage
                topic = rng.integers(100, 999)
                passage = f"Document {i}: This passage contains general information about topic {topic}."
            passages.append(passage)

        # Build context
        context = "\n".join(passages)
        question = "Based on the documents, what is the secret code?"
        full_prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

        # Vanilla path
        vanilla_correct += _run_single_litm_sample(
            model, tokenizer, full_prompt, code, vanilla=True
        )

        # Compressed path
        compressed_correct += _run_single_litm_sample(
            model, tokenizer, full_prompt, code, vanilla=False
        )

    return {
        "litm_vanilla": vanilla_correct / n_problems,
        "litm_compressed": compressed_correct / n_problems,
    }


def _run_single_litm_sample(model, tokenizer, prompt, code, vanilla):
    """Run a single LiT-M sample (vanilla or compressed path)."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import RotatingKVCache, make_prompt_cache

    prompt_ids = tokenizer.encode(prompt)

    if vanilla:
        # Vanilla path
        cache = make_prompt_cache(model)
        tokens = mx.array(prompt_ids)[None]
        _ = model(tokens, cache=cache)
        mx.eval(*[t for c in cache for t in [c.keys, c.values] if t is not None])

        # Generate up to 32 tokens
        tokens_out = []
        for token, _ in generate_step(tokens.flatten(), model, prompt_cache=cache, max_tokens=32):
            if token in tokenizer.eos_token_ids:
                break
            tokens_out.append(token)

        decoded = tokenizer.decode(tokens_out)
        return 1 if str(code) in decoded else 0

    else:
        # Compressed path
        cache = make_prompt_cache(model)
        tokens = mx.array(prompt_ids)[None]
        _ = model(tokens, cache=cache)
        mx.eval(*[t for c in cache for t in [c.keys, c.values] if t is not None])

        # Extract kv_cache
        offset = cache[0].offset
        kv_cache = [(c.keys[:, :, :offset, :], c.values[:, :, :offset, :]) for c in cache]

        # Compress and decompress
        blob = pipeline.compress(kv_cache)
        decompressed_layers, _ = pipeline.decompress(blob)

        # Inject into fresh cache
        new_cache = make_prompt_cache(model)
        for cache_obj, (keys, values) in zip(new_cache, decompressed_layers):
            if isinstance(cache_obj, RotatingKVCache):
                continue
            cache_obj.state = (keys, values)

        # Generate up to 32 tokens
        tokens_out = []
        for token, _ in generate_step(tokens.flatten(), model, prompt_cache=new_cache, max_tokens=32):
            if token in tokenizer.eos_token_ids:
                break
            tokens_out.append(token)

        decoded = tokenizer.decode(tokens_out)
        return 1 if str(code) in decoded else 0


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

    import mlx.core as mx
    import numpy as np

    sims = []
    for (ka, va), (kb, vb) in zip(layers_a, layers_b):
        # Convert to numpy arrays if needed, handling both numpy and MLX arrays
        # For numpy arrays, ensure float32 dtype; for MLX arrays, convert to float32 then numpy
        if isinstance(ka, np.ndarray):
            a = ka.astype(np.float32)
            b = kb.astype(np.float32)
            va_np = va.astype(np.float32)
            vb_np = vb.astype(np.float32)
        else:
            # MLX array - convert to float32 then numpy
            mx.eval(ka, va, kb, vb)
            a = ka.astype(mx.float32).numpy()
            b = kb.astype(mx.float32).numpy()
            va_np = va.astype(mx.float32).numpy()
            vb_np = vb.astype(mx.float32).numpy()

        # Flatten both tensors to 1D float32 arrays
        a_flat = a.ravel()
        b_flat = b.ravel()
        va_flat = va_np.ravel()
        vb_flat = vb_np.ravel()

        # Compute cosine similarity for keys
        denom_a = np.linalg.norm(a_flat)
        denom_b = np.linalg.norm(b_flat)
        denom = denom_a * denom_b + 1e-8
        if denom > 0:
            sim_k = np.dot(a_flat, b_flat) / denom
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
    import time

    import mlx.core as mx

    # Warmup loop
    for _ in range(n_warmup):
        layers, _ = pipeline.decompress(blob)
        mx.eval(*[t for pair in layers for t in pair])

    # Timed run
    t0 = time.perf_counter()
    layers, _ = pipeline.decompress(blob)
    mx.eval(*[t for pair in layers for t in pair])
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return elapsed_ms / n_layers


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
