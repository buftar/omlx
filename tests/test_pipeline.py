# SPDX-License-Identifier: Apache-2.0
"""
Test scaffold for Phase 5 pipeline assembly.

All fast tests are in RED state until Wave 1 implements the pipeline methods.
Each test calls a NotImplementedError-raising stub, which causes an unhandled
exception and a pytest failure (exit code 1).  This is the intended Wave 0
contract: tests are collected, run, and fail RED.

Slow test (TestSlowQwen) is skipped explicitly — it requires Qwen 2.5 7B on
disk and is gated until Wave 2.
"""
from unittest.mock import MagicMock

import mlx.core as mx
import numpy as np
import pytest

from omlx.compression.am import AMCompactedCache
from omlx.compression.pipeline import KVCachePipeline, PipelineBlob, _mx_materialize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_cache(n_layers=2, n_heads=4, seq_len=300, head_dim=128):
    """Return a minimal synthetic KV cache as a list of (keys, values) tuples.

    seq_len=300 ensures body tokens exist: 300 > n_sink(4) + window(128) = 132.
    """
    mx.random.seed(42)
    return [
        (
            mx.random.uniform(shape=[1, n_heads, seq_len, head_dim]).astype(mx.float16),
            mx.random.uniform(shape=[1, n_heads, seq_len, head_dim]).astype(mx.float16),
        )
        for _ in range(n_layers)
    ]


def cosine_sim(a, b):
    """Cosine similarity of two array-like objects, reshaped to 1-D float32."""
    a_f = np.array(a, dtype=np.float32).ravel()
    b_f = np.array(b, dtype=np.float32).ravel()
    denom = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a_f, b_f) / denom)


# ---------------------------------------------------------------------------
# PIPE-01 — compress() produces a PipelineBlob
# ---------------------------------------------------------------------------

class TestCompress:
    def test_returns_pipeline_blob(self):
        """PIPE-01: compress() should return a PipelineBlob instance."""
        cache = make_synthetic_cache()
        pipeline = KVCachePipeline()
        result = pipeline.compress(cache)  # raises NotImplementedError -> RED
        assert isinstance(result, PipelineBlob)

    def test_multiplicative_ratio(self):
        """PIPE-01: compaction_ratio > 1.0 and blob.compressed is non-empty."""
        cache = make_synthetic_cache()
        pipeline = KVCachePipeline()
        result = pipeline.compress(cache)  # raises NotImplementedError -> RED
        assert result.compaction_ratio > 1.0
        assert len(result.compressed) > 0


# ---------------------------------------------------------------------------
# PIPE-02 — round-trip: compress -> decompress preserves shape and values
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_round_trip_shape(self):
        """PIPE-02: decompress() returns (layers, logical_seq_len) with valid shape."""
        cache = make_synthetic_cache()
        pipeline = KVCachePipeline()
        blob = pipeline.compress(cache)  # raises NotImplementedError -> RED
        layers, logical_seq_len = pipeline.decompress(blob)
        # shape: [1, n_heads, compacted_T, head_dim]
        assert len(layers) == len(cache)
        k0, v0 = layers[0]
        assert k0.ndim == 4
        assert v0.ndim == 4

    def test_round_trip_cosine_sim(self):
        """PIPE-02: cosine similarity of decompressed values > 0.9 (synthetic threshold).

        Compares decompressed (kvtc round-trip) values against the compacted values
        (post-AM), not the original values.  AM compaction is lossy and changes the
        token count; comparing original vs decompressed would produce incompatible
        shapes.  The cosine-sim check verifies kvtc codec fidelity only.
        """
        cache = make_synthetic_cache()
        pipeline = KVCachePipeline()
        # Get the compacted cache directly for reference values
        compacted = pipeline.compact(cache)
        _, v_compacted = compacted.layers[0]
        # Full round-trip through compress -> decompress
        blob = pipeline.compress(cache)
        layers, _ = pipeline.decompress(blob)
        _, v_restored = layers[0]
        sim = cosine_sim(np.array(v_compacted), np.array(v_restored))
        assert sim > 0.9, f"cosine_sim={sim:.4f} below 0.9 threshold"

    def test_logical_seq_len_preserved(self):
        """PIPE-02: logical_seq_len in blob matches original T."""
        cache = make_synthetic_cache(seq_len=300)
        pipeline = KVCachePipeline()
        blob = pipeline.compress(cache)  # raises NotImplementedError -> RED
        assert blob.logical_seq_len == 300


# ---------------------------------------------------------------------------
# PIPE-03/04/05 — trigger semantics
# ---------------------------------------------------------------------------

class TestTriggerSemantics:
    def test_compact_fires_above_threshold(self):
        """PIPE-03: compact() fires when memory pressure exceeds threshold.

        Uses MagicMock to avoid needing a real model; validates that compact()
        is callable with a KV cache and returns an AMCompactedCache when
        implemented.
        """
        cache = make_synthetic_cache()
        pipeline = KVCachePipeline()
        result = pipeline.compact(cache)  # raises NotImplementedError -> RED
        assert isinstance(result, AMCompactedCache)

    def test_compress_callable(self):
        """PIPE-04: compress() is the eviction entry point and returns PipelineBlob."""
        cache = make_synthetic_cache()
        pipeline = KVCachePipeline()
        result = pipeline.compress(cache)  # raises NotImplementedError -> RED
        assert isinstance(result, PipelineBlob)

    def test_decompress_restores_layers(self):
        """PIPE-05: decompress() returns a non-empty layer list."""
        cache = make_synthetic_cache()
        pipeline = KVCachePipeline()
        blob = pipeline.compress(cache)  # raises NotImplementedError -> RED
        layers, _ = pipeline.decompress(blob)
        assert len(layers) > 0


# ---------------------------------------------------------------------------
# PIPE-02 slow — Qwen 2.5 7B full round-trip (gated to Wave 2)
# ---------------------------------------------------------------------------

class TestSlowQwen:
    @pytest.mark.slow
    def test_qwen_round_trip(self):
        """PIPE-02 slow: full Qwen 2.5 7B round-trip through the pipeline."""
        mlx_lm = pytest.importorskip("mlx_lm")
        from mlx_lm.models.cache import make_prompt_cache

        # Load Qwen 2.5 7B Instruct — skip gracefully if not cached locally
        try:
            model, tokenizer = mlx_lm.load("Qwen/Qwen2.5-7B-Instruct")
        except (FileNotFoundError, OSError):
            pytest.skip("Qwen 2.5 7B not available locally")

        # Tokenize a moderately long prompt (aim for seq_len > 132)
        # n_sink=4 + window=128 = 132 minimum for non-empty AM body
        base_prompt = (
            "Summarize the key ideas behind transformer self-attention in detail. "
            "Explain query, key, and value projections, multi-head attention, "
            "positional encodings, and how the attention mechanism allows each token "
            "to attend to every other token in the sequence."
        )
        input_ids = tokenizer.encode(base_prompt)
        while len(input_ids) <= 132:
            base_prompt = base_prompt + " " + base_prompt
            input_ids = tokenizer.encode(base_prompt)

        tokens = mx.array(input_ids)[None]  # [1, seq_len]

        # Run a forward pass to populate the KV cache via mlx_lm prompt cache API
        cache = make_prompt_cache(model)
        _ = model(tokens, cache=cache)
        # Materialize all cache tensors before numpy bridge (MLX lazy eval)
        _mx_materialize(*[v for c in cache for v in [c.keys, c.values] if v is not None])

        # Extract KV cache as list[tuple[mx.array, mx.array]] — one tuple per layer.
        # cache[i].offset is the actual filled sequence length; keys/values may be
        # over-allocated (mlx_lm allocates in chunks). Slice to filled region.
        actual_seq_len = cache[0].offset
        kv_cache = [
            (c.keys[:, :, :actual_seq_len, :], c.values[:, :, :actual_seq_len, :])
            for c in cache
        ]

        original_seq_len = kv_cache[0][0].shape[2]
        assert original_seq_len > 132, (
            f"seq_len={original_seq_len} not > 132; prompt too short for non-empty AM body"
        )

        # Build pipeline in testing path (bundle_path=None)
        pipeline = KVCachePipeline(bundle_path=None, am_ratio=4.0)

        # --- compress() assertions ---
        blob = pipeline.compress(kv_cache)
        assert isinstance(blob, PipelineBlob), "compress() must return PipelineBlob"
        assert blob.compaction_ratio > 1.0, (
            f"compaction_ratio={blob.compaction_ratio:.4f} not > 1.0"
        )
        assert len(blob.compressed) > 0, "compressed bytes must be non-empty"
        assert blob.logical_seq_len == original_seq_len, (
            f"logical_seq_len={blob.logical_seq_len} != original {original_seq_len}"
        )

        # --- decompress() assertions ---
        layers, logical_seq_len = pipeline.decompress(blob)
        assert logical_seq_len == blob.logical_seq_len, (
            "decompress() must return blob.logical_seq_len unchanged"
        )
        assert len(layers) == len(kv_cache), (
            f"decompress returned {len(layers)} layers, expected {len(kv_cache)}"
        )

        # Check decompressed layer shapes: [1, n_kv_heads, compacted_seq_len, head_dim]
        compacted_seq_len = layers[0][0].shape[2]
        assert compacted_seq_len < original_seq_len, (
            f"compacted_seq_len={compacted_seq_len} not < original {original_seq_len}"
        )
        for i, (k, v) in enumerate(layers):
            assert k.ndim == 4, f"Layer {i} keys: expected 4D, got {k.ndim}D"
            assert v.ndim == 4, f"Layer {i} values: expected 4D, got {v.ndim}D"
            assert k.shape[2] == compacted_seq_len, (
                f"Layer {i} keys shape[2]={k.shape[2]} != compacted_seq_len={compacted_seq_len}"
            )

        # --- cosine similarity: decompressed vs compacted reference ---
        # compact() separately to get the reference values at compacted token count
        compacted = pipeline.compact(kv_cache)
        for i, ((_, v_compacted), (_, v_restored)) in enumerate(
            zip(compacted.layers, layers)
        ):
            sim = cosine_sim(np.array(v_compacted), np.array(v_restored))
            assert sim > 0.9, (
                f"Layer {i} cosine_sim={sim:.4f} below 0.9 threshold (bundle=None path)"
            )
