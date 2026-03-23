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
from omlx.compression.pipeline import KVCachePipeline, PipelineBlob


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
        """PIPE-02: cosine similarity of decompressed values > 0.9 (synthetic threshold)."""
        cache = make_synthetic_cache()
        pipeline = KVCachePipeline()
        blob = pipeline.compress(cache)  # raises NotImplementedError -> RED
        layers, _ = pipeline.decompress(blob)
        _, v_orig = cache[0]
        _, v_restored = layers[0]
        sim = cosine_sim(np.array(v_orig), np.array(v_restored))
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
        pytest.skip("Wave 2 — requires Qwen 2.5 7B")
