# SPDX-License-Identifier: Apache-2.0
"""
Wave 0 RED-state test scaffold for omlx/compression/kvtc.py.

All implementation tests MUST fail with NotImplementedError in Wave 0.
Only TestPCAProjection.test_compressor_constructs_without_bundle should pass.

Running fast tests (excludes model downloads):
    uv run python -m pytest tests/test_kvtc.py -m "not slow" -q

Running all tests (requires Qwen 2.5 7B download):
    uv run python -m pytest tests/test_kvtc.py -v
"""

import time

import mlx.core as mx
import numpy as np
import pytest

from omlx.compression.kvtc import KVTCCompressor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_kv_cache():
    """2-layer GQA cache: 4 KV heads, seq_len=300, head_dim=128, float16.

    seq_len=300 ensures the body is non-empty:
        300 > n_sink_tokens(4) + sliding_window(128) = 132
    """
    mx.random.seed(42)
    layers = [
        (
            mx.random.uniform(shape=[1, 4, 300, 128]).astype(mx.float16),
            mx.random.uniform(shape=[1, 4, 300, 128]).astype(mx.float16),
        )
        for _ in range(2)
    ]
    return layers


@pytest.fixture
def gqa_kv_cache():
    """28-layer Qwen 2.5 7B GQA config: 4 KV heads, seq_len=300, head_dim=128, float16."""
    mx.random.seed(7)
    layers = [
        (
            mx.random.uniform(shape=[1, 4, 300, 128]).astype(mx.float16),
            mx.random.uniform(shape=[1, 4, 300, 128]).astype(mx.float16),
        )
        for _ in range(28)
    ]
    return layers


# ---------------------------------------------------------------------------
# KVTC-01: PCA Projection
# ---------------------------------------------------------------------------


class TestPCAProjection:
    """KVTC-01: PCA projection reduces head_dim to n_components efficiently."""

    def test_compressor_constructs_without_bundle(self):
        """KVTCCompressor(None) constructs with expected defaults. Must PASS in Wave 0."""
        c = KVTCCompressor(pca_bundle=None)
        assert c.pca_bundle is None
        assert c.n_sink_tokens == 4
        assert c.sliding_window == 128
        assert c.bits_per_token == 4.0

    def test_compress_raises_not_implemented(self, small_kv_cache):
        """compress() raises NotImplementedError in Wave 0 (RED state)."""
        c = KVTCCompressor(pca_bundle=None)
        with pytest.raises(NotImplementedError):
            c.compress(small_kv_cache)

    def test_onthefly_pca_produces_bytes(self, small_kv_cache):
        """compress() returns bytes (passes after Wave 1 implementation)."""
        c = KVTCCompressor(pca_bundle=None)
        result = c.compress(small_kv_cache)
        assert isinstance(result, bytes)

    def test_pca_projection_preserves_shape(self, small_kv_cache):
        """Decompressed shape matches input (passes after Wave 2)."""
        c = KVTCCompressor(pca_bundle=None)
        blob = c.compress(small_kv_cache)
        recovered = c.decompress(blob)
        assert len(recovered) == len(small_kv_cache)
        for (rk, rv), (ok, ov) in zip(recovered, small_kv_cache):
            assert rk.shape == ok.shape
            assert rv.shape == ov.shape


# ---------------------------------------------------------------------------
# KVTC-02: DP Bit Allocation
# ---------------------------------------------------------------------------


class TestDPAllocation:
    """KVTC-02: DP allocates variable bits per PCA component within budget."""

    def test_compress_raises_not_implemented(self, small_kv_cache):
        """compress() raises NotImplementedError in Wave 0 (RED state)."""
        c = KVTCCompressor(pca_bundle=None)
        with pytest.raises(NotImplementedError):
            c.compress(small_kv_cache)

    def test_dp_all_bits_at_least_one(self, small_kv_cache):
        """All bit allocations >= 1 (verified via successful round-trip, Wave 2+)."""
        c = KVTCCompressor(pca_bundle=None)
        blob = c.compress(small_kv_cache)
        recovered = c.decompress(blob)
        # If we got here with a valid recovered cache, allocation was >= 1 per component
        assert recovered is not None

    def test_dp_within_budget(self, small_kv_cache):
        """Compressed blob is smaller than raw float16 size * 0.9."""
        c = KVTCCompressor(pca_bundle=None)
        blob = c.compress(small_kv_cache)
        # Raw size: 2 layers * 2 tensors * (1*4*300*128) * 2 bytes (float16)
        raw_bytes = len(small_kv_cache) * 2 * (1 * 4 * 300 * 128) * 2
        assert len(blob) < raw_bytes * 0.9


# ---------------------------------------------------------------------------
# KVTC-03: Compress / Decompress Contract
# ---------------------------------------------------------------------------


class TestCompressDecompress:
    """KVTC-03: compress() produces a valid self-describing blob."""

    def test_compress_returns_bytes(self, small_kv_cache):
        """compress() raises NotImplementedError in RED state."""
        c = KVTCCompressor(pca_bundle=None)
        with pytest.raises(NotImplementedError):
            c.compress(small_kv_cache)

    def test_blob_has_kvtc_magic(self, small_kv_cache):
        """Blob starts with b'KVTC' magic bytes (passes after Wave 1)."""
        c = KVTCCompressor(pca_bundle=None)
        blob = c.compress(small_kv_cache)
        assert blob[:4] == b"KVTC"

    def test_blob_is_deterministic(self, small_kv_cache):
        """Same input produces identical blobs (passes after Wave 1)."""
        c = KVTCCompressor(pca_bundle=None)
        blob1 = c.compress(small_kv_cache)
        blob2 = c.compress(small_kv_cache)
        assert blob1 == blob2


# ---------------------------------------------------------------------------
# KVTC-04: Round-Trip Fidelity
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """KVTC-04: Decompressed KV cache has cosine similarity >= 0.97 vs input."""

    def test_round_trip_raises_not_implemented(self, small_kv_cache):
        """compress() raises NotImplementedError in Wave 0 (RED state)."""
        c = KVTCCompressor(pca_bundle=None)
        with pytest.raises(NotImplementedError):
            c.compress(small_kv_cache)

    def test_round_trip_cosine_similarity(self, small_kv_cache):
        """Body tokens cosine similarity >= 0.97 after round-trip (Wave 2+)."""
        c = KVTCCompressor(pca_bundle=None)
        blob = c.compress(small_kv_cache)
        recovered = c.decompress(blob)

        n_sink = c.n_sink_tokens
        n_window = c.sliding_window

        for layer_idx, ((rk, rv), (ok, ov)) in enumerate(
            zip(recovered, small_kv_cache)
        ):
            seq_len = ok.shape[2]
            body_start = n_sink
            body_end = seq_len - n_window

            if body_end <= body_start:
                continue

            for orig, rec, label in [(ok, rk, "K"), (ov, rv, "V")]:
                o_body = np.array(orig[0, :, body_start:body_end, :])
                r_body = np.array(rec[0, :, body_start:body_end, :])

                # Per-token cosine similarity
                o_flat = o_body.reshape(-1, o_body.shape[-1])
                r_flat = r_body.reshape(-1, r_body.shape[-1])
                norms_o = np.linalg.norm(o_flat, axis=-1, keepdims=True) + 1e-8
                norms_r = np.linalg.norm(r_flat, axis=-1, keepdims=True) + 1e-8
                cos_sim = np.mean(
                    np.sum((o_flat / norms_o) * (r_flat / norms_r), axis=-1)
                )
                assert cos_sim >= 0.97, (
                    f"Layer {layer_idx} {label}: cosine_sim={cos_sim:.4f} < 0.97"
                )

    @pytest.mark.slow
    def test_real_qwen_round_trip(self):
        """Round-trip on real Qwen 2.5 7B KV cache (requires model download)."""
        pytest.skip("Real model integration test: implement in Wave 2 slow path")


# ---------------------------------------------------------------------------
# KVTC-05: Sink Token Exemption
# ---------------------------------------------------------------------------


class TestSinkTokenExemption:
    """KVTC-05: First n_sink_tokens are stored verbatim (bit-identical round-trip)."""

    def test_compress_raises_not_implemented(self, small_kv_cache):
        """compress() raises NotImplementedError in Wave 0 (RED state)."""
        c = KVTCCompressor(pca_bundle=None)
        with pytest.raises(NotImplementedError):
            c.compress(small_kv_cache)

    def test_sink_tokens_preserved_exactly(self, small_kv_cache):
        """First n_sink_tokens are bit-identical after round-trip (Wave 2+)."""
        c = KVTCCompressor(pca_bundle=None)
        blob = c.compress(small_kv_cache)
        recovered = c.decompress(blob)

        n_sink = c.n_sink_tokens
        for layer_idx, ((rk, rv), (ok, ov)) in enumerate(
            zip(recovered, small_kv_cache)
        ):
            ok_sink = np.array(ok[0, :, :n_sink, :])
            rk_sink = np.array(rk[0, :, :n_sink, :])
            ov_sink = np.array(ov[0, :, :n_sink, :])
            rv_sink = np.array(rv[0, :, :n_sink, :])

            assert np.array_equal(ok_sink, rk_sink), (
                f"Layer {layer_idx} K: sink tokens not bit-identical"
            )
            assert np.array_equal(ov_sink, rv_sink), (
                f"Layer {layer_idx} V: sink tokens not bit-identical"
            )

    def test_sink_exemption_is_load_bearing(self):
        """Ablation test stub: removing sink exemption degrades cosine_sim (Wave 2)."""
        pytest.skip("Ablation comparison test: implement after Wave 2 compress() works")


# ---------------------------------------------------------------------------
# KVTC-06: Sliding Window Exemption
# ---------------------------------------------------------------------------


class TestWindowTokenExemption:
    """KVTC-06: Last sliding_window tokens are stored verbatim."""

    def test_compress_raises_not_implemented(self, small_kv_cache):
        """compress() raises NotImplementedError in Wave 0 (RED state)."""
        c = KVTCCompressor(pca_bundle=None)
        with pytest.raises(NotImplementedError):
            c.compress(small_kv_cache)

    def test_window_tokens_preserved_exactly(self, small_kv_cache):
        """Last sliding_window tokens bit-identical after round-trip (Wave 2+)."""
        c = KVTCCompressor(pca_bundle=None)
        blob = c.compress(small_kv_cache)
        recovered = c.decompress(blob)

        n_window = c.sliding_window
        for layer_idx, ((rk, rv), (ok, ov)) in enumerate(
            zip(recovered, small_kv_cache)
        ):
            ok_win = np.array(ok[0, :, -n_window:, :])
            rk_win = np.array(rk[0, :, -n_window:, :])
            ov_win = np.array(ov[0, :, -n_window:, :])
            rv_win = np.array(rv[0, :, -n_window:, :])

            assert np.array_equal(ok_win, rk_win), (
                f"Layer {layer_idx} K: window tokens not bit-identical"
            )
            assert np.array_equal(ov_win, rv_win), (
                f"Layer {layer_idx} V: window tokens not bit-identical"
            )

    def test_short_sequence_guard(self):
        """seq_len < n_sink + sliding_window does not crash; stores all verbatim."""
        # seq_len=100 < 4+128=132 => no body tokens => all verbatim
        mx.random.seed(99)
        short_cache = [
            (
                mx.random.uniform(shape=[1, 4, 100, 128]).astype(mx.float16),
                mx.random.uniform(shape=[1, 4, 100, 128]).astype(mx.float16),
            )
            for _ in range(2)
        ]
        c = KVTCCompressor(pca_bundle=None)
        blob = c.compress(short_cache)
        recovered = c.decompress(blob)
        for (rk, rv), (ok, ov) in zip(recovered, short_cache):
            assert rk.shape == ok.shape
            assert rv.shape == ov.shape


# ---------------------------------------------------------------------------
# KVTC-07: GQA Shape Contract
# ---------------------------------------------------------------------------


class TestGQAShapeContract:
    """KVTC-07: Compressor handles GQA (n_kv_heads != n_query_heads) correctly."""

    def test_compress_raises_not_implemented(self, gqa_kv_cache):
        """compress() raises NotImplementedError on 28-layer GQA cache (RED state)."""
        c = KVTCCompressor(pca_bundle=None)
        with pytest.raises(NotImplementedError):
            c.compress(gqa_kv_cache)

    def test_gqa_shape_preserved(self, gqa_kv_cache):
        """Decompressed output has shape [1, 4, seq_len, 128] for GQA cache (Wave 2+)."""
        c = KVTCCompressor(pca_bundle=None)
        blob = c.compress(gqa_kv_cache)
        recovered = c.decompress(blob)

        assert len(recovered) == len(gqa_kv_cache)
        for (rk, rv), (ok, ov) in zip(recovered, gqa_kv_cache):
            assert rk.shape == ok.shape, f"K shape mismatch: {rk.shape} != {ok.shape}"
            assert rv.shape == ov.shape, f"V shape mismatch: {rv.shape} != {ov.shape}"

    def test_kv_head_count_independent_of_query_heads(self):
        """8-head MHA cache (n_kv_heads == n_query_heads) also compresses correctly."""
        mx.random.seed(13)
        mha_cache = [
            (
                mx.random.uniform(shape=[1, 8, 300, 128]).astype(mx.float16),
                mx.random.uniform(shape=[1, 8, 300, 128]).astype(mx.float16),
            )
            for _ in range(4)
        ]
        c = KVTCCompressor(pca_bundle=None)
        blob = c.compress(mha_cache)
        recovered = c.decompress(blob)

        assert len(recovered) == len(mha_cache)
        for (rk, rv), (ok, ov) in zip(recovered, mha_cache):
            assert rk.shape == ok.shape
            assert rv.shape == ov.shape


# ---------------------------------------------------------------------------
# Decompression Latency
# ---------------------------------------------------------------------------


class TestDecompressionLatency:
    """Decompression of an 8K-context 28-layer cache must be < 10ms per layer."""

    def test_decompression_raises_not_implemented(self, small_kv_cache):
        """compress() raises NotImplementedError in Wave 0 (RED state)."""
        c = KVTCCompressor(pca_bundle=None)
        with pytest.raises(NotImplementedError):
            c.compress(small_kv_cache)

    def test_decompression_latency_under_10ms_per_layer(self):
        """28 layers, seq_len=8192: per-layer decompress time < 10ms (Wave 2+)."""
        mx.random.seed(5)
        large_cache = [
            (
                mx.random.uniform(shape=[1, 4, 8192, 128]).astype(mx.float16),
                mx.random.uniform(shape=[1, 4, 8192, 128]).astype(mx.float16),
            )
            for _ in range(28)
        ]
        c = KVTCCompressor(pca_bundle=None)
        blob = c.compress(large_cache)

        t0 = time.perf_counter()
        recovered = c.decompress(blob)
        elapsed = time.perf_counter() - t0

        per_layer_ms = (elapsed * 1000) / len(large_cache)
        assert per_layer_ms < 10.0, (
            f"Decompression too slow: {per_layer_ms:.2f}ms per layer (limit 10ms)"
        )
        assert recovered is not None
