# SPDX-License-Identifier: Apache-2.0
"""
Phase 6 cache integration test scaffold (Wave 0 RED state).

All tests are stubs that will fail on collection because the modules they
import do not exist yet. As plans 06-02 through 06-04 complete, these tests
will turn GREEN one class at a time.
"""

import subprocess

import numpy as np
import pytest
from omlx.compression.config import CompressionConfig
from omlx.compression.compressed_cache_manager import (
    CompressedPagedSSDCacheManager,
    CompressedTieredCacheManager,
)


def _make_kv_cache(num_layers=4, seq_len=64, num_heads=8, head_dim=128):
    """Return a list of (keys, values) tuples — synthetic float32 tensors."""
    import mlx.core as mx

    return [
        (
            mx.random.normal([1, seq_len, num_heads, head_dim]),
            mx.random.normal([1, seq_len, num_heads, head_dim]),
        )
        for _ in range(num_layers)
    ]


class TestCompressedSaveLoad:
    """PIPE-06: Compressed save/load round-trip via CompressedPagedSSDCacheManager."""

    def test_compressed_save_block(self, tmp_path):
        """Save a synthetic KV cache; assert return True and metadata marks compressed=true."""
        raise NotImplementedError("test_compressed_save_block not yet implemented")

    def test_compressed_load_block(self, tmp_path):
        """Save then load via CompressedPagedSSDCacheManager; assert non-None, correct layer count."""
        raise NotImplementedError("test_compressed_load_block not yet implemented")

    def test_decompression_failure_returns_cache_miss(self, tmp_path):
        """Mock pipeline.decompress to raise RuntimeError; assert load_block returns None."""
        raise NotImplementedError(
            "test_decompression_failure_returns_cache_miss not yet implemented"
        )


class TestNoOpPath:
    """PIPE-07: When compression is disabled, factory returns plain PagedSSDCacheManager."""

    def test_no_op_path_vanilla_factory(self, tmp_path):
        """CompressionConfig(enabled=False) → CacheFactory.create_full_cache_stack() returns PagedSSDCacheManager."""
        raise NotImplementedError("test_no_op_path_vanilla_factory not yet implemented")

    def test_existing_tests_unaffected(self):
        """Existing paged-SSD cache tests pass (exit code 0) when compression is disabled by default."""
        result = subprocess.run(
            [
                "pytest",
                "tests/test_paged_ssd_cache.py",
                "-x",
                "-q",
                "--tb=no",
            ],
            capture_output=True,
            text=True,
        )
        raise NotImplementedError(
            "test_existing_tests_unaffected stub — actual enforcement via full suite run"
        )


class TestRuntimeToggle:
    """PIPE-08: CompressionConfig.set_enabled() toggles compression at runtime."""

    def test_runtime_toggle(self):
        """set_enabled(False) then set_enabled(True) round-trip asserts on config.enabled."""
        raise NotImplementedError("test_runtime_toggle not yet implemented")


class TestCompressionConfig:
    """PIPE-09: CompressionConfig stores and exposes all required fields."""

    def test_compression_config_fields(self):
        """Custom am_ratio=8.0, n_components=32 are stored correctly."""
        raise NotImplementedError("test_compression_config_fields not yet implemented")

    def test_compression_config_defaults(self):
        """Default config: enabled=False, am_ratio=4.0, n_sink_tokens=4, sliding_window=128."""
        raise NotImplementedError("test_compression_config_defaults not yet implemented")


class TestAdminEndpoint:
    """PIPE-08 runtime: Admin endpoint for toggling compression."""

    def test_admin_endpoint(self):
        """Admin endpoint wires to runtime toggle (stub — endpoint not yet implemented)."""
        raise NotImplementedError("admin endpoint not yet wired")


class TestDecompressionLatency:
    """PIPE-10: Decompression latency must be under 10 ms for cached blocks."""

    @pytest.mark.slow
    def test_decompression_latency_under_10ms(self, tmp_path):
        """End-to-end decompression latency < 10 ms (stub — benchmark not yet implemented)."""
        raise NotImplementedError("latency test not yet implemented")
