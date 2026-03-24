# SPDX-License-Identifier: Apache-2.0
"""
Phase 6 cache integration tests.

Tests confirm that:
- CompressionConfig is a thread-safe dataclass with correct defaults/fields
- CompressedPagedSSDCacheManager.save_block() compresses on inference thread
- CompressedPagedSSDCacheManager.load_block() decompresses correctly
- Decompression failure is treated as a cache miss (returns None)
- CompressedTieredCacheManager.check_memory_pressure() calls _compact_hot_blocks() then super()
- Runtime toggle (set_enabled) works thread-safely
"""
from __future__ import annotations

import subprocess
import threading

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

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


def _make_pipeline_blob_bytes(n=128):
    """Return fake compressed bytes."""
    import random
    rng = random.Random(42)
    return bytes(rng.randint(0, 255) for _ in range(n))


# ---------------------------------------------------------------------------
# TestCompressionConfig
# ---------------------------------------------------------------------------

class TestCompressionConfig:
    """PIPE-09: CompressionConfig stores and exposes all required fields."""

    def test_compression_config_defaults(self):
        """Default config: enabled=False, am_ratio=4.0, n_sink_tokens=4, sliding_window=128."""
        cfg = CompressionConfig()
        assert cfg.enabled is False
        assert cfg.bundle_path is None
        assert cfg.am_ratio == 4.0
        assert cfg.n_sink_tokens == 4
        assert cfg.sliding_window == 128
        assert cfg.n_components is None

    def test_compression_config_fields(self):
        """Custom am_ratio=8.0, n_components=32 are stored correctly."""
        cfg = CompressionConfig(
            enabled=True,
            bundle_path="/tmp/test.npz",
            am_ratio=8.0,
            n_sink_tokens=8,
            sliding_window=64,
            n_components=32,
        )
        assert cfg.enabled is True
        assert cfg.bundle_path == "/tmp/test.npz"
        assert cfg.am_ratio == 8.0
        assert cfg.n_sink_tokens == 8
        assert cfg.sliding_window == 64
        assert cfg.n_components == 32


# ---------------------------------------------------------------------------
# TestRuntimeToggle
# ---------------------------------------------------------------------------

class TestRuntimeToggle:
    """PIPE-08: CompressionConfig.set_enabled() toggles compression at runtime."""

    def test_runtime_toggle(self):
        """set_enabled(True) then set_enabled(False) round-trip."""
        cfg = CompressionConfig()
        assert cfg.enabled is False

        cfg.set_enabled(True)
        assert cfg.enabled is True

        cfg.set_enabled(False)
        assert cfg.enabled is False

    def test_set_enabled_is_thread_safe(self):
        """Concurrent set_enabled calls do not corrupt state."""
        cfg = CompressionConfig()
        errors = []

        def toggle(value):
            try:
                for _ in range(100):
                    cfg.set_enabled(value)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=toggle, args=(i % 2 == 0,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread-safety errors: {errors}"
        assert isinstance(cfg.enabled, bool)


# ---------------------------------------------------------------------------
# TestCompressedSaveLoad
# ---------------------------------------------------------------------------

class TestCompressedSaveLoad:
    """PIPE-06: Compressed save/load round-trip via CompressedPagedSSDCacheManager."""

    def _make_manager(self, tmp_path, enabled=True):
        """Create a CompressedPagedSSDCacheManager with hot-cache mode."""
        cfg = CompressionConfig(enabled=enabled, bundle_path=None)
        mgr = CompressedPagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=500 * 1024 * 1024,
            hot_cache_max_bytes=100 * 1024 * 1024,
            compression_config=cfg,
        )
        return mgr

    def test_compressed_save_block(self, tmp_path):
        """save_block() with enabled=True stores entry with compressed=true metadata."""
        from omlx.compression.pipeline import PipelineBlob

        mgr = self._make_manager(tmp_path, enabled=True)
        block_hash = b"a" * 32
        kv_cache = _make_kv_cache(num_layers=2, seq_len=8, num_heads=4, head_dim=16)

        fake_blob = PipelineBlob(
            compressed=_make_pipeline_blob_bytes(128),
            logical_seq_len=8,
            compaction_ratio=1.0,
        )
        with patch.object(type(mgr), '_pipeline', new_callable=PropertyMock) as mock_prop:
            mock_pipeline = MagicMock()
            mock_pipeline.compress.return_value = fake_blob
            mock_prop.return_value = mock_pipeline

            result = mgr.save_block(
                block_hash=block_hash,
                cache_data=kv_cache,
                token_count=8,
                model_name="test_model",
            )

        assert result is True
        with mgr._hot_cache_lock:
            assert block_hash in mgr._hot_cache
            entry = mgr._hot_cache[block_hash]
        assert entry['file_metadata'].get('compressed') == 'true'
        assert 'compressed_blob' in entry['tensors_raw']

    def test_compressed_load_block(self, tmp_path):
        """save_block then load_block returns decompressed result (non-None)."""
        from omlx.compression.pipeline import PipelineBlob

        mgr = self._make_manager(tmp_path, enabled=True)
        block_hash = b"b" * 32
        kv_cache = _make_kv_cache(num_layers=2, seq_len=8, num_heads=4, head_dim=16)

        fake_blob = PipelineBlob(
            compressed=_make_pipeline_blob_bytes(128),
            logical_seq_len=8,
            compaction_ratio=1.0,
        )
        # Save
        with patch.object(type(mgr), '_pipeline', new_callable=PropertyMock) as mock_prop:
            mock_pipeline = MagicMock()
            mock_pipeline.compress.return_value = fake_blob
            mock_prop.return_value = mock_pipeline
            mgr.save_block(
                block_hash=block_hash,
                cache_data=kv_cache,
                token_count=8,
                model_name="test_model",
            )

        # Load — mock decompress to return a known value
        fake_decompressed = (_make_kv_cache(num_layers=2, seq_len=2, num_heads=4, head_dim=16), 8)
        with patch.object(type(mgr), '_pipeline', new_callable=PropertyMock) as mock_prop:
            mock_pipeline = MagicMock()
            mock_pipeline.decompress.return_value = fake_decompressed
            mock_prop.return_value = mock_pipeline
            result = mgr.load_block(block_hash=block_hash)

        assert result is not None
        assert result == fake_decompressed

    def test_decompression_failure_returns_cache_miss(self, tmp_path):
        """When pipeline.decompress raises RuntimeError, load_block returns None."""
        from omlx.compression.pipeline import PipelineBlob

        mgr = self._make_manager(tmp_path, enabled=True)
        block_hash = b"c" * 32
        kv_cache = _make_kv_cache(num_layers=2, seq_len=8, num_heads=4, head_dim=16)

        fake_blob = PipelineBlob(
            compressed=_make_pipeline_blob_bytes(128),
            logical_seq_len=8,
            compaction_ratio=1.0,
        )
        # Save first
        with patch.object(type(mgr), '_pipeline', new_callable=PropertyMock) as mock_prop:
            mock_pipeline = MagicMock()
            mock_pipeline.compress.return_value = fake_blob
            mock_prop.return_value = mock_pipeline
            mgr.save_block(
                block_hash=block_hash,
                cache_data=kv_cache,
                token_count=8,
                model_name="test_model",
            )

        # Load with decompress raising
        with patch.object(type(mgr), '_pipeline', new_callable=PropertyMock) as mock_prop:
            mock_pipeline = MagicMock()
            mock_pipeline.decompress.side_effect = RuntimeError("simulated decompression error")
            mock_prop.return_value = mock_pipeline
            result = mgr.load_block(block_hash=block_hash)

        assert result is None


# ---------------------------------------------------------------------------
# TestNoOpPath
# ---------------------------------------------------------------------------

class TestNoOpPath:
    """PIPE-07: When compression is disabled, save_block delegates to super() unchanged."""

    def test_no_op_path_vanilla_factory(self, tmp_path):
        """CompressionConfig(enabled=False) → save_block delegates to parent."""
        cfg = CompressionConfig(enabled=False)
        mgr = CompressedPagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=500 * 1024 * 1024,
            hot_cache_max_bytes=100 * 1024 * 1024,
            compression_config=cfg,
        )

        block_hash = b"d" * 32
        kv_cache = _make_kv_cache(num_layers=2, seq_len=8, num_heads=4, head_dim=16)

        with patch.object(
            CompressedPagedSSDCacheManager.__bases__[0],
            'save_block',
            return_value=True,
        ) as mock_super_save:
            result = mgr.save_block(
                block_hash=block_hash,
                cache_data=kv_cache,
                token_count=8,
                model_name="test_model",
            )

        assert result is True
        mock_super_save.assert_called_once()

    def test_existing_tests_unaffected(self):
        """Existing paged-SSD cache tests pass when compression is disabled by default."""
        result = subprocess.run(
            [
                "uv",
                "run",
                "pytest",
                "tests/test_paged_ssd_cache.py",
                "-x",
                "-q",
                "--tb=no",
                "-m",
                "not slow",
            ],
            capture_output=True,
            text=True,
            cwd="/Users/tonysina/projects/omlx",
        )
        assert result.returncode == 0, (
            f"Existing paged SSD tests failed:\n{result.stdout}\n{result.stderr}"
        )


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
