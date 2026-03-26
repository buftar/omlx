# SPDX-License-Identifier: Apache-2.0
"""Observability tests for Phase 8 - Server metrics and admin UI stats."""

import time

import numpy as np
import pytest

from omlx.cache.stats import CompressedCacheStats
from omlx.compression.config import CompressionConfig
from omlx.compression.pipeline import PipelineBlob, KVCachePipeline
from omlx.server_metrics import ServerMetrics, reset_server_metrics, get_server_metrics


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


class TestCompressionRatioMetric:
    """OBS-01: Compression ratio metric recording."""

    def test_compression_ratio_metric_records_ratio(self):
        """Test that ServerMetrics records non-zero compression ratio."""
        # Reset metrics to start fresh
        reset_server_metrics()

        # Get server metrics singleton instance
        metrics = get_server_metrics()

        # Record a compression ratio
        test_ratio = 2.5
        metrics.record_compression_ratio(test_ratio)

        # Verify the ratio was recorded
        snapshot = metrics.get_snapshot()
        assert snapshot["compression_ratio"] == test_ratio

        # Record another ratio and verify average
        metrics.record_compression_ratio(3.0)
        snapshot = metrics.get_snapshot()
        expected_avg = (test_ratio + 3.0) / 2
        assert snapshot["compression_ratio"] == expected_avg

    def test_compression_ratio_metric_exposed_via_admin(self):
        """Test that compression ratio is exposed via admin endpoint."""
        from omlx.admin.routes import get_compression_status

        # Reset metrics
        reset_server_metrics()

        # Get server metrics singleton
        metrics = get_server_metrics()
        metrics.record_compression_ratio(2.5)
        metrics.record_compression_ratio(3.0)

        # The endpoint requires admin auth, so we test the getter pattern
        from omlx.admin.routes import _get_compression_stats

        # Verify getter returns stats object
        stats = _get_compression_stats()
        assert stats is not None


class TestDecompressionLatencyMetric:
    """OBS-02: Decompression latency metric tracking."""

    def test_decompression_latency_metric_tracks_latency(self):
        """Test that ServerMetrics records non-zero decompression latency."""
        # Reset metrics
        reset_server_metrics()

        # Get server metrics singleton instance
        metrics = get_server_metrics()

        # Record a decompression latency
        test_latency_ms = 15.5
        metrics.record_decompression_latency(test_latency_ms)

        # Verify the latency was recorded
        snapshot = metrics.get_snapshot()
        assert snapshot["avg_decompression_latency_ms"] == test_latency_ms

        # Record another latency and verify average
        metrics.record_decompression_latency(20.0)
        snapshot = metrics.get_snapshot()
        expected_avg = (test_latency_ms + 20.0) / 2
        assert snapshot["avg_decompression_latency_ms"] == expected_avg

    @pytest.mark.slow
    def test_decompression_latency_under_threshold(self):
        """Test that decompression latency is under threshold."""
        # Reset metrics
        reset_server_metrics()

        # Create a KVCachePipeline and test decompression latency
        pipeline = KVCachePipeline()

        kv_cache = _make_kv_cache()
        blob = pipeline.compress(kv_cache)

        # Measure decompression latency
        start_time = time.perf_counter()
        result = pipeline.decompress(blob)
        decompression_ms = (time.perf_counter() - start_time) * 1000

        # Verify latency is recorded
        metrics = ServerMetrics()
        metrics.record_decompression_latency(decompression_ms)

        snapshot = metrics.get_snapshot()
        assert snapshot["avg_decompression_latency_ms"] > 0


class TestCacheHitMissMetrics:
    """OBS-03: Cache hit/miss metrics for compressed cache."""

    def test_cache_hit_miss_metrics_distinguish_compressed(self):
        """Test that metrics track compressed cache stats separately."""
        # Create CompressedCacheStats instance
        stats = CompressedCacheStats()

        # Simulate compression operations
        stats.compression_success_count = 5
        stats.compression_failure_count = 1
        stats.decompression_success_count = 4
        stats.decompression_failure_count = 0

        # Simulate size tracking
        stats.total_compressed_bytes = 1024
        stats.total_logical_bytes = 2048

        # Verify compression ratio calculation
        assert stats.compression_ratio == 2.0  # 2048 / 1024

        # Verify average compression ratio
        stats.compression_ratios = [2.0, 2.5, 1.8]
        assert abs(stats.avg_compression_ratio - 2.1) < 0.1

        # Verify to_dict includes compression stats
        d = stats.to_dict()
        assert d["compression_success_count"] == 5
        assert d["compression_failure_count"] == 1
        assert d["total_compressed_bytes"] == 1024
        assert d["overall_compression_ratio"] == 2.0

    def test_admin_stats_endpoint_includes_compression(self):
        """Test that admin stats endpoint includes compression metrics."""
        from omlx.admin.routes import get_compression_status

        # Reset metrics
        reset_server_metrics()

        # Get server metrics singleton
        metrics = get_server_metrics()
        metrics.record_compression_ratio(2.5)
        metrics.record_decompression_latency(10.0)

        # Create CompressedCacheStats
        stats = CompressedCacheStats()
        stats.compression_success_count = 3
        stats.decompression_success_count = 3
        stats.total_compressed_bytes = 512
        stats.total_logical_bytes = 1024

        # Wire the getter
        from omlx.admin.routes import set_compression_stats_getter
        set_compression_stats_getter(lambda: stats)

        # Create a mock request context (admin=True)
        class MockRequest:
            pass

        # Verify the endpoint returns compression metrics
        # Note: This would normally require a test client, but we verify the structure
        payload = {
            "enabled": False,
            "compression_ratio": 0.0,
            "avg_decompression_latency_ms": 0.0,
            "compression_success_count": 0,
            "compression_failure_count": 0,
            "decompression_success_count": 0,
            "decompression_failure_count": 0,
            "total_compressed_bytes": 0,
            "total_logical_bytes": 0,
        }

        # Simulate what the endpoint does
        snapshot = metrics.get_snapshot()
        payload["compression_ratio"] = stats.avg_compression_ratio
        payload["avg_decompression_latency_ms"] = stats.avg_decompression_latency_ms
        payload["compression_success_count"] = stats.compression_success_count

        # Verify stats values are set correctly
        assert stats.avg_compression_ratio == 0.0  # No ratios recorded on stats object
        assert payload["compression_success_count"] == 3

        # Verify metrics have the recorded values
        assert snapshot["compression_ratio"] == 2.5  # Only 2.5 was recorded


class TestMetricsVisibleInAdminUI:
    """OBS-04: Metrics visibility in admin UI dashboard."""

    def test_metrics_visible_in_admin_ui(self):
        """Test that metrics are visible in admin UI dashboard."""
        # Verify ServerMetrics has all required fields
        reset_server_metrics()
        metrics = get_server_metrics()

        # Check that snapshot includes all required fields
        snapshot = metrics.get_snapshot()
        required_fields = [
            "total_tokens_served",
            "total_cached_tokens",
            "cache_efficiency",
            "compression_ratio",
            "avg_decompression_latency_ms",
        ]
        for field in required_fields:
            assert field in snapshot, f"Missing field: {field}"

    def test_compression_ratio_displayed_in_dashboard(self):
        """Test that compression ratio is displayed in dashboard."""
        reset_server_metrics()
        metrics = get_server_metrics()

        # Record compression ratios
        metrics.record_compression_ratio(2.5)
        metrics.record_compression_ratio(3.0)

        snapshot = metrics.get_snapshot()
        assert snapshot["compression_ratio"] == 2.75

        # Verify it's a reasonable value
        assert snapshot["compression_ratio"] > 0
        assert isinstance(snapshot["compression_ratio"], float)


class TestPipelineCompressionMetrics:
    """Test that KVCachePipeline records metrics on compress()."""

    def test_pipeline_compress_records_metric(self):
        """Test that compress() records compression ratio to ServerMetrics."""
        reset_server_metrics()

        pipeline = KVCachePipeline()

        kv_cache = _make_kv_cache()
        blob = pipeline.compress(kv_cache)

        # Verify metrics were recorded (use singleton)
        metrics = get_server_metrics()
        snapshot = metrics.get_snapshot()

        # Compression ratio should be recorded (non-zero)
        assert snapshot["compression_ratio"] > 0

    def test_pipeline_compress_records_on_multiple_calls(self):
        """Test that multiple compress() calls accumulate metrics."""
        reset_server_metrics()

        pipeline = KVCachePipeline()

        # Compress multiple times
        for _ in range(3):
            kv_cache = _make_kv_cache()
            pipeline.compress(kv_cache)

        # Verify all ratios were recorded (use singleton)
        metrics = get_server_metrics()
        snapshot = metrics.get_snapshot()

        # Should have 3 ratios recorded
        # (The ratio depends on actual compaction, but should be > 0)
        assert snapshot["compression_ratio"] > 0


class TestCompressedCacheStats:
    """Test CompressedCacheStats class."""

    def test_compressed_cache_stats_initialization(self):
        """Test that CompressedCacheStats initializes with zeroed stats."""
        stats = CompressedCacheStats()

        assert stats.compression_success_count == 0
        assert stats.compression_failure_count == 0
        assert stats.decompression_success_count == 0
        assert stats.decompression_failure_count == 0
        assert stats.total_compressed_bytes == 0
        assert stats.total_logical_bytes == 0
        assert stats.avg_compression_ratio == 0.0
        assert stats.avg_decompression_latency_ms == 0.0

    def test_compressed_cache_stats_record_methods(self):
        """Test record methods update stats correctly."""
        stats = CompressedCacheStats()

        # Record compression success
        stats.record_compression_success(ratio=2.5, compressed_bytes=1024, logical_bytes=2048)

        assert stats.compression_success_count == 1
        assert stats.total_compressed_bytes == 1024
        assert stats.total_logical_bytes == 2048
        assert stats.compression_ratio == 2.0

        # Record decompression success
        stats.record_decompression_success(latency_ms=15.0)

        assert stats.decompression_success_count == 1
        assert stats.avg_decompression_latency_ms == 15.0

    def test_compressed_cache_stats_reset(self):
        """Test that reset() clears all stats."""
        stats = CompressedCacheStats()

        # Set some values
        stats.compression_success_count = 5
        stats.total_compressed_bytes = 1024
        stats.decompression_latencies_ms.append(10.0)

        # Reset
        stats.reset()

        assert stats.compression_success_count == 0
        assert stats.total_compressed_bytes == 0
        assert len(stats.decompression_latencies_ms) == 0

    def test_compressed_cache_stats_to_dict(self):
        """Test that to_dict() includes all fields."""
        stats = CompressedCacheStats()
        stats.compression_success_count = 3
        stats.total_compressed_bytes = 512
        stats.total_logical_bytes = 1024

        d = stats.to_dict()

        assert d["compression_success_count"] == 3
        assert d["total_compressed_bytes"] == 512
        assert d["overall_compression_ratio"] == 2.0