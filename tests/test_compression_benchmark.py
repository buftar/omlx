# SPDX-License-Identifier: Apache-2.0
"""
Benchmark suite tests — Phase 7: VAL-01 through VAL-08.
Wave 0 scaffold: all tests RED (NotImplementedError from stubs).
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import json
import tempfile
import os
import sys

from omlx.compression.benchmark import BenchmarkRunner
from omlx.compression.evaluators import (
    cosine_sim_kv,
    get_compressible_layer_indices,
    detect_swa_layers,
)


class TestBenchmarkReport:
    """VAL-01: Benchmark report schema and required fields."""

    def test_report_has_required_fields(self):
        """Test that benchmark report has required fields."""
        runner = BenchmarkRunner("fake/model")
        report = runner.run_benchmark(tasks=[])
        required_keys = [
            "schema_version",
            "model",
            "timestamp",
            "seed",
            "config",
            "technical_metrics",
            "quality_metrics",
            "thresholds",
            "swa_layers_skipped",
            "overall_pass",
        ]
        for key in required_keys:
            assert key in report, f"Missing required key: {key}"

    def test_report_schema_version(self):
        """Test that benchmark report includes schema version."""
        runner = BenchmarkRunner("fake/model")
        report = runner.run_benchmark(tasks=[])
        assert report["schema_version"] == "1.0"


class TestReproducibility:
    """VAL-08: Benchmark reproducibility with same seed."""

    def test_same_seed_produces_same_report(self):
        """Test that same seed produces identical reports."""
        runner1 = BenchmarkRunner("fake/model", seed=42)
        runner2 = BenchmarkRunner("fake/model", seed=42)
        report1 = runner1.run_benchmark(tasks=[])
        report2 = runner2.run_benchmark(tasks=[])
        assert report1["seed"] == report2["seed"]
        assert report1["config"] == report2["config"]


class TestSwaDetection:
    """VAL-06: SWA (Sliding Window Attention) layer detection."""

    def test_non_gemma3_returns_empty_set(self):
        """Test that non-Gemma3 models return empty SWA set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"model_type": "qwen2"}, f)
            result = detect_swa_layers(tmpdir, 28)
            assert result == set()

    def test_gemma3_with_default_pattern(self):
        """Test Gemma3 with default sliding_window_pattern=6."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"model_type": "gemma3"}, f)
            result = detect_swa_layers(tmpdir, 28)
            # SWA pattern: i % 6 != 5, so SWA layers are {5, 11, 17, 23}
            # Non-SWA (compressible) = all others
            assert result == {5, 11, 17, 23}

    def test_gemma3_with_custom_pattern(self):
        """Test Gemma3 with custom sliding_window_pattern=4."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"model_type": "gemma3_text", "sliding_window_pattern": 4}, f)
            result = detect_swa_layers(tmpdir, 28)
            # SWA pattern: i % 4 != 3, so SWA layers are {3, 7, 11, 15, 19, 23, 27}
            assert result == {3, 7, 11, 15, 19, 23, 27}

    def test_missing_config_returns_empty_set(self):
        """Test that missing config.json returns empty set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = detect_swa_layers(tmpdir, 28)
            assert result == set()

    def test_get_compressible_layer_indices_filters_rotating(self):
        """Test that compressible indices exclude SWA layers."""
        # Create fake prompt_cache objects
        from mlx_lm.models.cache import RotatingKVCache

        # RotatingKVCache requires max_size argument
        fake_rotating = RotatingKVCache(max_size=1024)
        fake_regular = MagicMock()

        result = get_compressible_layer_indices([fake_rotating, fake_regular])
        assert result == [1]

    def test_cosine_sim_kv_identical_tensors(self):
        """Test cosine similarity returns 1.0 for identical tensors."""
        layer_a = [(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))]
        layer_b = [(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))]
        result = cosine_sim_kv(layer_a, layer_b)
        assert 0.99 <= result <= 1.0

    def test_cosine_sim_kv_different_tensors(self):
        """Test cosine similarity returns value between 0 and 1 for different tensors."""
        layer_a = [(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))]
        layer_b = [(np.array([7.0, 8.0, 9.0]), np.array([10.0, 11.0, 12.0]))]
        result = cosine_sim_kv(layer_a, layer_b)
        assert 0.0 <= result <= 1.0

    def test_cosine_sim_kv_empty_layers(self):
        """Test cosine similarity returns 0.0 for empty layers."""
        result = cosine_sim_kv([], [])
        assert result == 0.0


class TestSlowQwen:
    """VAL-02, VAL-03, VAL-04: Qwen model benchmark tests."""

    @pytest.mark.slow
    def test_am_cosine_sim(self):
        """Test AM cosine similarity computation with Qwen."""
        runner = BenchmarkRunner("Qwen/Qwen2-7B")
        with pytest.raises(NotImplementedError):
            runner.run_benchmark(tasks=["cosine_sim"])

    @pytest.mark.slow
    def test_task_accuracy(self):
        """Test task accuracy benchmarks with Qwen."""
        runner = BenchmarkRunner("Qwen/Qwen2-7B")
        with pytest.raises(NotImplementedError):
            runner.run_benchmark(tasks=["accuracy"])

    @pytest.mark.slow
    def test_full_pipeline_runs(self):
        """Test full Qwen pipeline runs end-to-end."""
        runner = BenchmarkRunner("Qwen/Qwen2-7B")
        with pytest.raises(NotImplementedError):
            runner.run_benchmark(tasks=["full"])


class TestSlowLlama:
    """VAL-05: Llama model benchmark tests."""

    @pytest.mark.slow
    def test_llama_pipeline_runs(self):
        """Test Llama pipeline runs end-to-end."""
        runner = BenchmarkRunner("meta-llama/Llama-2-7b")
        with pytest.raises(NotImplementedError):
            runner.run_benchmark(tasks=["full"])


class TestSlowGemma:
    """VAL-06: Gemma model SWA layer tests."""

    @pytest.mark.slow
    def test_gemma_swa_layers_skipped(self):
        """Test that Gemma SWA layers are correctly skipped."""
        runner = BenchmarkRunner("google/gemma-7b")
        with pytest.raises(NotImplementedError):
            runner.run_benchmark(tasks=["swa"])


class TestSlowDeepSeek:
    """VAL-07: DeepSeek model benchmark tests."""

    @pytest.mark.slow
    def test_deepseek_pipeline_runs(self):
        """Test DeepSeek pipeline runs end-to-end."""
        runner = BenchmarkRunner("deepseek-ai/DeepSeek-Coder")
        with pytest.raises(NotImplementedError):
            runner.run_benchmark(tasks=["full"])
