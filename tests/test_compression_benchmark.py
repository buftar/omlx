# SPDX-License-Identifier: Apache-2.0
"""
Benchmark suite tests — Phase 7: VAL-01 through VAL-08.
Wave 0 scaffold: all tests RED (NotImplementedError from stubs).
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

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
        with pytest.raises(NotImplementedError):
            runner.run_benchmark(tasks=["cosine_sim"])

    def test_report_schema_version(self):
        """Test that benchmark report includes schema version."""
        runner = BenchmarkRunner("fake/model")
        with pytest.raises(NotImplementedError):
            runner.run_benchmark(tasks=["schema"])


class TestReproducibility:
    """VAL-08: Benchmark reproducibility with same seed."""

    def test_same_seed_produces_same_report(self):
        """Test that same seed produces identical reports."""
        runner1 = BenchmarkRunner("fake/model", seed=42)
        runner2 = BenchmarkRunner("fake/model", seed=42)
        with pytest.raises(NotImplementedError):
            runner1.run_benchmark()
        with pytest.raises(NotImplementedError):
            runner2.run_benchmark()


class TestSwaDetection:
    """VAL-06: SWA (Sliding Window Attention) layer detection."""

    def test_non_gemma3_returns_empty_set(self):
        """Test that non-Gemma3 models return empty SWA set."""
        with patch("omlx.compression.evaluators.detect_swa_layers") as mock_detect:
            mock_detect.side_effect = NotImplementedError("detect_swa_layers not yet implemented")
            with pytest.raises(NotImplementedError):
                detect_swa_layers("/fake/path/non-gemma", 28)

    def test_get_compressible_layer_indices_filters_rotating(self):
        """Test that compressible indices exclude SWA layers."""
        # Create fake prompt_cache objects
        fake_layer1 = MagicMock()
        fake_layer1.is_swa = False
        fake_layer2 = MagicMock()
        fake_layer2.is_swa = True

        with pytest.raises(NotImplementedError):
            get_compressible_layer_indices([fake_layer1, fake_layer2])


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
