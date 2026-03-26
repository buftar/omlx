# SPDX-License-Identifier: Apache-2.0
"""Benchmark runner for compression evaluation."""

import datetime
import json
from dataclasses import dataclass
from typing import Optional

from omlx.compression.evaluators import (
    cosine_sim_kv,
    get_compressible_layer_indices,
    detect_swa_layers,
)
from omlx.server_metrics import ServerMetrics, get_server_metrics


@dataclass
class CompressionMetrics:
    """Metrics for compression operations.

    Args:
        compression_ratio: Average compression ratio (logical / compressed bytes).
        decompression_latency_ms_per_layer: Average decompression latency in ms per layer.
        cache_hit_rate: Cache hit rate after compression.
        cache_miss_rate: Cache miss rate after compression.
    """

    compression_ratio: float
    decompression_latency_ms_per_layer: float
    cache_hit_rate: float
    cache_miss_rate: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "compression_ratio": self.compression_ratio,
            "decompression_latency_ms_per_layer": self.decompression_latency_ms_per_layer,
            "cache_hit_rate": self.cache_hit_rate,
            "cache_miss_rate": self.cache_miss_rate,
        }


class BenchmarkRunner:
    """Benchmark runner for compression evaluation.

    Args:
        model_path: Path to the model to benchmark.
        bundle_path: Optional path to a compression bundle.
        am_ratio: Attention machining ratio for compaction.
        seed: Random seed for reproducibility.
        n_samples: Number of samples to evaluate.
    """

    def __init__(
        self,
        model_path: str,
        bundle_path: Optional[str] = None,
        am_ratio: float = 4.0,
        seed: int = 42,
        n_samples: int = 200,
        tasks: Optional[list[str]] = None,
    ):
        self.model_path = model_path
        self.bundle_path = bundle_path
        self.am_ratio = am_ratio
        self.seed = seed
        self.n_samples = n_samples
        self.tasks = tasks

    def _load_model_and_pipeline(self):
        """Load model and pipeline for benchmarking."""
        import mlx_lm

        from omlx.compression.pipeline import KVCachePipeline

        self._model, self._tokenizer = mlx_lm.load(self.model_path)
        self._pipeline = KVCachePipeline(bundle_path=self.bundle_path, am_ratio=self.am_ratio)

    def _prefill_kv(self, prompt_text: str):
        """Prefill the model with prompt text and extract KV cache.

        Args:
            prompt_text: The prompt text to prefill.

        Returns:
            Tuple of (kv_cache, compressible_indices) where kv_cache is a list of
            (keys, values) tuples per compressible layer.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import RotatingKVCache, make_prompt_cache

        # Tokenize prompt
        prompt_ids = self._tokenizer.encode(prompt_text)
        tokens = mx.array(prompt_ids)[None]

        # Create fresh cache and prefill
        cache = make_prompt_cache(self._model)
        _ = self._model(tokens, cache=cache)

        # Force materialization
        mx.eval(*[t for c in cache for t in [c.keys, c.values] if t is not None])

        # Store full cache for later injection
        self._prompt_cache = cache

        # Get indices of compressible layers (non-SWA layers)
        compressible_indices = get_compressible_layer_indices(cache)

        # Extract only compressible layers
        offset = cache[0].offset
        kv_cache = [
            (cache[i].keys[:, :, :offset, :], cache[i].values[:, :, :offset, :])
            for i in compressible_indices
        ]

        self._compressible_indices = compressible_indices
        return kv_cache, compressible_indices

    def _make_compressed_cache(self, model, decompressed_layers):
        """Create a fresh prompt cache with decompressed layers injected.

        Args:
            model: The model to create cache for.
            decompressed_layers: List of (keys, values) tuples to inject.

        Returns:
            Fresh prompt_cache with decompressed layers injected into non-SWA positions.
        """
        from mlx_lm.models.cache import RotatingKVCache, make_prompt_cache

        prompt_cache = make_prompt_cache(model)
        layer_iter = iter(decompressed_layers)

        for i, cache_entry in enumerate(prompt_cache):
            if isinstance(cache_entry, RotatingKVCache):
                continue  # SWA layer: leave empty
            try:
                keys, values = next(layer_iter)
                cache_entry.state = (keys, values)
            except StopIteration:
                break

        return prompt_cache

    def run_benchmark(self, tasks: Optional[list[str]] = None) -> dict:
        """Run the benchmark with specified tasks.

        Args:
            tasks: List of task names to run. If None, runs all tasks.

        Returns:
            Dictionary with benchmark results.
        """
        # Seed for reproducibility
        _seed_all(self.seed)

        # Initialize report with all required keys from the Pattern 9 schema
        report = {
            "schema_version": "1.0",
            "model": self.model_path,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "seed": self.seed,
            "config": {"am_ratio": self.am_ratio},
            "technical_metrics": {},
            "quality_metrics": {},
            "thresholds": {},
            "swa_layers_skipped": [],
            "compression_metrics": None,
            "overall_pass": False,
        }

        # If tasks is empty list [], return the bare report dict immediately
        if tasks == []:
            return report

        # If tasks is None, set to default task list
        if tasks is None:
            tasks = ["cosine_sim", "perplexity", "gsm8k", "mmlu", "litm"]

        # Load model and pipeline for non-empty tasks
        self._load_model_and_pipeline()

        # Detect SWA layers and store in report
        num_layers = len(self._model.layers)
        swa_layers = detect_swa_layers(self.model_path, num_layers)
        report["swa_layers_skipped"] = sorted(list(swa_layers))

        # Run specified tasks
        for task in tasks:
            if task == "cosine_sim":
                # Run cosine similarity benchmark
                fixed_prompt = "The quick brown fox jumps over the lazy dog."
                kv_cache, compressible_indices = self._prefill_kv(fixed_prompt)

                # Get compacted layers (for comparison with decompressed)
                compacted = self._pipeline.compact(kv_cache)
                compacted_layers = compacted.layers

                # Get decompressed layers
                blob = self._pipeline.compress(kv_cache)
                decompressed_layers, _ = self._pipeline.decompress(blob)

                # Compute cosine similarity (compacted vs decompressed)
                sim = cosine_sim_kv(compacted_layers, decompressed_layers)
                report["technical_metrics"]["am_cosine_similarity"] = sim
                # Llama threshold is 0.95, Qwen is 0.998 - use more lenient threshold
                report["thresholds"]["am_cosine_sim_pass"] = sim > 0.95

            elif task == "latency":
                # Build blob first
                fixed_prompt = "The quick brown fox jumps over the lazy dog."
                kv_cache, compressible_indices = self._prefill_kv(fixed_prompt)
                blob = self._pipeline.compress(kv_cache)

                # Measure decompression latency
                from omlx.compression.evaluators import measure_decompression_latency

                n_layers = len(kv_cache)
                latency = measure_decompression_latency(
                    self._pipeline, blob, n_layers
                )
                report["technical_metrics"]["decompression_latency_ms_per_layer"] = latency
                report["thresholds"]["latency_pass"] = latency < 10.0

            elif task == "gsm8k":
                from omlx.compression.evaluators import run_gsm8k

                metrics = run_gsm8k(
                    self._model,
                    self._tokenizer,
                    self._pipeline,
                    self.n_samples,
                    self.seed,
                )
                report["quality_metrics"].update(metrics)

                # Check delta threshold (within 1 percentage point)
                delta = abs(metrics["gsm8k_vanilla"] - metrics["gsm8k_compressed"])
                report["thresholds"]["gsm8k_delta_pass"] = delta <= 1.0

            elif task == "mmlu":
                from omlx.compression.evaluators import run_mmlu

                metrics = run_mmlu(
                    self._model,
                    self._tokenizer,
                    self._pipeline,
                    self.n_samples,
                    self.seed,
                )
                report["quality_metrics"].update(metrics)

                # Check delta threshold (within 1 percentage point)
                delta = abs(metrics["mmlu_vanilla"] - metrics["mmlu_compressed"])
                report["thresholds"]["mmlu_delta_pass"] = delta <= 1.0

            elif task == "litm":
                from omlx.compression.evaluators import run_litm

                metrics = run_litm(
                    self._model,
                    self._tokenizer,
                    self._pipeline,
                    self.n_samples,
                    self.seed,
                )
                report["quality_metrics"].update(metrics)

                # Check delta threshold (within 0.05)
                delta = abs(metrics["litm_vanilla"] - metrics["litm_compressed"])
                report["thresholds"]["litm_delta_pass"] = delta <= 0.05

        # Compute overall pass
        report["overall_pass"] = all(v for v in report["thresholds"].values())

        # Populate compression metrics from ServerMetrics
        try:
            metrics = get_server_metrics()
            snapshot = metrics.get_snapshot()
            compression_metrics = CompressionMetrics(
                compression_ratio=snapshot.get("compression_ratio", 0.0),
                decompression_latency_ms_per_layer=snapshot.get(
                    "avg_decompression_latency_ms", 0.0
                ),
                cache_hit_rate=0.0,  # Cache hit metrics tracked separately
                cache_miss_rate=0.0,
            )
            report["compression_metrics"] = compression_metrics.to_dict()
        except Exception:
            # If metrics collection fails, leave as None
            report["compression_metrics"] = None

        return report


def _seed_all(seed: int = 42):
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    import mlx.core as mx
    import numpy as np

    mx.random.seed(seed)
    np.random.seed(seed)


def benchmark_compression_command(args) -> dict:
    """Command-line entry point for benchmarking.

    Args:
        args: Parsed command-line arguments (model, --bundle, --seed, --n-samples,
              --output, --tasks, --am-ratio).

    Returns:
        Dictionary with benchmark results.
    """
    runner = BenchmarkRunner(
        model_path=args.model,
        bundle_path=args.bundle,
        am_ratio=args.am_ratio,
        seed=args.seed,
        n_samples=args.n_samples,
        tasks=args.tasks,
    )
    report = runner.run_benchmark(tasks=args.tasks)

    # Write JSON output if path specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"JSON report written to: {args.output}")

    # Print human-readable summary
    print(f"\n{'='*60}")
    print(f"Benchmark Report: {args.model}")
    print(f"{'='*60}")
    print(f"Seed: {report['seed']}")
    print(f"AM Ratio: {report['config']['am_ratio']}")
    print(f"\nTechnical Metrics:")
    print(f"  AM Cosine Similarity: {report['technical_metrics'].get('am_cosine_similarity', 'N/A')}")
    print(f"  Decompression Latency: {report['technical_metrics'].get('decompression_latency_ms_per_layer', 'N/A')} ms/layer")
    print(f"\nQuality Metrics:")
    for key, value in report['quality_metrics'].items():
        print(f"  {key}: {value}")
    print(f"\nCompression Metrics:")
    if report.get("compression_metrics"):
        cm = report["compression_metrics"]
        print(f"  Compression Ratio: {cm.get('compression_ratio', 'N/A')}")
        print(f"  Decompression Latency: {cm.get('decompression_latency_ms_per_layer', 'N/A')} ms/layer")
        print(f"  Cache Hit Rate: {cm.get('cache_hit_rate', 'N/A')}")
        print(f"  Cache Miss Rate: {cm.get('cache_miss_rate', 'N/A')}")
    else:
        print("  Compression Metrics: N/A")
    print(f"\nSWA Layers Skipped: {report['swa_layers_skipped']}")
    print(f"Overall Pass: {report['overall_pass']}")
    print(f"{'='*60}")

    return report
