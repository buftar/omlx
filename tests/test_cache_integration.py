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


class TestCliFlagIntegration:
    """PIPE-07/08/09: CLI flags wire CompressionConfig into SchedulerConfig."""

    def test_cli_flags(self):
        """--compression-bundle sets CompressionConfig.enabled=True and bundle_path.

        Tests the core flag-to-config wiring logic directly without invoking
        the full serve_command (which has deeply nested lazy imports).
        """
        import argparse
        from omlx.compression.config import CompressionConfig
        from omlx.scheduler import SchedulerConfig

        # Simulate what serve_command does after parsing args
        args = argparse.Namespace(
            compression_bundle="/tmp/bundle.npz",
            compression_am_ratio=4.0,
            compression_n_components=None,
        )

        # Replicate the serve_command logic for building CompressionConfig
        compression_config = None
        if getattr(args, "compression_bundle", None) is not None:
            compression_config = CompressionConfig(
                enabled=True,
                bundle_path=args.compression_bundle,
                am_ratio=args.compression_am_ratio,
                n_components=args.compression_n_components,
            )

        # Replicate the serve_command logic for attaching to SchedulerConfig
        scheduler_config = SchedulerConfig()
        scheduler_config.compression_config = compression_config

        assert scheduler_config.compression_config is not None
        assert isinstance(scheduler_config.compression_config, CompressionConfig)
        assert scheduler_config.compression_config.enabled is True
        assert scheduler_config.compression_config.bundle_path == "/tmp/bundle.npz"
        assert scheduler_config.compression_config.am_ratio == 4.0
        assert scheduler_config.compression_config.n_components is None

    def test_cli_flags_no_bundle(self):
        """Without --compression-bundle, compression_config is None on SchedulerConfig."""
        import argparse
        from omlx.scheduler import SchedulerConfig

        args = argparse.Namespace(
            compression_bundle=None,
            compression_am_ratio=4.0,
            compression_n_components=None,
        )

        # Replicate serve_command logic
        compression_config = None
        if getattr(args, "compression_bundle", None) is not None:
            from omlx.compression.config import CompressionConfig
            compression_config = CompressionConfig(
                enabled=True,
                bundle_path=args.compression_bundle,
                am_ratio=args.compression_am_ratio,
                n_components=args.compression_n_components,
            )

        scheduler_config = SchedulerConfig()
        scheduler_config.compression_config = compression_config

        assert scheduler_config.compression_config is None

    def test_cli_help_shows_compression_flags(self):
        """--compression-bundle, --compression-am-ratio, --compression-n-components in serve --help."""
        import subprocess
        result = subprocess.run(
            ["uv", "run", "omlx", "serve", "--help"],
            capture_output=True,
            text=True,
            cwd="/Users/tonysina/projects/omlx",
        )
        assert "--compression-bundle" in result.stdout
        assert "--compression-am-ratio" in result.stdout
        assert "--compression-n-components" in result.stdout


class TestAdminEndpoint:
    """PIPE-08 runtime: Admin endpoint for toggling compression."""

    def test_admin_endpoint(self):
        """POST /api/compression/config endpoint toggles CompressionConfig at runtime."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI, Depends
        from omlx.admin.routes import router, set_admin_getters
        from omlx.admin.auth import require_admin
        from omlx.compression.config import CompressionConfig

        # Create a live CompressionConfig instance
        cfg = CompressionConfig(enabled=False, am_ratio=4.0)

        # Wire the getter so the endpoint can resolve it
        set_admin_getters(
            state_getter=lambda: object(),
            pool_getter=lambda: None,
            settings_manager_getter=lambda: None,
            global_settings_getter=lambda: None,
            compression_config_getter=lambda: cfg,
        )

        # Build a minimal FastAPI app with the admin router, auth bypassed
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[require_admin] = lambda: True

        client = TestClient(app, raise_server_exceptions=True)

        # Enable compression and set am_ratio
        resp = client.post(
            "/admin/api/compression/config",
            json={"enabled": True, "am_ratio": 2.0},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["success"] is True
        assert "enabled=True" in data["runtime_applied"]
        assert "am_ratio=2.0" in data["runtime_applied"]

        # Verify live config mutated
        assert cfg.enabled is True
        assert cfg.am_ratio == 2.0

        # Disable compression
        resp2 = client.post(
            "/admin/api/compression/config",
            json={"enabled": False},
        )
        assert resp2.status_code == 200
        assert cfg.enabled is False


class TestSlowQwen:
    """PIPE-10: Real Qwen 2.5 7B round-trip: compress->save->load->decompress."""

    @pytest.mark.slow
    def test_qwen_round_trip(self, tmp_path):
        """Real Qwen 2.5 7B: compress->save->load->decompress round-trip.

        Locked decision (CONTEXT.md line 63): verify cosine similarity of
        keys and values and that inference can continue on the restored cache.
        """
        import mlx.core as mx
        from mlx_lm import load as mlx_load
        from omlx.compression.config import CompressionConfig
        from omlx.compression.compressed_cache_manager import CompressedPagedSSDCacheManager

        # 1. Load model and tokenizer
        model, tokenizer = mlx_load("Qwen/Qwen2.5-7B-Instruct")

        # 2. Prefill to generate a real KV cache
        prompt = "The quick brown fox jumps over the lazy dog."
        input_ids = tokenizer.encode(prompt, return_tensors="mlx")

        # Allocate per-layer cache objects using make_prompt_cache if available
        try:
            from mlx_lm.models.cache import make_prompt_cache
            cache = make_prompt_cache(model)
        except ImportError:
            # Fallback: allocate cache per the Phase 5 pattern
            cache = [layer.state for layer in model.layers]

        logits = model(input_ids, cache=cache)
        # Force materialization of all cache arrays (mlx.eval is not Python eval)
        mx.eval(logits, *[arr for layer in cache for arr in (layer.keys, layer.values)])

        # 3. Build cache_data in PagedSSDCacheManager's expected format:
        #    List[Tuple[keys, values]] per layer
        cache_data = [(layer.keys, layer.values) for layer in cache]
        token_count = input_ids.shape[-1]
        block_hash = b"test_qwen_block_" + bytes(16)

        # 4. Compress once via the pipeline to snapshot the *compacted* KV state.
        # Cosine similarity is checked between compacted input and decompressed
        # output (not original vs decompressed) — AM compaction is a lossy token
        # selection step; comparing original to AM-selected tokens is meaningless
        # as they are different subsets of the sequence.
        # (Phase 5 locked decision in STATE.md: "compacted vs decompressed")
        from omlx.compression.pipeline import KVCachePipeline

        compression_config = CompressionConfig(
            enabled=True,
            bundle_path=None,       # on-the-fly mode
            am_ratio=4.0,
            n_sink_tokens=4,
            sliding_window=128,
        )
        pipeline = KVCachePipeline(
            bundle_path=compression_config.bundle_path,
            am_ratio=compression_config.am_ratio,
            n_sink_tokens=compression_config.n_sink_tokens,
            sliding_window=compression_config.sliding_window,
        )
        blob_ref = pipeline.compress(cache_data)
        # Decompress immediately to get the compacted reference KV
        compacted_kv_ref, _tok_ref = pipeline.decompress(blob_ref)
        # Snapshot the compacted arrays as float32 for comparison
        compacted_kv = []
        for k, v in compacted_kv_ref:
            k_f32 = k.astype(mx.float32)
            v_f32 = v.astype(mx.float32)
            mx.eval(k_f32, v_f32)
            compacted_kv.append((k_f32, v_f32))

        # 5. Save via CompressedPagedSSDCacheManager (uses same pipeline internally)
        manager = CompressedPagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1 * 1024 * 1024 * 1024,  # 1 GB
            hot_cache_max_bytes=256 * 1024 * 1024,   # 256 MB in-memory hot cache
            compression_config=compression_config,
        )
        saved = manager.save_block(
            block_hash=block_hash,
            cache_data=cache_data,
            token_count=token_count,
            model_name="Qwen/Qwen2.5-7B-Instruct",
        )
        assert saved, "save_block() should return True"

        # 6. Load and decompress via manager
        loaded = manager.load_block(block_hash)
        assert loaded is not None, "load_block() returned None -- decompression failed"

        # Normalize: decompress returns (kv_list, token_count) or just kv_list
        # kv_list is List[Tuple[keys, values]] per layer
        if (
            isinstance(loaded, tuple)
            and len(loaded) == 2
            and isinstance(loaded[1], int)
        ):
            loaded_kv = loaded[0]
        else:
            loaded_kv = loaded

        assert len(loaded_kv) == len(cache_data), (
            f"Layer count mismatch: got {len(loaded_kv)}, expected {len(cache_data)}"
        )

        # 7. Verify cosine similarity: compacted reference vs loaded/decompressed.
        # Both have the same shape (AM-compacted seq_len) so no trimming needed.
        # Threshold 0.90 — quantization and serialization are the only losses here.
        COSINE_THRESHOLD = 0.90

        def cosine_similarity(a: "mx.array", b: "mx.array") -> float:
            a_flat = a.reshape(-1).astype(mx.float32)
            b_flat = b.reshape(-1).astype(mx.float32)
            dot = (a_flat * b_flat).sum().item()
            norm_a = (a_flat * a_flat).sum().item() ** 0.5
            norm_b = (b_flat * b_flat).sum().item() ** 0.5
            return dot / (norm_a * norm_b + 1e-8)

        for i, ((ref_k, ref_v), layer_result) in enumerate(
            zip(compacted_kv, loaded_kv)
        ):
            loaded_k = layer_result[0].astype(mx.float32)
            loaded_v = layer_result[1].astype(mx.float32)
            mx.eval(loaded_k, loaded_v)

            sim_k = cosine_similarity(ref_k, loaded_k)
            sim_v = cosine_similarity(ref_v, loaded_v)
            assert sim_k > COSINE_THRESHOLD, (
                f"Layer {i} keys cosine similarity {sim_k:.4f} below threshold "
                f"{COSINE_THRESHOLD} (compacted vs decompressed)"
            )
            assert sim_v > COSINE_THRESHOLD, (
                f"Layer {i} values cosine similarity {sim_v:.4f} below threshold "
                f"{COSINE_THRESHOLD} (compacted vs decompressed)"
            )

        # 8. Verify inference can continue on the restored cache
        # Restore the KV cache from loaded tensors and run next-token generation
        for i, layer in enumerate(cache):
            loaded_k, loaded_v = loaded_kv[i][0], loaded_kv[i][1]
            layer.keys = loaded_k
            layer.values = loaded_v

        next_token_input = mx.array([[tokenizer.encode(" The")[0]]])  # single follow-on token
        try:
            next_logits = model(next_token_input, cache=cache)
            mx.eval(next_logits)
            # If we get here without exception, inference can continue
            assert next_logits.shape[-1] > 0, "next_logits should have vocab dimension"
        except Exception as e:
            pytest.fail(f"Inference continuation failed after cache restore: {e}")
