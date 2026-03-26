# SPDX-License-Identifier: Apache-2.0
"""
Tests for PCA calibration pipeline (Phase 4).

CAL-01: CLI dispatch wiring
CAL-02: RoPE stripping and PCA basis computation
CAL-03: Bundle serialisation round-trip
CAL-04: Head entropy integration with AMCompactor
CAL-05: Full calibration timing (slow, requires real model)
"""

import argparse
import pathlib
import tempfile
from unittest.mock import patch
import numpy as np
import mlx.core as mx
import pytest
from omlx.compression.calibrator import (
    strip_rope_from_keys,
    compute_pca_basis,
    save_calibration_bundle,
    load_calibration_bundle,
    assign_layer_groups,
    align_bases_to_reference,
    run_calibration,
)
from omlx.compression.am import AMCompactor
from omlx.cli import calibrate_kv_command

_mx_flush = mx.eval  # MLX graph materialization -- NOT Python eval()


class TestCLIDispatch:
    """CAL-01: CLI dispatch from calibrate-kv command to run_calibration()."""

    def test_dispatch_calls_run_calibration(self, monkeypatch):
        """calibrate_kv_command dispatches to run_calibration with correct args."""
        called_with = {}

        def mock_run_calibration(**kwargs):
            called_with.update(kwargs)

        monkeypatch.setattr(
            "omlx.compression.calibrator.run_calibration",
            mock_run_calibration,
        )

        args = argparse.Namespace(
            model="fake-model",
            n_components=64,
            n_groups=None,
            bits_per_token=4.0,
            output=None,
        )
        calibrate_kv_command(args)

        assert called_with["model_path"] == "fake-model"
        assert called_with["n_components"] == 64
        assert called_with["n_groups"] is None
        assert called_with["bits_per_token"] == 4.0
        assert called_with["output_path"] is None

    def test_cli_help_registered(self):
        """calibrate-kv subcommand is registered in omlx CLI."""
        import subprocess
        result = subprocess.run(
            ["uv", "run", "omlx", "calibrate-kv", "--help"],
            capture_output=True, text=True, cwd="/Users/tonysina/projects/omlx"
        )
        assert result.returncode == 0
        assert "--n-components" in result.stdout
        assert "--bits-per-token" in result.stdout


class TestRopeStrip:
    """CAL-02: RoPE stripping -- strip_rope_from_keys produces flat cosine similarity."""

    def test_rope_flatness_nontrad(self):
        """Non-traditional RoPE: per-token cosine similarity std < 1e-4."""
        rng = np.random.default_rng(0)
        head_dim = 128
        T = 300
        n_heads = 4
        rope_theta = 10000.0

        keys_raw_np = rng.standard_normal((1, n_heads, T, head_dim)).astype(np.float32)
        keys_raw_mx = mx.array(keys_raw_np)

        keys_rope_mx = mx.fast.rope(
            keys_raw_mx,
            head_dim,
            traditional=False,
            base=rope_theta,
            scale=1.0,
            offset=0,
        )
        _mx_flush(keys_rope_mx)
        keys_rope_np = np.array(keys_rope_mx)

        stripped = strip_rope_from_keys(keys_rope_np, rope_theta, traditional=False)

        orig = keys_raw_np[0, 0]
        strp = stripped[0, 0]
        dot = (orig * strp).sum(axis=-1)
        norm_orig = np.linalg.norm(orig, axis=-1)
        norm_strp = np.linalg.norm(strp, axis=-1)
        cos_per_token = dot / (norm_orig * norm_strp + 1e-8)

        assert cos_per_token.std() < 1e-4, f"std={cos_per_token.std():.6f}"
        assert cos_per_token.mean() > 0.999, f"mean={cos_per_token.mean():.6f}"

    def test_rope_flatness_traditional(self):
        """Traditional RoPE (consecutive pairs): same flatness guarantee."""
        rng = np.random.default_rng(1)
        head_dim = 128
        T = 300
        n_heads = 4
        rope_theta = 10000.0

        keys_raw_np = rng.standard_normal((1, n_heads, T, head_dim)).astype(np.float32)
        keys_raw_mx = mx.array(keys_raw_np)

        keys_rope_mx = mx.fast.rope(
            keys_raw_mx,
            head_dim,
            traditional=True,
            base=rope_theta,
            scale=1.0,
            offset=0,
        )
        _mx_flush(keys_rope_mx)
        keys_rope_np = np.array(keys_rope_mx)

        stripped = strip_rope_from_keys(keys_rope_np, rope_theta, traditional=True)

        orig = keys_raw_np[0, 0]
        strp = stripped[0, 0]
        dot = (orig * strp).sum(axis=-1)
        norm_orig = np.linalg.norm(orig, axis=-1)
        norm_strp = np.linalg.norm(strp, axis=-1)
        cos_per_token = dot / (norm_orig * norm_strp + 1e-8)

        assert cos_per_token.std() < 1e-4, f"std={cos_per_token.std():.6f}"
        assert cos_per_token.mean() > 0.999, f"mean={cos_per_token.mean():.6f}"


class TestPCABasis:
    """CAL-02: Deterministic SVD and layer grouping."""

    def test_shapes(self):
        """V=[head_dim, n_components], mu=[head_dim], sv=[n_components]."""
        rng = np.random.default_rng(42)
        head_dim = 128
        n_components = 32
        vectors = rng.standard_normal((500, head_dim)).astype(np.float32)
        V, mu, sv = compute_pca_basis(vectors, n_components)
        assert V.shape == (head_dim, n_components), f"V.shape={V.shape}"
        assert mu.shape == (head_dim,), f"mu.shape={mu.shape}"
        assert sv.shape == (n_components,), f"sv.shape={sv.shape}"

    def test_deterministic(self):
        """Same seed=42 on same data produces identical basis (cosine sim > 0.9999 per column)."""
        rng = np.random.default_rng(7)
        vectors = rng.standard_normal((500, 64)).astype(np.float32)
        V1, _, _ = compute_pca_basis(vectors, 16, seed=42)
        V2, _, _ = compute_pca_basis(vectors, 16, seed=42)
        dots = np.abs((V1 * V2).sum(axis=0) /
                      (np.linalg.norm(V1, axis=0) * np.linalg.norm(V2, axis=0) + 1e-12))
        assert dots.min() > 0.9999, f"min cosine={dots.min():.6f}"

    def test_subsampling(self):
        """5000 vectors triggers subsampling; output shape is still correct."""
        rng = np.random.default_rng(0)
        head_dim = 64
        n_components = 16
        vectors = rng.standard_normal((5000, head_dim)).astype(np.float32)
        V, mu, sv = compute_pca_basis(vectors, n_components)
        assert V.shape == (head_dim, n_components)
        assert mu.shape == (head_dim,)
        assert sv.shape == (n_components,)

    def test_assign_layer_groups(self):
        """assign_layer_groups sums to n_layers; handles remainder correctly."""
        groups_28 = assign_layer_groups(28, 7)
        assert len(groups_28) == 7
        assert sum(len(g) for g in groups_28) == 28

        groups_29 = assign_layer_groups(29, 7)
        assert len(groups_29) == 7
        assert sum(len(g) for g in groups_29) == 29


def _make_bundle_arrays(n_groups, head_dim, n_components, n_heads, layers_per_group):
    """Build a full npz array dict matching the 10-key bundle schema."""
    rng = np.random.default_rng(99)
    group_sizes = np.full(n_groups, layers_per_group, dtype=np.int32)
    return {
        "K_V": rng.standard_normal((n_groups, head_dim, n_components)).astype(np.float32),
        "V_V": rng.standard_normal((n_groups, head_dim, n_components)).astype(np.float32),
        "K_mu": rng.standard_normal((n_groups, head_dim)).astype(np.float32),
        "V_mu": rng.standard_normal((n_groups, head_dim)).astype(np.float32),
        "K_sv": np.abs(rng.standard_normal((n_groups, n_components))).astype(np.float32),
        "V_sv": np.abs(rng.standard_normal((n_groups, n_components))).astype(np.float32),
        "K_bit_alloc": rng.integers(2, 9, size=(n_groups, n_components), dtype=np.uint8),
        "V_bit_alloc": rng.integers(2, 9, size=(n_groups, n_components), dtype=np.uint8),
        "group_sizes": group_sizes,
        "head_entropy": rng.standard_normal(n_heads).astype(np.float32),
    }


class TestBundleSaveLoad:
    """CAL-03: Bundle serialisation -- all 10 npz keys survive round-trip."""

    REQUIRED_LAYER_KEYS = {
        "K_basis", "K_mean", "K_sv",
        "V_basis", "V_mean", "V_sv",
        "k_bit_alloc", "v_bit_alloc",
    }

    def test_round_trip(self):
        """All 10 keys survive save/load; pca_bundle[0] has 8 required keys."""
        n_groups = 7
        layers_per_group = 4
        n_heads = 4
        head_dim = 64
        n_components = 16

        arrays = _make_bundle_arrays(n_groups, head_dim, n_components, n_heads, layers_per_group)
        tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        tmp_path = pathlib.Path(tmp.name)
        tmp.close()
        try:
            save_calibration_bundle(tmp_path, arrays)
            pca_bundle, head_entropy = load_calibration_bundle(tmp_path)

            assert len(pca_bundle) == n_groups * layers_per_group
            assert self.REQUIRED_LAYER_KEYS.issubset(set(pca_bundle[0].keys()))
            assert isinstance(head_entropy, list)
            assert len(head_entropy) == n_heads
            assert all(isinstance(v, float) for v in head_entropy)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_layer_count(self):
        """7 groups x 4 layers each = 28 entries in pca_bundle."""
        n_groups = 7
        layers_per_group = 4
        arrays = _make_bundle_arrays(n_groups, 64, 16, 4, layers_per_group)
        tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        tmp_path = pathlib.Path(tmp.name)
        tmp.close()
        try:
            save_calibration_bundle(tmp_path, arrays)
            pca_bundle, _ = load_calibration_bundle(tmp_path)
            assert len(pca_bundle) == n_groups * layers_per_group
        finally:
            tmp_path.unlink(missing_ok=True)


class TestHeadEntropy:
    """CAL-04: Head entropy shape and AMCompactor non-uniform budget integration."""

    def test_nonuniform_budgets(self):
        """AMCompactor(head_entropy=loaded) produces non-uniform budgets."""
        n_groups = 2
        n_heads = 4
        arrays = _make_bundle_arrays(n_groups, 64, 16, n_heads, layers_per_group=1)
        arrays["head_entropy"] = np.array([0.34, 2.47, 1.12, 1.68], dtype=np.float32)

        tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        tmp_path = pathlib.Path(tmp.name)
        tmp.close()
        try:
            save_calibration_bundle(tmp_path, arrays)
            _, head_entropy = load_calibration_bundle(tmp_path)

            compactor = AMCompactor(head_entropy=head_entropy)
            budgets = compactor._compute_head_budgets(seq_len=400, ratio=4, n_heads=n_heads)
            assert len(set(budgets)) > 1, f"Budgets should be non-uniform: {budgets}"
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_head_entropy_shape(self):
        """head_entropy length matches n_heads from bundle."""
        n_heads = 4
        arrays = _make_bundle_arrays(2, 64, 16, n_heads, layers_per_group=1)
        tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        tmp_path = pathlib.Path(tmp.name)
        tmp.close()
        try:
            save_calibration_bundle(tmp_path, arrays)
            _, head_entropy = load_calibration_bundle(tmp_path)
            assert len(head_entropy) == n_heads
        finally:
            tmp_path.unlink(missing_ok=True)


@pytest.mark.slow
class TestCalibrationTiming:
    """CAL-05: Full calibration timing and determinism (slow -- requires real model).

    These tests require a real model on disk (mlx_lm must be installed and the
    model must be downloaded). They are excluded from CI fast runs via the slow
    marker. Run with: pytest -m slow tests/test_calibrator.py -v
    """

    def test_full_calibration_timing(self, tmp_path):
        """run_calibration() completes within 300 seconds for a 7B model."""
        pytest.importorskip("mlx_lm", reason="mlx_lm not installed; skip CAL-05")
        import time
        start = time.monotonic()
        run_calibration(
            model_path="Qwen/Qwen2.5-7B-Instruct",
            n_components=64,
            n_groups=None,
            bits_per_token=4.0,
            output_path=str(tmp_path),
        )
        elapsed = time.monotonic() - start
        assert elapsed < 300, f"Calibration too slow: {elapsed:.1f}s"

    def test_determinism(self, tmp_path):
        """Two calibration runs with the same seed produce identical bundles."""
        pytest.importorskip("mlx_lm", reason="mlx_lm not installed; skip CAL-05")
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        out1.mkdir()
        out2.mkdir()
        run_calibration(
            model_path="Qwen/Qwen2.5-7B-Instruct",
            n_components=64,
            n_groups=None,
            bits_per_token=4.0,
            output_path=str(out1),
        )
        run_calibration(
            model_path="Qwen/Qwen2.5-7B-Instruct",
            n_components=64,
            n_groups=None,
            bits_per_token=4.0,
            output_path=str(out2),
        )
        bundle1_path = out1 / "kv_pca_calibration.npz"
        bundle2_path = out2 / "kv_pca_calibration.npz"
        bundle1 = np.load(bundle1_path)
        bundle2 = np.load(bundle2_path)
        assert set(bundle1.files) == set(bundle2.files), "Bundle key mismatch"
        for key in bundle1.files:
            np.testing.assert_allclose(
                bundle1[key], bundle2[key], rtol=1e-5,
                err_msg=f"Determinism failure for key={key}",
            )
