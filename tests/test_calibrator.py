# SPDX-License-Identifier: Apache-2.0
"""
Test scaffold for PCA calibration pipeline (Phase 4).

All test classes cover CAL-01..CAL-05. Wave 0 RED state:
imports clean, all tests raise NotImplementedError (not ImportError).
"""

import argparse
import pathlib
import tempfile
import numpy as np
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
from omlx.cli import calibrate_kv_command


class TestCLIDispatch:
    """CAL-01: CLI dispatch wiring — calibrate_kv_command raises NotImplementedError."""

    def test_dispatch_calls_run_calibration(self):
        """Build a minimal Namespace and confirm dispatch wires to run_calibration."""
        args = argparse.Namespace(
            model="fake-model",
            n_components=64,
            n_groups=None,
            bits_per_token=4.0,
            output=None,
        )
        with pytest.raises(NotImplementedError):
            calibrate_kv_command(args)


class TestRopeStrip:
    """CAL-02: RoPE stripping — strip_rope_from_keys raises NotImplementedError."""

    def test_strip_rope_raises(self):
        with pytest.raises(NotImplementedError):
            strip_rope_from_keys(None, 10000.0, False)


class TestPCABasis:
    """CAL-02: Deterministic SVD — compute_pca_basis and assign_layer_groups raise NotImplementedError."""

    def test_compute_pca_basis_raises(self):
        with pytest.raises(NotImplementedError):
            compute_pca_basis(None, 64)

    def test_assign_layer_groups_raises(self):
        with pytest.raises(NotImplementedError):
            assign_layer_groups(28, 7)


class TestBundleSaveLoad:
    """CAL-03: Bundle serialisation — save/load raise NotImplementedError."""

    def test_save_raises(self):
        with pytest.raises(NotImplementedError):
            save_calibration_bundle(pathlib.Path("/tmp/x.npz"), {})

    def test_load_raises(self):
        with pytest.raises(NotImplementedError):
            load_calibration_bundle(pathlib.Path("/tmp/x.npz"))


class TestHeadEntropy:
    """CAL-04: Head entropy shape — run_calibration raises NotImplementedError."""

    def test_head_entropy_shape_raises(self):
        with pytest.raises(NotImplementedError):
            run_calibration(
                model_path="fake-model",
                n_components=64,
                n_groups=None,
                bits_per_token=4.0,
                output_path=None,
            )


@pytest.mark.slow
class TestCalibrationTiming:
    """CAL-05: Full calibration timing and determinism (slow — requires real model)."""

    def test_full_calibration_timing(self):
        pytest.raises(NotImplementedError, run_calibration,
                      model_path="Qwen/Qwen2.5-7B-Instruct",
                      n_components=64, n_groups=None,
                      bits_per_token=4.0, output_path=None)

    @pytest.mark.slow
    def test_determinism(self):
        pytest.raises(NotImplementedError, run_calibration,
                      model_path="Qwen/Qwen2.5-7B-Instruct",
                      n_components=64, n_groups=None,
                      bits_per_token=4.0, output_path=None)
