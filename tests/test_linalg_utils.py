# SPDX-License-Identifier: Apache-2.0
import re
from pathlib import Path

import numpy as np
import pytest
import mlx.core as mx

from omlx.compression.linalg_utils import svd_f32, pinv_f32, qr_f32, nnls_solve

# mx.eval() is the MLX graph materialization function (not Python's eval)
_mlx_eval = mx.eval


class TestEnsureF32:
    def test_float16_input_to_svd_returns_float32(self):
        a = mx.array([[1.0, 0.0], [0.0, 1.0]], dtype=mx.float16)
        U, S, Vt = svd_f32(a)
        _mlx_eval(U, S, Vt)
        assert U.dtype == mx.float32
        assert S.dtype == mx.float32
        assert Vt.dtype == mx.float32

    def test_bfloat16_input_to_svd_returns_float32(self):
        a = mx.array([[1.0, 0.0], [0.0, 1.0]], dtype=mx.bfloat16)
        U, S, Vt = svd_f32(a)
        _mlx_eval(U, S, Vt)
        assert U.dtype == mx.float32
        assert S.dtype == mx.float32
        assert Vt.dtype == mx.float32

    def test_float32_passthrough(self):
        a = mx.array([[1.0, 0.0], [0.0, 1.0]], dtype=mx.float32)
        U, S, Vt = svd_f32(a)
        _mlx_eval(U, S, Vt)
        assert U.dtype == mx.float32
        assert S.dtype == mx.float32
        assert Vt.dtype == mx.float32


class TestSvdF32:
    def test_basic_decomposition(self):
        a = mx.array([[3.0, 0.0], [0.0, 2.0], [0.0, 0.0]], dtype=mx.float32)
        U, S, Vt = svd_f32(a)
        _mlx_eval(U, S, Vt)
        assert U.dtype == mx.float32
        assert S.dtype == mx.float32
        assert Vt.dtype == mx.float32

    def test_float16_does_not_raise(self):
        a = mx.zeros((3, 2), dtype=mx.float16)
        U, S, Vt = svd_f32(a)
        _mlx_eval(U, S, Vt)

    def test_compute_uv_false(self):
        a = mx.array([[3.0, 0.0], [0.0, 2.0], [0.0, 0.0]], dtype=mx.float32)
        result = svd_f32(a, compute_uv=False)
        _mlx_eval(result)
        assert result.ndim == 1

    def test_reconstruction(self):
        a = mx.array([[3.0, 0.0], [0.0, 2.0], [0.0, 0.0]], dtype=mx.float32)
        U, S, Vt = svd_f32(a)
        _mlx_eval(U, S, Vt)
        reconstructed = U @ mx.diag(S) @ Vt
        _mlx_eval(reconstructed)
        np.testing.assert_allclose(
            np.array(reconstructed), np.array(a), atol=1e-4
        )


class TestPinvF32:
    def test_basic_pseudoinverse(self):
        a = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=mx.float32)
        p = pinv_f32(a)
        _mlx_eval(p)
        assert p.shape == (2, 3)
        assert p.dtype == mx.float32

    def test_float16_does_not_raise(self):
        a = mx.zeros((3, 2), dtype=mx.float16)
        p = pinv_f32(a)
        _mlx_eval(p)

    def test_moore_penrose_property(self):
        a = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=mx.float32)
        p = pinv_f32(a)
        result = a @ p @ a
        _mlx_eval(result)
        np.testing.assert_allclose(
            np.array(result), np.array(a), atol=1e-4
        )


class TestQrF32:
    def test_basic_decomposition(self):
        a = mx.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            dtype=mx.float32,
        )
        Q, R = qr_f32(a)
        _mlx_eval(Q, R)
        assert Q.dtype == mx.float32
        assert R.dtype == mx.float32

    def test_float16_does_not_raise(self):
        a = mx.zeros((4, 3), dtype=mx.float16)
        Q, R = qr_f32(a)
        _mlx_eval(Q, R)

    def test_bfloat16_does_not_raise(self):
        a = mx.zeros((4, 3), dtype=mx.bfloat16)
        Q, R = qr_f32(a)
        _mlx_eval(Q, R)


class TestNnlsSolve:
    def _make_inputs(self):
        A = mx.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            dtype=mx.float32,
        )
        b = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)
        return A, b

    def test_basic_solve(self):
        A, b = self._make_inputs()
        solution, residual = nnls_solve(A, b)
        _mlx_eval(solution)
        assert solution.dtype == mx.float32
        assert solution.shape == (3,)

    def test_residual_is_python_float(self):
        A, b = self._make_inputs()
        _, residual = nnls_solve(A, b)
        assert isinstance(residual, float)

    def test_solution_nonnegative(self):
        A, b = self._make_inputs()
        solution, _ = nnls_solve(A, b)
        _mlx_eval(solution)
        assert all(v >= 0.0 for v in solution.tolist())

    def test_b_not_1d_raises(self):
        b2d = mx.zeros((2, 2))
        with pytest.raises(ValueError):
            nnls_solve(mx.eye(3), b2d)

    def test_shape_mismatch_raises(self):
        A = mx.zeros((4, 3))
        b = mx.zeros((5,))
        with pytest.raises(ValueError):
            nnls_solve(A, b)

    def test_returns_mx_float32(self):
        A, b = self._make_inputs()
        solution, _ = nnls_solve(A, b)
        _mlx_eval(solution)
        assert solution.dtype == mx.float32

    def test_exact_solution_identity(self):
        A = mx.eye(3, dtype=mx.float32)
        b = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
        solution, residual = nnls_solve(A, b)
        _mlx_eval(solution)
        np.testing.assert_allclose(np.array(solution), [1.0, 2.0, 3.0], atol=1e-6)
        assert residual < 1e-6


def test_no_bare_linalg_calls():
    forbidden = re.compile(r"mx\.linalg\.(svd|pinv)\b")
    allowed = Path("omlx/compression/linalg_utils.py").resolve()
    violations = []
    for path in Path("omlx").rglob("*.py"):
        if path.resolve() == allowed:
            continue
        if forbidden.search(path.read_text()):
            violations.append(str(path))
    assert not violations, f"Bare mx.linalg.svd/pinv found outside linalg_utils: {violations}"
