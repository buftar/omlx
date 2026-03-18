# SPDX-License-Identifier: Apache-2.0
"""Float32-safe MLX linalg wrappers and scipy NNLS bridge for omlx compression."""
import mlx.core as mx
import numpy as _np
from scipy.optimize import nnls as _scipy_nnls

__all__ = ["svd_f32", "pinv_f32", "qr_f32", "nnls_solve"]


def _ensure_f32(tensor: mx.array) -> mx.array:
    """Cast float16 or bfloat16 tensors to float32; pass float32/float64 through unchanged."""
    if tensor.dtype in (mx.float16, mx.bfloat16):
        return tensor.astype(mx.float32)
    return tensor


def svd_f32(a: mx.array, compute_uv: bool = True):
    """SVD with automatic float32 cast and CPU stream routing."""
    a = _ensure_f32(a)
    return mx.linalg.svd(a, compute_uv=compute_uv, stream=mx.cpu)


def pinv_f32(a: mx.array) -> mx.array:
    """Moore-Penrose pseudoinverse with automatic float32 cast and CPU stream routing."""
    a = _ensure_f32(a)
    return mx.linalg.pinv(a, stream=mx.cpu)


def qr_f32(a: mx.array) -> tuple[mx.array, mx.array]:
    """QR decomposition with automatic float32 cast and CPU stream routing."""
    a = _ensure_f32(a)
    return mx.linalg.qr(a, stream=mx.cpu)


def nnls_solve(A: mx.array, b: mx.array) -> tuple[mx.array, float]:
    """Non-negative least squares via scipy bridge. Inputs and output are MLX tensors."""
    if b.ndim != 1:
        raise ValueError(f"b must be 1D, got shape {b.shape}")
    if A.shape[0] != b.shape[0]:
        raise ValueError(
            f"Shape mismatch: A has {A.shape[0]} rows but b has {b.shape[0]} elements"
        )
    A_np = _np.array(A, dtype=_np.float64)
    b_np = _np.array(b, dtype=_np.float64)
    x_np, residual = _scipy_nnls(A_np, b_np)
    return mx.array(x_np, dtype=mx.float32), float(residual)
