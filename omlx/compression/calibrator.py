# SPDX-License-Identifier: Apache-2.0
"""
PCA calibration pipeline for KV cache compression.

Generates kv_pca_calibration.npz alongside model weights, containing:
- Per-layer-group PCA bases (K and V sides)
- Pre-baked DP bit allocation tables
- Per-head entropy sensitivity for AMCompactor non-uniform budgets

Usage:
    omlx calibrate-kv <model> [--n-components 64] [--n-groups None]
                               [--bits-per-token 4.0] [--output /path/to/dir]
"""

import numpy as np
import mlx.core as mx
from scipy.linalg import orthogonal_procrustes
from omlx.compression.linalg_utils import svd_f32

# MLX tensor graph flush -- forces MLX lazy compute graph to materialize tensors.
# Named _mx_materialize to match convention in am.py (lines 29-30) and kvtc.py.
# This is NOT Python built-in evaluation -- it is mlx.core.eval(), the MLX graph flush.
_mx_materialize = mx.eval  # noqa: S307

N_SVD_TOKENS = 4000   # max tokens fed to SVD -- prevents OOM on unified memory
CALIB_SEED = 42       # fixed seed for deterministic subsampling


def strip_rope_from_keys(keys, rope_theta, traditional, offset=0):
    """Strip RoPE positional encoding from keys.

    Args:
        keys: float32 numpy [1, n_kv_heads, T, head_dim] with RoPE applied.
        rope_theta: base frequency from model config.
        traditional: True=consecutive pairs, False=half-dim split (MLX default).
        offset: starting position index (always 0 for fresh calibration prefill).

    Returns:
        float32 numpy array same shape as keys with positional encoding removed.
    """
    keys = np.asarray(keys, dtype=np.float32)
    head_dim = keys.shape[-1]
    half = head_dim // 2
    T = keys.shape[2]

    # Compute per-frequency thetas
    i = np.arange(half, dtype=np.float32)
    theta_freqs = (1.0 / (rope_theta ** (2 * i / head_dim))).astype(np.float32)

    # positions: [T]
    positions = np.arange(offset, offset + T, dtype=np.float32)
    # angles: [T, half]
    angles = np.outer(positions, theta_freqs).astype(np.float32)

    # cos/sin shapes for broadcasting: [1, 1, T, half]
    cos_a = np.cos(angles)[np.newaxis, np.newaxis, :, :]
    sin_a = np.sin(angles)[np.newaxis, np.newaxis, :, :]

    if not traditional:
        # Non-traditional (MLX default): first-half / second-half split
        k1 = keys[..., :half]
        k2 = keys[..., half:]
        stripped = np.concatenate(
            [k1 * cos_a + k2 * sin_a, -k1 * sin_a + k2 * cos_a], axis=-1
        )
    else:
        # Traditional: consecutive pairs (0::2 and 1::2)
        k_even = keys[..., 0::2]
        k_odd = keys[..., 1::2]
        stripped = np.empty_like(keys)
        stripped[..., 0::2] = k_even * cos_a + k_odd * sin_a
        stripped[..., 1::2] = -k_even * sin_a + k_odd * cos_a

    return stripped.astype(np.float32)


def compute_pca_basis(vectors, n_components, seed=42):
    """Compute PCA basis from token vectors via svd_f32.

    Args:
        vectors: float32 numpy [N, head_dim]
        n_components: number of principal components to retain.
        seed: RNG seed for deterministic subsampling (default 42).

    Returns:
        (V, mu, singular_values): all float32 numpy.
            V: [head_dim, n_components]
            mu: [head_dim]
            singular_values: [n_components]
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    rng = np.random.default_rng(seed=seed)

    # Subsample to N_SVD_TOKENS when input is large
    if len(vectors) > N_SVD_TOKENS:
        idx = rng.choice(len(vectors), N_SVD_TOKENS, replace=False)
        sample = vectors[idx]
    else:
        sample = vectors

    mu = sample.mean(axis=0)
    centered = (sample - mu).astype(np.float32)
    data_mlx = mx.array(centered)

    # Mandatory: use svd_f32 helper, not bare mx.linalg.svd
    U, S, Vt = svd_f32(data_mlx)
    _mx_materialize(Vt, S)

    Vt_np = np.array(Vt)   # [min(N, head_dim), head_dim]
    S_np = np.array(S)

    V = Vt_np[:n_components].T  # [head_dim, n_components]
    return (
        V.astype(np.float32),
        mu.astype(np.float32),
        S_np[:n_components].astype(np.float32),
    )


def assign_layer_groups(n_layers, n_groups):
    """Assign layers to groups of approximately equal size.

    Returns:
        list of lists of layer indices.
    """
    result = np.array_split(np.arange(n_layers), n_groups)
    return [g.tolist() for g in result]


def align_bases_to_reference(bases, reference_idx=0):
    """Align all bases in group to bases[reference_idx] via Procrustes rotation.

    Args:
        bases: list of [head_dim, n_components] float32 numpy arrays.

    Returns:
        list of aligned [head_dim, n_components] float32 arrays.
    """
    ref = bases[reference_idx].astype(np.float64)
    aligned = []
    for i, B in enumerate(bases):
        if i == reference_idx:
            aligned.append(bases[i])
        else:
            R, _ = orthogonal_procrustes(B.astype(np.float64), ref)
            aligned.append((B.astype(np.float64) @ R).astype(np.float32))
    return aligned


def save_calibration_bundle(output_path, bundle_arrays):
    """Serialize calibration bundle to .npz at output_path.

    Args:
        output_path: pathlib.Path -- destination file.
        bundle_arrays: dict mapping npz key names to numpy arrays.
    """
    np.savez(str(output_path), **bundle_arrays)
    print(f"Calibration bundle saved: {output_path}")


def load_calibration_bundle(path):
    """Load .npz calibration bundle. Returns (pca_bundle, head_entropy).

    Args:
        path: pathlib.Path to .npz file.

    Returns:
        pca_bundle: list[dict] compatible with KVTCCompressor.compress().
            Each dict has keys: K_basis, K_mean, K_sv, V_basis, V_mean, V_sv,
            k_bit_alloc, v_bit_alloc.
        head_entropy: list[float] of length n_heads for AMCompactor.
    """
    data = np.load(str(path))
    group_sizes = data["group_sizes"]

    # Build layer_to_group mapping
    layer_to_group = {}
    layer_idx = 0
    for g, size in enumerate(group_sizes):
        for _ in range(int(size)):
            layer_to_group[layer_idx] = g
            layer_idx += 1

    n_layers = layer_idx
    pca_bundle = []
    for l in range(n_layers):
        g = layer_to_group[l]
        pca_bundle.append({
            "K_basis": data["K_V"][g],
            "K_mean": data["K_mu"][g],
            "K_sv": data["K_sv"][g],
            "V_basis": data["V_V"][g],
            "V_mean": data["V_mu"][g],
            "V_sv": data["V_sv"][g],
            "k_bit_alloc": data["K_bit_alloc"][g],
            "v_bit_alloc": data["V_bit_alloc"][g],
        })

    return pca_bundle, data["head_entropy"].tolist()


def run_calibration(model_path, n_components=64, n_groups=None,
                    bits_per_token=4.0, output_path=None):
    """Run full PCA calibration pipeline for a model.

    Args:
        model_path: str -- HuggingFace repo ID or local directory path.
        n_components: int -- number of PCA components to retain (default 64).
        n_groups: int or None -- layer groups for cross-layer basis sharing.
            None = per-layer (n_groups = n_layers). Recommended default.
        bits_per_token: float -- global bit budget for DP allocation (default 4.0).
        output_path: str or None -- directory to write kv_pca_calibration.npz.
            None = write alongside model weights in resolved model directory.

    Writes:
        <output_dir>/kv_pca_calibration.npz with keys:
            K_V, V_V, K_mu, V_mu, K_sv, V_sv,
            K_bit_alloc, V_bit_alloc, group_sizes, head_entropy
    """
    raise NotImplementedError
