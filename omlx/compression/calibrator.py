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
    raise NotImplementedError


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
    raise NotImplementedError


def assign_layer_groups(n_layers, n_groups):
    """Assign layers to groups of approximately equal size.

    Returns:
        list of lists of layer indices.
    """
    raise NotImplementedError


def align_bases_to_reference(bases, reference_idx=0):
    """Align all bases in group to bases[reference_idx] via Procrustes rotation.

    Args:
        bases: list of [head_dim, n_components] float32 numpy arrays.

    Returns:
        list of aligned [head_dim, n_components] float32 arrays.
    """
    raise NotImplementedError


def save_calibration_bundle(output_path, bundle_arrays):
    """Serialize calibration bundle to .npz at output_path.

    Args:
        output_path: pathlib.Path — destination file.
        bundle_arrays: dict mapping npz key names to numpy arrays.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def run_calibration(model_path, n_components=64, n_groups=None,
                    bits_per_token=4.0, output_path=None):
    """Run full PCA calibration pipeline for a model.

    Args:
        model_path: str — HuggingFace repo ID or local directory path.
        n_components: int — number of PCA components to retain (default 64).
        n_groups: int or None — layer groups for cross-layer basis sharing.
            None = per-layer (n_groups = n_layers). Recommended default.
        bits_per_token: float — global bit budget for DP allocation (default 4.0).
        output_path: str or None — directory to write kv_pca_calibration.npz.
            None = write alongside model weights in resolved model directory.

    Writes:
        <output_dir>/kv_pca_calibration.npz with keys:
            K_V, V_V, K_mu, V_mu, K_sv, V_sv,
            K_bit_alloc, V_bit_alloc, group_sizes, head_entropy
    """
    raise NotImplementedError
