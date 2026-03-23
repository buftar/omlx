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

import pathlib
import numpy as np
import mlx.core as mx
from scipy.linalg import orthogonal_procrustes
from omlx.compression.linalg_utils import svd_f32
try:
    from tqdm import tqdm
except ImportError:  # tqdm not installed -- use identity wrapper
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable
try:
    from mlx_lm import load as _mlx_lm_load
    from mlx_lm.models.cache import make_prompt_cache as _make_prompt_cache
except ImportError:
    _mlx_lm_load = None  # type: ignore[assignment]
    _make_prompt_cache = None  # type: ignore[assignment]
try:
    from omlx.compression.kvtc import _dp_allocate_bits
except (ImportError, AttributeError):
    def _dp_allocate_bits(singular_values, bits_per_token_budget, n_tokens, n_components,  # type: ignore[misc]
                          min_bits=1, max_bits=8):
        """Fallback uniform allocation when kvtc._dp_allocate_bits is unavailable."""
        bits = int(round(bits_per_token_budget))
        return np.full(len(singular_values), max(1, min(8, bits)), dtype=np.uint8)

# MLX tensor graph flush -- forces MLX lazy compute graph to materialize tensors.
# Named _mx_materialize to match convention in am.py (lines 29-30) and kvtc.py.
# This is NOT Python built-in evaluation -- it is mlx.core.eval(), the MLX graph flush.
_mx_materialize = mx.eval  # noqa: S307

N_SVD_TOKENS = 4000   # max tokens fed to SVD -- prevents OOM on unified memory
CALIB_SEED = 42       # fixed seed for deterministic subsampling

# Built-in calibration prompt corpus (~25 diverse short prompts).
# No external download required. Covers instruction-following, factual, code, math.
CALIBRATION_PROMPTS: list = [
    "Explain the difference between machine learning and deep learning.",
    "Write a Python function to compute the Fibonacci sequence.",
    "What is the capital of France and what is it known for?",
    "Summarize the key principles of thermodynamics.",
    "How does attention mechanism work in transformer models?",
    "Write a SQL query to find the top 5 highest-paid employees.",
    "Explain gradient descent optimization in simple terms.",
    "What are the main differences between TCP and UDP protocols?",
    "Describe the water cycle and its importance to ecosystems.",
    "Solve: If x^2 + 5x + 6 = 0, what are the values of x?",
    "Write a function in JavaScript to reverse a string.",
    "What is the difference between supervised and unsupervised learning?",
    "Explain how a hash table works and its time complexity.",
    "Describe the process of photosynthesis step by step.",
    "What are the SOLID principles in software engineering?",
    "How does a convolutional neural network process images?",
    "Write a recursive function to compute factorial in Python.",
    "Explain the concept of entropy in information theory.",
    "What is the difference between RAM and ROM?",
    "Describe the Turing test and its significance in AI.",
    "How does backpropagation work in neural networks?",
    "What is memoization and when should you use it?",
    "Explain the concept of a pointer in C programming.",
    "What is the central limit theorem in statistics?",
    "Describe how the internet routes packets using IP addressing.",
]


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
    if _mlx_lm_load is None or _make_prompt_cache is None:
        raise RuntimeError(
            "mlx_lm is not installed. Install with: pip install mlx-lm"
        )

    # Step 1: Load model and config.
    model, tokenizer = _mlx_lm_load(model_path)

    # Extract rope parameters and architecture info with cross-model fallbacks.
    if hasattr(model, "args"):
        rope_theta = getattr(model.args, "rope_theta", 10000.0)
        rope_traditional = getattr(model.args, "rope_traditional", False)
        n_layers = model.args.num_hidden_layers
        # GQA models use num_key_value_heads; fall back to num_attention_heads.
        n_kv_heads = getattr(
            model.args,
            "num_key_value_heads",
            getattr(model.args, "num_attention_heads", None),
        )
    else:
        rope_theta = 10000.0
        rope_traditional = False
        n_layers = len(list(model.layers))
        n_kv_heads = None  # determined from first cache extraction

    # Step 2: Resolve output directory.
    if output_path is not None:
        out_dir = pathlib.Path(output_path)
    else:
        local = pathlib.Path(model_path)
        if local.is_dir():
            out_dir = local
        else:
            # HuggingFace hub ID: resolve to cache directory.
            import os
            hf_home = pathlib.Path(
                os.environ.get(
                    "HF_HOME",
                    str(pathlib.Path.home() / ".cache" / "huggingface"),
                )
            )
            cache_dir = hf_home / "hub"
            model_slug = model_path.replace("/", "--")
            candidates = sorted(
                cache_dir.glob(f"models--{model_slug}*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                snap_dirs = list((candidates[0] / "snapshots").iterdir())
                out_dir = sorted(
                    snap_dirs, key=lambda p: p.stat().st_mtime, reverse=True
                )[0]
            else:
                out_dir = pathlib.Path.cwd()
                print(
                    f"Warning: could not resolve model directory; writing to {out_dir}"
                )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 3: Determine layer grouping.
    effective_n_groups = n_groups if n_groups is not None else n_layers
    groups = assign_layer_groups(n_layers, effective_n_groups)
    group_sizes = np.array([len(g) for g in groups], dtype=np.int32)

    # Step 4: Collect KV activations across all prompts via model prefill.
    k_vectors: list = [[] for _ in range(n_layers)]
    v_vectors: list = [[] for _ in range(n_layers)]
    all_attentions: list = []  # per-prompt per-layer per-head entropy proxy arrays

    for prompt in tqdm(CALIBRATION_PROMPTS, desc="Prefill calibration prompts"):
        tokens = tokenizer.encode(prompt)
        tokens_mx = mx.array(tokens)[None]
        cache = _make_prompt_cache(model)
        # Forward pass populates cache with KV activations.
        model(tokens_mx, cache=cache)
        # Materialize all cache tensors before numpy conversion.
        for layer_cache in cache:
            _mx_materialize(*layer_cache.state)

        for layer_idx, layer_cache in enumerate(cache):
            keys, values = layer_cache.state
            # Cast to float32 (cache tensors may be float16).
            k_np = np.array(keys.astype(mx.float32))   # [1, n_kv_heads, T, head_dim]
            v_np = np.array(values.astype(mx.float32))

            if n_kv_heads is None:
                n_kv_heads = k_np.shape[1]

            # Strip RoPE positional encoding before SVD.
            k_stripped = strip_rope_from_keys(
                k_np, rope_theta, rope_traditional, offset=0
            )

            # Reshape to [T * n_kv_heads, head_dim] for PCA.
            head_dim = k_stripped.shape[3]
            k_vectors[layer_idx].append(k_stripped.reshape(-1, head_dim))
            v_vectors[layer_idx].append(v_np.reshape(-1, head_dim))

            # Compute per-head entropy proxy: variance of key norms across token positions.
            # Higher variance = spikier attention (more entropy-sensitive head).
            k_norms = np.linalg.norm(k_stripped[0], axis=-1)  # [n_kv_heads, T]
            head_ent = k_norms.var(axis=-1)                     # [n_kv_heads]
            all_attentions.append(head_ent)

    # Step 5: Compute per-group PCA bases.
    print("Computing PCA bases...")
    layer_to_group: dict = {}
    for g_idx, g_layers in enumerate(groups):
        for lyr in g_layers:
            layer_to_group[lyr] = g_idx

    group_k_vecs: list = [[] for _ in range(effective_n_groups)]
    group_v_vecs: list = [[] for _ in range(effective_n_groups)]
    for layer_idx in range(n_layers):
        g = layer_to_group[layer_idx]
        group_k_vecs[g].append(np.concatenate(k_vectors[layer_idx], axis=0))
        group_v_vecs[g].append(np.concatenate(v_vectors[layer_idx], axis=0))

    K_V_list, V_V_list = [], []
    K_mu_list, V_mu_list = [], []
    K_sv_list, V_sv_list = [], []
    K_bit_list, V_bit_list = [], []

    for g_idx in tqdm(range(effective_n_groups), desc="Computing PCA per group"):
        k_all = np.concatenate(group_k_vecs[g_idx], axis=0)
        v_all = np.concatenate(group_v_vecs[g_idx], axis=0)
        n_tok = k_all.shape[0]  # total token rows for this group

        kV, k_mu, k_sv = compute_pca_basis(k_all, n_components)
        vV, v_mu, v_sv = compute_pca_basis(v_all, n_components)

        k_bits = _dp_allocate_bits(k_sv, bits_per_token, n_tok, n_components)
        v_bits = _dp_allocate_bits(v_sv, bits_per_token, n_tok, n_components)

        K_V_list.append(kV)
        V_V_list.append(vV)
        K_mu_list.append(k_mu)
        V_mu_list.append(v_mu)
        K_sv_list.append(k_sv)
        V_sv_list.append(v_sv)
        K_bit_list.append(k_bits)
        V_bit_list.append(v_bits)

    # Step 6: Compute head entropy (mean across all prompts x layers).
    # all_attentions: list of [n_kv_heads] arrays.
    head_entropy = np.mean(np.array(all_attentions), axis=0).astype(np.float32)

    # Step 7: Pack and save bundle.
    bundle_arrays = {
        "K_V":          np.stack(K_V_list).astype(np.float32),       # [n_groups, head_dim, n_components]
        "V_V":          np.stack(V_V_list).astype(np.float32),
        "K_mu":         np.stack(K_mu_list).astype(np.float32),       # [n_groups, head_dim]
        "V_mu":         np.stack(V_mu_list).astype(np.float32),
        "K_sv":         np.stack(K_sv_list).astype(np.float32),       # [n_groups, n_components]
        "V_sv":         np.stack(V_sv_list).astype(np.float32),
        "K_bit_alloc":  np.stack(K_bit_list).astype(np.uint8),        # [n_groups, n_components]
        "V_bit_alloc":  np.stack(V_bit_list).astype(np.uint8),
        "group_sizes":  group_sizes,                                   # [n_groups] int32
        "head_entropy": head_entropy,                                  # [n_kv_heads] float32
    }

    out_path = out_dir / "kv_pca_calibration.npz"
    save_calibration_bundle(out_path, bundle_arrays)
    print(f"Done. Bundle has {effective_n_groups} groups covering {n_layers} layers.")
    print(
        f"Head entropy range: {head_entropy.min():.3f} -- {head_entropy.max():.3f}"
    )
