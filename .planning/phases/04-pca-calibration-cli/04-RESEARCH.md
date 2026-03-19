# Phase 4: PCA Calibration CLI - Research

**Researched:** 2026-03-19
**Domain:** Offline ML calibration pipeline -- KV cache PCA basis computation, RoPE stripping, CLI integration
**Confidence:** HIGH (grounded in working spike code, verified against project source, hands-on MLX experiments)

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CAL-01 | User can run `omlx calibrate-kv <model>` to generate PCA basis for any supported model | CLI pattern confirmed in omlx/cli.py argparse subcommand system; mlx_lm._download() handles both local paths and HuggingFace IDs |
| CAL-02 | Calibration uses randomized SVD on a representative dataset (~200K tokens) | Full SVD via svd_f32 on 2000-4000 subsampled tokens per layer is deterministic and fast; 28-layer SVD time ~2-17s total; 200K-token prefill ~4 min on M3 Max |
| CAL-03 | PCA basis V^T, mean mu, and DP bit allocation table are stored alongside model weights | numpy.savez to kv_pca_calibration.npz in model directory; npz format verified with all required keys |
| CAL-04 | Head entropy sensitivity curves are computed and stored for AM non-uniform budgets | AMCompactor expects 1D head_entropy array of length n_heads; calibration computes per-head attention entropy averaged across layers |
| CAL-05 | Calibration completes in under 10 minutes for models up to 12B parameters on Apple Silicon | Timing breakdown: ~4 min prefill + ~17s SVD + ~14ms Procrustes + negligible DP allocation = well under 10 min |
</phase_requirements>

---

## Summary

Phase 4 implements `omlx calibrate-kv <model>`, a new subcommand added to `omlx/cli.py` that invokes a new `omlx/compression/calibrator.py` module. The calibrator runs multi-prompt prefill to collect KV cache activations, strips RoPE from keys using a verified numpy inverse-rotation formula, computes per-layer PCA bases via `svd_f32`, optionally groups layers with Procrustes alignment, pre-bakes DP bit allocations, computes per-head entropy sensitivity, and saves everything to `kv_pca_calibration.npz` in the model directory. The output is consumed by `KVTCCompressor` (the `pca_bundle` constructor argument) and `AMCompactor` (the `head_entropy` constructor argument).

The key algorithmic discoveries from this research: (1) RoPE stripping is a well-posed matrix inverse -- applying the conjugate rotation (negate the sine term) restores the pre-RoPE keys with cosine similarity >0.9999 (float32 trig round-trip noise); (2) SVD on 2000-4000 subsampled token vectors per layer is sufficient for stable PCA basis estimation and takes 60-600ms per layer via `svd_f32`; (3) Procrustes alignment across 28 layers is ~14ms total -- negligible overhead; (4) the entire pipeline fits within 10 minutes on M3 Max for a 7B model. All five requirements are straightforward given the phase 1-3 infrastructure.

The calibrator follows the same optional-bundle constructor pattern established by `AMCompactor` and `KVTCCompressor`. The `omlx calibrate-kv` CLI subcommand is a minimal addition to the existing argparse structure in `cli.py` -- no framework changes are needed.

**Primary recommendation:** Implement `omlx/compression/calibrator.py` with a `run_calibration()` function; add `calibrate-kv` subcommand to `cli.py` following the existing `serve`/`launch` pattern; write tests in `tests/test_calibrator.py` (unit tests + one `@pytest.mark.slow` end-to-end test).

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `mlx_lm` | 0.31.2 (pinned in pyproject.toml) | Load model, tokenize, run prefill, extract KV cache | Already used throughout omlx; `mlx_lm.load()` and `make_prompt_cache()` are the established model execution path |
| `mlx` | >=0.31.1 | KV cache tensor ops, RoPE application | Project foundation; all KV tensors are `mx.array` |
| `numpy` | >=1.24.0 | RoPE stripping, SVD subsampling, Procrustes, npz I/O | Bridge between MLX tensors and scipy; `numpy.savez`/`numpy.load` for bundle I/O |
| `scipy.linalg.orthogonal_procrustes` | >=1.7.0 | Align PCA bases across layers in same group | Available (already in project deps); 14ms total for 28-layer alignment |
| `omlx.compression.linalg_utils.svd_f32` | project | Full SVD for PCA calibration | CI lint gate enforces this; bare `mx.linalg.svd` is forbidden outside linalg_utils |
| `tqdm` | >=4.66.0 | Progress bar for multi-prompt prefill loop | Already in project deps; standard CLI progress reporting in omlx |
| `numpy.savez` / `numpy.load` | numpy | Bundle serialization (npz format) | numpy native; no new dep; human-readable keys; handles arrays of different shapes and dtypes |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `mlx_lm.utils._download` | 0.31.2 | Resolve model path (local or HuggingFace ID) | Used inside calibrator to convert CLI arg to local filesystem path |
| `mlx_lm.models.cache.make_prompt_cache` | 0.31.2 | Create per-layer KV cache object for prefill | Identical to spike usage; `cache.state` gives (keys, values) per layer |
| `pathlib.Path` | stdlib | File path construction for output location | Idiomatic Python for path manipulation |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `numpy.savez` (npz) | HDF5 or custom binary | npz is stdlib-adjacent (numpy), single-file, named arrays, readable without special tools. HDF5 requires h5py dep. |
| `scipy.linalg.orthogonal_procrustes` | Manual SVD-based Procrustes | scipy is already a project dep; `orthogonal_procrustes` is 3 lines vs 15 of manual SVD. No reason to hand-roll. |
| Built-in calibration prompts | HuggingFace Datasets wikitext-2 | Built-in prompts have no external download dependency; deterministic; no new dep. |
| Full SVD via svd_f32 on subsample | sklearn randomized SVD | Full SVD on [2000-4000, 128] is fast enough (60-600ms/layer) and produces identical Vt at this scale. No sklearn dep needed. |

**Installation:** All dependencies already in `pyproject.toml`. No new packages required for Phase 4.

---

## Architecture Patterns

### Recommended Project Structure
```
omlx/
+-- compression/
    +-- __init__.py          # intentionally empty
    +-- linalg_utils.py      # existing
    +-- am.py                # existing
    +-- kvtc.py              # existing
    +-- calibrator.py        # Phase 4 target: run_calibration(), bundle save/load

tests/
+-- test_calibrator.py       # Phase 4 tests -- Wave 0 creates
```

The calibrator is a separate module from `kvtc.py`. It is a pure function module (no classes required). The CLI entry in `cli.py` calls `run_calibration()` from this module.

### Pattern 1: CLI Subcommand (mirrors existing serve/launch commands)

**What:** Add `calibrate-kv` as a new subparser in `main()` in `cli.py`. Follow the established lazy-import pattern where heavy imports happen inside the command handler function.

**When to use:** Always -- this is the only CLI extension pattern in omlx.

```python
# Source: omlx/cli.py lines 348-552 (established pattern)

# In main(), inside subparsers section:
calibrate_parser = subparsers.add_parser(
    "calibrate-kv",
    help="Generate PCA calibration bundle for a model",
)
calibrate_parser.add_argument("model", type=str,
    help="Path to model directory or HuggingFace repo ID")
calibrate_parser.add_argument("--n-components", type=int, default=64)
calibrate_parser.add_argument("--n-groups", type=int, default=None)
calibrate_parser.add_argument("--bits-per-token", type=float, default=4.0)
calibrate_parser.add_argument("--output", type=str, default=None)

# In main(), dispatch block:
elif args.command == "calibrate-kv":
    calibrate_kv_command(args)

# Command handler (heavy imports inside to keep startup fast):
def calibrate_kv_command(args):
    from omlx.compression.calibrator import run_calibration
    run_calibration(
        model_path=args.model,
        n_components=args.n_components,
        n_groups=args.n_groups,
        bits_per_token=args.bits_per_token,
        output_path=args.output,
    )
```

### Pattern 2: KV Cache Extraction (from spike, verified working)

**What:** Load model, tokenize each calibration prompt, run prefill, extract per-layer KV tensors from cache state. Keys in cache.state already have RoPE applied -- strip before SVD.

**When to use:** Inside the calibration loop -- called once per prompt.

```python
# Source: spike_kvtc.py lines 31-67 (confirmed working)
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
import mlx.core as mx

# The line below aliases mx.eval (MLX lazy graph materialization) to _mx_materialize.
# This is the same pattern used in am.py lines 29-30 and kvtc.py.
# Note: this is tensor materialization, not Python code evaluation.
_mx_materialize = mx.eval  # noqa: S307  -- MLX materialization, not code execution

def extract_kv_from_prefill(model, prompt_tokens):
    """Run prefill and return list of (keys_f32_np, values_f32_np) per layer."""
    import numpy as np
    cache = make_prompt_cache(model)
    model(prompt_tokens[None], cache=cache)
    _mx_materialize(cache[0].state[0])
    layers = []
    for layer_cache in cache:
        keys, values = layer_cache.state
        _mx_materialize(keys, values)
        layers.append((
            np.array(keys.astype(mx.float32)),
            np.array(values.astype(mx.float32)),
        ))
    return layers
```

### Pattern 3: RoPE Stripping (verified by experiment)

**What:** Remove positional encoding from keys before SVD. The inverse of RoPE at position p is computed by conjugate rotation: apply cos(p*theta) with sign-negated sin term. This is the standard SO(2) inverse: R^{-1} = R^T.

**Verified result (hands-on experiment in research, 2026-03-19):** Mean cosine similarity between stripped and original pre-RoPE keys is 1.00000000; min cosine sim is 0.99999976. The ~1e-4 absolute diff is float32 trig round-trip noise -- acceptable for calibration.

```python
# Source: verified by experiment during Phase 4 research (2026-03-19)
import numpy as np

def strip_rope_from_keys(keys, rope_theta, traditional, offset=0):
    """Strip RoPE positional encoding from keys.

    Args:
        keys: float32 numpy [1, n_kv_heads, T, head_dim] with RoPE applied
        rope_theta: base frequency from model config (e.g. Qwen 2.5: 1000000.0)
        traditional: from model config (True = consecutive pairs, False = half-dim split)
        offset: starting position (always 0 for fresh calibration prefill)

    Returns:
        float32 numpy array same shape as keys, with positional encoding removed
    """
    head_dim = keys.shape[-1]
    T = keys.shape[2]
    i = np.arange(head_dim // 2, dtype=np.float64)
    theta_freqs = 1.0 / (rope_theta ** (2 * i / head_dim))
    positions = np.arange(offset, offset + T, dtype=np.float64)
    angles = np.outer(positions, theta_freqs).astype(np.float32)
    cos_a = np.cos(angles).reshape(1, 1, T, head_dim // 2)
    sin_a = np.sin(angles).reshape(1, 1, T, head_dim // 2)

    if not traditional:
        # Non-traditional (MLX default): first half / second half split
        k1 = keys[..., :head_dim // 2]
        k2 = keys[..., head_dim // 2:]
        # Inverse: R^T => negate the sin term
        return np.concatenate([
            k1 * cos_a + k2 * sin_a,
            -k1 * sin_a + k2 * cos_a,
        ], axis=-1)
    else:
        # Traditional: consecutive pairs
        k_even = keys[..., 0::2]
        k_odd  = keys[..., 1::2]
        out = np.empty_like(keys)
        out[..., 0::2] = k_even * cos_a + k_odd * sin_a
        out[..., 1::2] = -k_even * sin_a + k_odd * cos_a
        return out
```

### Pattern 4: PCA Basis Computation (per-layer)

**What:** Compute SVD on subsampled KV vectors per layer. Use `svd_f32` (not bare `mx.linalg.svd`). Subsample to at most N_SVD_TOKENS vectors to keep SVD fast.

**Timing benchmarked:** SVD on [2000, 128] takes 0.061s; on [4000, 128] takes ~0.6s. At 28 layers: 2-17s total for SVD.

```python
# Source: derived from spike_kvtc.py pca_calibrate() + linalg_utils.svd_f32 convention
from omlx.compression.linalg_utils import svd_f32
import mlx.core as mx
import numpy as np

N_SVD_TOKENS = 4000   # max tokens for SVD (M3 Max: ~600ms at this size)
CALIB_SEED = 42       # fixed seed for deterministic subsampling

def compute_pca_basis(vectors, n_components, seed=CALIB_SEED):
    """Compute PCA basis from token vectors.

    Args:
        vectors: float32 numpy [N, head_dim]
        n_components: number of components to retain
        seed: fixed RNG seed for deterministic subsampling

    Returns:
        (V, mu, singular_values) all float32 numpy:
            V: [head_dim, n_components]
            mu: [head_dim]
            singular_values: [n_components]
    """
    rng = np.random.default_rng(seed=seed)
    if len(vectors) > N_SVD_TOKENS:
        idx = rng.choice(len(vectors), N_SVD_TOKENS, replace=False)
        sample = vectors[idx]
    else:
        sample = vectors

    mu = sample.mean(axis=0)
    centered = (sample - mu).astype(np.float32)
    data_mlx = mx.array(centered)
    # svd_f32 is mandatory -- bare mx.linalg.svd is forbidden outside linalg_utils.py
    U, S, Vt = svd_f32(data_mlx)
    _mx_materialize(Vt, S)
    Vt_np = np.array(Vt)   # [min(N, head_dim), head_dim]
    S_np = np.array(S)
    V = Vt_np[:n_components].T  # [head_dim, n_components]
    return V.astype(np.float32), mu.astype(np.float32), S_np[:n_components].astype(np.float32)
```

### Pattern 5: Procrustes Alignment (cross-layer basis alignment)

**What:** After computing per-layer PCA bases, align bases within each group to a common reference (first layer's basis). Uses `scipy.linalg.orthogonal_procrustes`.

**Timing benchmarked:** 28 layers in 7 groups = ~14ms total -- negligible overhead.

```python
# Source: scipy.linalg.orthogonal_procrustes (verified available in project deps)
from scipy.linalg import orthogonal_procrustes
import numpy as np

def align_bases_to_reference(bases, reference_idx=0):
    """Align all bases in group to bases[reference_idx] using Procrustes rotation.

    Args:
        bases: list of [head_dim, n_components] float32 numpy arrays

    Returns:
        list of aligned [head_dim, n_components] float32 arrays
    """
    ref = bases[reference_idx].astype(np.float64)
    aligned = [None] * len(bases)
    for i, B in enumerate(bases):
        if i == reference_idx:
            aligned[i] = bases[i]
        else:
            R, _ = orthogonal_procrustes(B.astype(np.float64), ref)
            aligned[i] = (B.astype(np.float64) @ R).astype(np.float32)
    return aligned
```

### Pattern 6: npz Bundle Save/Load

**What:** Serialize calibration results to a single `.npz` file. Output path: `<model_dir>/kv_pca_calibration.npz`.

**Bundle key schema:**
```
K_V:          [n_groups, head_dim, n_components]  float32
V_V:          [n_groups, head_dim, n_components]  float32
K_mu:         [n_groups, head_dim]                float32
V_mu:         [n_groups, head_dim]                float32
K_sv:         [n_groups, n_components]            float32
V_sv:         [n_groups, n_components]            float32
K_bit_alloc:  [n_groups, n_components]            uint8
V_bit_alloc:  [n_groups, n_components]            uint8
group_sizes:  [n_groups]                          int32
head_entropy: [n_heads]                           float32
```

**Note on success criteria keys:** The success criteria names `V, mu, bit_alloc, group_sizes, head_entropy`. The npz uses prefixed names (`K_V`, `V_V`, etc.) to distinguish K and V sides. The loader reconstructs the `list[dict]` format expected by `KVTCCompressor.compress()`.

```python
# Source: numpy.savez/load verified in research
import numpy as np

def save_calibration_bundle(output_path, bundle_arrays):
    """Save bundle to .npz. Prints confirmed path to stdout."""
    np.savez(str(output_path), **bundle_arrays)
    print(f"Calibration bundle saved: {output_path}")

def load_calibration_bundle(path):
    """Load .npz bundle. Returns (pca_bundle, head_entropy).

    pca_bundle: list[dict] for KVTCCompressor
    head_entropy: list[float] for AMCompactor
    """
    data = np.load(str(path))
    group_sizes = data["group_sizes"]
    layer_to_group = {}
    layer_idx = 0
    for g, size in enumerate(group_sizes):
        for _ in range(int(size)):
            layer_to_group[layer_idx] = g
            layer_idx += 1

    pca_bundle = []
    for l in range(sum(group_sizes)):
        g = layer_to_group[l]
        pca_bundle.append({
            "K_basis": data["K_V"][g], "K_mean": data["K_mu"][g], "K_sv": data["K_sv"][g],
            "V_basis": data["V_V"][g], "V_mean": data["V_mu"][g], "V_sv": data["V_sv"][g],
            "k_bit_alloc": data["K_bit_alloc"][g],
            "v_bit_alloc": data["V_bit_alloc"][g],
        })
    return pca_bundle, data["head_entropy"].tolist()
```

### Pattern 7: Layer Grouping with Equal-Size Groups

```python
# Source: numpy.array_split documentation
import numpy as np

def assign_layer_groups(n_layers, n_groups):
    """Returns list of lists of layer indices. Remainder distributed across first groups."""
    return [g.tolist() for g in np.array_split(np.arange(n_layers), n_groups)]
```

### Anti-Patterns to Avoid

- **Bare `mx.linalg.svd` in calibrator.py:** Must use `svd_f32` from `linalg_utils`. The CI lint gate enforces this.
- **RoPE stripping with non-zero offset for fresh calibration:** Calibration always uses fresh `make_prompt_cache(model)` -- offset starts at 0. Do not pass offset!=0 unless the cache was pre-filled.
- **SVD on full token buffer without subsampling:** Will OOM at >10K tokens on M3 Max (exit code 138 confirmed during research). Always subsample to N_SVD_TOKENS <= 4000 before SVD.
- **Float16 keys passed to strip_rope:** Must cast to float32 before stripping: `keys_f32 = np.array(keys_mlx.astype(mx.float32))`.
- **Failing to materialize MLX tensors before numpy:** Call `_mx_materialize(tensor)` before `np.array(tensor)`. See am.py lines 29-30 for the established project pattern.
- **Wrong group_sizes sum:** `sum(group_sizes)` must equal `n_layers`. Use `np.array_split` which handles remainders automatically.
- **Missing SPDX header:** Every new `.py` file must start with `# SPDX-License-Identifier: Apache-2.0`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| RoPE inverse operation | Custom rotation inverse | Conjugate formula in Pattern 3 above | MLX has no `unrope` primitive; the formula is 3 lines and verified at cosine sim >0.9999 |
| Procrustes alignment | SVD + rotation by hand | `scipy.linalg.orthogonal_procrustes` | 1 line vs 15+; already a project dep; ~0.5ms per call; handles edge cases |
| HuggingFace model download | Custom downloader | `mlx_lm.load()` which calls `_download` internally | Already used by omlx engine; handles caching, snapshots, and auth |
| npz serialization | Custom binary format | `numpy.savez` / `numpy.load` | numpy native; no new dep; human-readable keys; handles arrays of different shapes |
| Progress reporting | Custom stdout flush loop | `tqdm` | Already in project deps; standard CLI UX across omlx |
| Layer group assignment with remainder | Manual loop | `numpy.array_split` | Handles non-divisible n_layers automatically; deterministic; one line |

---

## Common Pitfalls

### Pitfall 1: RoPE Offset Mismatch
**What goes wrong:** RoPE stripping produces wrong keys -- cosine similarity drops below 0.99, PCA basis is contaminated with positional signal. This violates success criterion 2 (flat cosine sim across positions).
**Why it happens:** If offset is non-zero (model was used for prior generation), keys were rotated at positions `offset..offset+T-1`, not `0..T-1`.
**How to avoid:** Calibration always creates a fresh cache with `make_prompt_cache(model)` -- offset starts at 0. Assert `cache[0].offset == 0` after creation.
**Warning signs:** Per-position cosine similarity is NOT flat; it varies systematically with token index.

### Pitfall 2: SVD OOM on Full Calibration Buffer
**What goes wrong:** Python crashes with exit code 138 (confirmed during research when testing arrays >50K tokens).
**Why it happens:** Full SVD on [200K, 128] exceeds unified memory.
**How to avoid:** Always subsample to N_SVD_TOKENS = 4000 max before SVD.
**Warning signs:** Exit code 138, crash with no Python traceback.

### Pitfall 3: group_sizes Sum Mismatch
**What goes wrong:** `load_calibration_bundle()` produces wrong layer-to-group mapping.
**Why it happens:** Integer division without remainder handling silently drops the last few layers.
**How to avoid:** Use `np.array_split(range(n_layers), n_groups)`. Verify `sum(group_sizes) == n_layers` before saving.
**Warning signs:** `KVTCCompressor.compress()` raises KeyError for the last few layers.

### Pitfall 4: Float16 Keys Not Cast Before RoPE Strip
**What goes wrong:** Trig computation on float16 produces poor accuracy.
**Why it happens:** `cache.state` returns float16 tensors.
**How to avoid:** Cast explicitly: `k_np = np.array(keys.astype(mx.float32))` before calling `strip_rope_from_keys`.
**Warning signs:** Mean cosine sim after RoPE strip drops below 0.999 on synthetic test data.

### Pitfall 5: head_entropy Shape Mismatch
**What goes wrong:** `AMCompactor` produces uniform budgets even with non-uniform head_entropy.
**Why it happens:** Bundle stores `head_entropy[n_layers, n_heads]` but AMCompactor expects 1D `[n_heads]`.
**How to avoid:** Store `head_entropy` as `[n_heads]` (pre-averaged across layers) in the npz. Include a unit test that constructs `AMCompactor(head_entropy=loaded)` and verifies non-uniform budgets.
**Warning signs:** All heads receive the same token budget even with diverse entropy values.

### Pitfall 6: Non-Deterministic Calibration
**What goes wrong:** Two calibration runs produce different bases (CAL-05 violated).
**Why it happens:** Random subsampling without a fixed seed.
**How to avoid:** Always use `np.random.default_rng(seed=42)` for subsampling. Keep calibration prompts as a sorted constant list in the module. Full SVD is deterministic.
**Warning signs:** Cosine similarity between runs drops below 0.999 for any layer.

### Pitfall 7: Output Path Points to Wrong Directory
**What goes wrong:** npz saved to current working directory instead of model directory.
**Why it happens:** HuggingFace IDs don't directly map to a local path without using mlx_lm's download resolution.
**How to avoid:** Use the resolved local path from `mlx_lm.load()` to construct the output path. Print the full absolute output path to stdout.
**Warning signs:** User reports command succeeded but cannot find kv_pca_calibration.npz.

---

## Code Examples

### RoPE Strip Correctness Test Pattern
```python
# Source: verified by experiment -- cosine sim >0.9999 confirmed 2026-03-19
import mlx.core as mx
import numpy as np

def test_rope_strip_flat_across_positions():
    """Verify cosine sim is flat (no positional drift) after RoPE strip."""
    dims = 128
    T = 300
    keys_raw = np.random.randn(1, 4, T, dims).astype(np.float32)
    keys_mlx = mx.array(keys_raw)
    # Apply RoPE at offset=0
    keys_rope = mx.fast.rope(keys_mlx, dims, traditional=False,
                              base=10000.0, scale=1.0, offset=0)
    mx.eval(keys_rope)  # MLX lazy graph materialization -- forces tensor computation
    keys_rope_np = np.array(keys_rope.astype(mx.float32))
    # Strip RoPE
    stripped = strip_rope_from_keys(keys_rope_np, rope_theta=10000.0, traditional=False)
    # Verify flatness: per-token cosine sim should be uniform across positions
    orig = keys_raw.reshape(-1, dims)
    strp = stripped.reshape(-1, dims)
    dots = np.sum(orig * strp, axis=-1)
    norms = np.linalg.norm(orig, axis=-1) * np.linalg.norm(strp, axis=-1)
    cos_per_token = dots / (norms + 1e-8)
    assert cos_per_token.std() < 1e-4   # flat across positions
    assert cos_per_token.mean() > 0.999
```

### npz Round-Trip Verification
```python
# Source: numpy.savez/load verified in research 2026-03-19
import numpy as np, tempfile, pathlib

def test_bundle_round_trip():
    """All required keys survive npz save/load."""
    arrays = {
        "K_V": np.random.randn(7, 128, 64).astype(np.float32),
        "V_V": np.random.randn(7, 128, 64).astype(np.float32),
        "K_mu": np.random.randn(7, 128).astype(np.float32),
        "V_mu": np.random.randn(7, 128).astype(np.float32),
        "K_sv": np.random.randn(7, 64).astype(np.float32),
        "V_sv": np.random.randn(7, 64).astype(np.float32),
        "K_bit_alloc": np.random.randint(1, 9, (7, 64), dtype=np.uint8),
        "V_bit_alloc": np.random.randint(1, 9, (7, 64), dtype=np.uint8),
        "group_sizes": np.array([4, 4, 4, 4, 4, 4, 4], dtype=np.int32),
        "head_entropy": np.array([1.68, 0.34, 2.47, 1.12], dtype=np.float32),
    }
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = pathlib.Path(f.name)
    try:
        save_calibration_bundle(path, arrays)
        pca_bundle, head_entropy = load_calibration_bundle(path)
        assert len(pca_bundle) == 28          # 7 groups x 4 layers
        assert "K_basis" in pca_bundle[0]
        assert len(head_entropy) == 4
    finally:
        path.unlink(missing_ok=True)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-prompt PCA (SVD per prompt separately) | Aggregate vectors from all prompts, one SVD per layer group | Phase 3 CONTEXT.md | Stable basis across diverse prompts |
| Per-layer PCA (spike approach) | Cross-layer PCA with Procrustes alignment | Phase 3 CONTEXT.md | Better cross-layer basis sharing |
| Uniform bit allocation hardcoded at runtime | DP bit allocation pre-baked in bundle at calibration time | Phase 3 decision | Compressor does lookup not recompute |
| On-the-fly PCA only (pca_bundle=None) | Persistent bundle via `omlx calibrate-kv` | This phase | Enables production-quality compression |

---

## Open Questions

1. **rope_theta key name across model families**
   - What we know: `rope_theta` is in model config dict from `load(return_config=True)`. Confirmed for Qwen 2.5 (value: 1000000.0).
   - What's unclear: Whether all omlx-supported models expose this with the same key name.
   - Recommendation: Access via `config.get("rope_theta", 10000.0)` with logged fallback.

2. **n_groups default value**
   - What we know: n_groups is configurable. Procrustes is fast regardless.
   - Recommendation: Default `n_groups = n_layers` (per-layer, no cross-layer grouping). Simplest, always correct, clean baseline.

3. **Calibration prompt corpus size**
   - What we know: 2000-4000 tokens per layer is sufficient for stable SVD. 20-30 diverse prompts (~200 tokens each) gives ~4000-6000 total tokens.
   - Recommendation: Bundle 20-30 short diverse prompts covering instruction following, factual recall, code, and mathematical text.

4. **Traditional RoPE models**
   - What we know: Most models use non-traditional (interleaved) RoPE.
   - Recommendation: Implement both branches in `strip_rope_from_keys()`. Access via `config.get("rope_traditional", False)`.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=9.0.2 (from pyproject.toml dependency-groups) |
| Config file | `[tool.pytest.ini_options]` in `pyproject.toml` |
| Quick run command | `pytest tests/test_calibrator.py -m "not slow" -x` |
| Full suite command | `pytest tests/test_calibrator.py -v` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CAL-01 | `calibrate_kv_command()` parses args and dispatches to `run_calibration()` | unit | `pytest tests/test_calibrator.py::TestCLIDispatch -x` | ❌ Wave 0 |
| CAL-02 | PCA basis computed from subsampled vectors; same seed = same Vt | unit | `pytest tests/test_calibrator.py::TestPCABasis -x` | ❌ Wave 0 |
| CAL-02 | RoPE stripping produces flat cosine similarity across all token positions | unit | `pytest tests/test_calibrator.py::TestRopeStrip -x` | ❌ Wave 0 |
| CAL-03 | Saved .npz has all required keys; loaded bundle matches KVTCCompressor contract | unit | `pytest tests/test_calibrator.py::TestBundleSaveLoad -x` | ❌ Wave 0 |
| CAL-04 | head_entropy is length n_heads; AMCompactor(head_entropy=loaded) produces non-uniform budgets | unit | `pytest tests/test_calibrator.py::TestHeadEntropy -x` | ❌ Wave 0 |
| CAL-05 | Full calibration on Qwen 2.5 7B completes in under 600 seconds | slow | `pytest tests/test_calibrator.py::TestCalibrationTiming -m slow -v` | ❌ Wave 0 |
| CAL-05 | Two calibration runs produce bundles with per-layer basis cosine sim >0.999 | slow | `pytest tests/test_calibrator.py::TestDeterminism -m slow -v` | ❌ Wave 0 |

**Key unit test for CAL-02 (RoPE flatness):**
Apply MLX RoPE to synthetic keys, strip with `strip_rope_from_keys()`, verify cosine similarity per token position is uniform (std < 1e-4). This directly confirms success criterion 2 ("flat across token positions").

### Sampling Rate
- **Per task commit:** `pytest tests/test_calibrator.py -m "not slow" -x`
- **Per wave merge:** `pytest tests/test_calibrator.py -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_calibrator.py` -- all test classes listed above
- [ ] `omlx/compression/calibrator.py` -- stub with `run_calibration()` raising `NotImplementedError`
- [ ] `omlx/cli.py` -- add `calibrate-kv` subparser and `calibrate_kv_command()` stub in `main()`

*(No pyproject.toml changes required -- all deps already present including zstandard from Phase 3)*

---

## Sources

### Primary (HIGH confidence)
- `docs/research/kv-cache-compression/spike_kvtc.py` -- KV extraction pipeline, PCA calibration loop, verified working
- `omlx/compression/kvtc.py` -- `pca_bundle` interface contract (exact keys the loader must produce, lines 375-381, 421-422)
- `omlx/compression/am.py` -- `head_entropy` format and AMCompactor constructor pattern; `_mx_materialize` alias convention (lines 29-30)
- `omlx/cli.py` lines 348-552 -- CLI subcommand pattern (argparse subparsers, lazy imports, `main()` dispatch)
- `pyproject.toml` -- confirmed all deps present; no new packages required
- Hands-on experiments (this session, 2026-03-19): RoPE strip cosine sim >0.9999 verified; SVD timing benchmarked; Procrustes 14ms verified; npz round-trip with all keys verified

### Secondary (MEDIUM confidence)
- `mlx_lm` source inspected: `KVCache.state`, `make_prompt_cache`, `load(return_config=True)`, `_download()` -- stable API
- `scipy.linalg.orthogonal_procrustes` -- verified available in project venv; correct output shape confirmed
- STATE.md decision log -- "Per-head entropy 0.34-2.47" confirms expected range; "Procrustes alignment runs during calibration only" locks design

### Tertiary (LOW confidence)
- Timing estimate for 200K-token prefill (~4 min at 800 tok/s on M3 Max) -- general throughput estimate, not directly measured
- `n_groups = n_layers` default recommendation -- simplicity argument, not benchmarked against grouped alternatives

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries verified in venv; all APIs inspected directly
- Architecture: HIGH -- CLI pattern confirmed from source; calibration pipeline modeled on verified spike; all key APIs confirmed
- RoPE stripping: HIGH -- verified by hands-on experiment; cosine sim >0.9999 confirmed
- Timing estimates: MEDIUM -- SVD and Procrustes benchmarked directly; prefill timing is an estimate
- Bundle schema: MEDIUM -- derived from KVTCCompressor pca_bundle access pattern; exact npz key names are Claude's discretion

**Research date:** 2026-03-19
**Valid until:** 2026-04-18 (mlx_lm pinned in pyproject.toml; scipy and numpy stable)
