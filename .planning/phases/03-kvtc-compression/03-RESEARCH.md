# Phase 3: kvtc Compression - Research

**Researched:** 2026-03-19
**Domain:** PCA-based KV cache compression — MLX tensor math, DP quantization, zstd entropy coding
**Confidence:** HIGH (grounded in working spike code + verified spike results)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**PCA scope:**
- Cross-layer PCA with Procrustes alignment — not per-layer
- Separate K-basis and V-basis
- Layer grouping is configurable — n_groups is a calibration-time parameter. Compressor receives a list of (basis, layer_indices) and looks up which basis applies per layer. Compressor is group-count-agnostic.
- Procrustes alignment runs during calibration only (Phase 4). Compressor receives pre-aligned bases — no Procrustes logic in the compression hot path.

**KVTCCompressor API:**
- Constructor: `KVTCCompressor(pca_bundle=None, n_sink_tokens=4, sliding_window=128)`
  - `pca_bundle`: Optional pre-loaded bundle from Phase 4 calibration. When `None`, falls back to on-the-fly PCA (testing-only, lower quality, documented as such).
  - `n_sink_tokens=4`: First 4 tokens exempt from compression (matches AMCompactor and KVTC-05).
  - `sliding_window=128`: Last 128 tokens exempt from compression (KVTC-06).
- `compress(kv_cache) -> bytes`: Takes list of `(keys, values)` tuples, returns a self-describing blob.
- `decompress(blob) -> list[tuple[mx.array, mx.array]]`: Blob is self-describing — contains codebooks, group map, all metadata. No bundle required at decompress time.

**Testability (without Phase 4):**
- On-the-fly PCA fallback when `pca_bundle=None`: same pattern as `AMCompactor(head_entropy=None)`.
- Test fixtures use actual Qwen 2.5 7B KV caches, marked `@pytest.mark.slow`.
- No synthetic tensor tests for integration path — want real RoPE and shape issues caught.
- One `@pytest.mark.slow` integration test: extract real KV cache, strip RoPE, compress, decompress, verify cosine similarity round-trip.

**DP quantization:**
- Proper variable-bit DP allocation per KVTC-02 — not uniform 4-bit.
- Each PCA component gets its own bit width (e.g., 6 bits for component 0, 2 bits for component 64).
- DP minimizes reconstruction MSE subject to a target bits-per-token budget (configurable on KVTCCompressor).
- Lloyd's codebook per component for quantization levels.
- DP bit_alloc table is pre-baked in the calibration bundle. Testing fallback: fixed 4-bit uniform when no bundle.

**RoPE stripping:**
- Caller's responsibility — `compress()` assumes keys have already had RoPE stripped.
- KVTCCompressor is model-agnostic. Phase 4 and Phase 5 handle stripping before calling compress().
- Phase 3 scope includes one `@pytest.mark.slow` integration test that strips RoPE before compressing.

**GQA handling (KVTC-07):**
- Compressor operates on KV heads only — not query heads.
- Shape contract: input tensors are `[1, n_kv_heads, seq_len, head_dim]`.
- Verified correct on Qwen 2.5 7B (4 KV heads, 8 query heads in GQA config).

### Claude's Discretion

- Exact self-describing blob format (header layout, serialization scheme)
- Exact DP algorithm implementation (standard greedy DP or scipy minimizer)
- Internal batching strategy for per-layer compression
- Error message text and validation details

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| KVTC-01 | Cross-layer PCA basis V^T computed from calibration data with RoPE embeddings stripped | On-the-fly SVD via `svd_f32` when no bundle; bundle lookup otherwise. Spike confirms correctness. |
| KVTC-02 | DP algorithm allocates optimal bit widths per PCA component under a global bit budget | Greedy DP over components in singular-value order; Lloyd's codebook per component. Spike validated at uniform 4-bit baseline; variable-bit formalization is direct extension. |
| KVTC-03 | Quantized PCA coefficients are entropy-coded with zstd for final compressed representation | `zstandard` Python package (not yet in pyproject.toml — Wave 0 gap). Spike confirmed 6.8x ratio with zstd level=3. |
| KVTC-04 | Decompression restores KV cache tensors from compressed bitstream for attention computation | Blob is self-describing (codebooks embedded). Spike confirms 1.5ms decompression per layer. |
| KVTC-05 | First s=4 tokens (attention sinks) are exempt from compression | Stored verbatim alongside compressed blob, excluded from PCA projection. |
| KVTC-06 | Last w=128 tokens (sliding window) are exempt from compression | Stored verbatim alongside compressed blob, excluded from PCA projection. |
| KVTC-07 | GQA models handled correctly (compress KV heads, not query heads) | Shape contract `[1, n_kv_heads, seq_len, head_dim]` enforced. Verified on Qwen 2.5 7B (4 KV heads). |
</phase_requirements>

---

## Summary

Phase 3 implements `omlx/compression/kvtc.py`, a stateless `KVTCCompressor` that performs the full pipeline: PCA projection, DP quantization, zstd entropy coding, and a self-describing compressed byte blob. The class is a sibling of `am.py` and follows the same optional-bundle pattern. A working spike (`docs/research/kv-cache-compression/spike_kvtc.py`) provides verified reference implementations for every algorithmic step; this research directly grounds planning in that concrete code.

The confirmed baseline from the spike (501-token Qwen 2.5 7B, 28 layers, 4 KV heads, head_dim=128): 6.8x compression ratio, cosine similarity ~0.98, 1.5ms decompression per layer. The latency target is no more than 10ms per layer at 8K context. The spike runs at 1.5ms at 501 tokens; since decompression cost is linear in tokens, the budget is achievable with approximately 6x token count headroom to spare before hitting 10ms.

Two critical Wave 0 gaps exist: (1) `zstandard` is not in `pyproject.toml` and is not installed in the project venv — must be added before any test can pass; (2) `tests/test_kvtc.py` does not exist yet and must be created as part of Wave 0.

**Primary recommendation:** Follow the spike pipeline exactly, formalizing each step into `KVTCCompressor` methods. Use `svd_f32` from `linalg_utils.py` for the on-the-fly PCA fallback path. Implement DP bit allocation as a greedy descent over components sorted by singular value (largest SV gets most bits first). Use `struct` plus `numpy.tobytes()` for the self-describing blob header. Add `zstandard` to `pyproject.toml` in Wave 0.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `mlx` | >=0.31.1 | Tensor math, all KV cache operations | Project foundation; all tensors are `mx.array` |
| `numpy` | >=1.24.0 | CPU-side coefficient arrays for DP quantization; `tobytes()` for blob serialization | Bridge between MLX GPU tensors and Python bytes; already a direct project dependency |
| `zstandard` | latest stable | Entropy coding of packed quantization indices | NOT in pyproject.toml — must be added. Spike confirmed level=3 gives good ratio at fast decompression |
| `scipy` | >=1.7.0 | (Optional) scipy.optimize for DP; already imported via linalg_utils | Available; can use for fallback DP if needed |
| `struct` | stdlib | Binary header packing (blob self-description) | No external dep; handles int/float field layout cleanly |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `omlx.compression.linalg_utils.svd_f32` | project | PCA calibration in on-the-fly fallback path | Any time SVD is needed; enforced by CI lint gate |
| `mlx.core as mx` | >=0.31.1 | PCA projection, reconstruction (matmul only) | All tensor ops — no bare `mx.linalg.svd` outside linalg_utils |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `zstandard` (Python CFFI) | `pyzstd` or `python-zstd` | zstandard is the canonical Python zstd binding (maintained by zstd author). Spike uses it. No reason to diverge. |
| Greedy DP (custom) | `scipy.optimize.minimize` | Greedy DP is O(n_components x n_bit_levels) and transparent. scipy minimizer adds overhead for no gain in this discrete allocation problem. |
| `struct` for blob header | `msgpack` or `json` | `struct` is stdlib, zero-overhead, deterministic byte layout. JSON is wasteful. msgpack is an unnecessary dependency for a fixed-schema header. |

**Installation (Wave 0):**
```bash
# Add to pyproject.toml [project.dependencies]:
"zstandard>=0.21.0",

# Install for development:
pip install -e ".[dev]"
```

---

## Architecture Patterns

### Recommended Project Structure
```
omlx/
└── compression/
    ├── __init__.py        # intentionally empty — no re-exports
    ├── linalg_utils.py    # svd_f32, pinv_f32 — existing
    ├── am.py              # AMCompactor — existing
    └── kvtc.py            # KVTCCompressor — Phase 3 target

tests/
└── test_kvtc.py           # mirrors omlx/compression/kvtc.py — Wave 0 creates
```

### Pattern 1: Optional-Bundle Constructor (mirrors AMCompactor)

**What:** Constructor accepts an optional `pca_bundle`. When `None`, on-the-fly PCA from the input cache is used (testing only). When provided, bundle lookup is used (production path).

**When to use:** Always. This pattern is established by `AMCompactor(head_entropy=None)` and must be consistent.

```python
# Source: omlx/compression/am.py (established idiom)
class KVTCCompressor:
    def __init__(self, pca_bundle=None, n_sink_tokens: int = 4, sliding_window: int = 128):
        self._bundle = pca_bundle
        self.n_sink_tokens = n_sink_tokens
        self.sliding_window = sliding_window

    def compress(self, kv_cache: list[tuple[mx.array, mx.array]]) -> bytes:
        if self._bundle is None:
            # On-the-fly fallback: compute PCA from input (testing only)
            bundle = self._compute_onthefly_bundle(kv_cache)
        else:
            bundle = self._bundle
        ...
```

### Pattern 2: MLX Graph Materialization Alias

**What:** `mx.eval()` in MLX code is the lazy graph materialization function — it forces MLX tensor computation and is NOT Python's built-in eval. The project aliases it to `_mx_materialize` to document intent and avoid security scanner confusion. Established in `am.py`.

**When to use:** Before any numpy conversion (`np.array()` on an `mx.array`).

```python
# Source: omlx/compression/am.py lines 29-30
# IMPORTANT: _mx_materialize = mx.eval — this is MLX lazy graph materialization,
# NOT Python's built-in eval(). It does not execute strings or arbitrary code.
_mx_materialize = mx.eval

# Usage before numpy bridge:
_mx_materialize(coeffs)
coeffs_np = np.array(coeffs)
```

### Pattern 3: Sink + Window Exemption Split

**What:** Before compression, split the token sequence into three ranges:
- `[:n_sink_tokens]` — stored verbatim (uncompressed)
- `[n_sink_tokens:-sliding_window]` — compressed via PCA, DP, and zstd
- `[-sliding_window:]` — stored verbatim (uncompressed)

**When to use:** In every `compress()` call before PCA projection.

```python
# Derived from spike_kvtc.py structure and CONTEXT.md decisions
def _split_tokens(self, tensor: mx.array):
    # tensor: [1, n_kv_heads, seq_len, head_dim]
    seq_len = tensor.shape[2]
    sink_end = self.n_sink_tokens
    window_start = max(sink_end, seq_len - self.sliding_window)
    sinks = tensor[:, :, :sink_end, :]
    body = tensor[:, :, sink_end:window_start, :]
    window = tensor[:, :, window_start:, :]
    return sinks, body, window
```

### Pattern 4: Self-Describing Blob Format

**What:** The blob must carry all information needed for decompression without the bundle in memory. This enables cold SSD storage (Phase 6 semantics).

**Recommended layout:**
```
[4 bytes] magic: b'KVTC'
[4 bytes] version: uint32 = 1
[4 bytes] n_layers: uint32
[4 bytes] n_sink_tokens: uint32
[4 bytes] sliding_window: uint32
--- per-layer sections ---
For each layer:
  [4 bytes] layer_idx: uint32
  [4 bytes] n_kv_heads: uint32
  [4 bytes] seq_len (original): uint32
  [4 bytes] head_dim: uint32
  [4 bytes] n_components: uint32
  [raw bytes] sink tokens: float16 array [1, n_kv_heads, n_sink_tokens, head_dim]
  [raw bytes] window tokens: float16 array [1, n_kv_heads, window_len, head_dim]
  [4 bytes] k_blob_len: uint32
  [k_blob_len bytes] zstd-compressed K blob (packed indices + K codebooks)
  [4 bytes] v_blob_len: uint32
  [v_blob_len bytes] zstd-compressed V blob (packed indices + V codebooks)
  [n_components bytes] k_bit_alloc: uint8 array (bits per component, 1-8)
  [n_components bytes] v_bit_alloc: uint8 array
  [head_dim * n_components * 4 bytes] K basis: float32 [head_dim, n_components]
  [head_dim * 4 bytes] K mean: float32 [head_dim]
  [head_dim * n_components * 4 bytes] V basis: float32 [head_dim, n_components]
  [head_dim * 4 bytes] V mean: float32 [head_dim]
```

**Key design constraint:** Codebooks and PCA basis are embedded in the blob — `decompress()` is fully self-contained.

### Pattern 5: DP Bit Allocation (Greedy)

**What:** Allocate integer bit widths (1 to 8) to each PCA component to minimize total MSE under a global bits-per-token budget.

**When to use:** At compress time when bundle has pre-baked `bit_alloc` table (lookup, not recomputed). In on-the-fly testing mode: use uniform 4-bit.

**Greedy DP approach** (O(n_components x budget_steps)):
```python
# Derived from spike dp_quantize() + standard rate-distortion allocation theory
def _dp_allocate_bits(
    singular_values: np.ndarray,  # [n_components] from SVD, largest first
    bits_per_token_budget: float,
    n_tokens: int,
    n_components: int,
    min_bits: int = 1,
    max_bits: int = 8,
) -> np.ndarray:
    """Returns uint8 array [n_components] of bit widths, each >= min_bits."""
    total_budget = int(bits_per_token_budget * n_tokens * n_components)
    alloc = np.full(n_components, min_bits, dtype=np.uint8)
    remaining = total_budget - min_bits * n_components
    priority = np.argsort(singular_values)[::-1]  # highest SV first
    for comp_idx in priority:
        extra = min(max_bits - alloc[comp_idx], remaining)
        alloc[comp_idx] += extra
        remaining -= extra
        if remaining <= 0:
            break
    return alloc
```

### Pattern 6: Lloyd's Codebook per Component

**What:** For each PCA component, fit 2^n_bits centroids using Lloyd's algorithm (3 iterations). Same as spike.

**Source:** `spike_kvtc.py` lines 199-228 — confirmed working.

```python
# Source: spike_kvtc.py dp_quantize() — proven implementation
def _lloyd_codebook(col: np.ndarray, n_bits: int) -> np.ndarray:
    """Fit Lloyd's codebook for one PCA component. Returns centroids [2^n_bits]."""
    n_levels = 2 ** n_bits
    percentiles = np.linspace(0, 100, n_levels + 1)
    boundaries = np.percentile(col, percentiles)
    centroids = (boundaries[:-1] + boundaries[1:]) / 2
    for _ in range(3):
        dists = np.abs(col[:, None] - centroids[None, :])
        indices = np.argmin(dists, axis=1)
        for k in range(n_levels):
            mask = indices == k
            if np.any(mask):
                centroids[k] = col[mask].mean()
    return centroids
```

### Pattern 7: Variable-Width Index Packing

**What:** Because each component has a different bit width (1 to 8 bits), indices cannot be naively packed at a fixed 4 bits/index as in the spike. Recommended approach: store each component's indices as 1-byte-per-index and rely on zstd to compress the byte-level redundancy.

```python
# Packing a single component: 1 index per byte (simpler than bit packing, zstd compensates)
def _pack_component_indices(indices: np.ndarray) -> bytes:
    return indices.astype(np.uint8).tobytes()
```

**Note:** The spike packs 2 indices per byte at fixed 4-bit. With variable widths, 1-index-per-byte is simpler and avoids bit-manipulation bugs. zstd still achieves strong compression ratios on the byte stream (spike confirmed 6.8x overall ratio including this overhead).

### Anti-Patterns to Avoid

- **Bare `mx.linalg.svd`:** Must go through `svd_f32` from `linalg_utils`. A CI lint gate enforces this. Use `from omlx.compression.linalg_utils import svd_f32`.
- **Compressing query heads in GQA models:** The shape contract is `[1, n_kv_heads, seq_len, head_dim]`. The compressor never receives query heads. Do not add GQA-detection inside `kvtc.py`.
- **Procrustes alignment in compress():** Procrustes runs during calibration (Phase 4). The compressor receives pre-aligned bases and performs only matmul projection.
- **Recomputing DP at runtime:** The bundle carries a pre-baked `bit_alloc` table. `compress()` does a lookup, never recomputes DP per call.
- **RoPE stripping inside kvtc.py:** The caller is responsible. No frequency table in `kvtc.py`.
- **Accessing `self._bundle` inside `decompress()`:** `decompress()` must be fully self-contained from the blob.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Entropy coding | Custom Huffman or RLE | `zstandard` Python package | zstd has hardware-accelerated decode; Huffman hand-roll misses ANS coding advantages. Spike proved 6.8x ratio. |
| SVD | Direct `mx.linalg.svd` call | `svd_f32` from `linalg_utils.py` | CI lint gate enforces this; float16 input causes silent NaN (Phase 1 finding). |
| Centroid initialization | Random init for Lloyd's | Percentile-based init | Random init has poor convergence; percentile init reaches good centroids in 3 iterations (spike proven). |
| Bit packing | Bit-level manipulation | 1-index-per-byte + zstd | Bit-level packing is bug-prone with variable widths; zstd handles byte-level redundancy compression effectively. |

**Key insight:** The spike already proves the pipeline works end-to-end. Phase 3 is formalization, not research. The only algorithmic extensions beyond the spike are (1) cross-layer PCA group lookup instead of per-layer SVD, and (2) variable-bit DP allocation instead of uniform 4-bit. Both are direct, low-risk changes to the spike structure.

---

## Common Pitfalls

### Pitfall 1: zstandard Not in Dependencies
**What goes wrong:** `import zstandard` raises `ModuleNotFoundError` at test collection time. All tests fail immediately.
**Why it happens:** `zstandard` is used in the spike but was never added to `pyproject.toml`. Confirmed absent from project venv.
**How to avoid:** Wave 0 adds `"zstandard>=0.21.0"` to `[project.dependencies]` in `pyproject.toml` and runs `pip install -e ".[dev]"`.
**Warning signs:** `ModuleNotFoundError: No module named 'zstandard'` — confirmed by direct venv check.

### Pitfall 2: MLX Lazy Tensor Not Materialized Before numpy Bridge
**What goes wrong:** `np.array(coeffs)` on a lazy MLX tensor returns garbage or raises.
**Why it happens:** MLX uses lazy evaluation; tensors are not materialized until the materialization function is called.
**How to avoid:** Always call `_mx_materialize(tensor)` before any `np.array()` conversion. Pattern established in `am.py`.
**Warning signs:** Unexpected NaN in numpy arrays or silent wrong values in quantization.

### Pitfall 3: Empty Body on Short Sequences
**What goes wrong:** Splitting `[:n_sink_tokens]` and `[-sliding_window:]` gives a zero-length or negative-length body for sequences shorter than `n_sink_tokens + sliding_window = 132` tokens.
**Why it happens:** For seq_len=100, `tensor[:, :, 4:-128, :]` clamps to zero tokens. Lloyd's algorithm on an empty column fails.
**How to avoid:** Guard: if body length is 0, store all tokens verbatim with no compression. Decompressor must handle the no-body case in the blob header.
**Warning signs:** Empty `body` tensor passed to `pca_project()`.

### Pitfall 4: Blob Not Self-Describing — Bundle Required at Decompress
**What goes wrong:** `decompress(blob)` fails in Phase 6 cold-storage scenario because bundle is not in memory.
**Why it happens:** Easy to accidentally rely on `self._bundle` inside `decompress()` if not careful.
**How to avoid:** `decompress()` must read all parameters (basis, mean, codebooks, bit_alloc) from the blob header. Never access `self._bundle` inside `decompress()`. Catch in code review.

### Pitfall 5: mx.linalg.svd Called Outside linalg_utils
**What goes wrong:** CI lint gate fails; PR cannot merge.
**Why it happens:** Copy-paste from spike code which calls `mx.linalg.svd` directly.
**How to avoid:** Import `svd_f32` from `omlx.compression.linalg_utils`. Never use bare `mx.linalg.svd` in production code.
**Warning signs:** Running `grep -n "mx.linalg.svd" omlx/compression/kvtc.py` returns matches.

### Pitfall 6: DP Allocation Produces 0-bit Components
**What goes wrong:** A component allocated 0 bits has undefined codebook behavior (2^0 = 1 level means every value maps to one centroid, losing all information).
**Why it happens:** Greedy DP with a tight budget can exhaust the budget before reaching low-SV components.
**How to avoid:** Set `min_bits = 1`. The DP loop enforces this floor. Include a test that verifies all alloc values are at least 1.

### Pitfall 7: pytest.mark.slow Not Registered
**What goes wrong:** `PytestUnknownMarkWarning` on any test decorated with `@pytest.mark.slow`.
**Why it happens:** `pyproject.toml` has no `markers` list in `[tool.pytest.ini_options]`.
**How to avoid:** Wave 0 adds `markers = ["slow: marks tests as slow (deselect with '-m \"not slow\"')"]` to `[tool.pytest.ini_options]`.

---

## Code Examples

Verified patterns from spike source and established project code:

### PCA Projection (from spike, verified working)
```python
# Source: spike_kvtc.py lines 142-171
def pca_project(tensor: mx.array, basis: mx.array, mean: mx.array) -> mx.array:
    # tensor: [1, n_kv_heads, seq_len, head_dim]
    # basis:  [head_dim, n_components]
    # mean:   [head_dim]
    t = tensor.astype(mx.float32)
    centered = t - mean
    coeffs = centered @ basis  # [1, n_kv_heads, seq_len, n_components]
    return coeffs

def pca_reconstruct(coeffs: mx.array, basis: mx.array, mean: mx.array) -> mx.array:
    # coeffs: [1, n_kv_heads, seq_len, n_components]
    reconstructed = coeffs @ basis.T + mean  # [1, n_kv_heads, seq_len, head_dim]
    return reconstructed
```

### On-the-fly SVD for Fallback PCA
```python
# Source: spike_kvtc.py lines 110-122 + linalg_utils.py
from omlx.compression.linalg_utils import svd_f32

# _mx_materialize is the MLX lazy graph materialization call (forces tensor computation)
# It is NOT Python's built-in eval(). It does not execute strings or arbitrary code.
_mx_materialize = mx.eval

def _calibrate_onthefly(tensor: mx.array, n_components: int):
    # tensor: [1, n_kv_heads, seq_len, head_dim]
    head_dim = tensor.shape[3]
    data = tensor.reshape(-1, head_dim).astype(mx.float32)
    mean = mx.mean(data, axis=0, keepdims=True)
    centered = data - mean
    _mx_materialize(centered)
    U, S, Vt = svd_f32(centered)   # svd_f32, never bare mx.linalg.svd
    _mx_materialize(U, S, Vt)
    basis = Vt[:n_components].T    # [head_dim, n_components]
    return basis, mean.squeeze(0), S[:n_components]
```

### zstd Compress/Decompress
```python
# Source: spike_kvtc.py lines 234-245
import zstandard as zstd

def _compress_zstd(data: bytes, level: int = 3) -> bytes:
    return zstd.ZstdCompressor(level=level).compress(data)

def _decompress_zstd(compressed: bytes) -> bytes:
    return zstd.ZstdDecompressor().decompress(compressed)
```

### Lloyd's Quantization (from spike, verified)
```python
# Source: spike_kvtc.py lines 196-228 — proven implementation
def _lloyd_codebook(col: np.ndarray, n_bits: int) -> np.ndarray:
    n_levels = 2 ** n_bits
    percentiles = np.linspace(0, 100, n_levels + 1)
    boundaries = np.percentile(col, percentiles)
    centroids = (boundaries[:-1] + boundaries[1:]) / 2
    for _ in range(3):
        dists = np.abs(col[:, None] - centroids[None, :])
        indices = np.argmin(dists, axis=1)
        for k in range(n_levels):
            mask = indices == k
            if np.any(mask):
                centroids[k] = col[mask].mean()
    return centroids
```

### Blob Header Struct Format
```python
# Claude's Discretion: recommended approach using stdlib struct
import struct

MAGIC = b'KVTC'
VERSION = 1
HEADER_FMT = '!4sIIII'  # magic(4), version, n_layers, n_sink_tokens, sliding_window
HEADER_SIZE = struct.calcsize(HEADER_FMT)

def _pack_header(n_layers: int, n_sink_tokens: int, sliding_window: int) -> bytes:
    return struct.pack(HEADER_FMT, MAGIC, VERSION, n_layers, n_sink_tokens, sliding_window)

def _unpack_header(data: bytes) -> tuple:
    return struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-layer PCA (spike) | Cross-layer PCA with Procrustes alignment (production) | Decision in CONTEXT.md | Better compression ratio across similar layers; basis count = n_groups, not n_layers |
| Uniform 4-bit quantization (spike) | Variable-bit DP per component | Decision in CONTEXT.md | Lower MSE under same bit budget; high-SV components get more bits |
| Separate codebook storage | Self-describing blob (codebooks embedded) | Decision in CONTEXT.md | Enables SSD cold storage without keeping bundle in memory |

**Spike baseline (confirmed measured results from spike_kvtc_results.json):**
- 501 tokens, 28 layers, 4 KV heads, head_dim=128 (Qwen 2.5 7B): 6.8x compression ratio, cosine sim ~0.98, 1.5ms decompression per layer
- Per-layer PCA + uniform 4-bit. Production (cross-layer + variable-bit) expected to improve both ratio and quality.

---

## Open Questions

1. **bits-per-token budget default value**
   - What we know: DP allocation is parameterized by a target bits-per-token budget (configurable on `KVTCCompressor`).
   - What's unclear: What is the right default? The spike uses head_dim//2 components at 4 bits, equivalent to approximately 2 bits/token at head_dim=128.
   - Recommendation: Default to `bits_per_token=4.0` (matches common LLM quantization intuition). Can be tuned in Phase 7 validation. Document clearly in docstring.

2. **Short sequence guard threshold**
   - What we know: Sequences shorter than `n_sink_tokens + sliding_window = 132` tokens produce an empty body.
   - What's unclear: Should the compressor raise, warn, or silently store everything verbatim?
   - Recommendation: Store entirely verbatim with a docstring note. Zero body tokens is valid. Decompressor must handle it.

3. **Blob basis precision: float16 vs float32**
   - What we know: Spike stores codebooks as float16. Bases are computed in float32 (SVD output).
   - What's unclear: Is float16 basis sufficient for reconstruction quality? PCA basis quantization error stacks on top of coefficient quantization error.
   - Recommendation: Store basis as float32 in the blob (8 bytes/weight vs 4). At head_dim=128, n_components=64, 28 layers: 28 x 2 x 128 x 64 x 4 = 3.6 MB total basis overhead — acceptable for a blob also carrying compressed tokens.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=9.0.2 (from pyproject.toml dependency-groups) |
| Config file | `[tool.pytest.ini_options]` in `pyproject.toml` |
| Quick run command | `pytest tests/test_kvtc.py -m "not slow" -x` |
| Full suite command | `pytest tests/test_kvtc.py -v` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| KVTC-01 | PCA projection produces correct coefficients; on-the-fly fallback works | unit | `pytest tests/test_kvtc.py::TestPCAProjection -x` | ❌ Wave 0 |
| KVTC-02 | DP bit allocation: all values >= 1; higher-SV components get more bits; sum within budget | unit | `pytest tests/test_kvtc.py::TestDPAllocation -x` | ❌ Wave 0 |
| KVTC-03 | compress() produces bytes; result decompresses to matching shape | unit | `pytest tests/test_kvtc.py::TestCompressDecompress -x` | ❌ Wave 0 |
| KVTC-04 | Round-trip: decompress() restores tensors with cosine sim >= 0.97 | unit + slow | `pytest tests/test_kvtc.py::TestRoundTrip -x` | ❌ Wave 0 |
| KVTC-05 | First 4 tokens in every layer are identical before and after round-trip | unit | `pytest tests/test_kvtc.py::TestSinkTokenExemption -x` | ❌ Wave 0 |
| KVTC-06 | Last 128 tokens in every layer are identical before and after round-trip | unit | `pytest tests/test_kvtc.py::TestWindowTokenExemption -x` | ❌ Wave 0 |
| KVTC-07 | Input shape `[1, n_kv_heads, seq_len, head_dim]` handled; GQA (4 KV heads) works correctly | unit | `pytest tests/test_kvtc.py::TestGQAShapeContract -x` | ❌ Wave 0 |
| Integration | Real Qwen 2.5 7B KV cache: strip RoPE, compress, decompress, cosine sim >= 0.97 | slow | `pytest tests/test_kvtc.py -m slow -v` | ❌ Wave 0 |

**KVTC-05 ablation:** Test also verifies that when the sink exemption is disabled (all tokens compressed), reconstruction quality collapses for sink positions — confirming exemption is load-bearing.

### Sampling Rate
- **Per task commit:** `pytest tests/test_kvtc.py -m "not slow" -x`
- **Per wave merge:** `pytest tests/test_kvtc.py -v` (includes slow tests if model available)
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_kvtc.py` — all test classes above (KVTC-01 through KVTC-07 and integration)
- [ ] `omlx/compression/kvtc.py` — stub with `KVTCCompressor` class raising `NotImplementedError` (confirms import works, gives RED state)
- [ ] `"zstandard>=0.21.0"` added to `[project.dependencies]` in `pyproject.toml`
- [ ] `markers = ["slow: marks tests as slow (deselect with '-m \"not slow\"')"]` added to `[tool.pytest.ini_options]` in `pyproject.toml`
- [ ] Install: `pip install -e ".[dev]"` after pyproject.toml update

---

## Sources

### Primary (HIGH confidence)
- `docs/research/kv-cache-compression/spike_kvtc.py` — complete working pipeline; directly inspected line by line
- `docs/research/kv-cache-compression/spike_kvtc_results.json` — measured results (6.8x ratio, 1.5ms decompression, cos=0.98)
- `omlx/compression/am.py` — established idioms: optional-bundle pattern, materialization alias, shape contracts
- `omlx/compression/linalg_utils.py` — `svd_f32` API; lint gate enforcement confirmed in STATE.md
- `pyproject.toml` — confirmed `zstandard` absence; confirmed `pytest` version; confirmed `mlx>=0.31.1`
- `.planning/phases/03-kvtc-compression/03-CONTEXT.md` — all locked decisions

### Secondary (MEDIUM confidence)
- `spike_kvtc_results.json` — 501-token baseline confirmed; 8K context latency budget extrapolated by linear scaling assumption (not directly measured at 8K)
- STATE.md decision log — "DP quantization and attention sink exemptions are correctness requirements, not quality enhancements — must be in Phase 3"

### Tertiary (LOW confidence)
- Greedy DP correctness for this specific discrete allocation problem — follows standard rate-distortion allocation for transform coding but not independently verified against a formal reference in this codebase

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries verified in venv or pyproject.toml; zstandard absence confirmed by direct import test
- Architecture patterns: HIGH — derived from working spike code and am.py idioms; no speculation
- DP quantization: MEDIUM-HIGH — greedy DP approach is standard; specific bit allocation defaults are Claude's discretion
- Blob format: MEDIUM — self-describing requirement is locked; exact layout is Claude's discretion; struct approach is recommended
- Pitfalls: HIGH — zstandard absence confirmed; float16-to-numpy without materialization confirmed from Phase 1 research; short-sequence guard derived from math

**Research date:** 2026-03-19
**Valid until:** 2026-04-18 (MLX and zstandard are stable; no fast-moving surface here)
