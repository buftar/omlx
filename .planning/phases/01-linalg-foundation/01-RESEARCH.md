# Phase 1: Linalg Foundation - Research

**Researched:** 2026-03-18
**Domain:** MLX linear algebra wrappers, scipy NNLS bridge, Python package structure
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Package structure**
- Create `omlx/compression/` as the new package root; `__init__.py` is empty — no re-exports
- `linalg_utils.py` is the only file in Phase 1; future phases (am.py, kvtc.py, etc.) are added as siblings
- Module is pure in-memory tensor ops — no file path helpers, no I/O
- Include `qr_f32` alongside `svd_f32` and `pinv_f32` while the linalg pattern is being established

**Float16/bfloat16 handling**
- Auto-cast silently: both `float16` and `bfloat16` inputs are cast to `float32` before computation
- All wrappers delegate to a shared private `_ensure_f32(tensor)` function — one place to update if policy changes
- All wrappers always return `float32` — never cast back to the input dtype
- CPU stream routing (`stream=mx.cpu`) is applied inside the wrappers; callers never pass a stream

**NNLS bridge**
- Signature: `nnls_solve(A: mx.array, b: mx.array) -> tuple[mx.array, float]`
- Returns `(solution, residual)` — residual is `||Ax - b||` as a Python float
- Bridge converts to `np.float64` for the scipy call (scipy default), returns solution as `mx.float32`
- Input validation: assert `b` is 1D and shape-compatible with `A`; raise `ValueError` with a clear message on mismatch
- MLX->numpy->MLX conversion is fully internal — callers never import scipy or numpy

### Claude's Discretion
- Docstring style and verbosity
- Whether to add `__all__` to linalg_utils.py
- Exact ValueError message text

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MATH-01 | Linalg safety layer wraps MLX ops with automatic float32 cast and CPU stream routing | Confirmed: MLX raises `ValueError` for float16/bfloat16 inputs; GPU stream raises for SVD/pinv/QR. `_ensure_f32` + `stream=mx.cpu` pattern empirically validated. |
| MATH-02 | scipy NNLS wrapper for beta-fitting with numpy<->MLX bridge | Confirmed: `scipy.optimize.nnls` accepts np.float64 arrays, returns `(x, residual)`. `np.array(mx_array, dtype=np.float64)` and `mx.array(np_result, dtype=mx.float32)` both work. |
| MATH-03 | OLS value-fitting via `mx.linalg.pinv` with correct stream/dtype handling | Confirmed: `mx.linalg.pinv(A, stream=mx.cpu)` works on float32. GPU raises immediately. Covered by `pinv_f32` wrapper. |
</phase_requirements>

---

## Summary

Phase 1 creates `omlx/compression/linalg_utils.py` — the float32-safe linalg layer that all downstream compression phases (2, 3, 4) import. The domain is narrow and well-understood: three MLX linalg wrappers (`svd_f32`, `pinv_f32`, `qr_f32`), one scipy NNLS bridge (`nnls_solve`), one private helper (`_ensure_f32`), and a pytest-based CI lint gate.

All critical behaviors were empirically confirmed against the installed stack (MLX 0.31.0, scipy 1.17.1, numpy 2.4.2). MLX raises `ValueError` for float16 and bfloat16 inputs to `svd`, `pinv`, and `qr`. All three ops raise immediately when no `stream=mx.cpu` is given. The numpy bridge (`np.array(mx_array, dtype=np.float64)`) round-trips correctly. There are no surprises to navigate.

The only existing `mx.linalg.*` call in the codebase is `mx.linalg.norm` in `omlx/models/base_model.py:72`, which is float16-safe and must be excluded from the lint gate.

**Primary recommendation:** Implement the five public symbols (`_ensure_f32`, `svd_f32`, `pinv_f32`, `qr_f32`, `nnls_solve`) in a single task, then write tests in a second task. The lint gate (pytest scanning source files) is a third task.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| mlx | >=0.31.1 (installed: 0.31.0) | SVD, pinv, QR ops | Project dependency; only ML framework in use |
| scipy | 1.17.1 | NNLS solver (`scipy.optimize.nnls`) | Already in venv; proven in research spike |
| numpy | >=1.24.0 (installed: 2.4.2) | numpy<->MLX bridge for scipy | Project dependency; `np.array(mx_array)` works natively |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | >=7.0.0 | Unit tests + lint gate | All tests in this phase |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| scipy.optimize.nnls | mx-native NNLS | No native MLX NNLS exists; numpy bridge is the only option |
| auto-cast in `_ensure_f32` | raise on bad dtype | User decision: auto-cast. Raising was the alternative |

**Installation:** No new dependencies for scipy/numpy/mlx (already in `pyproject.toml`). See Open Questions — scipy is missing from declared dependencies.

---

## Architecture Patterns

### Recommended Project Structure
```
omlx/
└── compression/
    ├── __init__.py          # empty — no re-exports (locked decision)
    └── linalg_utils.py      # Phase 1 deliverable

tests/
└── test_linalg_utils.py     # mirrors omlx/compression/linalg_utils.py
```

### Pattern 1: `_ensure_f32` shared helper
**What:** Single private function that casts float16/bfloat16 to float32; passes float32 and float64 through unchanged.
**When to use:** Called at the top of every public wrapper — one place to change cast policy.
**Example:**
```python
# Verified empirically: MLX raises ValueError for float16 and bfloat16
# on svd, pinv, and qr. _ensure_f32 prevents the error by casting first.
def _ensure_f32(tensor: mx.array) -> mx.array:
    if tensor.dtype in (mx.float16, mx.bfloat16):
        return tensor.astype(mx.float32)
    return tensor
```

### Pattern 2: CPU-stream wrapper
**What:** Every linalg op passes `stream=mx.cpu`. The caller never passes a stream.
**When to use:** All SVD, pinv, and QR calls. GPU raises `[linalg::svd] This op is not yet supported on the GPU.`
**Example:**
```python
# Verified empirically: stream=mx.cpu required; bare call raises on GPU
def svd_f32(a: mx.array, compute_uv: bool = True):
    a = _ensure_f32(a)
    return mx.linalg.svd(a, compute_uv=compute_uv, stream=mx.cpu)
```

### Pattern 3: scipy NNLS numpy bridge
**What:** Convert MLX arrays to `np.float64`, call `scipy.optimize.nnls`, return `(mx.float32, float)`.
**When to use:** `nnls_solve` only. No other function should touch numpy or scipy.
**Example:**
```python
# Verified empirically: np.array(mx_array, dtype=np.float64) works correctly
# scipy.optimize.nnls returns (solution_array, residual_float)
from scipy.optimize import nnls as _scipy_nnls
import numpy as _np

def nnls_solve(A: mx.array, b: mx.array) -> tuple[mx.array, float]:
    if b.ndim != 1:
        raise ValueError(...)
    if A.shape[0] != b.shape[0]:
        raise ValueError(...)
    A_np = _np.array(A, dtype=_np.float64)
    b_np = _np.array(b, dtype=_np.float64)
    x_np, residual = _scipy_nnls(A_np, b_np)
    return mx.array(x_np, dtype=mx.float32), float(residual)
```

### Pattern 4: pytest source-scan lint gate
**What:** A pytest test that scans all `.py` files under `omlx/` for bare `mx.linalg.svd` or `mx.linalg.pinv`, excluding `omlx/compression/linalg_utils.py`.
**When to use:** One test function in `tests/test_linalg_utils.py`. Runs in milliseconds with no MLX required.
**Example:**
```python
# Pattern from CONTEXT.md specifics — grep source files, exclude the wrapper module
import re
from pathlib import Path

def test_no_bare_linalg_calls():
    forbidden = re.compile(r'mx\.linalg\.(svd|pinv)\b')
    allowed = Path("omlx/compression/linalg_utils.py").resolve()
    violations = []
    for path in Path("omlx").rglob("*.py"):
        if path.resolve() == allowed:
            continue
        if forbidden.search(path.read_text()):
            violations.append(str(path))
    assert not violations, f"Bare mx.linalg.svd/pinv found outside linalg_utils: {violations}"
```

### Anti-Patterns to Avoid
- **stream omitted:** Never call `mx.linalg.svd(a)` without `stream=mx.cpu` — raises on GPU at materialization time.
- **Lint gate includes `norm`:** `mx.linalg.norm` in `base_model.py:72` is float16-safe and must not be flagged.
- **Casting back to input dtype:** Wrappers always return `float32`, never cast back. Downstream callers own their dtype decisions.
- **scipy imported at module level without alias:** Import with underscore alias (`from scipy.optimize import nnls as _scipy_nnls`) so callers cannot accidentally reach it.
- **`__init__.py` with re-exports:** The locked decision is an empty `__init__.py`. Do not mirror the `omlx/cache/__init__.py` pattern.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Non-negative least squares | Custom NNLS loop | `scipy.optimize.nnls` | scipy uses FORTRAN LAWSON-HANSON; correctness is well-proven |
| Pseudoinverse | Matrix inversion + transpose | `mx.linalg.pinv` | Handles rank deficiency, numerically stable |
| Float type checking | Manual dtype string comparison | `tensor.dtype in (mx.float16, mx.bfloat16)` | MLX dtype objects compare by identity |

**Key insight:** NNLS is the only non-trivial algorithm here. The scipy implementation handles degenerate cases (all-zero columns, underdetermined systems) that a hand-rolled loop would miss.

---

## Common Pitfalls

### Pitfall 1: GPU-stream silent failure
**What goes wrong:** `mx.linalg.svd(a)` without `stream=mx.cpu` may appear to succeed but raises `[linalg::svd] This op is not yet supported on the GPU. Explicitly pass a CPU stream to run it.` when the default device is GPU.
**Why it happens:** MLX lazy computation defers the error until the graph is materialized. The call appears to succeed.
**How to avoid:** Always pass `stream=mx.cpu` inside wrappers. Tests should force graph materialization.
**Warning signs:** No error at call site, error at downstream materialization point.

### Pitfall 2: bfloat16 rejected (not just float16)
**What goes wrong:** Research spike only tested float16. bfloat16 also raises `ValueError` (empirically confirmed: `[linalg::svd] Arrays must have type float32, float64 or complex64. Received array with type bfloat16.`).
**Why it happens:** MLX linalg accepts only float32, float64, complex64 for SVD and float32/float64 for pinv/QR.
**How to avoid:** `_ensure_f32` must check `mx.bfloat16` alongside `mx.float16`.

### Pitfall 3: NNLS shape mismatch gives cryptic scipy error
**What goes wrong:** Passing `b` with wrong shape produces a confusing scipy C-extension error instead of a Python ValueError.
**Why it happens:** scipy's NNLS does not validate shapes clearly.
**How to avoid:** Validate `b.ndim == 1` and `A.shape[0] == b.shape[0]` before conversion, raise `ValueError` with a clear message.

### Pitfall 4: Lint gate catches `mx.linalg.norm`
**What goes wrong:** A broad pattern `mx\.linalg\.` flags `base_model.py:72` which uses `mx.linalg.norm`.
**Why it happens:** `norm` is in the `mx.linalg` namespace but is float16-safe and GPU-safe.
**How to avoid:** Pattern must match only `mx.linalg.svd` and `mx.linalg.pinv` (word-boundary anchored).

### Pitfall 5: SVD `full_matrices` parameter does not exist in MLX 0.31.x
**What goes wrong:** Calling `mx.linalg.svd(a, full_matrices=False, stream=mx.cpu)` raises `incompatible function arguments`.
**Why it happens:** MLX's SVD only supports `compute_uv` (bool) and `stream`. No economy/thin SVD mode.
**How to avoid:** `svd_f32` should only expose `compute_uv` as a passthrough argument. Do not expose `full_matrices`.

---

## Code Examples

Verified against MLX 0.31.0, scipy 1.17.1, numpy 2.4.2:

### MLX SVD signature (empirically verified)
```python
# mx.linalg.svd(a, compute_uv=True, *, stream=None) -> tuple[array, array, array]
# Returns (U, S, Vt) — full matrices only; no economy mode
# Accepts: float32, float64, complex64
# Rejects: float16, bfloat16 (ValueError)
# Requires stream=mx.cpu on GPU devices (raises at materialization time otherwise)
U, S, Vt = mx.linalg.svd(a_f32, stream=mx.cpu)
```

### MLX pinv signature (empirically verified)
```python
# mx.linalg.pinv(a, *, stream=None) -> array
# Accepts: float32, float64
# Rejects: float16, bfloat16, complex (ValueError)
# Requires stream=mx.cpu on GPU devices
P = mx.linalg.pinv(a_f32, stream=mx.cpu)
```

### MLX QR signature (empirically verified)
```python
# mx.linalg.qr(a, *, stream=None) -> tuple[array, array]
# Returns (Q, R)
# Accepts: float32, float64
# Rejects: float16, bfloat16 (ValueError)
# Requires stream=mx.cpu on GPU devices
Q, R = mx.linalg.qr(a_f32, stream=mx.cpu)
```

### numpy<->MLX bridge (empirically verified)
```python
# MLX -> numpy (works for all dtypes)
a_np = np.array(a_mx, dtype=np.float64)

# numpy -> MLX
a_mx = mx.array(a_np, dtype=mx.float32)
```

### scipy NNLS (empirically verified)
```python
from scipy.optimize import nnls
x, residual = nnls(A_np_f64, b_np_f64)
# x: np.ndarray, shape (n,), dtype float64, non-negative
# residual: float, ||Ax - b||
```

### Test structure pattern (matches existing project style)
```python
# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.compression.linalg_utils."""
import pytest
import mlx.core as mx
import numpy as np
from omlx.compression.linalg_utils import svd_f32, pinv_f32, qr_f32, nnls_solve

class TestEnsureF32:
    def test_float16_is_cast(self): ...
    def test_bfloat16_is_cast(self): ...
    def test_float32_is_unchanged(self): ...

class TestSvdF32:
    def test_float16_input_casts_and_succeeds(self): ...
    def test_returns_float32(self): ...

class TestNnlsSolve:
    def test_basic_solve(self): ...
    def test_b_not_1d_raises(self): ...
    def test_shape_mismatch_raises(self): ...
    def test_returns_mx_float32(self): ...

def test_no_bare_linalg_calls(): ...  # lint gate
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Inline cast + bare `mx.linalg.svd` call | Centralized `_ensure_f32` + `svd_f32` wrapper | Phase 1 (now) | Downstream phases never touch dtypes or streams |
| `scipy.optimize.nnls` called directly with numpy | `nnls_solve(mx.array, mx.array)` bridge | Phase 1 (now) | Callers stay in MLX tensor space |

**Deprecated/outdated:**
- Inline `data.astype(mx.float32)` before linalg calls (from spike files): replaced by `_ensure_f32`.
- Bare `mx.linalg.svd(centered, stream=mx.cpu)` in spike files: replaced by `svd_f32(centered)`.

---

## Open Questions

1. **scipy not in `pyproject.toml` dependencies**
   - What we know: scipy 1.17.1 is installed in the venv but not listed as a project dependency in `pyproject.toml`.
   - What's unclear: Was it installed transitively, or is it an oversight? Any downstream import of `linalg_utils` would fail in a fresh install without scipy.
   - Recommendation: Add `scipy>=1.7.0` to `pyproject.toml` `[project.dependencies]` as part of Phase 1. The planner should include this as an explicit task.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 7.x (project standard) |
| Config file | `pyproject.toml` — `[tool.pytest.ini_options]` (exists) |
| Quick run command | `pytest tests/test_linalg_utils.py -x` |
| Full suite command | `pytest tests/test_linalg_utils.py -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MATH-01 | `svd_f32` / `pinv_f32` / `qr_f32` auto-cast float16 to float32 | unit | `pytest tests/test_linalg_utils.py::TestSvdF32 -x` | Wave 0 |
| MATH-01 | `svd_f32` / `pinv_f32` / `qr_f32` use `stream=mx.cpu` (no GPU raises) | unit | `pytest tests/test_linalg_utils.py::TestCpuStream -x` | Wave 0 |
| MATH-01 | CI lint gate: no bare `mx.linalg.svd`/`pinv` outside `linalg_utils.py` | lint (pytest) | `pytest tests/test_linalg_utils.py::test_no_bare_linalg_calls -x` | Wave 0 |
| MATH-02 | `nnls_solve` accepts MLX tensors, returns `(mx.array, float)` | unit | `pytest tests/test_linalg_utils.py::TestNnlsSolve -x` | Wave 0 |
| MATH-02 | `nnls_solve` raises `ValueError` on bad shape | unit | `pytest tests/test_linalg_utils.py::TestNnlsSolve::test_b_not_1d_raises -x` | Wave 0 |
| MATH-03 | `pinv_f32` routes through `stream=mx.cpu`, returns float32 | unit | `pytest tests/test_linalg_utils.py::TestPinvF32 -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_linalg_utils.py -x`
- **Per wave merge:** `pytest tests/test_linalg_utils.py -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_linalg_utils.py` — covers all MATH-01, MATH-02, MATH-03 requirements (file does not exist yet)

---

## Sources

### Primary (HIGH confidence)
- Empirical — MLX 0.31.0 installed in `.venv` — all API signatures, dtype acceptance, stream requirements directly tested in project environment
- `docs/research/kv-cache-compression/spike_kvtc.py:115` — established `stream=mx.cpu` + float32 cast pattern
- `docs/research/kv-cache-compression/spike_am.py:157-172` — established scipy NNLS numpy bridge pattern
- `pyproject.toml` — confirmed scipy is absent from declared dependencies

### Secondary (MEDIUM confidence)
- MLX nanobind help output (from installed 0.31.0): `svd(a, compute_uv, *, stream)` — no `full_matrices` parameter

### Tertiary (LOW confidence)
- None

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — empirically verified against installed versions in project venv
- Architecture: HIGH — all patterns verified by running code in the project venv
- Pitfalls: HIGH — all pitfalls confirmed by triggering the actual errors

**Research date:** 2026-03-18
**Valid until:** 2026-09-18 (stable MLX APIs; revisit if MLX version is upgraded significantly)
