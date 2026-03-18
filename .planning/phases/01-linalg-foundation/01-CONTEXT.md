# Phase 1: Linalg Foundation - Context

**Gathered:** 2026-03-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Create `omlx/compression/linalg_utils.py` — a utility module with float32-safe MLX linalg wrappers (`svd_f32`, `pinv_f32`, `qr_f32`) and a scipy NNLS bridge (`nnls_solve`). This is pure in-memory tensor math. No file I/O, no cache integration, no calibration path logic. All downstream compressor phases (2, 3, 4) depend on this module.

</domain>

<decisions>
## Implementation Decisions

### Package structure
- Create `omlx/compression/` as the new package root; `__init__.py` is empty — no re-exports
- `linalg_utils.py` is the only file in Phase 1; future phases (am.py, kvtc.py, etc.) are added as siblings
- Module is pure in-memory tensor ops — no file path helpers, no I/O
- Include `qr_f32` alongside `svd_f32` and `pinv_f32` while the linalg pattern is being established

### Float16/bfloat16 handling
- Auto-cast silently: both `float16` and `bfloat16` inputs are cast to `float32` before computation
- All wrappers delegate to a shared private `_ensure_f32(tensor)` function — one place to update if policy changes
- All wrappers always return `float32` — never cast back to the input dtype
- CPU stream routing (`stream=mx.cpu`) is applied inside the wrappers; callers never pass a stream

### NNLS bridge
- Signature: `nnls_solve(A: mx.array, b: mx.array) -> tuple[mx.array, float]`
- Returns `(solution, residual)` — residual is `||Ax - b||` as a Python float
- Bridge converts to `np.float64` for the scipy call (scipy default), returns solution as `mx.float32`
- Input validation: assert `b` is 1D and shape-compatible with `A`; raise `ValueError` with a clear message on mismatch
- MLX→numpy→MLX conversion is fully internal — callers never import scipy or numpy

### Claude's Discretion
- Docstring style and verbosity
- Whether to add `__all__` to linalg_utils.py
- Exact ValueError message text

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `docs/research/kv-cache-compression/spike_kvtc.py`: established pattern for `stream=mx.cpu` SVD and float32 cast — Phase 1 formalizes these inline patterns into named helpers
- `docs/research/kv-cache-compression/spike_am.py`: `scipy.optimize.nnls` via numpy bridge already proven; `nnls_solve` wraps this pattern

### Established Patterns
- MLX linalg ops require `stream=mx.cpu` — confirmed by research spike (spike_kvtc.py:115)
- Float16 → float32 cast is done inline throughout spikes; Phase 1 centralizes this into `_ensure_f32()`
- `mx.linalg.norm` in `omlx/models/base_model.py:72` is float16-safe and should NOT be wrapped — the CI lint gate must exclude `norm`

### Integration Points
- No existing `omlx/compression/` package — Phase 1 creates it
- Phases 2, 3, 4 will import directly: `from omlx.compression.linalg_utils import svd_f32, pinv_f32, nnls_solve`

</code_context>

<specifics>
## Specific Ideas

- The CI lint gate (MATH-01 success criterion 4) should grep for bare `mx.linalg.svd` and `mx.linalg.pinv` only — not `mx.linalg.norm` — outside `omlx/compression/linalg_utils.py`. A pytest test scanning source files is the simplest approach.
- `qr_f32` follows the same pattern as `svd_f32`: `_ensure_f32` + `stream=mx.cpu` + return float32.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-linalg-foundation*
*Context gathered: 2026-03-18*
