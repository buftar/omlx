---
phase: 01-linalg-foundation
verified: 2026-03-18T12:00:00Z
status: passed
score: 5/5 must-haves verified
re_verification: false
---

# Phase 1: Linalg Foundation Verification Report

**Phase Goal:** Establish a float32-safe MLX linear algebra layer (omlx/compression/linalg_utils.py) that all downstream compression phases can import without risk of silent NaN failures or dtype/stream errors.
**Verified:** 2026-03-18T12:00:00Z
**Status:** passed
**Re-verification:** No - initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Calling svd_f32, pinv_f32, or qr_f32 with float16 or bfloat16 input succeeds - no ValueError, no silent NaN | VERIFIED | 6 tests across TestEnsureF32, TestSvdF32, TestPinvF32, TestQrF32 all pass; mx.eval() called after each op |
| 2 | All three wrappers use stream=mx.cpu - no bare mx.linalg.svd/pinv calls exist outside linalg_utils.py | VERIFIED | Lines 20, 26, 32 in linalg_utils.py each carry stream=mx.cpu; grep of omlx/ confirms zero violations; test_no_bare_linalg_calls passes |
| 3 | nnls_solve accepts mx.array inputs and returns (mx.array[float32], float) - callers never touch numpy or scipy directly | VERIFIED | Implementation bridges to _scipy_nnls internally, returns mx.array(x_np, dtype=mx.float32), float(residual); 7 tests in TestNnlsSolve pass including test_exact_solution_identity with residual < 1e-6 |
| 4 | A pytest lint gate scans omlx/ source files and fails if bare mx.linalg.svd or mx.linalg.pinv appear outside linalg_utils.py | VERIFIED | test_no_bare_linalg_calls at line 173 of test_linalg_utils.py uses pattern r'mx\.linalg\.(svd|pinv)\b'; passes in current run |
| 5 | scipy>=1.7.0 is declared in pyproject.toml [project.dependencies] - fresh installs do not fail | VERIFIED | tomllib parse confirms ['scipy>=1.7.0'] present at line 41 of pyproject.toml |

**Score:** 5/5 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| omlx/compression/__init__.py | Package root - license header only, no re-exports | VERIFIED | File is 2 lines: SPDX header + blank line. No imports, no __all__, no re-exports. |
| omlx/compression/linalg_utils.py | Float32-safe linalg wrappers and NNLS bridge; exports svd_f32, pinv_f32, qr_f32, nnls_solve | VERIFIED | 47 lines, substantive implementation. __all__ declares all 4 symbols. All 4 def statements present. stream=mx.cpu on all 3 linalg ops. scipy/numpy bridge in nnls_solve. |
| tests/test_linalg_utils.py | Full unit tests + lint gate including test_no_bare_linalg_calls | VERIFIED | 183 lines, 6 test classes + 1 module-level lint gate. test_no_bare_linalg_calls at line 173 confirmed present. |
| pyproject.toml | scipy>=1.7.0 in [project.dependencies] | VERIFIED | scipy>=1.7.0 at line 41, confirmed via tomllib parse. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| omlx/compression/linalg_utils.py | mlx.core.linalg | _ensure_f32 cast + stream=mx.cpu | VERIFIED | Lines 19-20, 25-26, 31-32 each cast via _ensure_f32 then call the MLX op with stream=mx.cpu |
| omlx/compression/linalg_utils.py | scipy.optimize.nnls | numpy bridge inside nnls_solve | VERIFIED | _scipy_nnls aliased at import; _np.array(A, dtype=_np.float64) bridge at lines 43-45; result converted back to mx.array at line 46 |
| tests/test_linalg_utils.py | omlx/compression/linalg_utils.py | direct import | VERIFIED | Line 9: from omlx.compression.linalg_utils import svd_f32, pinv_f32, qr_f32, nnls_solve - all 4 symbols imported and exercised |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| MATH-01 | 01-01-PLAN.md | Linalg safety layer wraps MLX ops with automatic float32 cast and CPU stream routing | SATISFIED | _ensure_f32 + stream=mx.cpu on all three wrappers; 13 tests covering float16/bfloat16 inputs across all three ops |
| MATH-02 | 01-01-PLAN.md | scipy NNLS wrapper for beta-fitting with numpy/MLX bridge | SATISFIED | nnls_solve bridges mx.array -> np.float64 -> scipy.optimize.nnls -> mx.float32; 7 tests pass including exact solution test |
| MATH-03 | 01-01-PLAN.md | OLS value-fitting via mx.linalg.pinv with correct stream/dtype handling | SATISFIED | pinv_f32 wraps mx.linalg.pinv with _ensure_f32 cast and stream=mx.cpu; Moore-Penrose property test passes with atol=1e-4 |

All 3 requirements for Phase 1 are SATISFIED. No orphaned requirements - REQUIREMENTS.md Traceability table maps MATH-01, MATH-02, MATH-03 exclusively to Phase 1.

---

### Anti-Patterns Found

None detected.

Scanned: omlx/compression/__init__.py, omlx/compression/linalg_utils.py, tests/test_linalg_utils.py

- No TODO/FIXME/HACK/PLACEHOLDER comments
- No stub return patterns
- No empty handler patterns
- No debug print-only implementations
- SPDX license header present in all 3 new .py files (required per CLAUDE.md contribution standards)

---

### Human Verification Required

None. All behaviors are programmatically verifiable and confirmed by the live pytest run.

---

### Test Run Summary

    pytest tests/test_linalg_utils.py -v -q
    21 passed in 0.22s

    Tests passing:
      TestEnsureF32 (3 tests)
      TestSvdF32 (4 tests)
      TestPinvF32 (3 tests)
      TestQrF32 (3 tests)
      TestNnlsSolve (7 tests)
      test_no_bare_linalg_calls (1 test)

---

### Gaps Summary

None. Phase goal is fully achieved. The float32-safe MLX linalg layer exists, is substantive, is wired correctly, passes all 21 tests, and satisfies all three declared requirements. Downstream compression phases (2, 3, 4) can safely import svd_f32, pinv_f32, qr_f32, and nnls_solve from omlx.compression.linalg_utils.

---

_Verified: 2026-03-18T12:00:00Z_
_Verifier: Claude (gsd-verifier)_
