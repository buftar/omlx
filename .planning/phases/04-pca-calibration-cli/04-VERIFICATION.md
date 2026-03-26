---
phase: 04-pca-calibration-cli
verified: 2026-03-22T00:00:00Z
status: gaps_found
score: 11/12 must-haves verified
gaps:
  - truth: "TestCalibrationTiming slow tests are valid CAL-05 integration tests"
    status: failed
    reason: "TestCalibrationTiming.test_full_calibration_timing and test_determinism assert pytest.raises(NotImplementedError, run_calibration, ...) but run_calibration is fully implemented and no longer raises NotImplementedError — these slow tests will fail when run with pytest -m slow"
    artifacts:
      - path: "tests/test_calibrator.py"
        issue: "Lines 303-314: TestCalibrationTiming uses stale Wave 0 stub contract (pytest.raises NotImplementedError) that contradicts the Plan 03 implementation"
    missing:
      - "Replace pytest.raises(NotImplementedError, ...) in both TestCalibrationTiming test methods with real run_calibration() integration calls that assert timing < 600s and determinism"
      - "Or mark the tests as xfail with a clear reason explaining they are deferred pending real model on disk"
human_verification:
  - test: "Full end-to-end calibration smoke test"
    expected: "uv run python -c \"from omlx.compression.calibrator import run_calibration; run_calibration('mlx-community/Qwen2.5-7B-Instruct-4bit', n_components=8, output_path='/tmp/cal_test')\" writes /tmp/cal_test/kv_pca_calibration.npz with all 10 keys and completes in under 10 minutes"
    why_human: "Requires a real model on disk; cannot verify pipeline timing or output bundle correctness programmatically without a live model"
---

# Phase 4: PCA Calibration CLI Verification Report

**Phase Goal:** Implement a PCA calibration CLI that loads a model, runs multi-prompt KV prefill, strips RoPE from keys, fits per-head-group PCA bases, and saves a .kvtc-cal calibration bundle. Expose via `omlx calibrate-kv` subcommand.
**Verified:** 2026-03-22
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

All truths are derived from the must_haves across Plans 01, 02, and 03.

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | All six test classes exist and import cleanly | VERIFIED | tests/test_calibrator.py lines 34-314, all 6 classes present, `python -m pytest` collected 14 items |
| 2 | calibrator.py exports all 7 public functions | VERIFIED | Lines 79, 127, 170, 180, 200, 211, 252 — all functions implemented, none raise NotImplementedError |
| 3 | cli.py has calibrate-kv subparser and calibrate_kv_command() | VERIFIED | cli.py lines 348-357 (handler), 552-594 (subparser + dispatch) |
| 4 | strip_rope_from_keys produces per-token cosine similarity std < 1e-4 | VERIFIED | test_rope_flatness_nontrad and test_rope_flatness_traditional PASSED |
| 5 | compute_pca_basis with seed=42 is deterministic | VERIFIED | test_deterministic PASSED; uses svd_f32, not bare mx.linalg.svd |
| 6 | save/load_calibration_bundle round-trip preserves all 10 npz keys | VERIFIED | test_round_trip PASSED; pca_bundle[0] has all 8 KVTCCompressor keys |
| 7 | load_calibration_bundle returns pca_bundle list[dict] and head_entropy list[float] | VERIFIED | test_round_trip, test_head_entropy_shape PASSED |
| 8 | assign_layer_groups(28, 7) produces 7 groups summing to 28 | VERIFIED | test_assign_layer_groups PASSED |
| 9 | calibrate_kv_command dispatches to run_calibration with correct kwargs | VERIFIED | test_dispatch_calls_run_calibration PASSED (monkeypatch confirms all 5 kwargs forwarded correctly) |
| 10 | omlx calibrate-kv --help shows model arg + 4 optional flags | VERIFIED | test_cli_help_registered PASSED; confirmed --n-components, --bits-per-token in stdout |
| 11 | run_calibration() implements full 7-step pipeline | VERIFIED | calibrator.py lines 252-437: model load, cache extraction, RoPE strip, SVD, DP allocation, head entropy, bundle save — all wired |
| 12 | TestCalibrationTiming slow tests are valid CAL-05 integration tests | FAILED | Lines 303-314: stale pytest.raises(NotImplementedError) assertions contradict implemented run_calibration; if run with -m slow, both tests will fail with AssertionError (run_calibration raises RuntimeError from mlx_lm check, not NotImplementedError) |

**Score:** 11/12 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `omlx/compression/calibrator.py` | All 7 functions implemented | VERIFIED | 438 lines; SPDX header present; all 7 functions substantive; run_calibration() is 185-line full pipeline |
| `tests/test_calibrator.py` | 6 test classes, real assertions | VERIFIED (partial) | 315 lines; 6 classes present; 12 unit tests pass; TestCalibrationTiming has stale stub assertions |
| `omlx/cli.py` | calibrate-kv subparser + dispatch | VERIFIED | Lines 348-357 + 552-594; subparser with all 5 args; elif dispatch branch; lazy import inside handler |
| `omlx/compression/linalg_utils.py` | svd_f32 exists as dependency | VERIFIED | Line 17: `def svd_f32(a: mx.array, compute_uv: bool = True)` — wraps mx.linalg.svd with float32 cast and cpu stream |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `omlx/cli.py` | `omlx/compression/calibrator.py` | lazy import inside calibrate_kv_command() | WIRED | Line 350: `from omlx.compression.calibrator import run_calibration` inside function body |
| `tests/test_calibrator.py` | `omlx/compression/calibrator.py` | module import | WIRED | Lines 19-27: full import of all 7 public symbols |
| `calibrator.py:compute_pca_basis` | `omlx/compression/linalg_utils.svd_f32` | `from omlx.compression.linalg_utils import svd_f32` | WIRED | Line 19 import; line 156 call: `U, S, Vt = svd_f32(data_mlx)` — no bare mx.linalg.svd calls |
| `calibrator.py:load_calibration_bundle` | `KVTCCompressor` pca_bundle contract | pca_bundle list[dict] with exact 8 keys | WIRED | Lines 238-248: K_basis, K_mean, K_sv, V_basis, V_mean, V_sv, k_bit_alloc, v_bit_alloc all set |
| `calibrator.py:load_calibration_bundle` | `AMCompactor` head_entropy | `data["head_entropy"].tolist()` | WIRED | Line 249: returns list[float]; test_nonuniform_budgets confirms AMCompactor(head_entropy=loaded) produces non-uniform budgets |
| `run_calibration()` | `mlx_lm.load()` | `from mlx_lm import load as _mlx_lm_load` | WIRED | Lines 26-30: try/except import with RuntimeError guard; line 276: `_mlx_lm_load(model_path)` |
| `run_calibration()` | `make_prompt_cache` | `from mlx_lm.models.cache import make_prompt_cache` | WIRED | Line 27 import; line 343: `cache = _make_prompt_cache(model)` |
| `run_calibration()` | `strip_rope_from_keys` | called per layer | WIRED | Lines 360-363: `strip_rope_from_keys(k_np, rope_theta, rope_traditional, offset=0)` |
| `run_calibration()` | `compute_pca_basis` | called per group | WIRED | Lines 399-400: called for both K and V vectors |
| `run_calibration()` | `save_calibration_bundle` | final bundle write | WIRED | Line 433: `save_calibration_bundle(out_path, bundle_arrays)` |
| `run_calibration()` | `_dp_allocate_bits` | DP bit allocation | WIRED | Lines 31-38: try/except import with fallback; lines 402-403: called with 4 args (matches actual kvtc.py signature) |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| CAL-01 | 04-01, 04-03 | User can run `omlx calibrate-kv <model>` for any supported model | SATISFIED | CLI subparser registered; calibrate_kv_command wired; test_dispatch_calls_run_calibration + test_cli_help_registered PASSED |
| CAL-02 | 04-01, 04-02 | Calibration uses SVD on representative dataset (~200K tokens) | SATISFIED | strip_rope_from_keys + compute_pca_basis implemented; 25 CALIBRATION_PROMPTS built-in; N_SVD_TOKENS=4000 cap; svd_f32 used exclusively |
| CAL-03 | 04-01, 04-02 | PCA basis V^T, mean µ, and DP bit allocation stored alongside model weights | SATISFIED | 10-key npz bundle (K_V, V_V, K_mu, V_mu, K_sv, V_sv, K_bit_alloc, V_bit_alloc, group_sizes, head_entropy); save/load round-trip verified |
| CAL-04 | 04-01, 04-02 | Head entropy sensitivity computed and stored for AM non-uniform budgets | SATISFIED | run_calibration() computes head entropy via key-norm variance proxy; load_calibration_bundle returns list[float]; AMCompactor integration test passes |
| CAL-05 | 04-01, 04-03 | Calibration completes in under 10 minutes for models up to 12B on Apple Silicon | PARTIAL | Implementation exists and is complete; BUT TestCalibrationTiming slow tests assert NotImplementedError (stale) — actual timing unverifiable without real model run |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `tests/test_calibrator.py` | 303-307 | `pytest.raises(NotImplementedError, run_calibration, ...)` in non-stub test | BLOCKER | TestCalibrationTiming.test_full_calibration_timing will FAIL when run with -m slow — run_calibration raises RuntimeError (mlx_lm guard), not NotImplementedError |
| `tests/test_calibrator.py` | 310-314 | Same stale stub in test_determinism | BLOCKER | Same failure mode — stale Wave 0 contract left in place after Plan 03 implementation landed |

No placeholder returns, no empty implementations, no bare mx.linalg.svd calls found. SPDX headers present on all new files.

### Human Verification Required

#### 1. Full End-to-End Calibration Smoke Test

**Test:** Run `uv run python -c "from omlx.compression.calibrator import run_calibration; run_calibration('mlx-community/Qwen2.5-7B-Instruct-4bit', n_components=8, output_path='/tmp/cal_test')"` with the model on disk.
**Expected:** Progress bars appear for 25 prompts (prefill) and 8 PCA groups; `/tmp/cal_test/kv_pca_calibration.npz` is written; print output shows bundle group count and head entropy range; total wall time under 10 minutes on M3 Max.
**Why human:** Requires real model on disk; pipeline timing and npz output quality cannot be verified without live model weights.

### Gaps Summary

One gap blocks full goal achievement:

**TestCalibrationTiming stale stub contract (CAL-05):** Plan 03 explicitly deferred updating these slow tests ("Leave TestCalibrationTiming stubs as-is"). The result is that both slow test methods assert `pytest.raises(NotImplementedError, run_calibration, ...)` but `run_calibration` now raises `RuntimeError` when `mlx_lm` is unavailable (not `NotImplementedError`), or completes successfully when it is available. Either way, `pytest.raises(NotImplementedError)` will fail. The SUMMARY marked CAL-05 complete, but the test contract for it is broken.

The fix is minimal: replace the two `pytest.raises(NotImplementedError, ...)` calls with either real integration assertions (actual timing test) or `pytest.mark.xfail` / `pytest.skip` with a clear reason documenting that a real model is required.

All other phase objectives are fully achieved:
- `omlx calibrate-kv` subcommand is registered and correctly dispatches
- All 7 calibrator functions are substantive (no stubs remain in the fast path)
- RoPE stripping, deterministic PCA, Procrustes alignment, and npz I/O are all verified GREEN
- run_calibration() wires all primitives into a complete 7-step pipeline
- KVTCCompressor and AMCompactor interface contracts are satisfied

---

_Verified: 2026-03-22_
_Verifier: Claude (gsd-verifier)_
