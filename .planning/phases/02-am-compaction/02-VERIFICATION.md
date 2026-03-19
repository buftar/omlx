---
phase: 02-am-compaction
verified: 2026-03-19T03:30:00Z
status: human_needed
score: 9/10 must-haves verified
human_verification:
  - test: "Load Qwen 2.5 7B and run AMCompactor.compact() with ratio=4.0 on a real KV cache using generate_reference_queries"
    expected: "Average attention output cosine similarity between compacted and original cache is >0.998"
    why_human: "Requires full 7B model load — too slow for CI. Automated test only validates shape contract and synthetic >0.98 threshold."
---

# Phase 2: AM Compaction Verification Report

**Phase Goal:** Implement AMCompactor — a production-quality attention-map compaction algorithm that selects high-attention tokens using HighestAttnKeys strategy, fits NNLS beta weights and OLS value weights, and supports entropy-proportional non-uniform head budgets.
**Verified:** 2026-03-19T03:30:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | AMCompactedCache dataclass has layers, logical_seq_len, diagnostics fields with correct types | VERIFIED | am.py lines 32-47: @dataclass with exactly these three fields, correct types |
| 2 | compact() with queries provided selects tokens by descending summed attention weight (HighestAttnKeys) | VERIFIED | _highest_attn_select() at line 257: importance = attn_full[0,0].sum(axis=0); argsort picks highest |
| 3 | First n_sink_tokens positions are always included in selected set regardless of attention weight | VERIFIED | _highest_attn_select() line 282: sink_idx = mx.array(list(range(n_sinks))); always concatenated into result |
| 4 | compact() falls back to uniform interval selection when queries=None | VERIFIED | _compact_head() line 210-215: if queries_h is None: calls _uniform_select |
| 5 | NNLS beta-fitting uses nnls_solve from linalg_utils; betas clipped to [-3,3] after solve | VERIFIED | am.py line 24 import confirmed; line 244-245: nnls_solve called, mx.clip(-3.0, 3.0) applied |
| 6 | OLS value-fitting uses pinv_f32 from linalg_utils; V_compact shape is [1,1,budget,head_dim] | VERIFIED | am.py line 251-252: A_pinv = pinv_f32(A_s); V_compact = V_compact[None,None] gives [1,1,budget,head_dim] |
| 7 | AMCompactedCache.logical_seq_len equals the original input sequence length T | VERIFIED | compact() line 92: seq_len = first_keys.shape[2]; returned as logical_seq_len at line 138 |
| 8 | Non-uniform head budgets use entropy proportions stored at __init__; compact() does not recompute entropy | VERIFIED | __init__ stores self._head_entropy (line 61); _compute_head_budgets uses self._head_entropy (line 168-171) |
| 9 | generate_reference_queries returns shape [1, n_heads, n_queries, head_dim] for both sample and random methods | VERIFIED | am.py lines 327-363: both paths produce correct shapes; TestGenerateReferenceQueries confirms |
| 10 | Attention output cosine similarity >0.998 on Qwen 2.5 7B at 4x compaction | ? UNCERTAIN | Automated test only verifies shape contract and synthetic >0.98 threshold. Real model validation deferred (requires full model load). |

**Score:** 9/10 truths verified (1 requires human verification)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `omlx/compression/am.py` | AMCompactor, AMCompactedCache, generate_reference_queries | VERIFIED | 363 lines (min_lines: 200 — passed); SPDX header present; all 3 exports confirmed |
| `tests/test_am.py` | All 10 test classes for AM-01..AM-08 | VERIFIED | 443 lines; all 10 test classes present; SPDX header present |
| `omlx/compression/__init__.py` | Package init | VERIFIED | Exists (38 bytes) |
| `omlx/compression/linalg_utils.py` | Dependency from Phase 1 | VERIFIED | Present, pinv_f32 and nnls_solve importable |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| omlx/compression/am.py | omlx/compression/linalg_utils.py | from omlx.compression.linalg_utils import pinv_f32, nnls_solve | WIRED | Line 24 of am.py; both symbols used at lines 244, 251 |
| tests/test_am.py | omlx/compression/am.py | from omlx.compression.am import AMCompactor, AMCompactedCache, generate_reference_queries | WIRED | Line 17 of test_am.py; all 3 symbols exercised in test methods |
| AMCompactor.__init__ | self._head_entropy | stored at init; entropy proportions used in _compute_head_budgets | WIRED | Line 61: self._head_entropy = head_entropy; line 168: if self._head_entropy is None |
| AMCompactor.compact | AMCompactor._compute_head_budgets | called once per compact() invocation using stored entropy proportions | WIRED | Line 97: head_budgets = self._compute_head_budgets(seq_len, ratio, n_heads) |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| AM-01 | 02-01, 02-02 | HighestAttnKeys selection with sink token preservation | SATISFIED | _highest_attn_select() + _uniform_select(); TestHighestAttnKeysSelection, TestUniformFallback |
| AM-02 | 02-01, 02-02 | NNLS beta fitting to preserve attention mass | SATISFIED (partial behavioral coverage) | nnls_solve called at line 244; betas clipped at 245. NOTE: TestNNLSBetaFitting assertions guarded by `diagnostics is not None` — always skipped in current implementation |
| AM-03 | 02-01, 02-02 | OLS value fitting via pinv to preserve attention outputs | SATISFIED | pinv_f32 at line 251; V_compact shape verified by TestOLSValueFitting::test_v_compact_shape and TestCompactIntegration |
| AM-04 | 02-01, 02-02 | Compacted cache retains logical_seq_len; physical size reduced | SATISFIED | logical_seq_len preserved at line 138; TestCompactedCacheShape passes |
| AM-05 | 02-01, 02-03 | Non-uniform head budgets precomputed per entropy | SATISFIED | _compute_head_budgets() implemented; TestHeadBudgets direct unit tests pass. NOTE: test_entropy_budgets_proportional and test_entropy_budgets_min_sinks are diagnostics-gated and trivially pass |
| AM-06 | 02-01, 02-03 | Budget schedule stored and reused across compactions | SATISFIED | self._head_entropy stored at __init__; _compute_head_budgets is deterministic given stored entropy. TestBudgetReuse verifies shape consistency across calls |
| AM-07 | 02-01, 02-03 | Reference queries generated via repeat-prefill strategy | SATISFIED | generate_reference_queries() with "sample" method; TestGenerateReferenceQueries passes |
| AM-08 | 02-01, 02-02 | Beta values box-constrained to [-3, 3] | SATISFIED (partial behavioral coverage) | mx.clip(betas, -3.0, 3.0) at line 245. NOTE: TestBetaBoxConstraint assertion guarded by `diagnostics is not None` — always skipped |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| tests/test_am.py | 130, 144 | TestNNLSBetaFitting assertions guarded by `if result.diagnostics is not None` | Warning | AM-02 behavioral test never executes — diagnostics is always None in current implementation |
| tests/test_am.py | 380 | TestBetaBoxConstraint assertion guarded by `if result.diagnostics is not None` | Warning | AM-08 beta clip behavioral test never executes |
| tests/test_am.py | 296, 310 | TestHeadBudgets::test_entropy_budgets_proportional and test_entropy_budgets_min_sinks guarded by `if result.diagnostics is not None` | Warning | These tests pass trivially; entropy budget behavior is only verified by the 4 direct _compute_head_budgets unit tests added in Plan 03 |

**Severity note:** These are Warnings, not Blockers. The behaviors *are* implemented in am.py; the tests simply cannot assert on them without accessing betas through diagnostics. The 4 direct `_compute_head_budgets` unit tests added in Plan 03 provide meaningful coverage for AM-05/AM-06. The NNLS/clip behavior (AM-02/AM-08) is verified via code inspection only.

---

### Lint Gate

| Check | Status | Details |
|-------|--------|---------|
| No bare mx.linalg.pinv in am.py | PASSED | Grep confirms 0 matches for `mx\.linalg\.pinv` in am.py |
| No bare mx.linalg.svd in am.py | PASSED | Grep confirms 0 matches for `mx\.linalg\.svd` in am.py |
| SPDX header in am.py | PASSED | Line 1: `# SPDX-License-Identifier: Apache-2.0` |
| SPDX header in tests/test_am.py | PASSED | Line 1: `# SPDX-License-Identifier: Apache-2.0` |

---

### Commit Verification

| Commit | Message | Status |
|--------|---------|--------|
| bb28bc4 | test(02-03): add failing tests for _compute_head_budgets method | EXISTS |
| 2ca0ea2 | feat(02-03): implement _compute_head_budgets and entropy-proportional budgets | EXISTS |
| cf801b7 | feat(02-03): update generate_reference_queries docstring per plan spec | EXISTS |

---

### Human Verification Required

#### 1. Cosine Similarity Quality Gate (ROADMAP Success Criterion #2)

**Test:** Load Qwen 2.5 7B weights. Build a representative KV cache by running a ~500-token prompt. Call `generate_reference_queries(keys, n_queries=64, method="sample")` to get reference queries. Run `AMCompactor().compact(kv_cache, ratio=4.0, queries=queries)`. Compute cosine similarity between the full-cache attention output and the compacted-cache attention output for all heads and layers.
**Expected:** Average cosine similarity > 0.998
**Why human:** Requires loading full 7B model weights (~14GB). Too slow for CI (estimated 5-15 minutes). The spike script at `docs/research/kv-cache-compression/spike_am.py` implements this test.

#### 2. OMP Key Pruning Clause (ROADMAP Success Criterion #5 partial)

**Test:** Not applicable for Phase 2. ROADMAP SC#5 includes "keys with beta < -7 are pruned when using OMP path." OMP key selection is deferred to v2 as AM-ADV-01 per REQUIREMENTS.md. Only the box-constraint clause ([-3, 3]) is in scope for Phase 2, and this is implemented at am.py line 245.
**Expected:** OMP pruning not required for Phase 2 gate.
**Why human:** Documented here for traceability. The ROADMAP SC#5 partially exceeds Phase 2 scope per the v2/v1 requirement split.

---

### Gaps Summary

No blocking gaps. All 9 automatically verifiable truths pass. The three "Warning" anti-patterns (diagnostics-gated assertions) represent incomplete test coverage rather than missing implementation — the behaviors ARE present in the code. The 4 direct `_compute_head_budgets` unit tests added in Plan 03 provide meaningful compensating coverage.

The only unresolved item is the human verification for >0.998 cosine similarity on a real model, which is documented in VALIDATION.md as a manual-only verification.

---

*Verified: 2026-03-19T03:30:00Z*
*Verifier: Claude (gsd-verifier)*
