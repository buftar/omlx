---
phase: 1
slug: linalg-foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-18
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `pyproject.toml` — `[tool.pytest.ini_options]` (exists) |
| **Quick run command** | `pytest tests/test_linalg_utils.py -x` |
| **Full suite command** | `pytest tests/test_linalg_utils.py -v` |
| **Estimated runtime** | ~2 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_linalg_utils.py -x`
- **After every plan wave:** Run `pytest tests/test_linalg_utils.py -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 0 | MATH-01, MATH-02, MATH-03 | unit + lint | `pytest tests/test_linalg_utils.py -x` | ❌ W0 | ⬜ pending |
| 1-01-02 | 01 | 1 | MATH-01 | unit | `pytest tests/test_linalg_utils.py::TestSvdF32 tests/test_linalg_utils.py::TestPinvF32 tests/test_linalg_utils.py::TestQrF32 -x` | ❌ W0 | ⬜ pending |
| 1-01-03 | 01 | 1 | MATH-02 | unit | `pytest tests/test_linalg_utils.py::TestNnlsSolve -x` | ❌ W0 | ⬜ pending |
| 1-01-04 | 01 | 1 | MATH-03 | unit | `pytest tests/test_linalg_utils.py::TestPinvF32 -x` | ❌ W0 | ⬜ pending |
| 1-01-05 | 01 | 2 | MATH-01 | lint | `pytest tests/test_linalg_utils.py::test_no_bare_linalg_calls -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_linalg_utils.py` — stubs for MATH-01, MATH-02, MATH-03 (file does not exist yet)

*Existing infrastructure (`pyproject.toml` pytest config) covers the framework — only the test file is missing.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
