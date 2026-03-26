---
phase: 2
slug: am-compaction
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-18
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `pytest.ini` — testpaths=tests, asyncio_mode=auto |
| **Quick run command** | `pytest tests/test_am.py -x -v -m "not slow"` |
| **Full suite command** | `pytest tests/test_am.py -v` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_am.py -x -v -m "not slow"`
- **After every plan wave:** Run `pytest tests/test_am.py -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 2-01-01 | 01 | 0 | AM-01..AM-08 | unit | `pytest tests/test_am.py -x -v` | ❌ W0 | ⬜ pending |
| 2-02-01 | 02 | 1 | AM-01 | unit | `pytest tests/test_am.py::TestHighestAttnKeysSelection -x` | ❌ W0 | ⬜ pending |
| 2-02-02 | 02 | 1 | AM-01 | unit | `pytest tests/test_am.py::TestUniformFallback -x` | ❌ W0 | ⬜ pending |
| 2-02-03 | 02 | 1 | AM-02 | unit | `pytest tests/test_am.py::TestNNLSBetaFitting -x` | ❌ W0 | ⬜ pending |
| 2-02-04 | 02 | 1 | AM-03 | unit | `pytest tests/test_am.py::TestOLSValueFitting -x` | ❌ W0 | ⬜ pending |
| 2-02-05 | 02 | 1 | AM-04 | unit | `pytest tests/test_am.py::TestCompactedCacheShape -x` | ❌ W0 | ⬜ pending |
| 2-02-06 | 02 | 1 | AM-05 | unit | `pytest tests/test_am.py::TestHeadBudgets -x` | ❌ W0 | ⬜ pending |
| 2-02-07 | 02 | 1 | AM-06 | unit | `pytest tests/test_am.py::TestBudgetReuse -x` | ❌ W0 | ⬜ pending |
| 2-02-08 | 02 | 1 | AM-07 | unit | `pytest tests/test_am.py::TestGenerateReferenceQueries -x` | ❌ W0 | ⬜ pending |
| 2-02-09 | 02 | 1 | AM-08 | unit | `pytest tests/test_am.py::TestBetaBoxConstraint -x` | ❌ W0 | ⬜ pending |
| 2-02-10 | 02 | 2 | AM-02+AM-03 | integration | `pytest tests/test_am.py::TestCompactIntegration -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_am.py` — test stubs for all AM-01 through AM-08 requirements

*`conftest.py` and `pytest.ini` already exist — no framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Cosine similarity >0.998 on Qwen 2.5 7B at 4x compaction | VAL-02 | Requires full model load (~7B weights); too slow and resource-heavy for CI | Run `python docs/research/kv-cache-compression/spike_am.py` with HighestAttnKeys path enabled; verify avg cosine sim in output JSON |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
