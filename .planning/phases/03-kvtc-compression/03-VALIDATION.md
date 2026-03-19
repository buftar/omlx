---
phase: 3
slug: kvtc-compression
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-19
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >=9.0.2 |
| **Config file** | `[tool.pytest.ini_options]` in `pyproject.toml` |
| **Quick run command** | `pytest tests/test_kvtc.py -m "not slow" -x` |
| **Full suite command** | `pytest tests/test_kvtc.py -v` |
| **Estimated runtime** | ~5 seconds (not slow) / ~60 seconds (full, model-dependent) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_kvtc.py -m "not slow" -x`
- **After every plan wave:** Run `pytest tests/test_kvtc.py -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 3-01-01 | 01 | 0 | KVTC-01..07 | scaffold | `pytest tests/test_kvtc.py --collect-only` | ❌ W0 | ⬜ pending |
| 3-02-01 | 02 | 1 | KVTC-01 | unit | `pytest tests/test_kvtc.py::TestPCAProjection -x` | ❌ W0 | ⬜ pending |
| 3-02-02 | 02 | 1 | KVTC-02 | unit | `pytest tests/test_kvtc.py::TestDPAllocation -x` | ❌ W0 | ⬜ pending |
| 3-02-03 | 02 | 1 | KVTC-03 | unit | `pytest tests/test_kvtc.py::TestCompressDecompress -x` | ❌ W0 | ⬜ pending |
| 3-02-04 | 02 | 1 | KVTC-04 | unit+slow | `pytest tests/test_kvtc.py::TestRoundTrip -x` | ❌ W0 | ⬜ pending |
| 3-02-05 | 02 | 1 | KVTC-05 | unit | `pytest tests/test_kvtc.py::TestSinkTokenExemption -x` | ❌ W0 | ⬜ pending |
| 3-02-06 | 02 | 1 | KVTC-06 | unit | `pytest tests/test_kvtc.py::TestWindowTokenExemption -x` | ❌ W0 | ⬜ pending |
| 3-02-07 | 02 | 1 | KVTC-07 | unit | `pytest tests/test_kvtc.py::TestGQAShapeContract -x` | ❌ W0 | ⬜ pending |
| 3-03-01 | 03 | 2 | Integration | slow | `pytest tests/test_kvtc.py -m slow -v` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_kvtc.py` — stubs for KVTC-01 through KVTC-07 and integration test classes
- [ ] `omlx/compression/kvtc.py` — stub with `KVTCCompressor` class raising `NotImplementedError`
- [ ] `"zstandard>=0.21.0"` added to `[project.dependencies]` in `pyproject.toml`
- [ ] `markers = ["slow: marks tests as slow (deselect with '-m \"not slow\"')"]` added to `[tool.pytest.ini_options]` in `pyproject.toml`
- [ ] `pip install -e ".[dev]"` run after pyproject.toml update

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Decompression latency < 10ms per layer for 8K context | KVTC-03 | Requires real Qwen 2.5 7B on M3 Max; measured in slow integration test but hardware-specific | Run `pytest tests/test_kvtc.py -m slow -v -s` on M3 Max and check printed timing |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
