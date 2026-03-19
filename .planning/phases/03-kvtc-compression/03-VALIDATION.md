---
phase: 3
slug: kvtc-compression
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-19
audited: 2026-03-19
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
| 3-01-01 | 01 | 0 | KVTC-01..07 | scaffold | `pytest tests/test_kvtc.py --collect-only` | ✅ | ✅ green |
| 3-02-01 | 02 | 1 | KVTC-01 | unit | `pytest tests/test_kvtc.py::TestPCAProjection -x` | ✅ | ✅ green |
| 3-02-02 | 02 | 1 | KVTC-02 | unit | `pytest tests/test_kvtc.py::TestDPAllocation -x` | ✅ | ✅ green |
| 3-02-03 | 02 | 1 | KVTC-03 | unit | `pytest tests/test_kvtc.py::TestCompressDecompress -x` | ✅ | ✅ green |
| 3-02-04 | 02 | 1 | KVTC-04 | unit+slow | `pytest tests/test_kvtc.py::TestRoundTrip -x` | ✅ | ✅ green |
| 3-02-05 | 02 | 1 | KVTC-05 | unit | `pytest tests/test_kvtc.py::TestSinkTokenExemption -x` | ✅ | ✅ green |
| 3-02-06 | 02 | 1 | KVTC-06 | unit | `pytest tests/test_kvtc.py::TestWindowTokenExemption -x` | ✅ | ✅ green |
| 3-02-07 | 02 | 1 | KVTC-07 | unit | `pytest tests/test_kvtc.py::TestGQAShapeContract -x` | ✅ | ✅ green |
| 3-03-01 | 03 | 2 | Latency | unit | `pytest tests/test_kvtc.py::TestDecompressionLatency -x` | ✅ | ✅ green |
| 3-03-02 | 03 | 2 | Integration | slow | `pytest tests/test_kvtc.py -m slow -v` | ✅ stub | ⬜ slow (manual) |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `tests/test_kvtc.py` — 8 test classes (22 fast + 1 slow skipped), 423 lines
- [x] `omlx/compression/kvtc.py` — 656 lines, fully implemented `KVTCCompressor`
- [x] `"zstandard>=0.21.0"` added to `[project.dependencies]` in `pyproject.toml`
- [x] `markers = ["slow: marks tests as slow (deselect with '-m \"not slow\"')"]` added to `[tool.pytest.ini_options]` in `pyproject.toml`
- [x] `pip install -e ".[dev]"` run after pyproject.toml update

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Decompression latency < 10ms per layer for 8K context | KVTC-03 | Requires real Qwen 2.5 7B on M3 Max; measured in slow integration test but hardware-specific | Run `pytest tests/test_kvtc.py -m slow -v -s` on M3 Max and check printed timing |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 10s (38s full suite, ~5s single class)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** 2026-03-19 — 22/22 fast tests GREEN, 0 gaps

## Validation Audit 2026-03-19
| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |
| Manual-only | 1 (real Qwen round-trip, requires model download) |
