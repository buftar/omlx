---
phase: 4
slug: pca-calibration-cli
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-19
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >=9.0.2 |
| **Config file** | `[tool.pytest.ini_options]` in `pyproject.toml` |
| **Quick run command** | `pytest tests/test_calibrator.py -m "not slow" -x` |
| **Full suite command** | `pytest tests/test_calibrator.py -v` |
| **Estimated runtime** | ~10 seconds (unit), ~600 seconds (with slow) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_calibrator.py -m "not slow" -x`
- **After every plan wave:** Run `pytest tests/test_calibrator.py -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds (unit tests only)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 4-01-01 | 01 | 0 | CAL-01..05 | unit | `pytest tests/test_calibrator.py -m "not slow" -x` | ❌ W0 | ⬜ pending |
| 4-02-01 | 02 | 1 | CAL-01 | unit | `pytest tests/test_calibrator.py::TestCLIDispatch -x` | ❌ W0 | ⬜ pending |
| 4-02-02 | 02 | 1 | CAL-02 | unit | `pytest tests/test_calibrator.py::TestRopeStrip -x` | ❌ W0 | ⬜ pending |
| 4-02-03 | 02 | 1 | CAL-02 | unit | `pytest tests/test_calibrator.py::TestPCABasis -x` | ❌ W0 | ⬜ pending |
| 4-02-04 | 02 | 1 | CAL-03 | unit | `pytest tests/test_calibrator.py::TestBundleSaveLoad -x` | ❌ W0 | ⬜ pending |
| 4-02-05 | 02 | 1 | CAL-04 | unit | `pytest tests/test_calibrator.py::TestHeadEntropy -x` | ❌ W0 | ⬜ pending |
| 4-03-01 | 03 | 2 | CAL-05 | slow | `pytest tests/test_calibrator.py::TestCalibrationTiming -m slow -v` | ❌ W0 | ⬜ pending |
| 4-03-02 | 03 | 2 | CAL-05 | slow | `pytest tests/test_calibrator.py::TestDeterminism -m slow -v` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_calibrator.py` — stubs for CAL-01 through CAL-05 (all test classes listed above)
- [ ] `omlx/compression/calibrator.py` — stub with `run_calibration()` raising `NotImplementedError`
- [ ] `omlx/cli.py` — add `calibrate-kv` subparser and `calibrate_kv_command()` stub in `main()`

*No new framework installs required — all deps already present including zstandard from Phase 3.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Progress bar displays during multi-prompt prefill | CAL-01 | tqdm output not captured by pytest | Run `omlx calibrate-kv <model> --n-components 8` and confirm tqdm bar appears |
| Bundle written alongside model weights | CAL-03 | Requires real model directory | Verify `kv_pca_calibration.npz` exists in `<model_dir>/` after calibration |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
