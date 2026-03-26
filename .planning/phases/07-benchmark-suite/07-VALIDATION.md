---
phase: 7
slug: benchmark-suite
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-24
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x + pytest-asyncio |
| **Config file** | `pytest.ini` (project root) |
| **Quick run command** | `pytest tests/test_compression_benchmark.py -m "not slow" -v` |
| **Full suite command** | `pytest tests/test_compression_benchmark.py -v` |
| **Estimated runtime** | ~30 seconds (fast), ~60 minutes (full with slow) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_compression_benchmark.py -m "not slow" -x`
- **After every plan wave:** Run `pytest tests/test_compression_benchmark.py -m "not slow" -v`
- **Before `/gsd:verify-work`:** Full suite must be green (including slow)
- **Max feedback latency:** 30 seconds (fast), ~3600 seconds (full with slow)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 07-01-01 | 01 | 0 | VAL-01 | unit | `pytest tests/test_compression_benchmark.py::TestBenchmarkReport -x` | ❌ W0 | ⬜ pending |
| 07-01-02 | 01 | 0 | VAL-08 | unit | `pytest tests/test_compression_benchmark.py::TestReproducibility -x` | ❌ W0 | ⬜ pending |
| 07-01-03 | 01 | 0 | VAL-06 | unit | `pytest tests/test_compression_benchmark.py::TestSwaDetection -x` | ❌ W0 | ⬜ pending |
| 07-02-01 | 02 | 1 | VAL-01 | unit | `pytest tests/test_compression_benchmark.py::TestBenchmarkReport -x` | ❌ W0 | ⬜ pending |
| 07-02-02 | 02 | 1 | VAL-06 | unit | `pytest tests/test_compression_benchmark.py::TestSwaDetection -x` | ❌ W0 | ⬜ pending |
| 07-02-03 | 02 | 1 | VAL-08 | unit | `pytest tests/test_compression_benchmark.py::TestReproducibility -x` | ❌ W0 | ⬜ pending |
| 07-03-01 | 03 | 2 | VAL-02 | slow | `pytest tests/test_compression_benchmark.py::TestSlowQwen::test_am_cosine_sim -m slow` | ❌ W0 | ⬜ pending |
| 07-03-02 | 03 | 2 | VAL-03 | slow | `pytest tests/test_compression_benchmark.py::TestSlowQwen::test_task_accuracy -m slow` | ❌ W0 | ⬜ pending |
| 07-03-03 | 03 | 2 | VAL-04 | slow | `pytest tests/test_compression_benchmark.py::TestSlowQwen -m slow` | ❌ W0 | ⬜ pending |
| 07-04-01 | 04 | 2 | VAL-05 | slow | `pytest tests/test_compression_benchmark.py::TestSlowLlama -m slow` | ❌ W0 | ⬜ pending |
| 07-04-02 | 04 | 2 | VAL-06 | slow | `pytest tests/test_compression_benchmark.py::TestSwaDetection -m slow` | ❌ W0 | ⬜ pending |
| 07-04-03 | 04 | 2 | VAL-07 | slow | `pytest tests/test_compression_benchmark.py::TestSlowDeepSeek -m slow` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_compression_benchmark.py` — stubs for all VAL-01 through VAL-08
- [ ] `omlx/compression/benchmark.py` — BenchmarkRunner stub (report fields, seed, CLI entry)
- [ ] `omlx/compression/evaluators.py` — GSM8K, MMLU, LITM evaluator stubs

*Framework install: already present — pytest in dev deps, no new requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `omlx benchmark-compression` CLI produces human-readable report | VAL-01 | Requires running CLI end-to-end with a live model | Run `omlx benchmark-compression --model <model_path> --seed 42`, inspect output |
| Benchmark results match between two separate CLI invocations | VAL-08 | Requires two full CLI runs with same seed | Run twice with `--seed 42`, diff the JSON output |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s (fast path)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
