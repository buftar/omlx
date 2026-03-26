---
phase: 5
slug: pipeline-assembly
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-22
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (configured in pytest.ini) |
| **Config file** | `pytest.ini` |
| **Quick run command** | `uv run python -m pytest tests/test_pipeline.py -m "not slow" -q` |
| **Full suite command** | `uv run python -m pytest tests/test_pipeline.py -v` |
| **Estimated runtime** | ~5 seconds (fast); ~120 seconds (slow, model load) |

---

## Sampling Rate

- **After every task commit:** Run `uv run python -m pytest tests/test_pipeline.py -m "not slow" -q`
- **After every plan wave:** Run `uv run python -m pytest tests/test_pipeline.py -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds (fast path)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 5-01-01 | 01 | 0 | PIPE-01..05 | scaffold | `uv run python -m pytest tests/test_pipeline.py -m "not slow" -q` | ❌ W0 | ⬜ pending |
| 5-02-01 | 02 | 1 | PIPE-01 | unit | `uv run python -m pytest tests/test_pipeline.py::TestCompress -x -q` | ❌ W0 | ⬜ pending |
| 5-02-02 | 02 | 1 | PIPE-01 | unit | `uv run python -m pytest tests/test_pipeline.py::TestCompress::test_multiplicative_ratio -x` | ❌ W0 | ⬜ pending |
| 5-02-03 | 02 | 1 | PIPE-02 | unit | `uv run python -m pytest tests/test_pipeline.py::TestRoundTrip -x -q` | ❌ W0 | ⬜ pending |
| 5-02-04 | 02 | 1 | PIPE-02 | unit | `uv run python -m pytest tests/test_pipeline.py::TestRoundTrip::test_logical_seq_len_preserved -x` | ❌ W0 | ⬜ pending |
| 5-02-05 | 02 | 1 | PIPE-03 | unit (mock) | `uv run python -m pytest tests/test_pipeline.py::TestTriggerSemantics::test_compact_fires_above_threshold -x` | ❌ W0 | ⬜ pending |
| 5-02-06 | 02 | 1 | PIPE-04 | unit | `uv run python -m pytest tests/test_pipeline.py::TestTriggerSemantics::test_compress_callable -x` | ❌ W0 | ⬜ pending |
| 5-02-07 | 02 | 1 | PIPE-05 | unit | `uv run python -m pytest tests/test_pipeline.py::TestTriggerSemantics::test_decompress_restores_layers -x` | ❌ W0 | ⬜ pending |
| 5-03-01 | 03 | 2 | PIPE-02 | slow | `uv run python -m pytest tests/test_pipeline.py -m slow -v` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `omlx/compression/pipeline.py` — `KVCachePipeline` class stub + `PipelineBlob` dataclass (no implementation yet)
- [ ] `tests/test_pipeline.py` — test scaffold with all test class/method stubs for PIPE-01 through PIPE-05

*Existing pytest infrastructure covers all phase requirements. No new framework installs needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Inference continues after decompressed cache restore | PIPE-02 (slow) | Requires Qwen 2.5 7B model loaded and generating tokens | Load model, run forward pass with decompressed KV cache, verify next-token prediction does not error and perplexity is within bounds |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
