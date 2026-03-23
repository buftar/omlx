---
phase: 6
slug: cache-integration
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-23
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `pyproject.toml` (existing) |
| **Quick run command** | `pytest tests/test_cache_integration.py -v -m "not slow"` |
| **Full suite command** | `pytest tests/test_cache_integration.py tests/test_paged_ssd_cache.py tests/test_prefix_cache.py -v -m "not slow"` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_cache_integration.py -v -m "not slow"`
- **After every plan wave:** Run `pytest tests/test_cache_integration.py tests/test_paged_ssd_cache.py tests/test_prefix_cache.py -v -m "not slow"`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 6-01-01 | 01 | 0 | PIPE-06 | unit stub | `pytest tests/test_cache_integration.py -v -k "not slow"` | ❌ W0 | ⬜ pending |
| 6-02-01 | 02 | 1 | PIPE-06 | unit | `pytest tests/test_cache_integration.py -v -k "test_compressed_save_block"` | ❌ W0 | ⬜ pending |
| 6-02-02 | 02 | 1 | PIPE-06 | unit | `pytest tests/test_cache_integration.py -v -k "test_compressed_load_block"` | ❌ W0 | ⬜ pending |
| 6-02-03 | 02 | 1 | PIPE-07 | unit | `pytest tests/test_paged_ssd_cache.py tests/test_prefix_cache.py -v -m "not slow"` | ✅ | ⬜ pending |
| 6-03-01 | 03 | 1 | PIPE-08 | unit | `pytest tests/test_cache_integration.py -v -k "test_runtime_toggle"` | ❌ W0 | ⬜ pending |
| 6-03-02 | 03 | 1 | PIPE-09 | unit | `pytest tests/test_cache_integration.py -v -k "test_compression_config"` | ❌ W0 | ⬜ pending |
| 6-04-01 | 04 | 2 | PIPE-08 | unit | `pytest tests/test_cache_integration.py -v -k "test_admin_endpoint"` | ❌ W0 | ⬜ pending |
| 6-04-02 | 04 | 2 | PIPE-09 | unit | `pytest tests/test_cache_integration.py -v -k "test_cli_flags"` | ❌ W0 | ⬜ pending |
| 6-05-01 | 05 | 3 | PIPE-10 | slow | `pytest tests/test_cache_integration.py -v -m "slow" -k "test_decompression_latency"` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_cache_integration.py` — stubs for PIPE-06, PIPE-07, PIPE-08, PIPE-09, PIPE-10
- [ ] Fixtures: `tmp_path` (built-in), synthetic KV tensor factories

*Existing infrastructure (`tests/test_paged_ssd_cache.py`, `tests/test_prefix_cache.py`) covers PIPE-07 regression checks.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| TTFT profile shows no compression on decode thread | PIPE-06 (async constraint) | Requires profiler trace inspection | Run `omlx serve` with `--compression-bundle`, generate tokens, capture MLX trace, verify no `compress()` call in decode span |
| Runtime toggle survives concurrent requests | PIPE-08 | Race condition — timing-sensitive | Toggle compression flag via admin API while inference is in-flight; verify no crash or corrupted output |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
