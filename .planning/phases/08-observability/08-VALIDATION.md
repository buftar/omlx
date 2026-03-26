---
phase: 8
slug: observability
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-24
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `pyproject.toml` (existing) |
| **Quick run command** | `pytest tests/test_observability.py -v -m "not slow"` |
| **Full suite command** | `pytest tests/test_observability.py tests/test_cache_integration.py -v -m "not slow"` |
| **Estimated runtime** | ~20 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_observability.py -v -m "not slow"`
- **After every plan wave:** Run `pytest tests/test_observability.py -v -m "not slow"`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 20 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 8-01-01 | 01 | 0 | OBS-01 | unit stub | `pytest tests/test_observability.py -v -k "not slow"` | ❌ W0 | ⬜ pending |
| 8-02-01 | 02 | 1 | OBS-01 | unit | `pytest tests/test_observability.py -v -k "test_compression_ratio_metric"` | ❌ W0 | ⬜ pending |
| 8-02-02 | 02 | 1 | OBS-02 | unit | `pytest tests/test_observability.py -v -k "test_decompression_latency_metric"` | ❌ W0 | ⬜ pending |
| 8-03-01 | 03 | 2 | OBS-03 | unit | `pytest tests/test_observability.py -v -k "test_cache_hit_miss_metrics"` | ❌ W0 | ⬜ pending |
| 8-03-02 | 03 | 2 | OBS-03 | integration | `pytest tests/test_observability.py -v -k "test_admin_stats_endpoint"` | ❌ W0 | ⬜ pending |
| 8-04-01 | 04 | 3 | OBS-04 | integration | `pytest tests/test_observability.py -v -k "test_metrics_visible_in_admin_ui"` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_observability.py` — stubs for OBS-01, OBS-02, OBS-03, OBS-04
- [ ] Fixtures: `tmp_path` (built-in), synthetic KV tensor factories, mock admin client

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Metrics appear in Prometheus scrape | OBS-01 | Requires actual metrics scraper | Start omlx server with compression enabled, configure Prometheus to scrape, verify `omlx_compression_*` metrics appear |
| Admin UI shows compression stats | OBS-03, OBS-04 | Requires browser-based UI inspection | Open admin dashboard, verify compression ratio, decompression latency, and cache hit/miss rates are displayed |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 20s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
