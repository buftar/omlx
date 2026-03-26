---
phase: 8
slug: observability
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-24
---

# Phase 8 — Research: Observability Architecture

> Investigate existing omlx metrics infrastructure and admin UI to determine how to integrate compression quality metrics.

---

## Research Questions

1. **What metrics infrastructure does omlx already have?**
   - Does omlx expose Prometheus-style metrics?
   - Are there existing admin endpoints for configuration/status?
   - What metrics are currently tracked (token counts, cache hits/misses, latency)?

2. **Where should compression metrics be emitted?**
   - Per-request (per-generation) metrics vs aggregate statistics?
   - Should metrics be emitted on every compression/decompression or only on cache events (save/load)?
   - How to track both AM compaction and kvtc compression separately?

3. **How should metrics be exposed?**
   - Prometheus endpoint (`/metrics`)?
   - Admin API endpoints (`/api/stats`, `/api/compression/status`)?
   - Both? Which is primary?

4. **What cache metrics exist that need to be extended?**
   - Current: cache hits, cache misses, eviction count
   - Needed: compression-enabled cache hits/misses, decompression latency per layer

5. **What admin UI exists?**
   - Is there an existing admin dashboard?
   - What data does it display?
   - How are new metrics integrated into the UI?

---

## Investigation Plan

### Step 1: Scan omlx source for existing metrics patterns

```bash
# Search for Prometheus-style metrics
rg "prometheus|PROMETHEUS|metrics" omlx/ --type py

# Search for admin endpoints
rg "admin|Admin" omlx/ --type py

# Search for metrics-related imports
rg "from prometheus|import prometheus" omlx/ --type py
```

### Step 2: Examine server.py for current endpoint structure

- What FastAPI routes exist?
- How are admin endpoints protected?
- Are there any stats/endpoints already?

### Step 3: Examine cache managers for existing metrics

- `PagedSSDCacheManager` - what metrics does it track?
- `PrefixCacheManager` - any compression-related instrumentation?

### Step 4: Review omlx configuration system

- How is config passed to cache managers?
- Can compression metrics be enabled/disabled independently?

**Findings:**
- `CompressionConfig` is created in `omlx/cli.py` when `--compression-bundle` is provided
- Config is attached to `scheduler_config.compression_config` and passed to `CacheFactory.create_paged_ssd_cache()`
- Factory checks `compression_config.enabled` and returns `CompressedPagedSSDCacheManager` when enabled
- Admin API exposes runtime toggle via `POST /admin/api/compression/config` (Phase 6 complete)
- Getter pattern: `_get_compression_config` accessor wired through `set_admin_getters()` in server.py

---

## Research Summary

### What Metrics Infrastructure Exists?

**No Prometheus-style metrics infrastructure exists.** omlx has:
- `omlx/server_metrics.py` - `ServerMetrics` singleton with thread-safe aggregation
- Session metrics (resets on restart) and all-time metrics (persisted via JSON)
- Basic cache statistics: hits, misses, evictions, saves, loads
- Speed metrics: avg_prefill_tps, avg_generation_tps

### Where Should Compression Metrics Be Emitted?

**Two-tier approach:**
1. **Per-request metrics**: Track compression ratio on every `compress()` call in `KVCachePipeline`
2. **Aggregate statistics**: Track totals via `ServerMetrics` singleton
3. **Emit on cache events**: save_block (compression) and load_block (decompression)

### How Should Metrics Be Exposed?

**Primary: Admin API endpoints**
- Extend `/admin/api/stats` to include compression metrics
- Add `/admin/api/compression/status` for detailed status
- Use existing getter pattern (`compression_config_getter`)

**Future: Prometheus endpoint**
- Add `/metrics` route if needed for external scraping

### What Cache Metrics Exist That Need Extension?

**Current stats classes** (`omlx/cache/stats.py`):
- `BaseCacheStats`: hits, misses, evictions
- `PrefixCacheStats`: extends with tokens_saved, partial_block_skips
- `PagedSSDCacheStats`: extends with saves, loads, errors, hot_cache stats

**New compression-aware stats needed:**
- `CompressedCacheStats`: extends PagedSSDCacheStats with:
  - compression_ratio (AM compaction ratio from PipelineBlob)
  - decompression_latency_ms (per-layer)
  - compression_success_count, compression_failure_count
  - total_compressed_bytes, total_logical_bytes

### What Admin UI Exists?

**Current dashboard** (`omlx/admin/templates/dashboard/_status.html`):
- Uses Alpine.js with `x-text` bindings to display stats from `stats` object
- Three stat cards: Total Tokens, Cached Tokens, Cache Efficiency
- Speed stats: avg_prefill_tps, avg_generation_tps

**Integration pattern:**
- Add compression stats to existing stats object
- New stat cards for compression ratio, decompression latency
- Toggle visibility based on `compression_config.enabled`

---

## Expected Outputs

1. **Metrics Design Document**: What metrics to track, their names, labels, and aggregation strategy
2. **Implementation Plan**: Which files need modification, in what order
3. **Admin UI Mockup** (optional): Sketch of where metrics appear in admin dashboard

---

## References

- omlx server.py and admin routes
- Existing cache manager implementations
- Any existing metrics or monitoring documentation
