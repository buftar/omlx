---
created: 2026-03-26T02:07:10.330Z
title: Add compression admin card for KVTC functionality
area: admin
files:
  - omlx/admin/templates/dashboard/_settings.html
  - omlx/admin/templates/dashboard/_status.html
  - omlx/admin/static/js/dashboard.js
  - omlx/admin/i18n/
---

## Problem

The compression feature has complete backend API infrastructure but **zero frontend UI**.

### What Backend Exists (functional, ready to consume)
- `GET /admin/api/compression/status` — returns `enabled`, `compression_ratio`, `decompression_latency_ms`, success/failure counts, bytes
- `POST /admin/api/compression/config` — accepts `enabled` (bool), `am_ratio` (float)
- Server wires compression config/stats getters at startup (`server.py` lines 407-408)

### What Frontend Is Missing
- No section in `omlx/admin/templates/dashboard/_settings.html` for compression settings
- No compression stats card in `omlx/admin/templates/dashboard/_status.html`
- No JavaScript in `omlx/admin/static/js/dashboard.js` to call compression endpoints
- No i18n strings for compression labels (`omlx/admin/i18n/`)
- macOS menubar app (`packaging/`) has no compression settings

## Solution

Add a "Compression" card to the admin dashboard that shows:
- Enabled toggle (calls `POST /admin/api/compression/config`)
- Live compression ratio
- Avg decompression latency (ms/layer)
- Success/failure counts
- `am_ratio` slider (1.0–8.0)

This is a self-contained frontend task — all data is already available from the backend.