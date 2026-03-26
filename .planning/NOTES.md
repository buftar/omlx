# Project Notes — Strategic Context for Future Phases

## TurboQuant Integration Strategy

TurboQuant does not exist in upstream omlx yet (current upstream: v0.2.18). If/when it ships,
KVTC and TurboQuant should be positioned as **complementary tiers**, not competing features.

### Dispatch Model
- `bundle_path=None` (no calibration) → TurboQuant path (data-oblivious, instant, ~3.5-bit zero-loss)
- `bundle_path=<path>` (post-calibration) → KVTC path (PCA-based, model-specific, DP bit allocation)

This dispatch is already architecturally supported by `CompressionConfig.bundle_path: Optional[str]`.

### PR Framing
When submitting the KVTC feature branch as a PR to jundot/omlx, frame it as:
> "Adds a two-stage KV cache compression pipeline (AM token compaction + PCA quantization)
> that activates when `--compression-bundle` is provided. Designed as the calibrated tier
> above any future data-oblivious quantizer (e.g. TurboQuant)."

This framing makes acceptance easier — KVTC is not competing with TurboQuant, it extends it.

### Key Differences

| Aspect | TurboQuant (future upstream) | KVTC (our feature branch) |
|--------|------------------------------|---------------------------|
| Setup | None — data-oblivious | `omlx calibrate-kv <model>` (~10 min) |
| Bit allocation | Uniform per coordinate | DP-allocated, model-specific |
| Decorrelation | Random rotation | Cross-layer PCA |
| Inner product bias | Explicitly corrected (QJL) | Not addressed (relies on PCA fidelity) |
| Quality at ratio | Good | Better (model-tuned) |

---

## KVTC Admin UI Gap (OBS-04 Extension)

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

### Recommended UI Addition (for a future phase or as part of Phase 9)
Add a "Compression" card to the admin dashboard that shows:
- Enabled toggle (calls `POST /admin/api/compression/config`)
- Live compression ratio
- Avg decompression latency (ms/layer)
- Success/failure counts
- `am_ratio` slider (1.0–8.0)

This is a self-contained frontend task — all data is already available from the backend.

---

## Phase 9 Update (2026-03-25)

Phase 9 now includes **two waves**:

### Wave 0 (COMPLETE)
- docs/compression/ directory with all documentation files
- BenchmarkReport extended with compression_metrics field
- CompressionMetrics dataclass defined

### Wave 1 (PENDING - Admin UI)
The admin UI work described above is now formalized as Phase 9 Wave 1:
- Compression settings card in `_settings.html`
- Compression stats card in `_status.html`
- JavaScript integration in `dashboard.js`
- i18n strings in `en.json`

All backend endpoints are already functional — this is purely a frontend UI implementation.
