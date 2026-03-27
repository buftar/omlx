# KV Cache Compression

A two-stage compression pipeline that reduces KV cache memory usage during long-context inference, with no changes to the public inference API.

## Overview

The pipeline consists of two stages applied on each cache eviction cycle:

1. **AM (Attention Matching) Compaction** — reduces token count by selecting the most important tokens per head based on attention patterns, using NNLS beta-fitting and OLS value-fitting. Sink tokens are always preserved.
2. **KVTC (KV Cache Transform Coding)** — PCA-based byte-level compression of key tensors using a calibration bundle generated offline.

```
┌──────────────┐    ┌────────────────────┐    ┌──────────────────────┐
│  Full KV     │ →  │  AM Compaction     │ →  │  KVTC Compression    │
│  Cache       │    │  (token reduction) │    │  (byte compression)  │
└──────────────┘    └────────────────────┘    └──────────────────────┘
```

Compression is opt-in and requires `--compression-bundle` + `--paged-cache-dir`. Existing behaviour is unchanged when neither flag is set.

### When to use

Enable when:
- Running long-context workloads (>8K tokens)
- GPU memory is constrained and you need more concurrent sessions
- You have a calibration bundle for the target model

Disable when:
- Running short-context workloads (<1K tokens)
- Latency is more critical than memory savings

### Typical performance

| Metric | Value |
|--------|-------|
| AM compaction ratio | 4x (configurable, 1–8x) |
| KVTC byte compression | ~16x |
| Combined ratio | ~64x |
| Decompression latency | <10ms/layer |
| Quality degradation | <1% on GSM8K/MMLU at 4x |

The AM ratio is a tunable parameter on a quality–memory Pareto frontier. 4x is the recommended starting point: it sits well within the conservative, quality-preserving range validated in the Attention Matching paper (which tests up to 50–200x). Increase toward 8x if memory pressure is severe; decrease toward 2x if quality is the priority.

## Quick start

```bash
# 1. Generate a calibration bundle for your model
omlx calibrate-kv Qwen/Qwen2.5-7B

# 2. Start the server with compression enabled
omlx serve \
  --compression-bundle ~/.omlx/calibration/kv_pca_calibration.npz \
  --compression-am-ratio 4.0 \
  --paged-cache-dir /path/to/ssd
```

Or enable at runtime without restarting:

```bash
curl -X POST http://localhost:8080/admin/api/compression/config \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "am_ratio": 4.0}'
```

## Calibration

Calibration generates PCA matrices and per-head entropy curves used by both pipeline stages.

```bash
# Basic
omlx calibrate-kv Qwen/Qwen2.5-7B

# Custom output path
omlx calibrate-kv --output /path/to/bundle.npz Qwen/Qwen2.5-7B

# All options
omlx calibrate-kv --help
```

| Option | Description |
|--------|-------------|
| `--output PATH` | Output path (default: `~/.omlx/calibration/kv_pca_calibration.npz`) |
| `--max-tokens N` | Maximum context length (reduce if OOM) |
| `--batch-size N` | Batch size |
| `--n-samples N` | Calibration samples (increase for better quality) |
| `--seed N` | Random seed for reproducibility |

The bundle contains:

| Key | Description |
|-----|-------------|
| `V` | PCA basis matrix (components × features) |
| `mu` | Mean vector for centering |
| `bit_alloc` | DP-allocated bit widths per component |
| `group_sizes` | Layer group sizes for shared PCA |
| `head_entropy` | Per-head entropy sensitivity curves |

Re-calibrate when the model checkpoint changes, quality degrades noticeably, or the input data distribution shifts significantly.

Approximate calibration times on M3 Max:

| Model | Context | Time |
|-------|---------|------|
| Qwen 2.5 7B | 8K | ~10 min |
| Llama 3 8B | 8K | ~12 min |
| Gemma 3 7B | 8K | ~8 min |

## Configuration

### Startup flags

```bash
omlx serve \
  --compression-bundle /path/to/bundle.npz \
  --compression-am-ratio 4.0 \
  --paged-cache-dir /path/to/ssd
```

### CompressionConfig

```python
@dataclass
class CompressionConfig:
    enabled: bool = False
    am_ratio: float = 4.0
    bundle_path: Optional[str] = None
```

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `False` | Enable/disable the pipeline |
| `am_ratio` | `4.0` | Token reduction ratio (4.0 = keep 25% of tokens) |
| `bundle_path` | `None` | Path to calibration bundle |

### Compression levels

| Level | `am_ratio` | Use case |
|-------|-----------|----------|
| Light | 2.0 | Quality-sensitive workloads |
| Balanced | 4.0 | General use (recommended default) |
| Aggressive | 8.0 | Severe memory pressure |

### Runtime API

```bash
# Update config
curl -X POST http://localhost:8080/admin/api/compression/config \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "am_ratio": 4.0}'

# Check status
curl http://localhost:8080/admin/api/compression/status
```

Status response includes `enabled`, `compression_ratio`, `avg_decompression_latency_ms`, `compression_success_count`, `compression_failure_count`.

## Troubleshooting

### Missing calibration bundle

```
ValueError: Compression bundle not found at ~/.omlx/calibration/kv_pca_calibration.npz
```

Run calibration first (`omlx calibrate-kv <model>`) or disable compression.

### Float16 input error

```
ValueError: Float16 input detected. Please cast to float32 before compression.
```

The pipeline requires float32 KV tensors. Cast before passing:

```python
kv_cache = [(k.astype(mx.float32), v.astype(mx.float32)) for k, v in kv_cache]
```

### Latency regression (>10ms/layer)

Check `avg_decompression_latency_ms` via the status endpoint. Mitigations:
1. Reduce `am_ratio` to 2.0
2. Disable compression temporarily
3. Re-calibrate with more samples (`--n-samples 500`)

### Quality degradation (>1% on benchmarks)

Run `omlx benchmark-compression --model <model> --bundle <path>` and check `am_cosine_similarity` (target >0.998). Mitigations:
1. Reduce `am_ratio`
2. Ensure a calibration bundle is in use
3. See known issues below

### Cache miss rate increase

Compression reduces token count which can affect cache key matching. Reduce `am_ratio` or monitor `cache_hit_rate` before/after enabling.

## Known issues

**Gemma 3 SWA layers** — Sliding Window Attention layers are automatically skipped. Check `swa_layers_skipped` in benchmark output.

**DeepSeek MLA architecture** — Multi-Latent Attention (MLA) is out of scope. Use distilled variants with standard GQA architecture instead.

## References

- [Fast KV Compaction via Attention Matching](https://arxiv.org/abs/2502.09398)
- [KV Cache Transform Coding](https://arxiv.org/abs/2502.09949)
