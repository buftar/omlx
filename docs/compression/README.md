# KV Cache Compression

A two-stage compression pipeline for KV caches in oMLX that reduces memory usage while maintaining inference quality.

## Overview

oMLX's KV cache compression pipeline consists of two stages:

1. **AM (Attention Matching) Compaction** - Reduces token count by selecting the most important tokens based on attention patterns
2. **KVTC (KV Cache Transform Coding)** - Applies PCA-based quantization and entropy coding for byte-level compression

### Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Original KV   │ →  │  AM Compaction │ →  │  KVTC Compression│
│    Cache        │    │   (Token Red.) │    │  (Byte Compress) │
└─────────────────┘    └──────────────────┘    └──────────────────┘
```

### Compression Pipeline

The pipeline works as follows:

1. **Input**: Full KV cache with keys and values for each layer
2. **AM Compaction**: Selects top-k tokens per layer based on attention scores, reducing token count by configured ratio (e.g., 4x)
3. **KVTC Compression**: Applies PCA rotation, quantization with DP bit allocation, and zstd entropy coding

### When to Use

Enable compression when:
- GPU memory is constrained
- Running long context workloads (>8K tokens)
- Need to keep more conversations in memory

Disable compression when:
- Running short context workloads (<1K tokens)
- Latency is more critical than memory savings
- Debugging compression-related issues

### Performance Expectations

| Metric | Typical Value |
|--------|---------------|
| AM Compaction Ratio | 4x (configurable) |
| KVTC Compression Ratio | 16x (configurable) |
| Combined Ratio | 64x (4x × 16x) |
| Decompression Latency | <10ms/layer |
| Quality Degradation | <1% on GSM8K/MMLU |

## Quick Start

### Enable Compression

```bash
# With pre-calibrated bundle
omlx serve --compression-bundle /path/to/kv_pca_calibration.npz

# Or enable at runtime via admin API
curl -X POST http://localhost:8080/admin/api/compression/config \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "am_ratio": 4.0}'
```

### Calibration

For optimal compression, run calibration first:

```bash
omlx calibrate-kv Qwen/Qwen2.5-7B
```

This generates a calibration bundle at `~/.omlx/calibration/kv_pca_calibration.npz`.

## Configuration

See [CONFIGURATION.md](./CONFIGURATION.md) for all available options.

## Calibration Workflow

See [CALIBRATION.md](./CALIBRATION.md) for detailed calibration instructions.

## Troubleshooting

See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for common issues and solutions.