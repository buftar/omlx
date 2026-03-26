# Calibration Workflow

Calibration generates PCA matrices for optimal KVTC compression.

## Overview

Calibration computes:
- **PCA Basis (V)**: Cross-layer principal components for decorrelation
- **DP Bit Allocation**: Optimal bit widths per PCA component
- **Head Entropy**: Per-head sensitivity curves for AM non-uniform budgets

## Running Calibration

```bash
# Basic usage
omlx calibrate-kv <model-path>

# Example for Qwen 2.5 7B
omlx calibrate-kv Qwen/Qwen2.5-7B

# With custom output path
omlx calibrate-kv --output /custom/path/calibration.npz Qwen/Qwen2.5-7B
```

## Calibration Output

The calibration bundle contains:

| Key | Description |
|-----|-------------|
| `V` | PCA basis matrix (components × features) |
| `mu` | Mean vector for centering |
| `bit_alloc` | DP-allocated bit widths per component |
| `group_sizes` | Layer group sizes for shared PCA |
| `head_entropy` | Per-head entropy sensitivity |

## Calibration Workflow

### Step 1: Run Calibration

```bash
omlx calibrate-kv Qwen/Qwen2.5-7B
```

Output:
```
Loading model...
Extracting KV caches from training data...
Computing PCA basis (this may take a few minutes)...
Allocating bits with DP algorithm...
Saving calibration bundle to ~/.omlx/calibration/kv_pca_calibration.npz
Calibration complete!
```

### Step 2: Verify Bundle

Check bundle contents:

```python
import numpy as np
bundle = np.load("~/.omlx/calibration/kv_pca_calibration.npz")
print("Keys:", list(bundle.keys()))
print("V shape:", bundle["V"].shape)
print("Bit allocation:", bundle["bit_alloc"])
```

### Step 3: Use Bundle

```bash
# Start server with calibration bundle
omlx serve --compression-enabled --compression-bundle ~/.omlx/calibration/kv_pca_calibration.npz
```

## When to Re-calibrate

Re-calibrate when:
- **Model update**: New model version or checkpoint
- **Quality degradation**: Compression quality drops significantly
- **Different dataset**: Training on different data distribution

## Calibration Performance

| Model | Context | Time |
|-------|---------|------|
| Qwen 2.5 7B | 8K | ~10 min |
| Llama 3 8B | 8K | ~12 min |
| Gemma 3 7B | 8K | ~8 min |

## Calibration Troubleshooting

### Out of Memory

Reduce context length:

```bash
omlx calibrate-kv --max-tokens 4096 Qwen/Qwen2.5-7B
```

### Slow Calibration

Increase batch size:

```bash
omlx calibrate-kv --batch-size 32 Qwen/Qwen2.5-7B
```

### Poor Quality

Increase calibration samples:

```bash
omlx calibrate-kv --n-samples 500 Qwen/Qwen2.5-7B
```

## Advanced Options

```bash
# Full usage
omlx calibrate-kv --help
```

| Option | Description |
|--------|-------------|
| `--output PATH` | Output path for calibration bundle |
| `--max-tokens N` | Maximum context length |
| `--batch-size N` | Batch size for calibration |
| `--n-samples N` | Number of calibration samples |
| `--seed N` | Random seed for reproducibility |