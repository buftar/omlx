# Configuration Reference

Compression settings can be configured at startup or runtime.

## Startup Configuration

### Command Line

```bash
# Enable compression with default settings
omlx serve --compression-enabled

# Enable compression with custom AM ratio
omlx serve --compression-enabled --am-ratio 4.0

# Use custom calibration bundle
omlx serve --compression-bundle /path/to/calibration.npz
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OMLX_COMPRESSION_ENABLED` | Enable compression pipeline | `false` |
| `OMLX_AM_RATIO` | AM compaction ratio | `4.0` |
| `OMLX_COMPRESSION_BUNDLE` | Path to PCA calibration bundle | `None` |

## Runtime Configuration

### Admin API

```bash
# Enable compression
curl -X POST http://localhost:8080/admin/api/compression/config \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "am_ratio": 4.0}'

# Disable compression
curl -X POST http://localhost:8080/admin/api/compression/config \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'

# Get current status
curl http://localhost:8080/admin/api/compression/status
```

## CompressionConfig

The `CompressionConfig` dataclass controls all compression settings:

```python
@dataclass
class CompressionConfig:
    enabled: bool = False        # Enable/disable compression pipeline
    am_ratio: float = 4.0        # AM compaction ratio (token reduction)
    bundle_path: Optional[str] = None  # Path to PCA calibration bundle
```

### Field Descriptions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable or disable the entire compression pipeline |
| `am_ratio` | `float` | `4.0` | AM compaction ratio. 4.0 = reduce tokens to 25% of original |
| `bundle_path` | `Optional[str]` | `None` | Path to PCA calibration bundle. If `None`, uses TurboQuant-style data-oblivious quantization |

## Compression Levels

### Light (2x-4x total)
```python
CompressionConfig(
    enabled=True,
    am_ratio=2.0,  # 50% token reduction
    bundle_path=None  # Data-oblivious quantization
)
```

### Balanced (16x total)
```python
CompressionConfig(
    enabled=True,
    am_ratio=4.0,  # 25% token reduction
    bundle_path="/path/to/calibration.npz"  # Model-specific PCA
)
```

### Aggressive (64x+ total)
```python
CompressionConfig(
    enabled=True,
    am_ratio=8.0,  # 12.5% token reduction
    bundle_path="/path/to/calibration.npz"
)
```

## Default Values

| Setting | Default | Notes |
|---------|---------|-------|
| `enabled` | `false` | Compression disabled by default |
| `am_ratio` | `4.0` | 4x compaction is standard |
| `bundle_path` | `None` | Uses TurboQuant path if not specified |

## Runtime Toggle

Compression can be toggled at runtime without server restart:

```python
from omlx.admin.routes import set_compression_config

# Disable compression
set_compression_config(CompressionConfig(enabled=False))

# Re-enable with custom settings
set_compression_config(CompressionConfig(
    enabled=True,
    am_ratio=4.0,
    bundle_path="/path/to/calibration.npz"
))
```

## Monitoring

Check compression status via admin API:

```bash
curl http://localhost:8080/admin/api/compression/status
```

Response includes:
- `enabled`: Current enablement state
- `compression_ratio`: Average compression ratio
- `avg_decompression_latency_ms`: Average decompression latency
- `compression_success_count` / `compression_failure_count`
- `decompression_success_count` / `decompression_failure_count`

## Best Practices

1. **Start with defaults**: Use `am_ratio=4.0` for balanced compression
2. **Calibrate for quality**: Run `omlx calibrate-kv <model>` for best quality
3. **Monitor latency**: Ensure decompression stays under 10ms/layer
4. **Check quality**: Verify task accuracy (GSM8K/MMLU) doesn't degrade >1%
5. **Adjust ratio**: Increase `am_ratio` if memory is constrained, decrease for quality