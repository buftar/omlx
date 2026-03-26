# Troubleshooting

Common issues and their solutions.

## Missing Calibration Bundle

### Error
```
ValueError: Compression bundle not found at ~/.omlx/calibration/kv_pca_calibration.npz
```

### Solution

Run calibration first:

```bash
omlx calibrate-kv Qwen/Qwen2.5-7B
```

Or disable compression:

```bash
omlx serve --compression-enabled=false
```

## Float16 Input Error

### Error
```
ValueError: Float16 input detected. Please cast to float32 before compression.
```

### Solution

The compression pipeline requires float32 inputs. Ensure your KV cache is float32:

```python
# Before compression, verify dtype
assert kv_cache[0][0].dtype == mx.float32, "KV cache must be float32"
```

If using mlx_lm, the cache should already be float32. If not:

```python
# Cast to float32
kv_cache = [
    (k.astype(mx.float32), v.astype(mx.float32))
    for k, v in kv_cache
]
```

## Latency Regression

### Symptom
Decompression latency exceeds 10ms/layer.

### Diagnosis

Check metrics:
```bash
curl http://localhost:8080/admin/api/compression/status
```

Look for:
- High `avg_decompression_latency_ms`
- Low `decompression_success_count`

### Solutions

1. **Reduce AM ratio**:
   ```bash
   curl -X POST http://localhost:8080/admin/api/compression/config \
     -d '{"am_ratio": 2.0}'
   ```

2. **Disable compression temporarily**:
   ```bash
   curl -X POST http://localhost:8080/admin/api/compression/config \
     -d '{"enabled": false}'
   ```

3. **Re-calibrate** with more samples:
   ```bash
   omlx calibrate-kv --n-samples 500 <model>
   ```

## Quality Degradation

### Symptom
Task accuracy drops >1% after compression.

### Diagnosis

Run benchmark:
```bash
omlx benchmark-compression --model <model> --bundle <path>
```

Check:
- `am_cosine_similarity` (should be >0.998)
- `gsm8k_delta` (should be <1.0)
- `mmlu_delta` (should be <1.0)

### Solutions

1. **Reduce AM ratio**:
   ```bash
   curl -X POST http://localhost:8080/admin/api/compression/config \
     -d '{"am_ratio": 2.0}'
   ```

2. **Use calibration bundle** (if not using one):
   ```bash
   omlx calibrate-kv <model>
   omlx serve --compression-bundle <path>
   ```

3. **Check SWA layers**: Gemma 3 models with SWA layers need special handling

## Cache Miss Rate Increase

### Symptom
Cache hit rate drops after enabling compression.

### Cause
Compression changes token count, which can affect cache key matching.

### Solutions

1. **Check compression ratio**:
   ```bash
   curl http://localhost:8080/admin/api/compression/status
   ```

2. **Reduce AM ratio** for better cache compatibility:
   ```bash
   curl -X POST http://localhost:8080/admin/api/compression/config \
     -d '{"am_ratio": 2.0}'
   ```

3. **Monitor hit/miss separately**:
   - `cache_hit_rate`: Before compression
   - `compression_cache_hit_rate`: After compression (if tracked)

## Server Crashes on Compression

### Symptom
Server crashes or hangs during compression.

### Diagnosis

Check logs:
```bash
tail -f ~/.omlx/logs/server.log
```

Look for:
- Memory allocation errors
- CUDA out of memory (if using GPU)
- Decompression failures

### Solutions

1. **Reduce context length**:
   ```bash
   curl -X POST http://localhost:8080/admin/api/compression/config \
     -d '{"am_ratio": 2.0}'
   ```

2. **Disable compression**:
   ```bash
   curl -X POST http://localhost:8080/admin/api/compression/config \
     -d '{"enabled": false}'
   ```

3. **Increase system memory** or use smaller model

## Known Issues

### SWA Layers in Gemma 3

Gemma 3 models use SwiGLU Attention (SWA) layers that don't support compression.

**Workaround**: Compression automatically skips SWA layers. Check `swa_layers_skipped` in benchmark output.

### DeepSeek MLA Architecture

DeepSeek models use Multi-Latent Attention (MLA) which is out of scope.

**Workaround**: Use distilled variants with standard GQA architecture.

## Getting Help

1. Check logs: `~/.omlx/logs/server.log`
2. Run benchmark: `omlx benchmark-compression --model <model>`
3. Check metrics: `curl http://localhost:8080/admin/api/compression/status`
4. Open issue on GitHub with error logs