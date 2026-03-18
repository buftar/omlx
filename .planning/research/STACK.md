# Stack Research

**Domain:** KV cache compression pipeline for Apple Silicon LLM inference (AM + kvtc)
**Researched:** 2026-03-18
**Confidence:** HIGH (spike prototypes validated; versions verified against installed environment and PyPI)

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| mlx | 0.31.1 | All tensor operations: attention, matrix math, softmax, elementwise ops | The only viable tensor framework for Apple Silicon GPU (Metal). Unified memory means zero-copy between CPU/GPU tensors. No CUDA, no PyTorch MPS workarounds. The spike ran on 0.31.0; 0.31.1 is the March 12 2026 stable release with no breaking changes to linalg. |
| mlx-lm | 0.30.7 | KV cache extraction via `make_prompt_cache` and `KVCache`/`QuantizedKVCache` interfaces | Already a direct dependency of omlx. The `mlx_lm.models.cache` module is the canonical way to interact with KV cache objects produced by mlx_lm-backed models. Don't reimplement; hook into existing interfaces. |
| numpy | 1.26.4 | Bridge layer: convert MLX tensors to numpy arrays for scipy calls; pack quantized arrays to bytes | Seamless interop via `np.array(mx_tensor)`. The spike uses numpy extensively for this bridge role. No alternative — scipy requires numpy arrays. Pin to 1.x; numpy 2.x has breaking API changes in memory layout that affect tobytes() patterns. |
| scipy | 1.15.2 | `scipy.optimize.nnls` for beta-fitting in AM (NNLS solves `min ||Aw - m||^2  s.t. w>=0`) | No MLX-native NNLS exists. The spike measured ~1ms per head for NNLS — fast enough. scipy 1.15.2 is the installed version. For OLS/value-fitting, use `mx.linalg.pinv` instead of scipy — the spike confirmed `pinv` is available in MLX with `stream=mx.cpu`. |
| zstandard | 0.23.0 | Entropy coding (lossless) in kvtc pipeline | The kvtc paper codec comparison (Table B.8) shows Zstandard is the top performer (34.6–46.3x total CR vs DEFLATE's 34.7–42.9x). nvCOMP is NVIDIA-only and ruled out. `zstandard` Python package wraps Meta's libzstd. The spike uses `zstandard` at level 3 (fast, good ratio). |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| mlx.core.linalg (built-in) | mlx 0.31.1 | SVD for PCA calibration (`mx.linalg.svd`), pseudoinverse for OLS value fitting (`mx.linalg.pinv`), normal equations via `mx.linalg.solve` | All linalg ops in the compression pipeline. Critical constraints: SVD is CPU-only in 0.31.x (Metal GPU SVD PR #2290 was closed as stale in Aug 2025). Always cast to float32 before SVD/pinv — float16 inputs cause silent numerical failures. Always pass `stream=mx.cpu` to both `svd` and `pinv`. |
| mlx.core (built-in) | mlx 0.31.1 | General tensor ops: `mx.matmul`, `mx.softmax`, `mx.exp`, `mx.log`, array slicing, dtype casting | All hot-path compute. Use `mx.eval(tensor)` to force lazy graph materialization before timing or numpy conversion. Cast queries/keys to float32 before softmax — the spike confirmed float16 softmax overflows for moderate sequence lengths. |
| psutil | 5.9.0+ | Memory pressure detection — already in omlx dependencies | Read `psutil.virtual_memory().percent` to trigger AM compaction. Already installed; no new dependency. |
| struct / numpy tobytes (stdlib + numpy) | Python 3.10+ / numpy 1.26.4 | Pack quantized int4/int2/fp8 coefficients into bytes for zstd input | Use `np.ndarray.tobytes()` directly — fastest for bulk arrays. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| black 23+ | Code formatting (already in omlx dev deps) | Line length 88, target py310–py313 per existing pyproject.toml |
| ruff 0.1+ | Linting (already in omlx dev deps) | Config already in pyproject.toml; includes E, F, W, I, N, UP, B, SIM |
| pytest + pytest-asyncio | Testing compression pipeline (already in omlx dev deps) | asyncio_mode = auto per existing pytest.ini |
| mypy | Type checking (already in omlx dev deps) | python_version = "3.10", ignore_missing_imports = true |

---

## Installation

All core dependencies are either already in omlx's `pyproject.toml` or need a single addition.

```bash
# Already present in pyproject.toml (no changes needed):
# mlx>=0.31.1  (pin to >=0.31.1 for stability; 0.31.0 works per spike)
# numpy>=1.24.0
# psutil>=5.9.0

# Add to pyproject.toml dependencies:
# "zstandard>=0.23.0",
# "scipy>=1.15.0",

# Install for development:
pip install -e ".[dev]"
```

The only net-new production dependency is `zstandard`. `scipy` should be added explicitly rather than relying on it being present transitively — it is not currently listed in pyproject.toml.

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| `scipy.optimize.nnls` | Pure MLX NNLS (iterative projected gradient) | If scipy dependency is unacceptable. Not recommended: no validated MLX NNLS exists; writing a correct NNLS from scratch introduces numerical risk. The ~1ms/head cost of scipy NNLS is fast enough that there is no performance argument for a custom implementation. |
| `zstandard` (python-zstandard 0.23) | `compression.zstd` (Python 3.14 stdlib) | If omlx drops Python 3.10–3.13 support and targets Python 3.14+ exclusively. Python 3.14 was released October 2025 but omlx currently supports 3.10–3.13. Use `zstandard` now; the stdlib module is a valid future migration path. |
| `zstandard` | `pyzstd` | pyzstd has a similar API. No advantage over `zstandard` for this use case; `zstandard` has wider adoption and is the de-facto standard Python zstd binding. |
| `mx.linalg.pinv` (MLX, CPU stream) | `np.linalg.lstsq` (numpy) | The AM paper uses `torch.linalg.lstsq` and notes it produces better quality than pinv. If you observe numerical quality issues in value fitting (Cv), switch to `np.linalg.lstsq` via numpy bridge. The spike validated `mx.linalg.pinv` at sufficient quality for 4x compaction (cos=0.9987). |
| Cross-layer PCA calibration (all layers concatenated) | Per-layer PCA | Per-layer is strictly worse: the kvtc paper ablation (B.10) shows cross-layer concatenation is load-bearing — using 1 layer gives LITM 0.0 at 32x CR vs 44.4 for 32 layers. One-time PCA is also the only approach with viable compression ratios (per-prompt V^T storage eats gains; see kvtc paper B.11). |
| `mx.linalg.svd` (MLX, CPU stream, float32) | Randomized SVD via sklearn `randomized_svd` | The paper (Halko et al. 2011 randomized SVD) is worth considering for large models where full SVD is too slow. The spike calibrated 28 layers in 4s with full SVD on Qwen2.5-7B (4 KV heads). For models with many KV heads (e.g., MHA with 32+ heads), `sklearn.utils.extmath.randomized_svd` would be faster. Flag this as a scalability consideration for MHA models. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| nvCOMP | NVIDIA GPU-only compression library. Hardcoded to CUDA. Not available on macOS. The kvtc paper uses it but explicitly calls it an implementation detail for their H100 hardware. | `zstandard` — matches or exceeds nvCOMP/DEFLATE compression ratio per Table B.8 of kvtc paper |
| PyTorch (torch) | Not in omlx. Adding torch for NNLS or lstsq would pull in a ~2GB CUDA-optional package. The AM paper uses `torch.linalg.lstsq` but that is their training environment, not a requirement. | scipy + numpy for equivalent linear algebra |
| `mx.linalg.svd` on float16 input | Silent numerical failure — MLX SVD does not raise on float16 input but produces garbage results. Confirmed in the spike: must cast KV tensors (float16 in inference) to float32 before SVD. | Always: `kv_f32 = kv.astype(mx.float32)` before calling `mx.linalg.svd(kv_f32, stream=mx.cpu)` |
| `mx.linalg.svd` without `stream=mx.cpu` | SVD runs CPU-only in MLX 0.31.x. Omitting `stream=mx.cpu` causes a stream scheduling error when the default device is GPU. The GPU Metal SVD PR (#2290) was closed stale in August 2025 and is not available. | `mx.linalg.svd(a, stream=mx.cpu)` — always explicit |
| `mx.linalg.pinv` without `stream=mx.cpu` | Same constraint as SVD — pinv is CPU-only in 0.31.x. | `mx.linalg.pinv(a, stream=mx.cpu)` — always explicit |
| float16 for attention recomputation in AM | Overflow at moderate sequence lengths confirmed in spike. The AM compaction step recomputes attention scores `exp(q @ Ck.T)` — this must be in float32. | Cast queries and compacted keys to float32 before attention score computation, cast output back to float16 if needed |
| Per-prompt PCA calibration | Storing a separate V^T matrix per conversation prompt collapses compression ratios to 1.3–12.4x vs 54–63x for one-time PCA (kvtc paper, appendix B.11). Also generalizes poorly to conversation continuation. | One-time PCA calibration per model, stored alongside model weights as a `.npz` file |
| numpy 2.x (unvalidated) | Breaking changes in memory layout and array interface that may affect `.tobytes()`, structured array handling, and MLX interop patterns. omlx pins `numpy>=1.24.0` which admits 2.x. | Keep `numpy>=1.24.0,<2.0` in compression module until 2.x interop is explicitly validated. |

---

## Stack Patterns by Variant

**For AM compaction (hot cache, memory pressure trigger):**
- All ops in MLX (`mx.matmul`, `mx.softmax`, `mx.exp`, `mx.log`)
- NNLS via scipy (numpy bridge for input/output)
- OLS/Cv fitting via `mx.linalg.pinv(stream=mx.cpu)` with float32 cast
- Key selection via attention score ranking in MLX
- No zstd needed — AM output is a reduced MLX tensor, not bytes

**For kvtc calibration (one-time CLI command, `omlx calibrate-kv <model>`):**
- SVD via `mx.linalg.svd(stream=mx.cpu)` with float32 cast
- For models with large KV head counts (>16 heads), consider `sklearn.utils.extmath.randomized_svd` as a faster alternative
- DP quantization table computed in numpy (pure Python loops + numpy arrays)
- Store calibration artifact as `numpy.savez_compressed()` (keys: `V`, `mu`, `bit_alloc`, `group_sizes`)

**For kvtc compression (eviction path trigger):**
- PCA projection: `(kv_f16.astype(np.float32) - mu) @ V` in numpy (kv already evicted from GPU)
- Quantize: numpy integer casting per DP bit allocation
- Pack + compress: `np.ndarray.tobytes()` + `zstandard.ZstdCompressor(level=3).compress()`

**For kvtc decompression (cache miss path):**
- Decompress: `zstandard.ZstdDecompressor().decompress()`
- Unpack + dequantize: numpy unpack + scale/shift
- PCA inverse: `D_float @ V.T + mu`, convert back to float16 MLX tensor
- Target: <10ms per layer (spike achieved 1.5ms)

**For sliding window / attention sink exemptions:**
- Do not compress first `s=4` tokens (attention sinks) — compressing these at high CR causes catastrophic accuracy loss confirmed in kvtc paper appendix B.3 (LITM drops from 90.2 to 0.0 at 64x CR)
- Do not compress last `w=128` tokens (sliding window) — active tokens with high attention mass
- Implement as index slicing before compression; re-concatenate after decompression

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| mlx 0.31.1 | mlx-lm 0.30.7 | Installed pair, verified working in spike on M3 Max |
| mlx 0.31.1 | numpy 1.26.4 | `np.array(mx_tensor)` interop confirmed working |
| scipy 1.15.2 | numpy 1.26.4 | Standard scipy/numpy pairing, no known issues |
| zstandard 0.23.0 | Python 3.10–3.13 | Supports all omlx-targeted Python versions |
| scipy 1.15.2 | Python 3.10–3.13 | Supports all omlx-targeted Python versions |
| numpy 1.26.4 (1.x) | MLX tensor interop | numpy 2.x interop with MLX has not been validated in this project — avoid until tested |

---

## Sources

- Spike results: `/Users/tonysina/projects/omlx/docs/research/kv-cache-compression/SPIKE-RESULTS.md` — MLX 0.31.0, all linalg constraints confirmed empirically (HIGH confidence)
- MLX PyPI: https://pypi.org/project/mlx/ — version 0.31.1, released March 12 2026 (HIGH confidence)
- MLX linalg docs: https://ml-explore.github.io/mlx/build/html/python/linalg.html — full linalg API listing confirmed (HIGH confidence)
- MLX SVD GPU PR (closed stale): https://github.com/ml-explore/mlx/pull/2290 — GPU SVD unavailable in 0.31.x (HIGH confidence)
- MLX SVD CPU-only issue: https://github.com/ml-explore/mlx/issues/847 — CPU-only constraint confirmed (HIGH confidence)
- kvtc paper (ICLR 2026, arXiv 2511.01815v2): codec comparison Table B.8 — Zstandard is best lossless codec (HIGH confidence, primary source)
- AM paper (arXiv 2602.16284, Feb 2026): solver comparison — lstsq better than pinv, but pinv validated sufficient in spike (HIGH confidence, primary source)
- python-zstandard: https://pypi.org/project/zstandard/ — current stable 0.25.0; installed 0.23.0 confirmed working (HIGH confidence)
- Python 3.14 compression.zstd: https://docs.python.org/3/library/compression.zstd.html — available in 3.14+ only; not usable with omlx's 3.10–3.13 support matrix (HIGH confidence)
- Installed versions confirmed via `conda list` on M3 Max (HIGH confidence)

---

*Stack research for: KV cache compression pipeline (AM + kvtc) on Apple Silicon / MLX*
*Researched: 2026-03-18*
