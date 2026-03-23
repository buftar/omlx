# Phase 5: Pipeline Assembly - Research

**Researched:** 2026-03-22
**Domain:** Python dataclass composition, MLX tensor pipeline, pytest mock patterns
**Confidence:** HIGH

## Summary

Phase 5 is a pure composition phase — no new algorithms. `KVCachePipeline` is a thin wrapper that sequences `AMCompactor.compact()` -> RoPE stripping -> `KVTCCompressor.compress()` and wraps the result in a `PipelineBlob` dataclass. All three upstream components (`am.py`, `kvtc.py`, `calibrator.py`) are complete, fully tested, and have stable APIs verified by reading the source directly.

The key design challenge is the RoPE stripping step between the two compressors. `strip_rope_from_keys()` in `calibrator.py` accepts a numpy array and returns numpy, but `AMCompactedCache.layers` contains `mx.array` tensors. The pipeline must convert (mx -> numpy), strip, and feed the result back to `KVTCCompressor` which expects `mx.array` inputs. The `bundle_path=None` testing path skips stripping entirely, matching the `None`-fallback contracts of both sub-compressors.

The trigger semantics (PIPE-03/04/05) are tested in Phase 5 via a lightweight mock callable, not by wiring to the omlx `MemoryMonitor` or eviction queue. Phase 6 handles actual hook registration. This scope boundary is locked in CONTEXT.md and must not be crossed.

**Primary recommendation:** Implement `pipeline.py` as a single-file module with one `PipelineBlob` dataclass and one `KVCachePipeline` class. Keep it thin — delegate all math to the sub-compressors and `strip_rope_from_keys`.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Constructor**
- `KVCachePipeline(bundle_path=None, am_ratio=4.0, n_sink_tokens=4, sliding_window=128)`
- Path-based: pipeline loads the `.npz` calibration bundle itself, constructs both `AMCompactor` and `KVTCCompressor` internally. Phase 6 passes one path.
- When `bundle_path=None`: both compressors use their bundle=None fallbacks (testing-only path, documented as lower quality).

**Method surface — two call paths**
- `compact(kv_cache, queries=None) -> AMCompactedCache` — Memory-pressure path. Runs AM compaction only. Returns `AMCompactedCache` (same as `AMCompactor.compact()` output). Phase 6 triggers this on GPU memory pressure.
- `compress(kv_cache, queries=None) -> PipelineBlob` — Eviction path. Runs full AM->kvtc pipeline: compact first, then kvtc compress. Returns `PipelineBlob`. Phase 6 triggers this on eviction to SSD.
- `decompress(blob: PipelineBlob) -> tuple[list[tuple[mx.array, mx.array]], int]` — Returns `(compacted_layers, logical_seq_len)`. Compacted layers are at reduced token count (not original). `logical_seq_len` is the original T, preserved for RoPE position continuity.

**compress() input and RoPE handling**
- `compress(kv_cache, queries=None)` — accepts raw KV cache (same shape as `AMCompactor.compact()` input).
- Pipeline handles RoPE stripping internally before passing keys to `KVTCCompressor`. Stripping uses `strip_rope_from_keys()` from `calibrator.py` with params loaded from the bundle.
- When `bundle_path=None` (testing): RoPE stripping is skipped. Keys passed to kvtc as-is. Consistent with `AMCompactor(head_entropy=None)` and `KVTCCompressor(pca_bundle=None)` patterns.

**PipelineBlob dataclass**
- `PipelineBlob(compressed: bytes, logical_seq_len: int, compaction_ratio: float)`
- `compressed`: self-describing kvtc blob (contains all decompression metadata — no bundle needed at decompress time, per Phase 3 design).
- `logical_seq_len`: original sequence length T before AM compaction — preserved for RoPE position indices in continued inference.
- `compaction_ratio`: actual AM ratio achieved (for observability / Phase 8).

**Decompress fidelity**
- `decompress()` returns the compacted cache at reduced token count — AM compaction is lossy, no reconstruction to original token count.
- Contract: decompressed cache achieves >0.998 cosine similarity (AM quality guarantee) at compacted token count. Suitable for continued inference.
- `logical_seq_len` flows through the blob so callers can set correct RoPE position offsets.

**Trigger semantics (Phase 5 scope)**
- Phase 5 delivers the callable pipeline only. No callback registration, no ABC, no hook points.
- PIPE-03/04/05 trigger behavior is tested in Phase 5 via a mock memory monitor passed as an optional dep to tests — pipeline fires `compact()` above threshold and not below.
- Phase 6 registers the pipeline's `compact()` and `compress()` with omlx's actual memory monitor and eviction path.

**Testing**
- Fast (CI): Synthetic round-trip: `compress(synthetic_kv_cache)` -> `PipelineBlob` -> `decompress()` -> cosine similarity check + `logical_seq_len` match. No model loading required.
- Slow (`@pytest.mark.slow`): Load Qwen 2.5 7B, generate real KV cache, run full `compress()` -> `decompress()` round-trip, verify cosine similarity and that inference can continue. Consistent with Phase 3 slow test precedent.
- Mock-based trigger test: mock memory monitor fires above/below threshold; verify `compact()` is called only above threshold.

### Claude's Discretion
- Exact `PipelineBlob` serialization if needed for blob-within-blob storage
- Internal batching strategy across layers
- Error message text and edge-case validation
- Whether `compact()` accepts the same `ratio` override as constructor default

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PIPE-01 | AM->kvtc combined pipeline compresses KV cache with multiplicative ratio (token reduction x byte compression) | `AMCompactor.compact()` returns `AMCompactedCache`; `KVTCCompressor.compress()` takes that layer list; `compaction_ratio` field in `PipelineBlob` enables ratio reporting |
| PIPE-02 | Full round-trip compress->decompress restores a cache usable for continued inference | `KVTCCompressor.decompress()` returns `list[tuple[mx.array, mx.array]]`; `logical_seq_len` in blob enables correct RoPE offset at call site; >0.998 cosine similarity guaranteed by AM math |
| PIPE-03 | AM compaction is triggered automatically when GPU memory pressure exceeds threshold | Phase 5 tests with mock callable that fires compact() above threshold; verified pattern only, Phase 6 wires to real monitor |
| PIPE-04 | kvtc compression is triggered on cache eviction to SSD cold storage | Phase 5 exposes `compress()` as the eviction entry point; trigger registration deferred to Phase 6 |
| PIPE-05 | Decompression is triggered on cache miss when restoring from SSD | Phase 5 exposes `decompress()` as the cache-miss entry point; trigger registration deferred to Phase 6 |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `mlx.core` | 0.31.x | Tensor operations for KV cache data | Already the project's GPU compute layer |
| `numpy` | any | Array bridge for `strip_rope_from_keys()`, which accepts and returns numpy | Already used across am.py, kvtc.py, calibrator.py |
| `dataclasses` (stdlib) | stdlib | `PipelineBlob` definition | Zero deps, same pattern as `AMCompactedCache` |
| `pathlib` (stdlib) | stdlib | `bundle_path` handling in constructor | Already used in calibrator.py |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `unittest.mock` | stdlib | `MagicMock` / callable mocks for trigger tests | Fast unit tests only, no model loading |
| `pytest` | current | Test runner, `@pytest.mark.slow` marker | All tests — already configured in pytest.ini |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| stdlib `dataclasses.dataclass` | `attrs` or Pydantic | No reason to add deps; project already uses bare dataclasses (see `AMCompactedCache`) |

**Installation:** No new packages required. All dependencies already present.

## Architecture Patterns

### Recommended Project Structure
```
omlx/compression/
    __init__.py       # intentionally empty -- no re-exports
    linalg_utils.py   # Phase 1 math primitives
    am.py             # Phase 2 AMCompactor + AMCompactedCache
    kvtc.py           # Phase 3 KVTCCompressor
    calibrator.py     # Phase 4 load_calibration_bundle + strip_rope_from_keys
    pipeline.py       # Phase 5 KVCachePipeline + PipelineBlob  <- NEW

tests/
    test_pipeline.py  # Phase 5 tests <- NEW
```

### Pattern 1: Constructor Bundle Loading
**What:** Pipeline constructor accepts `bundle_path: str | None`. When not None, calls `load_calibration_bundle(path)` to get `(pca_bundle, head_entropy)` and constructs sub-compressors with those. When None, constructs sub-compressors with their `None` fallbacks.
**When to use:** Always — this is the locked constructor signature.
**Example:**
```python
# Source: calibrator.py load_calibration_bundle (verified by reading source)
from omlx.compression.calibrator import load_calibration_bundle, strip_rope_from_keys
from omlx.compression.am import AMCompactor
from omlx.compression.kvtc import KVTCCompressor
import mlx.core as mx

# MLX graph materialization alias -- mx.eval() forces lazy compute graph execution.
# This is NOT Python string evaluation. Named _mx_materialize to document intent.
_mx_materialize = mx.eval  # noqa: S307

class KVCachePipeline:
    def __init__(self, bundle_path=None, am_ratio=4.0, n_sink_tokens=4, sliding_window=128):
        self._am_ratio = am_ratio
        self._rope_params = None  # (rope_theta, rope_traditional) when bundle loaded

        if bundle_path is not None:
            pca_bundle, head_entropy = load_calibration_bundle(bundle_path)
            self._am = AMCompactor(head_entropy=head_entropy, n_sink_tokens=n_sink_tokens)
            self._kvtc = KVTCCompressor(
                pca_bundle=pca_bundle,
                n_sink_tokens=n_sink_tokens,
                sliding_window=sliding_window,
            )
            # rope_theta and rope_traditional must be provided separately
            # (not currently stored in the .npz bundle -- see Open Questions)
        else:
            self._am = AMCompactor(head_entropy=None, n_sink_tokens=n_sink_tokens)
            self._kvtc = KVTCCompressor(
                pca_bundle=None,
                n_sink_tokens=n_sink_tokens,
                sliding_window=sliding_window,
            )
```

### Pattern 2: RoPE Strip Before kvtc
**What:** `AMCompactedCache.layers` are `mx.array` tensors with RoPE in keys. `strip_rope_from_keys()` accepts and returns numpy float32. Must convert mx -> numpy -> strip -> convert back to mx for `KVTCCompressor.compress()`.
**When to use:** In `compress()`, between the AM compact step and the kvtc compress step. Skip entirely when `bundle_path=None`.
**Example:**
```python
# Source: calibrator.py strip_rope_from_keys signature (verified by reading source)
# strip_rope_from_keys(keys: np.ndarray, rope_theta, traditional, offset=0) -> np.ndarray
import numpy as np

def _strip_rope(self, compacted_layers):
    """Strip RoPE from compacted cache keys. No-op when bundle_path=None."""
    if self._rope_params is None:
        return compacted_layers
    rope_theta, rope_traditional = self._rope_params
    stripped = []
    for keys, values in compacted_layers:
        _mx_materialize(keys)
        keys_np = np.array(keys.astype(mx.float32))
        # offset=0: PCA treats compacted body as position-agnostic vectors
        keys_stripped_np = strip_rope_from_keys(
            keys_np, rope_theta, rope_traditional, offset=0
        )
        stripped.append((mx.array(keys_stripped_np.astype(np.float16)), values))
    return stripped
```

### Pattern 3: PipelineBlob Assembly
**What:** `compress()` assembles `PipelineBlob` from the outputs of the two stages.
**When to use:** At the end of `compress()`.
**Example:**
```python
# Source: AMCompactedCache dataclass fields (verified by reading am.py)
from dataclasses import dataclass

@dataclass
class PipelineBlob:
    compressed: bytes        # self-describing KVTC blob
    logical_seq_len: int     # original T before AM compaction
    compaction_ratio: float  # actual ratio achieved

def compress(self, kv_cache, queries=None):
    compacted = self._am.compact(kv_cache, ratio=self._am_ratio, queries=queries)
    original_seq_len = compacted.logical_seq_len
    compacted_seq_len = compacted.layers[0][0].shape[2]  # physical token count
    actual_ratio = original_seq_len / compacted_seq_len if compacted_seq_len > 0 else 1.0

    stripped_layers = self._strip_rope(compacted.layers)
    compressed_bytes = self._kvtc.compress(stripped_layers)

    return PipelineBlob(
        compressed=compressed_bytes,
        logical_seq_len=original_seq_len,
        compaction_ratio=actual_ratio,
    )
```

### Pattern 4: Decompress Return Contract
**What:** `decompress()` calls `KVTCCompressor.decompress()` and returns `(layers, logical_seq_len)`.
**When to use:** Implementing `decompress()`.
**Example:**
```python
# Source: kvtc.py KVTCCompressor.decompress() returns list[tuple[mx.array, mx.array]]
def decompress(self, blob: PipelineBlob):
    layers = self._kvtc.decompress(blob.compressed)
    return layers, blob.logical_seq_len
```

### Pattern 5: Mock Trigger Test
**What:** Trigger tests use a simple callable with configurable threshold, not a full `MemoryMonitor` mock. Pattern is to wrap the pipeline's `compact()` method as a callback and assert call count above/below threshold.
**When to use:** PIPE-03/04/05 test coverage.
**Example:**
```python
from unittest.mock import MagicMock

def test_compact_fires_above_threshold():
    pipeline = KVCachePipeline(bundle_path=None)
    compact_spy = MagicMock(wraps=pipeline.compact)

    def pressure_callback(pressure_value, threshold, kv_cache):
        if pressure_value > threshold:
            return compact_spy(kv_cache)
        return None

    cache = make_synthetic_cache()
    pressure_callback(0.9, 0.8, cache)   # above threshold -> compact fires
    pressure_callback(0.5, 0.8, cache)   # below threshold -> compact silent
    assert compact_spy.call_count == 1
```

### Anti-Patterns to Avoid
- **Calling `strip_rope_from_keys()` when `bundle_path=None`:** The function needs `rope_theta` from the bundle. Skipping is the correct behavior for the None path, consistent with both sub-compressor None contracts.
- **Returning the original-shape cache from `decompress()`:** AM compaction is lossy. The decompressed cache has `compacted_seq_len` tokens, not `logical_seq_len`. The `logical_seq_len` integer is returned separately for RoPE offset use.
- **Passing `AMCompactedCache` directly to `KVTCCompressor.compress()`:** `KVTCCompressor.compress()` expects `list[tuple[mx.array, mx.array]]`. Use `compacted.layers`, not the `AMCompactedCache` object itself.
- **Re-exporting from `__init__.py`:** The project pattern is intentionally empty `__init__.py`. Callers import: `from omlx.compression.pipeline import KVCachePipeline`.
- **Using bare `mx.linalg.*` inside `pipeline.py`:** Only `linalg_utils.py` may call raw MLX linalg ops. Pipeline does no math — it delegates.
- **Forgetting `_mx_materialize = mx.eval` alias:** All three existing files use this alias with the comment that it is NOT Python string evaluation. Pipeline must match the pattern.
- **Forgetting SPDX header:** `# SPDX-License-Identifier: Apache-2.0` must be the first line of both `pipeline.py` and `test_pipeline.py`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Token selection / attention fitting | Custom compaction logic | `AMCompactor.compact()` | All AM math lives in am.py; tested and correct |
| PCA projection + quantization + zstd | Custom compression | `KVTCCompressor.compress()` | All kvtc math lives in kvtc.py; blob format is self-describing |
| RoPE inversion | Custom inverse RoPE | `strip_rope_from_keys()` from calibrator.py | Handles both `traditional` and non-traditional splits correctly |
| Bundle loading | Custom npz parsing | `load_calibration_bundle()` from calibrator.py | Returns `(pca_bundle, head_entropy)` in the exact formats sub-compressors expect |
| Blob deserialization | Custom byte parser | `KVTCCompressor.decompress()` | Self-describing blob; no bundle needed at decompress time |

**Key insight:** Phase 5 adds zero new math. Every computation is delegated to Phase 2/3/4 components. The pipeline's value is sequencing, RoPE bridging, and the `PipelineBlob` envelope.

## Common Pitfalls

### Pitfall 1: RoPE Offset in Stripped Keys
**What goes wrong:** `strip_rope_from_keys()` is called with `offset=0` for calibration (fresh prefill from position 0). Compacted keys are a subset of original positions, not necessarily starting at position 0. Using `offset=0` may look wrong but is actually correct.
**Why it happens:** kvtc treats the compacted sequence as position-agnostic token vectors for PCA purposes. The kvtc paper and calibrator implementation assume RoPE has been stripped and do not interpret position indices after stripping.
**How to avoid:** Use `offset=0` consistently, matching how `calibrator.py` uses it. The kvtc compressor compresses the vector content, not positional structure.
**Warning signs:** Cosine similarity below 0.99 in round-trip tests with a real bundle suggests RoPE stripping is incorrect, not the offset.

### Pitfall 2: mx.array vs numpy at the RoPE Bridge
**What goes wrong:** `strip_rope_from_keys()` returns numpy float32. Passing it directly to `KVTCCompressor.compress()` causes failures because kvtc does `body_k.astype(mx.float32)` — an MLX method call that numpy arrays don't have.
**Why it happens:** The kvtc compressor expects `mx.array` tensors in the layer list.
**How to avoid:** Wrap numpy output of `strip_rope_from_keys()` back into `mx.array(keys_stripped_np.astype(np.float16))` before passing layers to `KVTCCompressor.compress()`. Cast to float16 to match the dtype convention of existing cache tensors.
**Warning signs:** `AttributeError: 'numpy.ndarray' object has no attribute 'astype'` with an MLX dtype argument.

### Pitfall 3: compaction_ratio Computation
**What goes wrong:** Using `1 / am_ratio` (the constructor parameter) instead of the actual ratio gives wrong observability data — AM may achieve a different ratio due to `n_sink_tokens` floor and rounding.
**Why it happens:** `AMCompactor._compact_head()` applies `budget = max(n_sink_tokens, min(budget, seq_len))` which can deviate from the nominal ratio for short sequences.
**How to avoid:** Compute `compaction_ratio = original_seq_len / compacted_seq_len` from the actual `compacted.layers[0][0].shape[2]` value after calling compact.
**Warning signs:** `compaction_ratio` does not match the empirical token reduction observed in round-trip tests.

### Pitfall 4: compact() vs compress() Return Type Confusion
**What goes wrong:** `compact()` returns `AMCompactedCache`; `compress()` returns `PipelineBlob`. Mixing these types causes attribute errors.
**Why it happens:** Two paths with similar names; both return named dataclasses with different fields.
**How to avoid:** Tests should assert `isinstance(result, PipelineBlob)` and `isinstance(result, AMCompactedCache)` respectively. `PipelineBlob` has `compressed`, `logical_seq_len`, `compaction_ratio`. `AMCompactedCache` has `layers`, `logical_seq_len`, `diagnostics`.
**Warning signs:** `AttributeError: 'AMCompactedCache' object has no attribute 'compressed'` in a test.

### Pitfall 5: Forgetting MLX Graph Materialization Before Numpy Bridge
**What goes wrong:** `np.array(keys)` on a lazy MLX array returns garbage or raises if the compute graph has not been flushed.
**Why it happens:** MLX uses deferred/lazy evaluation. Tensors are not materialized until `_mx_materialize()` is called.
**How to avoid:** Call `_mx_materialize(keys, values)` before any `np.array(...)` conversion in the RoPE stripping helper. Follow the same pattern as `am.py` line 244 and `kvtc.py` line 406.
**Warning signs:** Intermittent wrong values in numpy arrays; `mlx.core.mlx_error` at numpy bridge time.

## Code Examples

Verified patterns from official sources (reading actual project source files):

### AMCompactedCache fields (from am.py)
```python
# Source: omlx/compression/am.py
@dataclass
class AMCompactedCache:
    layers: list[tuple[mx.array, mx.array]]   # [1, n_heads, budget, head_dim] each
    logical_seq_len: int                        # original T before compaction
    diagnostics: Optional[dict] = None
```

### KVTCCompressor.compress() signature (from kvtc.py)
```python
# Source: omlx/compression/kvtc.py
def compress(self, kv_cache: list[tuple[mx.array, mx.array]]) -> bytes:
    # kv_cache: list of (keys, values) with shape [1, n_kv_heads, seq_len, head_dim]
    # Returns: self-describing bytes blob starting with b"KVTC" magic
    # Keys must have RoPE stripped by the caller before this call.
```

### KVTCCompressor.decompress() signature (from kvtc.py)
```python
# Source: omlx/compression/kvtc.py
def decompress(self, blob: bytes) -> list[tuple[mx.array, mx.array]]:
    # Returns float16 tensors, shape [1, n_kv_heads, seq_len, head_dim]
    # seq_len here = compacted_seq_len (post-AM), not original T
    # No pca_bundle needed -- blob is self-describing
```

### load_calibration_bundle() return (from calibrator.py)
```python
# Source: omlx/compression/calibrator.py
pca_bundle, head_entropy = load_calibration_bundle(path)
# pca_bundle: list[dict] -- one dict per layer
#   keys: K_basis, K_mean, K_sv, V_basis, V_mean, V_sv, k_bit_alloc, v_bit_alloc
# head_entropy: list[float] of length n_kv_heads
```

### strip_rope_from_keys() signature (from calibrator.py)
```python
# Source: omlx/compression/calibrator.py
def strip_rope_from_keys(keys, rope_theta, traditional, offset=0):
    # keys: float32 numpy [1, n_kv_heads, T, head_dim] with RoPE applied
    # Returns: float32 numpy array, same shape, RoPE removed
```

### Synthetic KV cache fixture pattern (from test_kvtc.py)
```python
# Source: tests/test_kvtc.py -- standard fixture shape for fast tests
import mlx.core as mx
mx.random.seed(42)
kv_cache = [
    (
        mx.random.uniform(shape=[1, 4, 300, 128]).astype(mx.float16),
        mx.random.uniform(shape=[1, 4, 300, 128]).astype(mx.float16),
    )
    for _ in range(2)
]
# seq_len=300 ensures non-empty body: 300 > n_sink(4) + window(128) = 132
```

### Cosine similarity check pattern (from test_am.py convention)
```python
import mlx.core as mx
import numpy as np

def cosine_sim(a, b):
    a_np = np.array(a.astype(mx.float32)).reshape(-1)
    b_np = np.array(b.astype(mx.float32)).reshape(-1)
    return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np) + 1e-8))

# Use >0.9 for fast CI tests on synthetic data (bundle=None path)
# The >0.998 contract (VAL-02) applies only to production bundle path, validated in Phase 7
assert cosine_sim(recovered_values, original_values) > 0.9
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single-stage KV compression | Two-stage AM->kvtc pipeline | Phase 5 | Multiplicative ratio: token reduction x byte compression |
| kvtc blob requires pca_bundle at decompress | Self-describing blob (all metadata embedded) | Phase 3 | SSD cold storage works without keeping bundle in memory |
| Manual callback registration per phase | Phase 5 delivers callable, Phase 6 wires | Architecture decision | Clean separation of algorithm from infrastructure |

**Deprecated/outdated:**
- Nothing from prior phases is deprecated. Phase 5 extends, does not replace.

## Open Questions

1. **Where do rope_theta and rope_traditional come from?**
   - What we know: `load_calibration_bundle()` returns `(pca_bundle, head_entropy)` only. No rope params in the `.npz` bundle as-is. `run_calibration()` reads these from `model.args` but does not persist them.
   - What's unclear: `strip_rope_from_keys()` requires `rope_theta` and `traditional`. These must come from somewhere at pipeline construction time.
   - Recommendation: Add `rope_theta: float = 10000.0, rope_traditional: bool = False` constructor params to `KVCachePipeline`. Defaults match Qwen 2.5 7B and are the most common values. Phase 6 passes actual model config values. This avoids modifying the Phase 4 bundle format.

2. **Should `compact()` accept a `ratio` override?**
   - What we know: CONTEXT.md marks this as Claude's discretion.
   - What's unclear: Whether Phase 6 needs dynamic ratio adjustment.
   - Recommendation: Accept `ratio: float | None = None` with `None` meaning "use constructor default". Adds minimal complexity, preserves constructor as the single configuration point.

3. **Cosine similarity threshold for fast CI tests**
   - What we know: VAL-02 requires >0.998 with real model data. Synthetic random data will not achieve this with bundle=None and on-the-fly PCA.
   - What's unclear: What threshold is achievable with synthetic data.
   - Recommendation: Use >0.9 for fast CI (bundle=None, synthetic data). Document the >0.998 contract applies only to the production bundle path and is validated in Phase 7.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (configured in pytest.ini) |
| Config file | `/Users/tonysina/projects/omlx/pytest.ini` |
| Quick run command | `uv run python -m pytest tests/test_pipeline.py -m "not slow" -q` |
| Full suite command | `uv run python -m pytest tests/test_pipeline.py -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PIPE-01 | compress() returns PipelineBlob with compaction_ratio > 1.0 and non-empty compressed bytes | unit | `uv run python -m pytest tests/test_pipeline.py::TestCompress::test_returns_pipeline_blob -x` | ❌ Wave 0 |
| PIPE-01 | combined ratio exceeds either stage alone | unit | `uv run python -m pytest tests/test_pipeline.py::TestCompress::test_multiplicative_ratio -x` | ❌ Wave 0 |
| PIPE-02 | decompress() returns (layers, logical_seq_len); layers shape matches compacted shape | unit | `uv run python -m pytest tests/test_pipeline.py::TestRoundTrip::test_round_trip_shape -x` | ❌ Wave 0 |
| PIPE-02 | cosine similarity of decompressed vs compacted values above threshold | unit | `uv run python -m pytest tests/test_pipeline.py::TestRoundTrip::test_round_trip_cosine_sim -x` | ❌ Wave 0 |
| PIPE-02 | logical_seq_len in blob matches original T | unit | `uv run python -m pytest tests/test_pipeline.py::TestRoundTrip::test_logical_seq_len_preserved -x` | ❌ Wave 0 |
| PIPE-03 | compact() fires above pressure threshold, silent below | unit (mock) | `uv run python -m pytest tests/test_pipeline.py::TestTriggerSemantics::test_compact_fires_above_threshold -x` | ❌ Wave 0 |
| PIPE-04 | compress() is the eviction-path callable | unit | `uv run python -m pytest tests/test_pipeline.py::TestTriggerSemantics::test_compress_callable -x` | ❌ Wave 0 |
| PIPE-05 | decompress() restores usable cache layers on miss | unit | `uv run python -m pytest tests/test_pipeline.py::TestTriggerSemantics::test_decompress_restores_layers -x` | ❌ Wave 0 |
| PIPE-02 | Full round-trip with Qwen 2.5 7B; inference continues | slow | `uv run python -m pytest tests/test_pipeline.py -m slow -v` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run python -m pytest tests/test_pipeline.py -m "not slow" -q`
- **Per wave merge:** `uv run python -m pytest tests/test_pipeline.py -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_pipeline.py` — covers all PIPE-01 through PIPE-05 tests (does not exist)
- [ ] `omlx/compression/pipeline.py` — the implementation file (does not exist, no stub)

*(No new framework install needed — pytest is already configured)*

## Sources

### Primary (HIGH confidence)
- `omlx/compression/am.py` — Read directly; `AMCompactor`, `AMCompactedCache`, `generate_reference_queries` APIs verified
- `omlx/compression/kvtc.py` — Read directly; `KVTCCompressor.compress()` and `.decompress()` signatures, blob format, and `_mx_materialize` alias verified
- `omlx/compression/calibrator.py` — Read directly; `load_calibration_bundle()` return type, `strip_rope_from_keys()` signature, rope params absence from bundle confirmed
- `tests/test_kvtc.py` — Read directly; fixture shapes, `@pytest.mark.slow` usage patterns verified
- `tests/test_am.py` — Read directly; test class structure, cosine similarity assertion pattern verified
- `pytest.ini` — Read directly; `slow` marker registered, default `-m "not slow and not integration"` confirmed

### Secondary (MEDIUM confidence)
- `.planning/phases/05-pipeline-assembly/05-CONTEXT.md` — All locked decisions verified against source files for consistency

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all imports read directly from source files; no new deps required
- Architecture: HIGH — all API signatures confirmed by reading implementation files; patterns derived from existing code not documentation
- Pitfalls: HIGH for structural issues (mx vs numpy bridge, materialization before numpy conversion); MEDIUM for rope offset semantics (behavior correct but exact impact on compacted keys not empirically tested in this research)

**Research date:** 2026-03-22
**Valid until:** 2026-06-22 (stable — no fast-moving dependencies)
