# Phase 7: Benchmark Suite - Research

**Researched:** 2026-03-23
**Domain:** ML benchmark harness, KV cache quality evaluation, multi-model validation
**Confidence:** HIGH

## Summary

Phase 7 builds a reproducible CLI benchmark (`omlx benchmark-compression`) that measures compression quality across four model families. The harness must produce a structured JSON report covering technical metrics (compression ratio, cosine similarity, decompression latency) and task-accuracy metrics (perplexity, GSM8K, MMLU, LITM recall).

The key technical challenge is correctly handling Gemma 3's Sliding Window Attention (SWA) layers. mlx-lm's `make_cache()` assigns `RotatingKVCache` to SWA layers and `KVCache` to global layers. The compression pipeline must detect and skip SWA layers at runtime using `isinstance(cache[i], RotatingKVCache)`. For DeepSeek R1 validation, the benchmark should target distill variants (Qwen2 or Llama backbone) — not the full MLA 671B model, which is explicitly out of scope.

The evaluation strategy uses `datasets` (already installed) for GSM8K and MMLU data, a custom LITM synthetic harness for recall testing, and `mlx_lm.perplexity.eval_ppl` for perplexity. KV cache injection for compressed inference uses `KVCache.state` setter and `generate_step(prompt_cache=...)`.

**Primary recommendation:** Implement a single `omlx/compression/benchmark.py` module with a standalone `BenchmarkRunner` class, plus a `tests/test_compression_benchmark.py` test scaffold following the established Wave 0 RED-state pattern.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| VAL-01 | Benchmark suite measures compression ratio, cosine similarity, downstream task accuracy, and decompression latency | BenchmarkRunner collects all metrics; JSON report structure defined below |
| VAL-02 | AM compaction at 4x maintains >0.998 attention output cosine similarity | Per-layer KV vector cosine similarity on Qwen 2.5 7B; AMCompactor.compact() returns compacted layers; compare KV vectors directly |
| VAL-03 | kvtc at 16x stays within 1 point of vanilla on GSM8K, MMLU, LITM | Two-pass evaluation: vanilla prefill + compressed prefill; compare accuracy; datasets library available |
| VAL-04 | Pipeline validated against Qwen 2.5 7B (GQA, spiked) | Standard GQA; no special handling; mlx_lm.load() works directly |
| VAL-05 | Pipeline validated against Llama 3.x 8B (GQA, popular baseline) | Standard GQA; no special handling; mlx_lm.load() works directly |
| VAL-06 | Pipeline validated against Gemma 3 variants (SWA handling) | SWA detection via isinstance(cache[i], RotatingKVCache); skip SWA layers; pattern documented below |
| VAL-07 | Pipeline validated against DeepSeek R1 (long reasoning chains) | Use distill variants (Qwen2/Llama backbone); not the MLA 671B model; standard GQA pipeline |
| VAL-08 | Benchmark results are reproducible via a single CLI command | Fixed seed via mx.random.seed(N) + np.random.seed(N); deterministic dataset sampling |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `mlx_lm` | 0.31.2 (pinned commit) | Model loading, prefill, generation | Already in project deps; `make_prompt_cache`, `generate_step`, `eval_ppl` all available |
| `datasets` | 4.8.0 | GSM8K and MMLU data loading | Already installed; HF Hub integration; dataset caching |
| `numpy` | >=1.24.0 | Cosine similarity computation, accuracy aggregation | Already in deps |
| `mlx.core` | >=0.31.1 | Tensor ops, random seed | Already in deps; `mx.random.seed()` for reproducibility |
| `zstandard` | >=0.21.0 | Already in kvtc; no change | Part of existing pipeline |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `mlx_lm.perplexity.eval_ppl` | same | Perplexity evaluation on wikitext | Use for VAL-01 perplexity metric |
| `mlx_lm.models.cache.KVCache` | same | State injection for compressed inference | `cache.state = (k, v)` setter injects decompressed layers |
| `mlx_lm.models.cache.RotatingKVCache` | same | SWA layer detection at runtime | `isinstance(cache[i], RotatingKVCache)` identifies SWA layers |
| `json` (stdlib) | -- | JSON report serialization | Deterministic output format |
| `time` (stdlib) | -- | Decompression latency measurement | `time.perf_counter()` for sub-millisecond resolution |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `datasets` + custom runner | `lm_eval` harness via `mlx_lm.evaluate` | `lm_eval` is not installed and adds a heavy optional dep; custom runner is simpler and sufficient for 3 tasks |
| Custom LITM | `HELMET` or `RULER` benchmark datasets | No standard public LITM dataset; synthetic generation is reproducible and controllable |

**Installation:**
```bash
# All deps already present -- no new dependencies required
# datasets, mlx, mlx-lm, numpy, zstandard already in pyproject.toml
```

## Architecture Patterns

### Recommended Project Structure
```
omlx/
├── compression/
│   ├── benchmark.py        # BenchmarkRunner class + CLI entry point
│   └── evaluators.py       # GSM8K, MMLU, LITM evaluator functions
omlx/
└── cli.py                  # Add benchmark-compression subparser

tests/
└── test_compression_benchmark.py  # Phase 7 test scaffold
```

### Pattern 1: KV Cache Injection for Compressed Inference
**What:** Inject decompressed KV layers into `KVCache` objects that `generate_step` can use.
**When to use:** Every compressed-path inference call.
**Example:**
```python
# Source: mlx_lm.models.cache.KVCache.state setter (verified in mlx_lm 0.31.2)
from mlx_lm.models.cache import KVCache, make_prompt_cache, RotatingKVCache

def make_compressed_prompt_cache(model, decompressed_layers):
    """Build a prompt_cache from decompressed KV layers."""
    prompt_cache = make_prompt_cache(model)
    for cache_entry, (keys, values) in zip(prompt_cache, decompressed_layers):
        if isinstance(cache_entry, RotatingKVCache):
            continue  # Skip SWA layers -- left empty
        # state setter sets keys, values, AND updates offset
        cache_entry.state = (keys, values)
    return prompt_cache

# Then pass to generate_step:
from mlx_lm.generate import generate_step
token_gen = generate_step(
    question_tokens, model,
    prompt_cache=compressed_prompt_cache,
    max_tokens=256,
)
```

### Pattern 2: SWA Layer Detection and Skip
**What:** Detect Gemma 3 sliding window attention layers and exclude them from compression.
**When to use:** Any multi-model benchmark path that includes Gemma 3.
**Example:**
```python
# Source: verified from mlx_lm.models.gemma3_text make_cache() and is_sliding logic
from mlx_lm.models.cache import RotatingKVCache

def get_compressible_layer_indices(prompt_cache):
    """Return indices of layers that use KVCache (not RotatingKVCache)."""
    return [
        i for i, c in enumerate(prompt_cache)
        if not isinstance(c, RotatingKVCache)
    ]

# For Gemma 3 with sliding_window_pattern=6:
# Layers 0,1,2,3,4 -> RotatingKVCache (SWA, skip compression)
# Layer 5          -> KVCache (global, compress)
# Layers 6,7,8,9,10 -> RotatingKVCache (SWA, skip)
# Layer 11         -> KVCache (global, compress)
```

### Pattern 3: Gemma 3 SWA Detection from config.json
**What:** Detect SWA layers before running inference, directly from model config.
**When to use:** When pre-computing which layers to compress, or for benchmark reporting.
**Example:**
```python
import json
from pathlib import Path

def detect_swa_layers(model_path, num_hidden_layers):
    """Return set of layer indices that are SWA layers for Gemma 3."""
    config_path = Path(model_path) / "config.json"
    with open(config_path) as f:
        config = json load(f)

    model_type = config.get("model_type", "")
    if model_type not in ("gemma3", "gemma3_text"):
        return set()  # No SWA for non-Gemma3 models

    pattern = config.get("sliding_window_pattern", 6)
    # A layer is SWA if: i % pattern != pattern - 1
    return {i for i in range(num_hidden_layers) if i % pattern != pattern - 1}
```

### Pattern 4: Reproducible Benchmark Seeding
**What:** Fix all random sources for deterministic results.
**When to use:** Top of `benchmark_compression_command()` handler.
**Example:**
```python
import mlx.core as mx
import numpy as np

def seed_all(seed: int = 42):
    mx.random.seed(seed)
    np.random.seed(seed)
    # No torch; no additional seeding needed
```

### Pattern 5: Decompression Latency Measurement
**What:** Measure `decompress()` wall time with MLX materialization forced.
**When to use:** Technical metrics section of report.
**Example:**
```python
import time
import mlx.core as mx

def measure_decompression_latency(pipeline, blob, n_layers, n_warmup=1):
    """Return latency in ms per layer."""
    # Warmup: force JIT compilation on first pass
    for _ in range(n_warmup):
        layers, _ = pipeline.decompress(blob)
        mx.eval(*[t for pair in layers for t in pair])

    # Timed run
    t0 = time.perf_counter()
    layers, _ = pipeline.decompress(blob)
    mx.eval(*[t for pair in layers for t in pair])  # Force MLX lazy execution
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return elapsed_ms / n_layers
```

### Pattern 6: Cosine Similarity at KV Vector Level
**What:** Measure per-stage fidelity of KV cache transformations.
**When to use:** VAL-02 (AM stage) and VAL-03 technical metrics (kvtc stage).
**Example:**
```python
import numpy as np

def cosine_sim_kv(layers_a, layers_b):
    """Mean cosine similarity across all layers/heads/tokens."""
    sims = []
    for (ka, va), (kb, vb) in zip(layers_a, layers_b):
        for tensor_a, tensor_b in [(ka, kb), (va, vb)]:
            a = np.array(tensor_a, dtype=np.float32).ravel()
            b = np.array(tensor_b, dtype=np.float32).ravel()
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom > 0:
                sims.append(float(np.dot(a, b) / denom))
    return float(np.mean(sims)) if sims else 0.0
```

### Pattern 7: GSM8K Accuracy Evaluation
**What:** 8-shot accuracy on GSM8K test set using generation.
**When to use:** VAL-03 quality metrics.
**Example:**
```python
from datasets import load_dataset
import re

def load_gsm8k(n_samples=200, seed=42):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), min(n_samples, len(ds)), replace=False)
    return [ds[int(i)] for i in indices]

def parse_gsm8k_answer(text):
    """Extract final numeric answer after '####'."""
    match = re.search(r'####\s*([\d,.-]+)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    return numbers[-1].replace(',', '') if numbers else None
```

### Pattern 8: LITM (Lost in the Middle) Recall Evaluation
**What:** Synthetic multi-document QA; measures whether compressed KV retains mid-context tokens.
**When to use:** VAL-03 quality metrics.
**Example:**
```python
def generate_litm_problems(n_problems=100, n_docs=20, seed=42):
    """Generate LITM problems with planted fact at middle position."""
    rng = np.random.default_rng(seed)
    problems = []
    for i in range(n_problems):
        target_pos = n_docs // 2  # Bury fact in the middle
        passages = [_gen_distractor_passage(rng) for _ in range(n_docs)]
        target_fact = f"The answer code is {rng.integers(1000, 9999)}."
        passages[target_pos] = target_fact
        question = "What is the answer code mentioned in the documents?"
        answer = str(rng.integers(1000, 9999))  # Extract from target_fact
        problems.append({"context": "\n\n".join(passages), "question": question, "answer": answer})
    return problems
```

### Pattern 9: Report JSON Structure
**What:** Standardized output format for `omlx benchmark-compression`.
**When to use:** Written to disk and printed to stdout on completion.
**Example:**
```python
report = {
    "schema_version": "1.0",
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "timestamp": "2026-03-23T10:00:00Z",
    "seed": 42,
    "config": {"am_ratio": 4.0, "kvtc_bits": 4.0},
    "technical_metrics": {
        "compression_ratio": 15.8,
        "am_cosine_similarity": 0.9983,
        "kvtc_cosine_similarity": 0.9951,
        "decompression_latency_ms_per_layer": 7.2,
    },
    "quality_metrics": {
        "perplexity_vanilla": 7.24,
        "perplexity_compressed": 7.31,
        "gsm8k_vanilla": 84.2,
        "gsm8k_compressed": 83.8,
        "mmlu_vanilla": 74.1,
        "mmlu_compressed": 73.8,
        "litm_vanilla": 0.82,
        "litm_compressed": 0.81,
    },
    "thresholds": {
        "am_cosine_sim_pass": True,   # > 0.998
        "gsm8k_delta_pass": True,     # abs diff <= 1.0 percentage point
        "mmlu_delta_pass": True,
        "litm_delta_pass": True,
        "latency_pass": True,         # < 10ms/layer
    },
    "swa_layers_skipped": [],         # Empty for non-Gemma3
    "overall_pass": True,
}
```

### Anti-Patterns to Avoid
- **Running benchmark without warmup:** MLX JIT compilation on first pass skews latency by 10-100x. Always run 1 warmup decompress before timing.
- **Comparing compressed vs vanilla KV shapes directly for Gemma 3:** SWA layers have different cache types; only compare global layers.
- **Using the full DeepSeek-R1 MLA model:** MLA uses `kv_lora_rank` compression natively, making the pipeline incompatible. Use distill variants (Qwen2/Llama backbone).
- **Not calling `mx.eval()` before timing decompression:** MLX is lazy; timing without forcing materialization measures graph build, not execution.
- **Measuring cosine similarity on zero-padded AM outputs:** AM pads short heads with zeros for concatenation. Compare unpadded tensors per-head or mask out padding.
- **Sharing a single cache object between vanilla and compressed paths:** `KVCache.state` setter is in-place. Always create fresh caches for each path.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Model loading | Custom mlx model loader | `mlx_lm.load()` | Handles tokenizer, config, weights, all model types |
| KV cache creation | Manual dict-based cache | `make_prompt_cache(model)` | Handles Gemma 3 SWA cache types automatically |
| Perplexity | Cross-entropy loop | `mlx_lm.perplexity.eval_ppl()` | Batch evaluation, standard error, correct reduction |
| Generation | Token sampling loop | `generate_step(prompt_cache=...)` | Handles EOS, KV growth, max_tokens, sampling |
| Dataset loading | Raw HF API calls | `load_dataset("openai/gsm8k", "main")` | Caching, splits, type-safe |

**Key insight:** mlx_lm provides `prompt_cache` injection via `generate_step` — this is the critical hook for benchmarking compressed inference without modifying the model.

## Common Pitfalls

### Pitfall 1: MLX Lazy Evaluation Corrupts Timing
**What goes wrong:** `time.perf_counter()` around `pipeline.decompress()` measures graph construction, not GPU execution. Result: 0.1ms reported instead of actual 7ms.
**Why it happens:** MLX uses deferred execution; tensors are computed only when needed (at materialization time, numpy conversion, etc.).
**How to avoid:** Always call `mx.eval(*tensors)` immediately after decompress before stopping the timer.
**Warning signs:** Suspiciously fast decompression (<1ms/layer for 8K context).

### Pitfall 2: Gemma 3 Cache Type Mismatch
**What goes wrong:** Injecting decompressed KV into `RotatingKVCache` objects (SWA layers) causes shape errors or silent quality collapse.
**Why it happens:** `RotatingKVCache` has a bounded circular buffer; it doesn't accept arbitrary-length state injection the same way `KVCache` does.
**How to avoid:** Use `isinstance(cache[i], RotatingKVCache)` guard; skip SWA layers entirely during compression and injection.
**Warning signs:** Quality collapse at specific layers with pattern period 6 (layers 0-4, 6-10, etc.).

### Pitfall 3: DeepSeek R1 MLA Model
**What goes wrong:** Loading `deepseek-ai/DeepSeek-R1` (full 671B) or attempting pipeline.compress() on MLA KV caches fails because MLA uses `kv_lora_rank` latent space, not standard key/value tensors.
**Why it happens:** MLA is explicitly out of scope (REQUIREMENTS.md Out of Scope section).
**How to avoid:** Target `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` or `DeepSeek-R1-Distill-Llama-8B` -- these use standard Qwen2/Llama GQA architectures.
**Warning signs:** `kv_lora_rank` attribute present in model config; `deepseek_v3` model_type.

### Pitfall 4: Non-Reproducible LITM Problems
**What goes wrong:** Different LITM problems generated on each run because document content is random without seeding.
**Why it happens:** `random` or `numpy.random` called without seed.
**How to avoid:** Use `np.random.default_rng(seed)` for all document generation; derive all content from the fixed seed. Store the generated problems alongside the report.
**Warning signs:** LITM accuracy changes between benchmark runs with same `--seed`.

### Pitfall 5: AM Cosine Similarity Measured on Wrong Quantities
**What goes wrong:** Comparing post-AM KV tensors against pre-AM tensors gives misleading similarity because AM token selection changes the sequence length.
**Why it happens:** After AM compaction, `layers[i][0].shape[2]` (compacted seq_len) != original seq_len. Direct tensor comparison fails on incompatible shapes.
**How to avoid:** For VAL-02, compare KV vectors per retained token index. Alternatively compare compacted vs decompressed (established project pattern per STATE.md). Do not compare original vs compacted directly -- shapes are incompatible.
**Warning signs:** Shape mismatch errors during cosine similarity computation.

### Pitfall 6: MMLU Subject Sampling Bias
**What goes wrong:** Sampling only from one MMLU subject introduces bias; reported accuracy may not reflect general capability.
**Why it happens:** `cais/mmlu` has 57 subjects; loading "all" is slow and may not be needed.
**How to avoid:** Sample uniformly across 5-10 representative subjects (e.g., `elementary_mathematics`, `high_school_physics`, `moral_scenarios`, `professional_medicine`, `computer_security`). Use fixed seed for selection.

### Pitfall 7: Zero-Padded AM Head Confusion
**What goes wrong:** Cosine similarity computed on zero-padded AM output heads is artificially high (zeros dominate numerator/denominator).
**Why it happens:** Non-uniform head budgets cause shorter heads to be padded with zeros before concatenation (STATE.md decision for AM Phase 2).
**How to avoid:** Slice to `head_budget[h]` tokens per head before cosine similarity, or use unpadded per-head computation.

## Code Examples

Verified patterns from official sources:

### Loading a Model and Extracting KV Cache
```python
# Source: mlx_lm.load() + make_prompt_cache (verified mlx_lm 0.31.2)
import mlx.core as mx
import mlx_lm
from mlx_lm.models.cache import make_prompt_cache

model, tokenizer = mlx_lm.load("Qwen/Qwen2.5-7B-Instruct")
cache = make_prompt_cache(model)

prompt_ids = tokenizer.encode("Solve: 3x + 7 = 22")
tokens = mx.array(prompt_ids)[None]  # [1, seq_len]
_ = model(tokens, cache=cache)
mx.eval(*[t for c in cache for t in [c.keys, c.values] if t is not None])

actual_seq_len = cache[0].offset
kv_cache = [
    (c.keys[:, :, :actual_seq_len, :], c.values[:, :, :actual_seq_len, :])
    for c in cache
]
```

### Injecting Decompressed KV and Running Generation
```python
# Source: mlx_lm.generate.generate_step + KVCache.state setter (verified)
from mlx_lm.models.cache import KVCache, make_prompt_cache, RotatingKVCache
from mlx_lm.generate import generate_step

def run_with_compressed_cache(model, tokenizer, question_text, decompressed_layers):
    prompt_cache = make_prompt_cache(model)
    for cache_obj, (keys, values) in zip(prompt_cache, decompressed_layers):
        if isinstance(cache_obj, RotatingKVCache):
            continue  # Skip SWA layers
        cache_obj.state = (keys, values)  # Sets keys, values, and offset

    question_ids = mx.array(tokenizer.encode(question_text))[None]
    tokens = []
    for token, _ in generate_step(
        question_ids.flatten(), model,
        prompt_cache=prompt_cache,
        max_tokens=256,
    ):
        if token in tokenizer.eos_token_ids:
            break
        tokens.append(token)
    return tokenizer.decode(tokens)
```

### Gemma 3 SWA Detection
```python
# Source: mlx_lm.models.gemma3_text make_cache() and Attention.__init__ (verified)
# is_swa = (layer_idx + 1) % sliding_window_pattern != 0  (from Attention.__init__)
# make_cache uses: i % pattern == pattern - 1 for global (equivalent formula)

def build_compressible_layer_mask(prompt_cache):
    """Return bool list: True = can compress, False = SWA (skip)."""
    from mlx_lm.models.cache import RotatingKVCache
    return [not isinstance(c, RotatingKVCache) for c in prompt_cache]
```

### GSM8K Few-Shot Prompt Format
```python
# Standard format for GSM8K few-shot evaluation
SYSTEM_PROMPT = "Solve the following math problem step by step. End with: #### <answer>"

def format_gsm8k_prompt(problem_text, few_shot_examples):
    shots = "\n\n".join(
        f"Problem: {ex['question']}\nSolution: {ex['answer']}"
        for ex in few_shot_examples
    )
    return f"{SYSTEM_PROMPT}\n\n{shots}\n\nProblem: {problem_text}\nSolution:"
```

### Adding benchmark-compression CLI Subcommand
```python
# Source: omlx/cli.py pattern from calibrate-kv (verified)
bench_parser = subparsers.add_parser(
    "benchmark-compression",
    help="Run compression quality benchmark suite",
)
bench_parser.add_argument("model", type=str,
    help="Model path or HuggingFace repo ID")
bench_parser.add_argument("--bundle", type=str, default=None,
    help="PCA calibration bundle path (.npz)")
bench_parser.add_argument("--seed", type=int, default=42)
bench_parser.add_argument("--n-samples", type=int, default=200,
    help="Problems per task (default: 200)")
bench_parser.add_argument("--output", type=str, default=None,
    help="JSON report output path")
bench_parser.add_argument("--tasks", nargs="+",
    default=["cosine_sim", "perplexity", "gsm8k", "mmlu", "litm"],
    help="Tasks to run")
bench_parser.add_argument("--am-ratio", type=float, default=4.0)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Run lm-eval harness externally | Embed lightweight evaluators in-process | This phase | No extra dep; faster iteration |
| Manual cache dict for inference | `generate_step(prompt_cache=...)` | mlx_lm 0.x | Clean cache injection API |
| Separate calibration + benchmark scripts | Single `omlx benchmark-compression` CLI | This phase | Reproducible with one command |

**Deprecated/outdated:**
- `mlx_lm.evaluate` module: Requires `lm_eval` (not installed); use `mlx_lm.perplexity.eval_ppl` for perplexity and custom runners for GSM8K/MMLU/LITM instead.

## Open Questions

1. **AM cosine similarity measurement for VAL-02**
   - What we know: VAL-02 requires >0.998 on attention output cosine similarity. AMCompactor returns `AMCompactedCache` without attention output vectors (only compacted KV tensors). Diagnostics dict is `None` by default.
   - What's unclear: Should VAL-02 compare (a) attention output vectors requiring a forward pass with reference queries, or (b) KV vector cosine similarity between vanilla and compacted tensors?
   - Recommendation: Use KV vector cosine similarity (compacted vs decompressed round-trip, following established project pattern from STATE.md). If threshold not met with this approach, escalate to attention output comparison.

2. **DeepSeek R1 model identifier**
   - What we know: Full DeepSeek-R1 uses MLA (out of scope). Distill variants use Qwen2/Llama.
   - What's unclear: Which exact model HF ID to use for VAL-07 -- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` or `DeepSeek-R1-Distill-Llama-8B`? Long context requirement suggests Qwen2 variant.
   - Recommendation: Use `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (7B, Qwen2 backbone, commonly available, fits in 16GB). Mark as `@pytest.mark.slow`.

3. **LITM dataset design**
   - What we know: No standard public LITM dataset exists. Must be synthetic. Recall accuracy is the metric.
   - What's unclear: Optimal context length and document count for discriminating compressed vs vanilla.
   - Recommendation: 20 passages of ~100 tokens each (~2K total context), target buried at position 10 (middle). Use substring match for answer extraction.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 7.x + pytest-asyncio |
| Config file | `pytest.ini` (project root) |
| Quick run command | `pytest tests/test_compression_benchmark.py -m "not slow" -v` |
| Full suite command | `pytest tests/test_compression_benchmark.py -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| VAL-01 | BenchmarkRunner produces report dict with all required fields | unit | `pytest tests/test_compression_benchmark.py::TestBenchmarkReport -x` | No -- Wave 0 |
| VAL-02 | AM cosine similarity > 0.998 on Qwen 2.5 7B | slow | `pytest tests/test_compression_benchmark.py::TestSlowQwen::test_am_cosine_sim -m slow` | No -- Wave 0 |
| VAL-03 | Accuracy delta <= 1pt on GSM8K/MMLU/LITM | slow | `pytest tests/test_compression_benchmark.py::TestSlowQwen::test_task_accuracy -m slow` | No -- Wave 0 |
| VAL-04 | Qwen 2.5 7B runs without error | slow | `pytest tests/test_compression_benchmark.py::TestSlowQwen -m slow` | No -- Wave 0 |
| VAL-05 | Llama 3.x 8B runs without error | slow | `pytest tests/test_compression_benchmark.py::TestSlowLlama -m slow` | No -- Wave 0 |
| VAL-06 | Gemma 3 SWA layers detected and skipped | unit + slow | `pytest tests/test_compression_benchmark.py::TestSwaDetection -x` | No -- Wave 0 |
| VAL-07 | DeepSeek R1 Distill Qwen 7B runs without error | slow | `pytest tests/test_compression_benchmark.py::TestSlowDeepSeek -m slow` | No -- Wave 0 |
| VAL-08 | Two runs with same seed produce identical report | unit | `pytest tests/test_compression_benchmark.py::TestReproducibility -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_compression_benchmark.py -m "not slow" -x`
- **Per wave merge:** `pytest tests/test_compression_benchmark.py -m "not slow" -v`
- **Phase gate:** Full suite green (including slow) before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_compression_benchmark.py` -- covers all VAL requirements above
- [ ] `omlx/compression/benchmark.py` -- BenchmarkRunner stub
- [ ] `omlx/compression/evaluators.py` -- GSM8K, MMLU, LITM evaluator stubs

*(Framework install: already present -- pytest in dev deps, no new requirements)*

## Sources

### Primary (HIGH confidence)
- mlx_lm source code (v0.31.2, commit 564281f) -- `make_prompt_cache`, `generate_step`, `KVCache.state`, `eval_ppl`, `gemma3_text.make_cache`, SWA layer pattern all verified by direct inspection
- `omlx/compression/pipeline.py` -- `KVCachePipeline.compress()`, `decompress()`, `PipelineBlob` API verified by reading source
- `omlx/compression/am.py` -- `AMCompactor.compact()`, `AMCompactedCache` fields verified by reading source
- `pyproject.toml` -- dependency versions, no `lm_eval` present, `datasets` present

### Secondary (MEDIUM confidence)
- `.planning/STATE.md` decisions -- confirmed cosine similarity comparison strategy (compacted vs decompressed, not original vs decompressed)
- `tests/test_pipeline.py` -- confirmed test patterns and thresholds used in prior phases
- HuggingFace `openai/gsm8k` and `cais/mmlu` datasets -- standard benchmark datasets, widely documented

### Tertiary (LOW confidence)
- DeepSeek R1 distill variant identification -- inferred from MLA out-of-scope requirement and known model families; not directly verified by running on device

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries verified installed; API signatures checked directly
- Architecture: HIGH -- KVCache injection via `.state` setter verified in mlx_lm source; SWA detection verified from gemma3_text source
- Pitfalls: HIGH -- MLX lazy eval pitfall confirmed by prior phase decisions in STATE.md; SWA type detection verified
- Gemma 3 SWA pattern: HIGH -- verified directly from mlx_lm.models.gemma3_text source; both detection methods (config.json and isinstance) confirmed
- DeepSeek R1 variant: MEDIUM -- MLA exclusion confirmed; distill variant recommendation is standard knowledge

**Research date:** 2026-03-23
**Valid until:** 2026-04-23 (mlx_lm is pinned to a commit; stable for 30 days)
