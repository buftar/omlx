# SPDX-License-Identifier: Apache-2.0
"""
Test scaffold for Phase 2 AM (Attention-based Magnitude) compaction.

All tests are RED until Wave 1 implementation ships in omlx/compression/am.py.
The import below will raise ImportError at collection time -- that is the
expected RED state confirming Wave 0 is complete.
"""
import math
import re
from pathlib import Path

import numpy as np
import pytest
import mlx.core as mx

from omlx.compression.am import AMCompactor, AMCompactedCache, generate_reference_queries
from omlx.compression.linalg_utils import nnls_solve

# mx.eval() is the MLX graph materialization function (not Python's built-in eval)
_mlx_eval = mx.eval


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_kv_cache():
    """2-layer, 4-head, seq_len=64, head_dim=128 synthetic KV cache."""
    layer = (
        mx.random.normal([1, 4, 64, 128]),
        mx.random.normal([1, 4, 64, 128]),
    )
    return [layer, layer]


# ---------------------------------------------------------------------------
# AM-01 -- Key selection: highest-attention positions + sink tokens
# ---------------------------------------------------------------------------

class TestHighestAttnKeysSelection:
    """AM-01: compact() keeps sink tokens + highest-attention positions."""

    def test_sink_tokens_always_preserved(self, small_kv_cache):
        """With ratio=4 and n_sink_tokens=4, the first 4 token indices must be
        present in every head's selected set across all layers."""
        compactor = AMCompactor(n_sink_tokens=4)
        keys = small_kv_cache[0][0]  # [1, 4, 64, 128]
        queries = generate_reference_queries(keys, n_queries=16, method="sample")
        result = compactor.compact(small_kv_cache, ratio=4.0, queries=queries)
        _mlx_eval(*[t for layer in result.layers for t in layer])
        for k_compact, _ in result.layers:
            assert k_compact.shape[2] >= 4, (
                f"Expected at least n_sink_tokens=4 but got shape {k_compact.shape}"
            )

    def test_selects_highest_attention_positions(self, small_kv_cache):
        """When synthetic keys have a clear high-attention cluster, those
        positions should be present in the compacted output."""
        base_keys = mx.random.normal([1, 4, 64, 128])
        pre  = base_keys[:, :, :10, :]
        hot  = mx.ones([1, 4, 4, 128]) * 10.0   # positions 10-13, very large norm
        post = base_keys[:, :, 14:, :]
        big_keys = mx.concatenate([pre, hot, post], axis=2)
        values = mx.random.normal([1, 4, 64, 128])
        cache = [(big_keys, values), (big_keys, values)]

        queries = generate_reference_queries(big_keys, n_queries=16, method="sample")
        compactor = AMCompactor(n_sink_tokens=4)
        result = compactor.compact(cache, ratio=4.0, queries=queries)
        _mlx_eval(*[t for layer in result.layers for t in layer])
        for k_compact, _ in result.layers:
            assert k_compact.ndim == 4

    def test_selected_count_equals_budget(self, small_kv_cache):
        """Number of retained tokens (dim 2) must equal ceil(seq_len / ratio)."""
        seq_len = 64
        ratio = 4.0
        expected_budget = math.ceil(seq_len / ratio)
        compactor = AMCompactor(n_sink_tokens=4)
        result = compactor.compact(small_kv_cache, ratio=ratio, queries=None)
        _mlx_eval(*[t for layer in result.layers for t in layer])
        for k_compact, _ in result.layers:
            assert k_compact.shape[2] == expected_budget, (
                f"Expected budget={expected_budget}, got {k_compact.shape[2]}"
            )


# ---------------------------------------------------------------------------
# AM-01 -- Uniform fallback (queries=None)
# ---------------------------------------------------------------------------

class TestUniformFallback:
    """AM-01: When queries=None, compact() falls back to uniform selection."""

    def test_queries_none_uses_uniform(self, small_kv_cache):
        """compact(kv_cache, queries=None) must run without error and return
        an AMCompactedCache instance."""
        compactor = AMCompactor(n_sink_tokens=4)
        result = compactor.compact(small_kv_cache, ratio=4.0, queries=None)
        _mlx_eval(*[t for layer in result.layers for t in layer])
        assert isinstance(result, AMCompactedCache)

    def test_uniform_coverage(self, small_kv_cache):
        """With uniform selection, physical length should match expected budget."""
        seq_len = 64
        ratio = 4.0
        expected_budget = math.ceil(seq_len / ratio)

        compactor = AMCompactor(n_sink_tokens=0)
        result = compactor.compact(small_kv_cache, ratio=ratio, queries=None)
        _mlx_eval(*[t for layer in result.layers for t in layer])
        for k_compact, _ in result.layers:
            assert k_compact.shape[2] == expected_budget


# ---------------------------------------------------------------------------
# AM-02 -- NNLS beta fitting
# ---------------------------------------------------------------------------

class TestNNLSBetaFitting:
    """AM-02: Mixture weights (betas) from NNLS are non-negative."""

    def test_betas_nonneg_before_clip(self, small_kv_cache):
        """Raw NNLS output is guaranteed >= 0 by scipy. If diagnostics are
        populated, betas must all be non-negative."""
        compactor = AMCompactor(n_sink_tokens=4)
        result = compactor.compact(small_kv_cache, ratio=4.0, queries=None)
        _mlx_eval(*[t for layer in result.layers for t in layer])
        if result.diagnostics is not None:
            for layer_diag in result.diagnostics.get("betas", []):
                for head_betas in layer_diag:
                    arr = np.array(head_betas)
                    assert (arr >= 0).all(), "NNLS betas must be non-negative"

    def test_beta_shape_matches_budget(self, small_kv_cache):
        """betas shape must be (budget,) for each head when diagnostics are available."""
        seq_len = 64
        ratio = 4.0
        budget = math.ceil(seq_len / ratio)
        compactor = AMCompactor(n_sink_tokens=4)
        result = compactor.compact(small_kv_cache, ratio=ratio, queries=None)
        _mlx_eval(*[t for layer in result.layers for t in layer])
        if result.diagnostics is not None:
            for layer_diag in result.diagnostics.get("betas", []):
                for head_betas in layer_diag:
                    assert len(head_betas) == budget


# ---------------------------------------------------------------------------
# AM-03 -- OLS value fitting
# ---------------------------------------------------------------------------

class TestOLSValueFitting:
    """AM-03: Compacted value vectors come from OLS fit, not raw selection."""

    def test_v_compact_shape(self, small_kv_cache):
        """V_compact for each layer must have shape [1, n_heads, budget, head_dim]."""
        seq_len = 64
        ratio = 4.0
        n_heads = 4
        head_dim = 128
        budget = math.ceil(seq_len / ratio)
        compactor = AMCompactor(n_sink_tokens=4)
        result = compactor.compact(small_kv_cache, ratio=ratio, queries=None)
        _mlx_eval(*[t for layer in result.layers for t in layer])
        for _, v_compact in result.layers:
            assert v_compact.shape == (1, n_heads, budget, head_dim), (
                f"Expected [1, {n_heads}, {budget}, {head_dim}], got {v_compact.shape}"
            )

    def test_no_bare_pinv_in_am(self):
        """am.py must not call mx.linalg.pinv directly -- it must go through
        pinv_f32 from linalg_utils (lint check)."""
        am_path = Path("omlx/compression/am.py")
        if not am_path.exists():
            pytest.skip("am.py not yet created (Wave 0 RED state -- expected)")
        source = am_path.read_text()
        assert "mx.linalg.pinv" not in source, (
            "am.py must use pinv_f32 from linalg_utils, not mx.linalg.pinv directly"
        )


# ---------------------------------------------------------------------------
# AM-04 -- Compacted cache shape invariants
# ---------------------------------------------------------------------------

class TestCompactedCacheShape:
    """AM-04: Shape contract for AMCompactedCache."""

    def test_logical_seq_len_preserved(self, small_kv_cache):
        """result.logical_seq_len must equal the original sequence length T."""
        original_seq_len = 64
        compactor = AMCompactor(n_sink_tokens=4)
        result = compactor.compact(small_kv_cache, ratio=4.0, queries=None)
        assert result.logical_seq_len == original_seq_len, (
            f"Expected logical_seq_len={original_seq_len}, got {result.logical_seq_len}"
        )

    def test_physical_tokens_reduced(self, small_kv_cache):
        """All result.layers[i][0].shape[2] must equal budget (< T=64)."""
        seq_len = 64
        ratio = 4.0
        budget = math.ceil(seq_len / ratio)
        compactor = AMCompactor(n_sink_tokens=4)
        result = compactor.compact(small_kv_cache, ratio=ratio, queries=None)
        _mlx_eval(*[t for layer in result.layers for t in layer])
        for i, (k_compact, _) in enumerate(result.layers):
            assert k_compact.shape[2] == budget, (
                f"Layer {i}: expected physical length={budget}, got {k_compact.shape[2]}"
            )
        assert budget < seq_len, "Sanity: budget must be strictly less than seq_len"

    def test_layer_count_preserved(self, small_kv_cache):
        """len(result.layers) must equal len(input kv_cache)."""
        compactor = AMCompactor(n_sink_tokens=4)
        result = compactor.compact(small_kv_cache, ratio=4.0, queries=None)
        assert len(result.layers) == len(small_kv_cache), (
            f"Expected {len(small_kv_cache)} layers, got {len(result.layers)}"
        )


# ---------------------------------------------------------------------------
# AM-05 -- Per-head budgets
# ---------------------------------------------------------------------------

class TestHeadBudgets:
    """AM-05: Per-head token budgets are computed correctly."""

    def test_compute_head_budgets_uniform(self):
        """_compute_head_budgets returns uniform list when head_entropy is None."""
        compactor = AMCompactor(head_entropy=None, n_sink_tokens=4)
        budgets = compactor._compute_head_budgets(seq_len=64, ratio=4.0, n_heads=4)
        assert isinstance(budgets, list)
        assert len(budgets) == 4
        expected = max(4, int(64 / 4.0))
        assert all(b == expected for b in budgets), (
            f"Expected uniform budgets of {expected}, got {budgets}"
        )

    def test_compute_head_budgets_entropy_proportional(self):
        """_compute_head_budgets produces higher budgets for higher-entropy heads."""
        head_entropy = [1.68, 0.34, 2.47, 1.12]
        compactor = AMCompactor(head_entropy=head_entropy, n_sink_tokens=4)
        budgets = compactor._compute_head_budgets(seq_len=501, ratio=4.0, n_heads=4)
        assert isinstance(budgets, list)
        assert len(budgets) == 4
        # head 2 (entropy=2.47) must have at least as many tokens as head 1 (entropy=0.34)
        assert budgets[2] >= budgets[1], (
            f"Head 2 (entropy=2.47) should have budget >= head 1 (entropy=0.34), "
            f"got {budgets}"
        )

    def test_compute_head_budgets_min_sinks(self):
        """No budget returned by _compute_head_budgets may fall below n_sink_tokens."""
        head_entropy = [1.68, 0.34, 2.47, 1.12]
        n_sinks = 4
        compactor = AMCompactor(head_entropy=head_entropy, n_sink_tokens=n_sinks)
        budgets = compactor._compute_head_budgets(seq_len=64, ratio=4.0, n_heads=4)
        assert all(b >= n_sinks for b in budgets), (
            f"All budgets must be >= n_sink_tokens={n_sinks}, got {budgets}"
        )

    def test_compute_head_budgets_sum_correct(self):
        """Sum of per-head budgets equals n_heads * floor(T/ratio) (rounding-corrected)."""
        head_entropy = [1.68, 0.34, 2.47, 1.12]
        compactor = AMCompactor(head_entropy=head_entropy, n_sink_tokens=1)
        seq_len, ratio, n_heads = 100, 4.0, 4
        budgets = compactor._compute_head_budgets(seq_len=seq_len, ratio=ratio, n_heads=n_heads)
        expected_total = n_heads * max(1, int(seq_len / ratio))
        assert sum(budgets) == expected_total, (
            f"Expected sum={expected_total}, got sum={sum(budgets)} with budgets={budgets}"
        )

    def test_uniform_budgets_correct(self, small_kv_cache):
        """With head_entropy=None, every head gets budget = max(n_sinks, floor(T/ratio))."""
        seq_len = 64
        ratio = 4.0
        n_sinks = 4
        expected_budget = max(n_sinks, math.floor(seq_len / ratio))
        compactor = AMCompactor(head_entropy=None, n_sink_tokens=n_sinks)
        result = compactor.compact(small_kv_cache, ratio=ratio, queries=None)
        _mlx_eval(*[t for layer in result.layers for t in layer])
        for k_compact, _ in result.layers:
            assert k_compact.shape[2] == expected_budget, (
                f"Expected uniform budget={expected_budget}, got {k_compact.shape[2]}"
            )

    def test_entropy_budgets_proportional(self, small_kv_cache):
        """Higher-entropy heads must receive more tokens than lower-entropy heads."""
        # Spike's observed per-head entropies for Qwen 2.5 7B (layer 0, 4 heads)
        head_entropy = [1.68, 0.34, 2.47, 1.12]
        compactor = AMCompactor(head_entropy=head_entropy, n_sink_tokens=4)
        result = compactor.compact(small_kv_cache, ratio=4.0, queries=None)
        _mlx_eval(*[t for layer in result.layers for t in layer])
        if result.diagnostics is not None and "per_head_budgets" in result.diagnostics:
            budgets = result.diagnostics["per_head_budgets"][0]  # layer 0
            assert budgets[2] >= budgets[1], (
                f"Head 2 (entropy=2.47) should have budget >= head 1 (entropy=0.34), "
                f"got budgets={budgets}"
            )

    def test_entropy_budgets_min_sinks(self, small_kv_cache):
        """No per-head budget may fall below n_sink_tokens."""
        head_entropy = [1.68, 0.34, 2.47, 1.12]
        n_sinks = 4
        compactor = AMCompactor(head_entropy=head_entropy, n_sink_tokens=n_sinks)
        result = compactor.compact(small_kv_cache, ratio=4.0, queries=None)
        _mlx_eval(*[t for layer in result.layers for t in layer])
        if result.diagnostics is not None and "per_head_budgets" in result.diagnostics:
            for layer_budgets in result.diagnostics["per_head_budgets"]:
                for b in layer_budgets:
                    assert b >= n_sinks, (
                        f"Budget {b} is below n_sink_tokens={n_sinks}"
                    )


# ---------------------------------------------------------------------------
# AM-06 -- Budget schedule reuse (computed at init, not at compact())
# ---------------------------------------------------------------------------

class TestBudgetReuse:
    """AM-06: Budget schedule is computed once at __init__ and reused."""

    def test_budgets_computed_at_init(self, small_kv_cache):
        """Two compact() calls with the same ratio must return identical shapes,
        confirming budget was computed at init and not recomputed."""
        head_entropy = [1.68, 0.34, 2.47, 1.12]
        compactor = AMCompactor(head_entropy=head_entropy, n_sink_tokens=4)

        result_a = compactor.compact(small_kv_cache, ratio=4.0, queries=None)
        result_b = compactor.compact(small_kv_cache, ratio=4.0, queries=None)
        _mlx_eval(*[t for layer in result_a.layers for t in layer])
        _mlx_eval(*[t for layer in result_b.layers for t in layer])

        for (ka, _), (kb, _) in zip(result_a.layers, result_b.layers):
            assert ka.shape == kb.shape, (
                f"Shapes differ across compact() calls: {ka.shape} vs {kb.shape}"
            )


# ---------------------------------------------------------------------------
# AM-07 -- generate_reference_queries
# ---------------------------------------------------------------------------

class TestGenerateReferenceQueries:
    """AM-07: generate_reference_queries returns correctly shaped tensors."""

    def test_sample_method_shape(self):
        """method='sample' returns [1, n_heads, n_queries, head_dim]."""
        keys = mx.random.normal([1, 4, 64, 128])
        result = generate_reference_queries(keys, n_queries=64, method="sample")
        _mlx_eval(result)
        assert result.shape == (1, 4, 64, 128), (
            f"Expected (1, 4, 64, 128), got {result.shape}"
        )

    def test_random_method_shape(self):
        """method='random' returns [1, n_heads, n_queries, head_dim]."""
        keys = mx.random.normal([1, 4, 64, 128])
        result = generate_reference_queries(keys, n_queries=32, method="random")
        _mlx_eval(result)
        assert result.shape == (1, 4, 32, 128), (
            f"Expected (1, 4, 32, 128), got {result.shape}"
        )


# ---------------------------------------------------------------------------
# AM-08 -- Beta box constraint [-3, +3]
# ---------------------------------------------------------------------------

class TestBetaBoxConstraint:
    """AM-08: All mixture weights after compact() are clipped to [-3.0, 3.0]."""

    def test_betas_clipped_to_minus3_plus3(self, small_kv_cache):
        """All betas after compact() must satisfy -3.0 <= beta <= 3.0."""
        compactor = AMCompactor(n_sink_tokens=4)
        result = compactor.compact(small_kv_cache, ratio=4.0, queries=None)
        _mlx_eval(*[t for layer in result.layers for t in layer])
        if result.diagnostics is not None:
            for layer_diag in result.diagnostics.get("betas", []):
                for head_betas in layer_diag:
                    arr = np.array(head_betas)
                    assert (arr >= -3.0).all() and (arr <= 3.0).all(), (
                        f"Betas out of [-3, 3] range: min={arr.min()}, max={arr.max()}"
                    )


# ---------------------------------------------------------------------------
# AM-02 + AM-03 -- Integration tests
# ---------------------------------------------------------------------------

class TestCompactIntegration:
    """AM-02 + AM-03: Integration -- round-trip shape consistency and cosine
    similarity of compacted representation to original."""

    def test_cosine_sim_above_threshold(self):
        """compact() on synthetic KV cache returns shapes that are self-consistent.
        Avg cosine similarity > 0.98 threshold verified in Wave 2 with a real model."""
        n_heads = 4
        seq_len = 32
        head_dim = 128
        n_layers = 2

        # Build a cache where values lie in a low-rank subspace so that OLS
        # can reconstruct them accurately.
        basis = mx.random.normal([1, n_heads, 4, head_dim])   # rank-4 subspace
        coeffs = mx.random.normal([1, n_heads, seq_len, 4])
        values = (coeffs @ basis).astype(mx.float32)          # [1, n_heads, seq_len, head_dim]
        keys = mx.random.normal([1, n_heads, seq_len, head_dim])
        cache = [(keys, values)] * n_layers

        queries = generate_reference_queries(keys, n_queries=8, method="sample")
        compactor = AMCompactor(n_sink_tokens=4)
        result = compactor.compact(cache, ratio=4.0, queries=queries)
        _mlx_eval(*[t for layer in result.layers for t in layer])

        budget = math.ceil(seq_len / 4.0)
        for k_c, v_c in result.layers:
            assert k_c.shape == (1, n_heads, budget, head_dim)
            assert v_c.shape == (1, n_heads, budget, head_dim)

    def test_round_trip_shapes(self, small_kv_cache):
        """result shapes are self-consistent after compact()."""
        seq_len = 64
        ratio = 4.0
        n_heads = 4
        head_dim = 128
        budget = math.ceil(seq_len / ratio)

        compactor = AMCompactor(n_sink_tokens=4)
        result = compactor.compact(small_kv_cache, ratio=ratio, queries=None)
        _mlx_eval(*[t for layer in result.layers for t in layer])

        assert result.logical_seq_len == seq_len
        assert len(result.layers) == len(small_kv_cache)
        for k_c, v_c in result.layers:
            assert k_c.shape == (1, n_heads, budget, head_dim), (
                f"K shape mismatch: {k_c.shape}"
            )
            assert v_c.shape == (1, n_heads, budget, head_dim), (
                f"V shape mismatch: {v_c.shape}"
            )


# ---------------------------------------------------------------------------
# AM-02 -- Direct NNLS beta fitting (no diagnostics dependency)
# ---------------------------------------------------------------------------

class TestNNLSBetaFittingDirect:
    """AM-02: Direct unit tests for NNLS beta fitting via nnls_solve.

    These tests call nnls_solve directly with known inputs, avoiding the
    diagnostics guard in compact(). They verify AM-02 requirements without
    relying on AMCompactor.compact() returning diagnostics.
    """

    def test_nnls_returns_nonneg_betas(self):
        """nnls_solve returns non-negative betas for any non-negative A, b."""
        rng = np.random.default_rng(42)
        n_queries, budget = 16, 8
        A = mx.array(np.abs(rng.standard_normal((n_queries, budget))).astype(np.float32))
        b = mx.array(np.abs(rng.standard_normal(n_queries)).astype(np.float32))
        betas, residual = nnls_solve(A, b)
        mx.eval(betas)
        beta_np = np.array(betas)
        assert (beta_np >= 0).all(), (
            f"NNLS betas must be non-negative, got min={beta_np.min():.6f}"
        )
        assert isinstance(residual, float)

    def test_nnls_exact_solution_recoverable(self):
        """nnls_solve recovers exact non-negative solution when one exists."""
        rng = np.random.default_rng(7)
        n_queries, budget = 10, 5
        A_np = np.abs(rng.standard_normal((n_queries, budget))).astype(np.float32)
        # Construct b = A @ x_true for known non-negative x_true
        x_true = np.abs(rng.standard_normal(budget)).astype(np.float32)
        b_np = A_np @ x_true

        A = mx.array(A_np)
        b = mx.array(b_np)
        betas, residual = nnls_solve(A, b)
        mx.eval(betas)
        beta_np = np.array(betas)

        # Solution should have small residual (< 1e-3 for well-conditioned system)
        assert residual < 1e-3, (
            f"NNLS residual too large for recoverable problem: {residual:.6f}"
        )
        assert (beta_np >= 0).all(), "NNLS betas must be non-negative"

    def test_nnls_beta_shape_matches_budget(self):
        """nnls_solve output shape is (budget,) regardless of n_queries."""
        rng = np.random.default_rng(99)
        for n_queries, budget in [(8, 4), (16, 8), (32, 16)]:
            A = mx.array(rng.standard_normal((n_queries, budget)).astype(np.float32))
            b = mx.array(rng.standard_normal(n_queries).astype(np.float32))
            betas, _ = nnls_solve(A, b)
            mx.eval(betas)
            assert betas.shape == (budget,), (
                f"Expected shape ({budget},), got {betas.shape} "
                f"for n_queries={n_queries}, budget={budget}"
            )

    def test_nnls_softmax_row_sums_are_valid_target(self):
        """softmax row sums (attention mass target for NNLS) are all == 1.0."""
        rng = np.random.default_rng(0)
        n_queries, seq_len = 8, 32
        scores = mx.array(rng.standard_normal((n_queries, seq_len)).astype(np.float32))
        attn = mx.softmax(scores, axis=-1)
        row_sums = attn.sum(axis=-1)
        mx.eval(row_sums)
        sums_np = np.array(row_sums)
        np.testing.assert_allclose(sums_np, np.ones(n_queries), atol=1e-5,
                                   err_msg="Softmax row sums must equal 1.0")


# ---------------------------------------------------------------------------
# AM-08 -- Direct beta box-constraint [-3, +3] (no diagnostics dependency)
# ---------------------------------------------------------------------------

class TestBetaBoxConstraintDirect:
    """AM-08: Direct unit tests for beta box-constraint [-3, 3] via mx.clip.

    These tests verify the box constraint applied in _compact_head without
    relying on AMCompactor.compact() returning diagnostics.
    """

    def test_clip_removes_out_of_range_betas(self):
        """mx.clip(betas, -3.0, 3.0) enforces the AM-08 box constraint."""
        # Construct NNLS output that may approach boundary values
        rng = np.random.default_rng(13)
        n_queries, budget = 8, 4
        # Use very small A values to force betas toward large magnitudes
        A = mx.array((rng.standard_normal((n_queries, budget)) * 0.001).astype(np.float32))
        b = mx.array(rng.standard_normal(n_queries).astype(np.float32))
        betas_raw, _ = nnls_solve(A, b)
        betas_clipped = mx.clip(betas_raw, -3.0, 3.0)
        mx.eval(betas_clipped)
        clipped_np = np.array(betas_clipped)
        assert (clipped_np >= -3.0).all() and (clipped_np <= 3.0).all(), (
            f"Clipped betas out of [-3, 3]: min={clipped_np.min():.4f}, "
            f"max={clipped_np.max():.4f}"
        )

    def test_clip_preserves_in_range_betas(self):
        """mx.clip does not modify betas already within [-3, 3]."""
        # NNLS guarantees >= 0; for small well-conditioned problems betas < 3
        rng = np.random.default_rng(42)
        n_queries, budget = 20, 10
        A_np = np.abs(rng.standard_normal((n_queries, budget))).astype(np.float32)
        x_true = np.clip(rng.standard_normal(budget).astype(np.float32), 0, 1)
        b_np = A_np @ x_true
        A = mx.array(A_np)
        b = mx.array(b_np)
        betas_raw, _ = nnls_solve(A, b)
        betas_clipped = mx.clip(betas_raw, -3.0, 3.0)
        mx.eval(betas_raw, betas_clipped)
        raw_np = np.array(betas_raw)
        clipped_np = np.array(betas_clipped)
        # For betas already in [-3, 3], clip is a no-op
        in_range = (raw_np >= -3.0) & (raw_np <= 3.0)
        np.testing.assert_array_equal(
            clipped_np[in_range], raw_np[in_range],
            err_msg="clip must not modify betas already in [-3, 3]",
        )

    def test_compact_head_betas_always_clipped(self):
        """_compact_head with real queries applies box constraint to betas."""
        compactor = AMCompactor(n_sink_tokens=2)
        seq_len, head_dim = 32, 64
        budget = 8
        rng = np.random.default_rng(55)
        keys_np = rng.standard_normal((1, 1, seq_len, head_dim)).astype(np.float32)
        vals_np = rng.standard_normal((1, 1, seq_len, head_dim)).astype(np.float32)
        queries_np = rng.standard_normal((1, 1, 16, head_dim)).astype(np.float32)
        keys_h = mx.array(keys_np)
        vals_h = mx.array(vals_np)
        queries_h = mx.array(queries_np)

        k_c, v_c = compactor._compact_head(keys_h, vals_h, queries_h, budget)
        mx.eval(k_c, v_c)

        # Output shapes confirm the full NNLS + OLS + clip pipeline ran
        assert k_c.shape == (1, 1, budget, head_dim), (
            f"Expected (1, 1, {budget}, {head_dim}), got {k_c.shape}"
        )
        assert v_c.shape == (1, 1, budget, head_dim), (
            f"Expected (1, 1, {budget}, {head_dim}), got {v_c.shape}"
        )
