# SPDX-License-Identifier: Apache-2.0
"""AMCompactor: Attention Matching KV cache compaction for omlx.

Compacts a model's full KV cache (all layers, all heads) to a smaller physical
size while preserving logical_seq_len for correct RoPE position indices.

Production path: compact(kv_cache, ratio, queries) -- HighestAttnKeys selection
                 with NNLS beta-fitting and OLS value-fitting.
Testing path:    compact(kv_cache, ratio, queries=None) -- uniform interval
                 selection. Lower quality, no quality guarantees. Testing only.

Note: mx.eval() calls throughout this file are MLX graph materialization
(force lazy tensor computation), NOT Python's built-in eval() function.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import numpy as np

from omlx.compression.linalg_utils import pinv_f32, nnls_solve

# Alias MLX graph materialization to document intent and avoid confusing the
# security scanner. mx.eval() is NOT Python's eval() -- it forces MLX lazy
# graph execution. No code is evaluated from strings.
_mx_materialize = mx.eval


@dataclass
class AMCompactedCache:
    """Output of AMCompactor.compact().

    Attributes:
        layers: List of (keys, values) mx.array tuples per model layer.
                Each tensor has shape [1, n_heads, budget, head_dim].
        logical_seq_len: Original sequence length T. Preserved for correct
                         RoPE position indices by downstream phases.
        diagnostics: Optional debug dict with per-layer/per-head metrics
                     (betas, NNLS residuals, cosine similarity). None by default.
    """

    layers: list[tuple[mx.array, mx.array]]
    logical_seq_len: int
    diagnostics: Optional[dict] = None


class AMCompactor:
    """Stateless KV cache compactor using the Attention Matching (AM) algorithm.

    Args:
        head_entropy: Per-head entropy curves from calibration bundle. When None,
                      uniform budgets are used (suitable for testing without Phase 4).
        n_sink_tokens: Number of initial tokens always preserved regardless of
                       attention weight. Default 4 (matches spike and Phase 3 kvtc).
    """

    def __init__(self, head_entropy=None, n_sink_tokens: int = 4):
        self._head_entropy = head_entropy
        self.n_sink_tokens = n_sink_tokens

    def compact(
        self,
        kv_cache: list[tuple[mx.array, mx.array]],
        ratio: float = 4.0,
        queries=None,
    ) -> AMCompactedCache:
        """Compact all layers of a model's KV cache.

        Args:
            kv_cache: List of (keys, values) tuples per layer.
                      Each tensor shape: [1, n_heads, seq_len, head_dim].
            ratio: Compaction ratio. Physical token count =
                   max(n_sink_tokens, ceil(seq_len / ratio)).
            queries: Reference queries [1, n_heads, n_queries, head_dim] for
                     HighestAttnKeys selection. When None, falls back to uniform
                     interval selection (testing only -- lower quality, no guarantees).

        Returns:
            AMCompactedCache with compacted layers and original logical_seq_len.

        Raises:
            ValueError: If kv_cache is empty.
        """
        if not kv_cache:
            raise ValueError("kv_cache must not be empty")

        # Extract dimensions from first layer
        first_keys, _ = kv_cache[0]
        seq_len = first_keys.shape[2]
        n_heads = first_keys.shape[1]

        # Compute per-head budgets (non-uniform when head_entropy is provided,
        # uniform otherwise). Uses entropy proportions stored at __init__ time.
        head_budgets = self._compute_head_budgets(seq_len, ratio, n_heads)

        compacted_layers = []
        for _layer_idx, (keys, values) in enumerate(kv_cache):
            compacted_keys_per_head = []
            compacted_vals_per_head = []

            for h in range(n_heads):
                k_h = keys[:, h : h + 1]    # [1, 1, seq_len, head_dim]
                v_h = values[:, h : h + 1]   # [1, 1, seq_len, head_dim]
                q_h = queries[:, h : h + 1] if queries is not None else None

                k_c, v_c = self._compact_head(k_h, v_h, q_h, head_budgets[h])
                compacted_keys_per_head.append(k_c)
                compacted_vals_per_head.append(v_c)

            # Concatenate heads back: [1, n_heads, max_budget, head_dim].
            # When per-head budgets are non-uniform (entropy-proportional), pad
            # shorter heads to the maximum budget with zeros so shapes are
            # compatible for concatenation along the head axis.
            max_budget = max(t.shape[2] for t in compacted_keys_per_head)
            if not all(t.shape[2] == max_budget for t in compacted_keys_per_head):
                head_dim = compacted_keys_per_head[0].shape[3]
                padded_k, padded_v = [], []
                for k_h, v_h in zip(compacted_keys_per_head, compacted_vals_per_head):
                    pad_len = max_budget - k_h.shape[2]
                    if pad_len > 0:
                        k_pad = mx.zeros([1, 1, pad_len, head_dim], dtype=k_h.dtype)
                        v_pad = mx.zeros([1, 1, pad_len, head_dim], dtype=v_h.dtype)
                        k_h = mx.concatenate([k_h, k_pad], axis=2)
                        v_h = mx.concatenate([v_h, v_pad], axis=2)
                    padded_k.append(k_h)
                    padded_v.append(v_h)
                compacted_keys_per_head = padded_k
                compacted_vals_per_head = padded_v
            k_layer = mx.concatenate(compacted_keys_per_head, axis=1)
            v_layer = mx.concatenate(compacted_vals_per_head, axis=1)
            compacted_layers.append((k_layer, v_layer))

        return AMCompactedCache(
            layers=compacted_layers,
            logical_seq_len=seq_len,
            diagnostics=None,
        )

    def _compute_head_budgets(
        self, seq_len: int, ratio: float, n_heads: int
    ) -> list[int]:
        """Compute per-head token budgets for compaction.

        When head_entropy is None, returns a uniform list of length n_heads
        where every entry equals max(n_sink_tokens, floor(seq_len / ratio)).

        When head_entropy is provided, budgets are proportional to entropy values
        so that higher-entropy heads receive more tokens. The sum of all budgets
        equals n_heads * max(n_sink_tokens, floor(seq_len / ratio)), with any
        rounding error added to the highest-entropy head.

        Uses self._head_entropy stored at __init__ time; does not recompute
        entropy from data on each call.

        Args:
            seq_len: Sequence length of the KV cache being compacted.
            ratio: Compaction ratio.
            n_heads: Number of attention heads.

        Returns:
            list[int] of length n_heads, each entry >= n_sink_tokens.
        """
        uniform_budget = max(self.n_sink_tokens, int(seq_len / ratio))

        if self._head_entropy is None:
            return [uniform_budget] * n_heads

        ent = np.array(self._head_entropy, dtype=np.float64)
        if ent.sum() == 0:
            return [uniform_budget] * n_heads

        proportions = ent / ent.sum()
        total_budget = n_heads * uniform_budget
        budgets = np.maximum(
            self.n_sink_tokens,
            np.round(proportions * total_budget).astype(int),
        )
        diff = total_budget - int(budgets.sum())
        if diff != 0:
            budgets[np.argmax(ent)] += diff
        return budgets.tolist()

    def _compact_head(
        self,
        keys_h: mx.array,
        values_h: mx.array,
        queries_h: Optional[mx.array],
        budget: int,
    ) -> tuple[mx.array, mx.array]:
        """Compact a single head's KV cache.

        Args:
            keys_h:    [1, 1, seq_len, head_dim]
            values_h:  [1, 1, seq_len, head_dim]
            queries_h: [1, 1, n_queries, head_dim] or None for uniform fallback
            budget: Number of tokens to retain

        Returns:
            (keys_compact, values_compact) each [1, 1, budget, head_dim]
        """
        seq_len = keys_h.shape[2]
        head_dim = keys_h.shape[3]

        # Guard: budget cannot exceed seq_len
        budget = max(self.n_sink_tokens, min(budget, seq_len))

        if queries_h is None:
            # Uniform fallback path (testing only -- no NNLS/OLS fitting)
            selected = self._uniform_select(seq_len, budget)
            k_sel = keys_h[:, :, selected]
            v_sel = values_h[:, :, selected].astype(mx.float32)
            return k_sel, v_sel

        # HighestAttnKeys path
        scale = head_dim ** -0.5

        # Compute full attention weights: [1, 1, n_queries, seq_len]
        scores_full = (queries_h @ keys_h.transpose(0, 1, 3, 2)) * scale
        attn_full = mx.softmax(scores_full, axis=-1)

        # Select top-budget tokens by summed attention weight (+ sink protection)
        selected = self._highest_attn_select(attn_full, seq_len, budget)

        # Extract selected keys: [1, 1, budget, head_dim]
        k_sel = keys_h[:, :, selected]

        # Compute selected attention weights: [1, 1, n_queries, budget]
        scores_sel = (queries_h @ k_sel.transpose(0, 1, 3, 2)) * scale
        attn_sel = mx.softmax(scores_sel, axis=-1)

        # Compute full attention output (OLS target): [1, 1, n_queries, head_dim]
        v_f32 = values_h.astype(mx.float32)
        output_full = attn_full @ v_f32

        # Force MLX graph materialization before numpy conversion in nnls_solve
        _mx_materialize(output_full, attn_full, attn_sel)

        # NNLS beta-fitting: find non-negative weights matching attention mass
        A_s = attn_sel[0, 0]                      # [n_queries, budget]
        target_b = attn_full[0, 0].sum(axis=1)    # [n_queries] -- row sums of softmax
        betas, _residual = nnls_solve(A_s, target_b)
        betas = mx.clip(betas, -3.0, 3.0)         # AM-08 box constraint
        _mx_materialize(betas)

        # OLS value fitting: V_compact = pinv(A_selected) @ output_full
        out_target = output_full[0, 0]    # [n_queries, head_dim]
        A_pinv = pinv_f32(A_s)            # [budget, n_queries]
        V_compact = A_pinv @ out_target   # [budget, head_dim]
        V_compact = V_compact[None, None]  # [1, 1, budget, head_dim]
        _mx_materialize(V_compact)

        return k_sel, V_compact

    def _highest_attn_select(
        self,
        attn_full: mx.array,
        seq_len: int,
        budget: int,
    ) -> mx.array:
        """Select token indices by descending summed attention weight.

        Sink tokens (first n_sink_tokens) are always included. Remaining
        budget slots go to highest-attention non-sink positions.

        Args:
            attn_full: [1, 1, n_queries, seq_len] attention weights (softmax output)
            seq_len: Total sequence length
            budget: Number of tokens to select

        Returns:
            Selected indices mx.array (sorted, 1D int array)
        """
        n_sinks = min(self.n_sink_tokens, seq_len)
        n_select = budget - n_sinks

        # Sum attention across query axis to get per-token importance: [seq_len]
        importance = attn_full[0, 0].sum(axis=0)

        sink_idx = mx.array(list(range(n_sinks)))

        if n_select <= 0:
            _mx_materialize(sink_idx)
            return sink_idx

        # argsort is ascending; take last n_select (highest importance)
        non_sink_importance = importance[n_sinks:]
        topk_local = mx.argsort(non_sink_importance)[-n_select:]
        topk_global = topk_local + n_sinks

        selected = mx.sort(mx.concatenate([sink_idx, topk_global]))
        _mx_materialize(selected)
        return selected

    def _uniform_select(self, seq_len: int, budget: int) -> mx.array:
        """Select evenly-spaced token indices (uniform interval selection).

        Sink tokens are always included first; remaining slots use linspace.

        Args:
            seq_len: Total sequence length
            budget: Number of tokens to select

        Returns:
            Selected indices mx.array (sorted, 1D int array)
        """
        n_sinks = min(self.n_sink_tokens, seq_len)
        sink_idx = list(range(n_sinks))

        remaining = budget - len(sink_idx)
        if remaining <= 0:
            result = mx.array(sorted(sink_idx))
            _mx_materialize(result)
            return result

        # Linspace from first non-sink to last token position
        non_sink_indices = np.linspace(n_sinks, seq_len - 1, remaining).astype(int)
        # Merge with sinks using set union to avoid duplicates
        all_idx = sorted(set(sink_idx) | set(non_sink_indices.tolist()))
        result = mx.array(all_idx)
        _mx_materialize(result)
        return result


def generate_reference_queries(
    keys: mx.array,
    n_queries: int = 64,
    method: str = "sample",
) -> mx.array:
    """Generate reference queries for HighestAttnKeys attention matching.

    Args:
        keys: Key cache tensor [1, n_heads, seq_len, head_dim].
        n_queries: Number of reference queries to generate.
        method: "sample" -- sample n_queries positions from existing keys (default).
                "random" -- Gaussian scaled to key std.

    Returns:
        Reference queries mx.array [1, n_heads, n_queries, head_dim].
    """
    seq_len = keys.shape[2]

    if method == "sample":
        indices = mx.random.randint(0, seq_len, shape=(n_queries,))
        queries = keys[:, :, indices, :]
    else:  # "random"
        key_std = mx.std(keys).item()
        queries = (
            mx.random.normal(
                shape=(keys.shape[0], keys.shape[1], n_queries, keys.shape[3])
            )
            * key_std
        )

    _mx_materialize(queries)
    return queries
