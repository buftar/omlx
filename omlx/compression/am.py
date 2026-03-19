# SPDX-License-Identifier: Apache-2.0
"""AMCompactor: Attention Matching KV cache compaction for omlx.

Compacts a model's full KV cache (all layers, all heads) to a smaller physical
size while preserving logical_seq_len for correct RoPE position indices.

Production path: compact(kv_cache, ratio, queries) — HighestAttnKeys selection
                 with NNLS beta-fitting and OLS value-fitting.
Testing path:    compact(kv_cache, ratio, queries=None) — uniform interval
                 selection. Lower quality, no quality guarantees. Testing only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import numpy as np

from omlx.compression.linalg_utils import pinv_f32, nnls_solve


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
            ratio: Compaction ratio. Physical token count = max(n_sink_tokens, ceil(seq_len / ratio)).
            queries: Reference queries [1, n_heads, n_queries, head_dim] for
                     HighestAttnKeys selection. When None, falls back to uniform
                     interval selection (testing only — lower quality, no guarantees).

        Returns:
            AMCompactedCache with compacted layers and original logical_seq_len.

        Raises:
            ValueError: If kv_cache is empty.
        """
        raise NotImplementedError


def generate_reference_queries(
    keys: mx.array,
    n_queries: int = 64,
    method: str = "sample",
) -> mx.array:
    """Generate reference queries for HighestAttnKeys attention matching.

    Args:
        keys: Key cache tensor [1, n_heads, seq_len, head_dim].
        n_queries: Number of reference queries to generate.
        method: "sample" — sample n_queries positions from existing keys (default).
                "random" — Gaussian scaled to key std.

    Returns:
        Reference queries mx.array [1, n_heads, n_queries, head_dim].
    """
    raise NotImplementedError
