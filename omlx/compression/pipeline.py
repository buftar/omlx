# SPDX-License-Identifier: Apache-2.0
"""
KV-cache compression pipeline for omlx.

Two-path design:

  compact(kv_cache)   — memory-pressure path.
      Runs AM compaction to reduce the in-memory token count while keeping the
      model context alive.  No bytes are written to SSD.  Returns an
      AMCompactedCache.

  compress(kv_cache)  — eviction path.
      Runs AM compaction then KVTC quantisation, producing a self-describing
      byte blob suitable for SSD storage.  Returns a PipelineBlob.

  decompress(blob)    — restoration path.
      Deserialises a PipelineBlob back into (layers, logical_seq_len) ready for
      re-injection into the paged attention engine.

The pipeline is intentionally stateless beyond its constructor arguments so
that multiple concurrent requests can share a single instance.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

# MLX graph materialisation alias (NOT Python built-in eval)
_mx_materialize = mx.eval  # noqa: S307


@dataclass
class PipelineBlob:
    """Self-describing byte container produced by the eviction path.

    Fields
    ------
    compressed      : self-describing KVTC byte blob from KVTCCompressor.compress()
    logical_seq_len : original sequence length T before AM compaction
    compaction_ratio: actual AM ratio achieved (stored for Phase 8 observability)
    """

    compressed: bytes
    logical_seq_len: int
    compaction_ratio: float


class KVCachePipeline:
    """Two-path KV-cache compression pipeline.

    Parameters
    ----------
    bundle_path      : path to a calibration .npz produced by ``omlx calibrate-kv``.
                       When None the pipeline runs in on-the-fly mode (no PCA basis,
                       no head-entropy weighting).
    am_ratio         : AM compaction budget ratio passed to AMCompactor.compact().
    n_sink_tokens    : attention-sink token count shared by AM and KVTC.
    sliding_window   : KVTC sliding window (recent tokens exempt from quantisation).
    rope_theta       : base frequency used for the model's RoPE embedding.
                       Stored on the pipeline so _strip_rope() can invert RoPE.
    rope_traditional : whether the model uses traditional (consecutive-pair) RoPE.
    """

    def __init__(
        self,
        bundle_path=None,
        am_ratio: float = 4.0,
        n_sink_tokens: int = 4,
        sliding_window: int = 128,
        rope_theta: float = 10000.0,
        rope_traditional: bool = False,
    ) -> None:
        # Lazy imports — keep import order clean and avoid circular import risk
        from omlx.compression.am import AMCompactor
        from omlx.compression.kvtc import KVTCCompressor

        self._am_ratio = am_ratio
        self._n_sink_tokens = n_sink_tokens
        self._sliding_window = sliding_window
        self._rope_theta = rope_theta
        self._rope_traditional = rope_traditional

        if bundle_path is None:
            self._rope_params = None
            self._am = AMCompactor(
                head_entropy=None,
                n_sink_tokens=n_sink_tokens,
            )
            self._kvtc = KVTCCompressor(
                pca_bundle=None,
                n_sink_tokens=n_sink_tokens,
                sliding_window=sliding_window,
            )
        else:
            from omlx.compression.calibrator import load_calibration_bundle

            pca_bundle, head_entropy = load_calibration_bundle(Path(bundle_path))
            self._rope_params = (rope_theta, rope_traditional)
            self._am = AMCompactor(
                head_entropy=head_entropy,
                n_sink_tokens=n_sink_tokens,
            )
            self._kvtc = KVTCCompressor(
                pca_bundle=pca_bundle,
                n_sink_tokens=n_sink_tokens,
                sliding_window=sliding_window,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compact(self, kv_cache, ratio=None, queries=None):
        """Memory-pressure path: run AM compaction only.

        Parameters
        ----------
        kv_cache : list[tuple[mx.array, mx.array]]
        ratio    : override self._am_ratio for this call
        queries  : optional attention queries for entropy scoring

        Returns
        -------
        AMCompactedCache
        """
        effective_ratio = ratio if ratio is not None else self._am_ratio
        return self._am.compact(kv_cache, ratio=effective_ratio, queries=queries)

    def compress(self, kv_cache, queries=None) -> PipelineBlob:
        """Eviction path: AM compaction followed by KVTC quantisation.

        Parameters
        ----------
        kv_cache : list[tuple[mx.array, mx.array]]
        queries  : optional attention queries for AM entropy scoring

        Returns
        -------
        PipelineBlob
        """
        compacted = self._am.compact(kv_cache, ratio=self._am_ratio, queries=queries)
        original_seq_len = compacted.logical_seq_len
        compacted_seq_len = compacted.layers[0][0].shape[2]
        actual_ratio = original_seq_len / compacted_seq_len if compacted_seq_len > 0 else 1.0

        # Record compression metric
        from omlx.server_metrics import get_server_metrics
        get_server_metrics().record_compression_ratio(actual_ratio)

        stripped_layers = self._strip_rope(compacted.layers)
        compressed_bytes = self._kvtc.compress(stripped_layers)

        return PipelineBlob(
            compressed=compressed_bytes,
            logical_seq_len=original_seq_len,
            compaction_ratio=actual_ratio,
        )

    def decompress(self, blob: PipelineBlob):
        """Restore a PipelineBlob to KV-cache layers.

        Parameters
        ----------
        blob : PipelineBlob

        Returns
        -------
        tuple[list[tuple[mx.array, mx.array]], int]
            (layers, logical_seq_len)
        """
        layers = self._kvtc.decompress(blob.compressed)
        return layers, blob.logical_seq_len

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _strip_rope(self, compacted_layers):
        """Remove RoPE from compacted key tensors before KVTC PCA rotation.

        When bundle_path=None (self._rope_params is None), returns compacted_layers
        unchanged — stripping is skipped on the no-bundle testing path.

        Parameters
        ----------
        compacted_layers : list[tuple[mx.array, mx.array]]
            Compacted (keys, values) from AMCompactor; keys have RoPE applied.

        Returns
        -------
        list[tuple[mx.array, mx.array]]
            Same shape, RoPE removed from keys (or unchanged when rope_params is None).
        """
        if self._rope_params is None:
            return compacted_layers

        from omlx.compression.calibrator import strip_rope_from_keys

        rope_theta, rope_traditional = self._rope_params
        stripped = []
        for keys, values in compacted_layers:
            # Force MLX lazy graph execution before numpy bridge (Pitfall 5)
            _mx_materialize(keys)
            keys_np = np.array(keys.astype(mx.float32))
            # offset=0: kvtc treats compacted body as position-agnostic vectors
            keys_stripped_np = strip_rope_from_keys(
                keys_np, rope_theta, rope_traditional, offset=0
            )
            # Convert back to mx.array float16 — kvtc expects mx.array (Pitfall 2)
            stripped_keys = mx.array(keys_stripped_np.astype(np.float16))
            stripped.append((stripped_keys, values))
        return stripped
