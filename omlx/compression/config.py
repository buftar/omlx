# SPDX-License-Identifier: Apache-2.0
"""
CompressionConfig — thread-safe configuration dataclass for KV cache compression.

This is the single source of truth for compression parameters used by
CompressedPagedSSDCacheManager and CompressedTieredCacheManager.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CompressionConfig:
    """Configuration for KV cache compression pipeline.

    Fields
    ------
    enabled        : Whether compression is active. Toggle at runtime via set_enabled().
    bundle_path    : Path to calibration .npz produced by ``omlx calibrate-kv``.
                     None = on-the-fly mode (no PCA basis, no head-entropy weighting).
    am_ratio       : AM compaction budget ratio. Default 4.0.
    n_sink_tokens  : Attention-sink token count (exempt from compaction/quantisation).
    sliding_window : KVTC sliding window (recent tokens exempt from quantisation).
    n_components   : Number of PCA components. None = use full head_dim (on-the-fly).

    The ``_lock`` field is an RLock used only by set_enabled(). It is excluded from
    __init__, __repr__, and equality comparison.
    """

    enabled: bool = False
    bundle_path: Optional[str] = None
    am_ratio: float = 4.0
    n_sink_tokens: int = 4
    sliding_window: int = 128
    n_components: Optional[int] = None

    _lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
        compare=False,
    )

    def set_enabled(self, value: bool) -> None:
        """Toggle compression on or off in a thread-safe manner.

        Parameters
        ----------
        value : bool
            True to enable compression, False to disable.
        """
        with self._lock:
            self.enabled = value
