# SPDX-License-Identifier: Apache-2.0
"""
Compressed cache manager subclasses — stub (RED state).

Filled in by plan 06-02 Task 2.
"""
from __future__ import annotations

from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.tiered_manager import TieredCacheManager
from omlx.compression.config import CompressionConfig


class CompressedPagedSSDCacheManager(PagedSSDCacheManager):
    """Stub — not yet implemented."""

    def __init__(self, *args, compression_config: CompressionConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self._compression_config = compression_config

    def save_block(self, *args, **kwargs):
        raise NotImplementedError("CompressedPagedSSDCacheManager.save_block not implemented")

    def load_block(self, *args, **kwargs):
        raise NotImplementedError("CompressedPagedSSDCacheManager.load_block not implemented")


class CompressedTieredCacheManager(TieredCacheManager):
    """Stub — not yet implemented."""

    def __init__(self, *args, compression_config: CompressionConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self._compression_config = compression_config

    def check_memory_pressure(self) -> bool:
        raise NotImplementedError("CompressedTieredCacheManager.check_memory_pressure not implemented")
