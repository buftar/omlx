# SPDX-License-Identifier: Apache-2.0
"""
Compressed cache manager subclasses.

CompressedPagedSSDCacheManager — wraps PagedSSDCacheManager with KVCachePipeline
compression/decompression on save_block / load_block.

CompressedTieredCacheManager — wraps TieredCacheManager with compaction under
memory pressure via KVCachePipeline.compact().
"""
from __future__ import annotations

import logging
import queue
import time
from typing import Any, List, Optional, Tuple

from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.tiered_manager import TieredCacheManager
from omlx.compression.config import CompressionConfig

logger = logging.getLogger(__name__)


class CompressedPagedSSDCacheManager(PagedSSDCacheManager):
    """PagedSSDCacheManager subclass that compresses KV blocks via KVCachePipeline.

    When compression_config.enabled is True, save_block() compresses the KV cache
    on the calling (inference) thread before enqueueing to the background writer.
    load_block() decompresses when the stored metadata marks the block as compressed.

    When disabled, all methods delegate to the parent class unchanged (PIPE-07 no-op).
    """

    def __init__(
        self,
        *args,
        compression_config: CompressionConfig,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._compression_config = compression_config
        self.__pipeline = None  # lazy init — avoids mlx import at class load time

    # ------------------------------------------------------------------
    # Lazy pipeline property
    # ------------------------------------------------------------------

    @property
    def _pipeline(self):
        if self.__pipeline is None:
            from omlx.compression.pipeline import KVCachePipeline

            cfg = self._compression_config
            kwargs: dict = dict(
                bundle_path=cfg.bundle_path,
                am_ratio=cfg.am_ratio,
                n_sink_tokens=cfg.n_sink_tokens,
                sliding_window=cfg.sliding_window,
            )
            if cfg.n_components is not None:
                kwargs["n_components"] = cfg.n_components
            self.__pipeline = KVCachePipeline(**kwargs)
        return self.__pipeline

    # ------------------------------------------------------------------
    # save_block override
    # ------------------------------------------------------------------

    def save_block(
        self,
        block_hash: bytes,
        cache_data: List[Any],
        token_count: int,
        model_name: str = "",
        layer_cache_types: Optional[List[str]] = None,
        layer_meta_states: Optional[List[Tuple]] = None,
    ) -> bool:
        """Save a KV cache block, optionally compressing it first.

        When compression is disabled, delegates to parent unchanged (PIPE-07 no-op).
        When compression is enabled, replaces the array-building loop with a single
        PipelineBlob that is stored as 'compressed_blob' in tensors_raw.
        """
        if not self._compression_config.enabled:
            return super().save_block(
                block_hash,
                cache_data,
                token_count,
                model_name,
                layer_cache_types,
                layer_meta_states,
            )

        # -- Compressed path --

        import numpy as np
        import mlx.core as mx
        from omlx.cache.paged_ssd_cache import _extract_tensor_bytes, PagedSSDBlockMetadata

        # Step 1: duplicate-check index
        if self._index.contains(block_hash):
            self._index.touch(block_hash)
            self._stats["hits"] += 1
            return True

        # Step 2: duplicate-check hot cache
        with self._hot_cache_lock:
            if block_hash in self._hot_cache:
                self._stats["hits"] += 1
                return True

        # Step 3: queue-capacity guard (SSD path only)
        if not self._hot_cache_enabled and self._write_queue.full():
            logger.warning(
                f"Write queue full, dropping block {block_hash.hex()[:16]}"
            )
            return False

        # Step 4: resolve file path
        file_path = self._get_file_path(block_hash)

        # Step 5: enforce size limit (SSD path only)
        if not self._hot_cache_enabled:
            self._enforce_size_limit_for_new_block()

        # Steps 6-8: compress and extract raw bytes (OVERRIDE)
        try:
            blob = self._pipeline.compress(cache_data)
            blob_arr = mx.array(np.frombuffer(blob.compressed, dtype=np.uint8))
            mx.eval(blob_arr)
            tensors_raw = {"compressed_blob": _extract_tensor_bytes(blob_arr)}
        except Exception as exc:
            logger.error(
                f"Compression failed for block {block_hash.hex()[:16]}: {exc}; "
                f"falling back to uncompressed save"
            )
            return super().save_block(
                block_hash,
                cache_data,
                token_count,
                model_name,
                layer_cache_types,
                layer_meta_states,
            )

        metadata = {
            "block_hash": block_hash.hex(),
            "token_count": str(token_count),
            "num_layers": str(len(cache_data)),
            "model_name": model_name,
            "created_at": str(time.time()),
            "compressed": "true",
            "logical_seq_len": str(blob.logical_seq_len),
        }

        # Step 9: build block metadata
        estimated_size = (
            sum(len(raw) for raw, _, _ in tensors_raw.values()) + 1024
        )
        now = time.time()
        block_metadata = PagedSSDBlockMetadata(
            block_hash=block_hash,
            file_path=file_path,
            file_size=estimated_size,
            token_count=token_count,
            created_at=now,
            last_access=now,
            num_layers=len(cache_data),
            model_name=model_name,
            layer_cache_types=layer_cache_types,
            layer_meta_states=layer_meta_states,
        )

        # Step 10: build cache entry dict
        cache_entry = {
            "tensors_raw": tensors_raw,
            "file_metadata": metadata,
            "num_layers": len(cache_data),
            "layer_cache_types": layer_cache_types,
            "block_metadata": block_metadata,
        }

        # Step 11a: hot-cache path
        if self._hot_cache_enabled:
            self._hot_cache_put(block_hash, cache_entry)
            self._stats["saves"] += 1
            return True

        # Step 11b: SSD path
        self._index.add(block_metadata)
        with self._hot_cache_lock:
            self._hot_cache[block_hash] = cache_entry
        with self._pending_write_hashes_lock:
            self._pending_write_hashes.add(block_hash)
        try:
            self._write_queue.put_nowait(
                (block_hash, tensors_raw, metadata, file_path)
            )
        except queue.Full:
            logger.warning(
                f"Write queue full after enqueue attempt, reverting block "
                f"{block_hash.hex()[:16]}"
            )
            self._index.remove(block_hash)
            self._hot_cache_remove(block_hash)
            return False

        self._stats["saves"] += 1
        return True

    # ------------------------------------------------------------------
    # load_block override
    # ------------------------------------------------------------------

    def load_block(self, block_hash: bytes) -> Optional[Any]:
        """Load a KV cache block, decompressing if necessary.

        Hot-cache path: if the entry has compressed=true, reconstruct PipelineBlob
        from stored raw bytes and decompress. On error, returns None (cache miss).

        Disk path: if the .safetensors file has compressed=true in metadata,
        reconstruct PipelineBlob and decompress. On error, returns None.

        Non-compressed entries fall through to super().load_block().
        """
        # Step 1: check hot cache
        entry = self._hot_cache_get(block_hash)
        if entry is not None:
            if entry["file_metadata"].get("compressed") == "true":
                raw_bytes, dtype_str, shape = entry["tensors_raw"]["compressed_blob"]
                try:
                    from omlx.compression.pipeline import PipelineBlob

                    blob = PipelineBlob(
                        compressed=raw_bytes,
                        logical_seq_len=int(entry["file_metadata"]["logical_seq_len"]),
                        compaction_ratio=1.0,
                    )
                    result = self._pipeline.decompress(blob)
                    self._index.touch(block_hash)
                    self._stats["loads"] += 1
                    self._stats["hits"] += 1
                    self._stats["hot_cache_hits"] += 1
                    return result
                except Exception as exc:
                    logger.error(
                        f"Decompression failed (hot cache) for "
                        f"{block_hash.hex()[:16]}: {exc}"
                    )
                    return None
            else:
                # Not compressed — delegate to parent (which will re-check hot cache)
                return super().load_block(block_hash)

        # Step 2: check index
        metadata = self._index.get(block_hash)
        if metadata is None:
            self._stats["misses"] += 1
            return None

        # Step 3: check file exists
        file_path = metadata.file_path
        if not file_path.exists():
            logger.warning(f"SSD cache file missing: {file_path}")
            self._index.remove(block_hash)
            self._stats["misses"] += 1
            return None

        # Step 4: load from disk
        import mlx.core as mx

        arrays, file_metadata = mx.load(str(file_path), return_metadata=True)

        if file_metadata.get("compressed") == "true":
            try:
                import numpy as np
                from omlx.compression.pipeline import PipelineBlob

                blob_arr = arrays["compressed_blob"]
                mx.eval(blob_arr)
                raw_bytes = bytes(memoryview(np.array(blob_arr)))
                blob = PipelineBlob(
                    compressed=raw_bytes,
                    logical_seq_len=int(file_metadata["logical_seq_len"]),
                    compaction_ratio=1.0,
                )
                result = self._pipeline.decompress(blob)
                self._index.touch(block_hash)
                self._stats["loads"] += 1
                return result
            except Exception as exc:
                logger.error(
                    f"Decompression failed (disk) for {block_hash.hex()[:16]}: {exc}"
                )
                return None
        else:
            # Not compressed — fall through to parent's full disk reconstruction
            return super().load_block(block_hash)


class CompressedTieredCacheManager(TieredCacheManager):
    """TieredCacheManager subclass that compacts hot blocks under memory pressure.

    When compression_config.enabled is True, check_memory_pressure() calls
    _compact_hot_blocks() (best-effort AM compaction of LRU blocks) before
    delegating to super().check_memory_pressure().
    """

    def __init__(
        self,
        *args,
        compression_config: CompressionConfig,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._compression_config = compression_config
        self.__pipeline = None  # lazy init

    # ------------------------------------------------------------------
    # Lazy pipeline property
    # ------------------------------------------------------------------

    @property
    def _pipeline(self):
        if self.__pipeline is None:
            from omlx.compression.pipeline import KVCachePipeline

            cfg = self._compression_config
            kwargs: dict = dict(
                bundle_path=cfg.bundle_path,
                am_ratio=cfg.am_ratio,
                n_sink_tokens=cfg.n_sink_tokens,
                sliding_window=cfg.sliding_window,
            )
            if cfg.n_components is not None:
                kwargs["n_components"] = cfg.n_components
            self.__pipeline = KVCachePipeline(**kwargs)
        return self.__pipeline

    # ------------------------------------------------------------------
    # _compact_hot_blocks
    # ------------------------------------------------------------------

    def _compact_hot_blocks(self) -> None:
        """Compact up to 3 LRU evictable blocks in hot memory (best-effort).

        Calls self._pipeline.compact() on each block's cache data and replaces
        it in the hot-memory block store. All errors are caught and logged at
        WARNING level — compaction failure must never prevent normal eviction.
        """
        if self.paged_cache_manager is None:
            return

        try:
            evictable = self.paged_cache_manager.get_evictable_blocks(3)
        except Exception as exc:
            logger.warning(f"_compact_hot_blocks: get_evictable_blocks failed: {exc}")
            return

        if not evictable:
            return

        compacted = 0
        for block in evictable:
            try:
                cache_data = getattr(block, "cache_data", None)
                if cache_data is None:
                    continue
                compacted_data = self._pipeline.compact(cache_data)
                block.cache_data = compacted_data
                compacted += 1
            except Exception as exc:
                logger.warning(
                    f"_compact_hot_blocks: compaction failed for block "
                    f"{getattr(block, 'block_id', '?')}: {exc}"
                )

        logger.debug(
            f"Compacted {compacted} hot blocks under memory pressure"
        )

    # ------------------------------------------------------------------
    # check_memory_pressure override
    # ------------------------------------------------------------------

    def check_memory_pressure(self) -> bool:
        """Compact hot blocks then delegate to parent's pressure check."""
        if self._compression_config.enabled:
            self._compact_hot_blocks()
        return super().check_memory_pressure()
