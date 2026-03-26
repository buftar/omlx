# SPDX-License-Identifier: Apache-2.0
"""Admin dashboard helpers for the oMLX menubar app.

Provides compression settings and stats cards for the native macOS menubar,
surfacing the same data that the web admin dashboard shows for compression.

The web admin dashboard (HTML/JS) already exposes compression settings and
stats in full detail.  This module provides a lightweight native counterpart:
a pair of helper functions that build NSMenuItem entries for the menubar
"Compression" submenu, polling the same backend admin API endpoints.

Compression settings card (Task 1)
-----------------------------------
- Fetches enabled/disabled state, AM ratio, and component count from
  GET /admin/api/compression/status.
- Exposes ``set_compression_enabled`` and ``set_compression_am_ratio`` helpers
  that POST to /admin/api/compression/config for runtime configuration.
- ``build_compression_settings_items`` returns a list of (label, value) pairs
  suitable for NSMenuItem display.

Compression stats card (Task 2)
---------------------------------
- Fetches compression ratio, decompression latency, and cache hit/miss
  counts from GET /admin/api/compression/status.
- ``build_compression_stats_items`` returns a list of (label, value) pairs
  for the stats section of the Compression submenu.
- Designed for real-time refresh: caller re-invokes on each polling cycle.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal API helpers
# ---------------------------------------------------------------------------

_TIMEOUT = 2  # seconds — mirrors the existing _fetch_stats timeout in app.py


def _get_compression_status(
    session: requests.Session,
    base_url: str,
) -> Optional[dict[str, Any]]:
    """Fetch compression status from the admin API.

    Args:
        session: Authenticated requests session (logged-in admin cookie).
        base_url: Server base URL, e.g. ``"http://127.0.0.1:8000"``.

    Returns:
        Parsed JSON payload from ``GET /admin/api/compression/status``, or
        ``None`` if the request fails or the server is unreachable.
    """
    try:
        resp = session.get(
            f"{base_url}/admin/api/compression/status",
            timeout=_TIMEOUT,
        )
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException as exc:
        logger.debug("Failed to fetch compression status: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Compression settings card (Task 1)
# ---------------------------------------------------------------------------


def build_compression_settings_items(
    status: Optional[dict[str, Any]],
) -> list[tuple[str, str]]:
    """Build display entries for the compression settings card.

    Returns a list of ``(label, value)`` pairs that the menubar can render
    as non-interactive NSMenuItems.  The caller controls the actual menu
    construction so that PyObjC import is not required in this module.

    Args:
        status: Payload from :func:`_get_compression_status`, or ``None``
                when compression data is unavailable.

    Returns:
        List of ``(label, value)`` tuples::

            [
                ("Compression", "Enabled"),
                ("AM Ratio", "4.0x"),
                ("Components", "64"),
            ]

        When *status* is ``None`` an empty list is returned so the caller
        can show a "Compression unavailable" placeholder instead.
    """
    if status is None:
        return []

    enabled: bool = status.get("enabled", False)
    am_ratio: float = float(status.get("am_ratio", 4.0))
    n_components: int = int(status.get("n_components", 0))

    entries: list[tuple[str, str]] = [
        ("Compression", "Enabled" if enabled else "Disabled"),
        ("AM Ratio", f"{am_ratio:.1f}x"),
    ]
    if n_components > 0:
        entries.append(("Components", str(n_components)))

    return entries


def set_compression_enabled(
    session: requests.Session,
    base_url: str,
    enabled: bool,
    am_ratio: float = 4.0,
) -> bool:
    """Toggle compression on/off via the admin API.

    Args:
        session: Authenticated requests session.
        base_url: Server base URL.
        enabled: ``True`` to enable, ``False`` to disable.
        am_ratio: Current AM ratio value (preserved while toggling).

    Returns:
        ``True`` if the request succeeded, ``False`` otherwise.
    """
    try:
        resp = session.post(
            f"{base_url}/admin/api/compression/config",
            json={"enabled": enabled, "am_ratio": am_ratio},
            timeout=_TIMEOUT,
        )
        return resp.status_code == 200
    except requests.RequestException as exc:
        logger.debug("Failed to set compression enabled=%s: %s", enabled, exc)
        return False


def set_compression_am_ratio(
    session: requests.Session,
    base_url: str,
    am_ratio: float,
    enabled: bool = True,
) -> bool:
    """Update the AM compaction ratio via the admin API.

    Args:
        session: Authenticated requests session.
        base_url: Server base URL.
        am_ratio: New ratio value (1.0–8.0).
        enabled: Current enabled state (preserved while updating ratio).

    Returns:
        ``True`` if the request succeeded, ``False`` otherwise.
    """
    try:
        resp = session.post(
            f"{base_url}/admin/api/compression/config",
            json={"enabled": enabled, "am_ratio": am_ratio},
            timeout=_TIMEOUT,
        )
        return resp.status_code == 200
    except requests.RequestException as exc:
        logger.debug("Failed to set compression am_ratio=%s: %s", am_ratio, exc)
        return False


# ---------------------------------------------------------------------------
# Compression stats card (Task 2)
# ---------------------------------------------------------------------------


def build_compression_stats_items(
    status: Optional[dict[str, Any]],
) -> list[tuple[str, str]]:
    """Build display entries for the compression stats card.

    Returns a list of ``(label, value)`` pairs for the stats section of
    the Compression submenu, covering compression ratio, decompression
    latency, and cache hit/miss rates.

    Args:
        status: Payload from :func:`_get_compression_status`, or ``None``
                when stats data is unavailable.

    Returns:
        List of ``(label, value)`` tuples::

            [
                ("Compression Ratio", "3.21x"),
                ("Decompress Latency", "1.4 ms"),
                ("Cache Hit", "142"),
                ("Cache Miss", "8"),
            ]

        When *status* is ``None`` or there is no active compression data
        an empty list is returned.
    """
    if status is None:
        return []

    compression_ratio: float = float(status.get("compression_ratio", 0.0))
    decomp_latency: float = float(status.get("avg_decompression_latency_ms", 0.0))
    success_count: int = int(status.get("compression_success_count", 0))
    failure_count: int = int(status.get("compression_failure_count", 0))
    decomp_success: int = int(status.get("decompression_success_count", 0))
    decomp_failure: int = int(status.get("decompression_failure_count", 0))

    # Suppress stats section when compression has never fired this session
    if compression_ratio == 0.0 and success_count == 0 and decomp_success == 0:
        return []

    entries: list[tuple[str, str]] = [
        ("Compression Ratio", f"{compression_ratio:.2f}x"),
        ("Decompress Latency", f"{decomp_latency:.1f} ms"),
    ]

    # Cache hit/miss: decompression success = cache hit, failure = miss
    total_cache_ops = decomp_success + decomp_failure
    if total_cache_ops > 0:
        hit_rate = decomp_success / total_cache_ops * 100
        entries.append(("Cache Hit Rate", f"{hit_rate:.1f}%"))

    entries.extend(
        [
            ("Compress Success", str(success_count)),
            ("Compress Failure", str(failure_count)),
        ]
    )

    return entries


# ---------------------------------------------------------------------------
# Public facade — single call for menubar refresh cycle
# ---------------------------------------------------------------------------


def fetch_compression_dashboard(
    session: requests.Session,
    base_url: str,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Fetch and build both compression cards in one call.

    Convenience wrapper used by the menubar app's periodic refresh cycle.
    Returns both the settings card entries and the stats card entries in
    a single round-trip to the admin API.

    Args:
        session: Authenticated requests session.
        base_url: Server base URL.

    Returns:
        ``(settings_items, stats_items)`` — each is a list of
        ``(label, value)`` tuples as returned by
        :func:`build_compression_settings_items` and
        :func:`build_compression_stats_items`.
    """
    status = _get_compression_status(session, base_url)
    settings_items = build_compression_settings_items(status)
    stats_items = build_compression_stats_items(status)
    return settings_items, stats_items
