"""
memory/consolidator.py — STM → LTM Memory Consolidator.

Implements systems consolidation: drains the Redis short-term buffer into the
ChromaDB long-term store, skipping texts that are already present (by SHA-256
hash), then logs the result.

Public surface
--------------
consolidate(...)          — run one consolidation pass; returns count written.
maybe_consolidate(state)  — call consolidate() only when state["should_consolidate"]
                            is True.
start_scheduler(interval) — start a background APScheduler job for periodic
                            consolidation.

All functions accept optional ``embedding_fn`` and ``chroma_client`` kwargs so
unit tests can inject in-memory alternatives without live services.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable, Optional

from config import SESSION_ID, STM_TTL
from memory.long_term import _http_client, add_to_ltm
from memory.short_term import get_short_term_memory
from models.brain_state import BrainState

logger = logging.getLogger(__name__)

DEFAULT_WINDOW: int = 50
DEFAULT_COLLECTION: str = "episodic"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _text_hash(text: str) -> str:
    """Return a 32-char hex SHA-256 prefix for *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _is_duplicate(text_hash: str, collection_name: str, chroma_client: Any) -> bool:
    """Return True if a document with ``hash == text_hash`` already exists.

    Uses the raw ChromaDB client to query the collection metadata directly —
    faster than a vector search and avoids false positives.  Returns ``False``
    on any error so that a missing or empty collection is never treated as a
    duplicate.
    """
    try:
        col = chroma_client.get_or_create_collection(collection_name)
        result = col.get(
            where={"hash": {"$eq": text_hash}},
            include=["metadatas"],
        )
        return len(result["ids"]) > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def consolidate(
    *,
    session_id: str = SESSION_ID,
    ttl: int = STM_TTL,
    window: int = DEFAULT_WINDOW,
    collection: str = DEFAULT_COLLECTION,
    embedding_fn: Optional[Callable] = None,
    chroma_client: Any = None,
) -> int:
    """Replay the last *window* STM messages into LTM, skipping duplicates.

    Parameters
    ----------
    session_id:    Redis session key.
    ttl:           Redis TTL (passed through to STM helpers).
    window:        Maximum number of recent messages to consider.
    collection:    Target ChromaDB collection (default: ``"episodic"``).
    embedding_fn:  Embeddings callable for ChromaDB (uses live model if None).
    chroma_client: ChromaDB client (uses HTTP client from config if None).

    Returns
    -------
    Number of new documents written to LTM.
    """
    _client = chroma_client if chroma_client is not None else _http_client()
    messages = get_short_term_memory(window, session_id=session_id, ttl=ttl)

    written = 0
    for text in messages:
        h = _text_hash(text)
        if _is_duplicate(h, collection, _client):
            logger.debug("Consolidator: skipping duplicate hash=%s", h[:8])
            continue
        add_to_ltm(
            text,
            {"source": "consolidated", "hash": h},
            collection,
            embedding_fn=embedding_fn,
            chroma_client=_client,
        )
        written += 1

    logger.info(
        "Consolidator: wrote %d/%d messages → collection=%r",
        written,
        len(messages),
        collection,
    )
    return written


def maybe_consolidate(state: BrainState, **kwargs: Any) -> None:
    """Trigger consolidation when ``state["should_consolidate"]`` is ``True``.

    Passes all *kwargs* through to :func:`consolidate`, so callers can inject
    ``embedding_fn`` and ``chroma_client`` for testing.
    """
    if state.get("should_consolidate"):
        logger.info("Consolidator: should_consolidate flag set — running consolidation")
        consolidate(**kwargs)


def start_scheduler(
    interval_seconds: int = 3600,
    **consolidate_kwargs: Any,
):
    """Start an APScheduler background job that calls :func:`consolidate` on a
    fixed interval.

    Parameters
    ----------
    interval_seconds:   How often to run consolidation (default: 1 hour).
    **consolidate_kwargs: Forwarded to :func:`consolidate` on each run.

    Returns
    -------
    The running :class:`apscheduler.schedulers.background.BackgroundScheduler`
    instance — callers must call ``scheduler.shutdown()`` when done.
    """
    from apscheduler.schedulers.background import BackgroundScheduler

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        consolidate,
        "interval",
        seconds=interval_seconds,
        kwargs=consolidate_kwargs,
        id="memory_consolidator",
        replace_existing=True,
    )
    scheduler.start()
    logger.info(
        "Consolidator scheduler started (interval=%ds)", interval_seconds
    )
    return scheduler
