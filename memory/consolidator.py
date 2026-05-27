"""
memory/consolidator.py — STM → LTM Memory Consolidator.

Implements systems consolidation: drains the Redis short-term buffer into the
ChromaDB long-term store as a single LLM-generated summary (or raw, when
``summarize=False``), skipping batches whose content hash is already stored,
then logs the result.

Public surface
--------------
consolidate(...)          — run one consolidation pass; returns count written.
maybe_consolidate(state)  — call consolidate() only when state["should_consolidate"]
                            is True.
start_scheduler(interval) — start a background APScheduler job for periodic
                            consolidation.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage

from config import CONSOLIDATION_WINDOW, SESSION_ID, STM_TTL
from memory.long_term import _http_client, add_to_ltm
from memory.short_term import get_short_term_memory
from models.brain_state import BrainState

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION: str = "episodic"

_SUMMARY_PROMPT = (
    "You are a memory consolidation system. Summarize the following conversation "
    "messages into a single concise episodic memory that preserves key facts, "
    "decisions, and context. Be factual and concise.\n\n"
    "Messages:\n{messages}\n\n"
    "Episodic memory summary:"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _text_hash(text: str) -> str:
    """Return a 32-char hex SHA-256 prefix for *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _is_duplicate(text_hash: str, collection_name: str, chroma_client: Any) -> bool:
    """Return True if a document with ``hash == text_hash`` already exists."""
    try:
        col = chroma_client.get_or_create_collection(collection_name)
        result = col.get(
            where={"hash": {"$eq": text_hash}},
            include=["metadatas"],
        )
        return len(result["ids"]) > 0
    except Exception:
        return False


def _summarize_messages(texts: list[str], llm: Any = None) -> str:
    """Call the LLM to produce a single episodic-memory summary of *texts*.

    Parameters
    ----------
    texts:  List of raw STM message strings to summarise.
    llm:    Optional pre-built LLM instance; uses the configured provider if
            omitted.  Injected by tests to avoid live network calls.

    Returns
    -------
    The summary string returned by the LLM.
    """
    from providers.llm import get_llm

    _llm = llm or get_llm(temperature=0.3)
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    prompt = _SUMMARY_PROMPT.format(messages=numbered)
    response = _llm.invoke([HumanMessage(content=prompt)])
    return str(response.content).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def consolidate(
    *,
    session_id: str = SESSION_ID,
    ttl: int = STM_TTL,
    window: int = CONSOLIDATION_WINDOW,
    collection: str = DEFAULT_COLLECTION,
    summarize: bool = True,
    embedding_fn: Embeddings | None = None,
    chroma_client: Any = None,
    llm: Any = None,
) -> int:
    """Consolidate recent STM messages into LTM.

    When *summarize* is ``True`` (default) all messages in *window* are fed to
    the LLM which produces a single episodic-memory summary.  That summary is
    stored as one LTM document, keyed by a hash of the raw message batch so
    the same batch is never stored twice.

    When *summarize* is ``False`` each message is stored individually (legacy
    behaviour), also with per-message dedup.

    Parameters
    ----------
    session_id:    Redis session key.
    ttl:           Redis TTL (passed through to STM helpers).
    window:        Maximum number of recent messages to consider.
    collection:    Target ChromaDB collection (default: ``"episodic"``).
    summarize:     Summarise the batch via LLM before storing (default: True).
    embedding_fn:  Embeddings callable for ChromaDB.
    chroma_client: ChromaDB client.
    llm:           LLM instance for summarisation (uses configured provider if
                   None and *summarize* is True).

    Returns
    -------
    Number of new documents written to LTM (0 or 1 when summarizing, 0–N
    when storing raw messages).
    """
    _client = chroma_client if chroma_client is not None else _http_client()
    messages = get_short_term_memory(window, session_id=session_id, ttl=ttl)

    if not messages:
        logger.info("Consolidator: no messages in STM — nothing to consolidate")
        return 0

    if summarize:
        # Hash the full ordered batch to detect already-consolidated windows.
        batch_text = "\n".join(messages)
        h = _text_hash(batch_text)
        if _is_duplicate(h, collection, _client):
            logger.debug("Consolidator: batch already consolidated (hash=%s)", h[:8])
            return 0

        summary = _summarize_messages(messages, llm=llm)
        add_to_ltm(
            summary,
            {
                "source": "summarized",
                "hash": h,
                "message_count": len(messages),
            },
            collection,
            embedding_fn=embedding_fn,
            chroma_client=_client,
        )
        logger.info(
            "Consolidator: summarized %d messages → 1 LTM document (hash=%s)",
            len(messages),
            h[:8],
        )
        return 1

    # Raw (legacy) path — store each message individually.
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

    Passes all *kwargs* through to :func:`consolidate`.
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
    interval_seconds:     How often to run consolidation (default: 1 hour).
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
        "Consolidator scheduler started (interval=%ds, summarize=%s)",
        interval_seconds,
        consolidate_kwargs.get("summarize", True),
    )
    return scheduler
