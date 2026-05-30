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
consolidate_ltm(...)       — compress old LTM docs into a meta-summary (LTM→LTM).
start_scheduler(interval) — start a background APScheduler job for periodic
                            consolidation.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage

from config import (
    CONSOLIDATION_COLLECTION,
    CONSOLIDATION_INTERVAL_SECONDS,
    CONSOLIDATION_LLM_TEMPERATURE,
    CONSOLIDATION_MODEL,
    CONSOLIDATION_WINDOW,
    LTM_SUMMARIZATION_INTERVAL_SECONDS,
    LTM_SUMMARIZATION_MIN_DOCS,
    SESSION_ID,
    STM_TTL,
)
from memory.long_term import _http_client, add_to_ltm
from memory.short_term import get_short_term_memory
from models.brain_state import BrainState

logger = logging.getLogger(__name__)

_last_consolidation_time: float = 0.0  # monotonic clock; updated after each on-demand run

DEFAULT_COLLECTION: str = CONSOLIDATION_COLLECTION

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

    _llm = llm or get_llm(model_name=CONSOLIDATION_MODEL, temperature=CONSOLIDATION_LLM_TEMPERATURE)
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

    .. warning::
        The ``summarize=False`` path stores raw conversation turns verbatim. It
        exists only for testing and back-compat. All production call-sites must
        use the default ``summarize=True`` to satisfy the design invariant that
        LTM contains only LLM-generated summaries, never raw chat logs.

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

    Enforces a cooldown of ``CONSOLIDATION_INTERVAL_SECONDS`` so that rapid
    back-to-back emotional turns do not each trigger a full consolidation pass
    mid-conversation.  The scheduler path (``start_scheduler``) is unaffected.

    Passes all *kwargs* through to :func:`consolidate`.
    """
    global _last_consolidation_time
    if not state.get("should_consolidate"):
        return
    elapsed = time.monotonic() - _last_consolidation_time
    if elapsed < CONSOLIDATION_INTERVAL_SECONDS:
        logger.debug(
            "Consolidator: skipping on-demand run — only %.0fs since last consolidation (min %ds)",
            elapsed,
            CONSOLIDATION_INTERVAL_SECONDS,
        )
        return
    logger.info("Consolidator: should_consolidate flag set — running consolidation")
    consolidate(**kwargs)
    _last_consolidation_time = time.monotonic()




_LTM_COMPRESS_PROMPT = (
    "You are a long-term memory compression system. The following documents are "
    "episodic memories stored over time. Compress them into a single high-level "
    "summary that preserves key facts, patterns, decisions, and context. "
    "Be factual and concise.\n\nDocuments:\n{documents}\n\nCompressed memory:"
)


def consolidate_ltm(
    *,
    collection: str = DEFAULT_COLLECTION,
    min_docs: int = LTM_SUMMARIZATION_MIN_DOCS,
    embedding_fn: Embeddings | None = None,
    chroma_client: Any = None,
    llm: Any = None,
) -> int:
    """Compress all documents in an LTM collection into a single meta-summary.

    Reads every document currently in *collection*, asks the LLM to produce a
    high-level compressed memory, deletes the originals, and stores the
    meta-summary back in the same collection.

    Skips (returns 0) when the document count is below *min_docs*, preventing
    compression of an already-small collection.

    Parameters
    ----------
    collection:    Target ChromaDB collection (default: ``"episodic"``).
    min_docs:      Minimum number of documents required to trigger compression.
    embedding_fn:  Embeddings callable for ChromaDB.
    chroma_client: ChromaDB client.
    llm:           LLM instance (uses configured provider if None).

    Returns
    -------
    Number of documents replaced by the meta-summary (0 when skipped).
    """
    _client = chroma_client if chroma_client is not None else _http_client()
    col = _client.get_or_create_collection(collection)

    result = col.get(include=["documents", "metadatas"])
    docs: list[str] = result.get("documents") or []
    ids: list[str] = result.get("ids") or []

    if len(docs) < min_docs:
        logger.info(
            "LTM consolidate: %d docs < min_docs=%d in collection=%r — skipping",
            len(docs),
            min_docs,
            collection,
        )
        return 0

    from providers.llm import get_llm

    _llm = llm or get_llm(model_name=CONSOLIDATION_MODEL, temperature=CONSOLIDATION_LLM_TEMPERATURE)
    numbered = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(docs))
    prompt = _LTM_COMPRESS_PROMPT.format(documents=numbered)
    from langchain_core.messages import HumanMessage as _HumanMessage
    response = _llm.invoke([_HumanMessage(content=prompt)])
    meta_summary = str(response.content).strip()

    col.delete(ids=ids)
    add_to_ltm(
        meta_summary,
        {
            "source": "ltm_compressed",
            "doc_count": len(docs),
        },
        collection,
        embedding_fn=embedding_fn,
        chroma_client=_client,
    )
    logger.info(
        "LTM consolidate: compressed %d docs \u2192 1 meta-summary (collection=%r)",
        len(docs),
        collection,
    )
    return len(docs)

def start_scheduler(
    interval_seconds: int = 3600,
    ltm_interval_seconds: int = LTM_SUMMARIZATION_INTERVAL_SECONDS,
    **consolidate_kwargs: Any,
):
    """Start APScheduler background jobs for STM\u2192LTM and LTM\u2192LTM consolidation.

    Two jobs are registered:
    - ``memory_consolidator``: runs :func:`consolidate` (STM \u2192 LTM summary) every
      *interval_seconds*.
    - ``ltm_consolidator``: runs :func:`consolidate_ltm` (LTM \u2192 compressed LTM) every
      *ltm_interval_seconds* (default: daily).

    Parameters
    ----------
    interval_seconds:     STM\u2192LTM interval (default: 30 minutes).
    ltm_interval_seconds: LTM\u2192LTM compression interval (default: 24 hours).
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
    scheduler.add_job(
        consolidate_ltm,
        "interval",
        seconds=ltm_interval_seconds,
        id="ltm_consolidator",
        replace_existing=True,
    )
    scheduler.start()
    logger.info(
        "Consolidator scheduler started (stm_interval=%ds, ltm_interval=%ds, summarize=%s)",
        interval_seconds,
        ltm_interval_seconds,
        consolidate_kwargs.get("summarize", True),
    )
    return scheduler
