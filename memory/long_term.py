"""
memory/long_term.py — ChromaDB-backed long-term memory (LTM).

Exposes three named collections (episodic, semantic, procedural) and two
public helpers consumed by agent nodes and the consolidator:

    add_to_ltm(text, metadata, collection)
    add_batch_to_ltm(texts, metadatas, collection)
    search_ltm(query, collection, k, search_type)

All functions accept optional *chroma_client* and *embedding_fn* kwargs so
tests can inject an in-memory client and deterministic embeddings without
touching the real ChromaDB container.

Metadata schema (all fields optional at call-site; defaults applied here):
    {
        "source":           str   — e.g. "user", "consolidated", "agent"
        "emotional_weight": float — Amygdala score at time of storage
        "timestamp":        str   — ISO-8601 UTC
        "type":             str   — "episodic" | "semantic" | "procedural"
    }
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from config import CHROMA_HOST, CHROMA_PORT
from providers.llm import get_embedding_model

# ── Constants ──────────────────────────────────────────────────────────────────

COLLECTIONS: tuple[str, ...] = ("episodic", "semantic", "procedural")

_DEFAULT_METADATA: dict[str, Any] = {
    "source": "unknown",
    "emotional_weight": 0.0,
    "timestamp": "",
    "type": "episodic",
}


# ── Internal helpers ───────────────────────────────────────────────────────────

def _http_client() -> chromadb.ClientAPI:
    """Return an HttpClient pointing at the configured ChromaDB service."""
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _fill_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Return a metadata dict with all required keys, applying defaults."""
    base = dict(_DEFAULT_METADATA)
    base["timestamp"] = _now_iso()
    if metadata:
        base.update(metadata)
    return base


def _get_store(
    collection: str,
    *,
    embedding_fn: Embeddings | None = None,
    chroma_client: chromadb.ClientAPI | None = None,
) -> Chroma:
    """
    Return a Chroma vector store for *collection*.

    Args:
        collection:    One of COLLECTIONS.
        embedding_fn:  Embeddings instance (defaults to providers.llm.get_embedding_model()).
        chroma_client: ChromaDB client (defaults to HttpClient from config).
    """
    if collection not in COLLECTIONS:
        raise ValueError(f"Unknown collection {collection!r}; valid: {COLLECTIONS}")
    return Chroma(
        collection_name=collection,
        embedding_function=embedding_fn or get_embedding_model(),
        client=chroma_client or _http_client(),
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def add_to_ltm(
    text: str,
    metadata: dict[str, Any] | None = None,
    collection: str = "episodic",
    *,
    embedding_fn: Embeddings | None = None,
    chroma_client: chromadb.ClientAPI | None = None,
) -> None:
    """Store a single *text* with optional *metadata* in *collection*."""
    store = _get_store(collection, embedding_fn=embedding_fn, chroma_client=chroma_client)
    store.add_texts([text], metadatas=[_fill_metadata(metadata)])


def add_batch_to_ltm(
    texts: list[str],
    metadatas: list[dict[str, Any]] | None = None,
    collection: str = "episodic",
    *,
    embedding_fn: Embeddings | None = None,
    chroma_client: chromadb.ClientAPI | None = None,
) -> None:
    """Store a batch of texts with optional per-item metadata in *collection*."""
    if not texts:
        return
    nones: list[dict[str, Any] | None] = [None] * len(texts)
    filled = [_fill_metadata(m) for m in (metadatas or nones)]
    store = _get_store(collection, embedding_fn=embedding_fn, chroma_client=chroma_client)
    store.add_texts(texts, metadatas=filled)


def search_ltm(
    query: str,
    collection: str = "episodic",
    k: int = 5,
    search_type: str = "mmr",
    *,
    embedding_fn: Embeddings | None = None,
    chroma_client: chromadb.ClientAPI | None = None,
) -> list[str]:
    """
    Retrieve the top-*k* documents from *collection* relevant to *query*.

    Args:
        query:       Natural-language search string.
        collection:  Which collection to search.
        k:           Number of results to return.
        search_type: "mmr" (diverse) or "similarity".

    Returns:
        List of document content strings; empty list when nothing is stored.
    """
    store = _get_store(collection, embedding_fn=embedding_fn, chroma_client=chroma_client)

    if search_type == "mmr":
        docs = store.max_marginal_relevance_search(query, k=k)
    else:
        docs = store.similarity_search(query, k=k)

    return [doc.page_content for doc in docs]


def search_ltm_with_metadata(
    query: str,
    collection: str = "episodic",
    k: int = 5,
    search_type: str = "mmr",
    *,
    embedding_fn: Embeddings | None = None,
    chroma_client: chromadb.ClientAPI | None = None,
) -> list[dict[str, Any]]:
    """
    Like search_ltm but returns dicts with both *content* and *metadata* keys.
    Used by the Hippocampus agent and tests that verify metadata persistence.
    """
    store = _get_store(collection, embedding_fn=embedding_fn, chroma_client=chroma_client)

    if search_type == "mmr":
        docs = store.max_marginal_relevance_search(query, k=k)
    else:
        docs = store.similarity_search(query, k=k)

    return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
