"""
memory/vectorstore.py — Swappable vector store interface for LTM.

Defines a ``VectorStoreClient`` Protocol so that any compliant backend
(ChromaDB, Weaviate, Pinecone, or the built-in in-memory stub) can be used
without changing agent code.

Usage
-----
    from memory.vectorstore import ChromaVectorStore, InMemoryVectorStore

    # In production (uses real ChromaDB):
    store = ChromaVectorStore("episodic", embedding_fn=embed, chroma_client=client)

    # In tests / local dev (zero external deps):
    store = InMemoryVectorStore()

    store.add(["Paris is the capital of France"], [{"source": "test"}])
    results = store.search("capital of France", k=3)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class VectorStoreClient(Protocol):
    """Minimal interface every vector-store backend must satisfy."""

    def add(self, texts: list[str], metadatas: list[dict[str, Any]]) -> None:
        """Persist *texts* alongside their *metadatas*."""
        ...

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        """Return up to *k* documents most relevant to *query*.

        Each element of the returned list must have at least:
            ``{"content": str, "metadata": dict}``
        """
        ...

    def reset(self) -> None:
        """Remove all stored documents (useful in tests)."""
        ...


# ---------------------------------------------------------------------------
# In-memory stub
# ---------------------------------------------------------------------------

class InMemoryVectorStore:
    """Pure-Python in-memory vector store — no external services required.

    Similarity is approximated by word-overlap scoring, which is good enough
    for testing interface contracts and unit tests that care about recall
    rather than semantic precision.
    """

    def __init__(self) -> None:
        self._documents: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # VectorStoreClient interface
    # ------------------------------------------------------------------

    def add(self, texts: list[str], metadatas: list[dict[str, Any]]) -> None:
        for text, meta in zip(texts, metadatas or [{}] * len(texts)):
            self._documents.append({"content": text, "metadata": meta or {}})

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        query_words = set(query.lower().split())

        def _score(doc: dict[str, Any]) -> int:
            doc_words = set(doc["content"].lower().split())
            return len(query_words & doc_words)

        ranked = sorted(self._documents, key=_score, reverse=True)
        return ranked[:k]

    def reset(self) -> None:
        self._documents.clear()


# ---------------------------------------------------------------------------
# ChromaDB adapter
# ---------------------------------------------------------------------------

class ChromaVectorStore:
    """Adapter that wraps the existing LTM helpers to satisfy ``VectorStoreClient``.

    This allows any code written against ``VectorStoreClient`` to use the real
    ChromaDB backend without modification.
    """

    def __init__(
        self,
        collection: str = "episodic",
        *,
        embedding_fn: Any = None,
        chroma_client: Any = None,
    ) -> None:
        self._collection = collection
        self._embedding_fn = embedding_fn
        self._chroma_client = chroma_client

    # ------------------------------------------------------------------
    # VectorStoreClient interface
    # ------------------------------------------------------------------

    def add(self, texts: list[str], metadatas: list[dict[str, Any]]) -> None:
        from memory.long_term import add_batch_to_ltm

        add_batch_to_ltm(
            texts,
            metadatas,
            self._collection,
            embedding_fn=self._embedding_fn,
            chroma_client=self._chroma_client,
        )

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        from memory.long_term import search_ltm_with_metadata

        return search_ltm_with_metadata(
            query,
            self._collection,
            k,
            "similarity",
            embedding_fn=self._embedding_fn,
            chroma_client=self._chroma_client,
        )

    def reset(self) -> None:
        if self._chroma_client and hasattr(self._chroma_client, "reset"):
            self._chroma_client.reset()
