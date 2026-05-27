"""
agents/hippocampus.py — Memory Encoding & Retrieval Node.

The Hippocampus runs after the Amygdala.  It does two things atomically:

1. **Retrieve** — query the LTM (ChromaDB) for the top-k memories most
   similar to the current input; write them to ``state["retrieved_memories"]``.

2. **Encode** — store the current input in the ``episodic`` LTM collection
   with the emotional weight set by the Amygdala as metadata.

Usage as a LangGraph node::

    from agents.hippocampus import hippocampus_node
    graph.add_node("hippocampus", hippocampus_node)
"""
from __future__ import annotations

from typing import Any

import chromadb
from langchain_core.embeddings import Embeddings

from memory.long_term import add_to_ltm, search_ltm_with_metadata
from models.brain_state import BrainState

# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULT_K: int = 5
DEFAULT_SEARCH_TYPE: str = "mmr"
DEFAULT_COLLECTION: str = "episodic"


# ── LangGraph node ─────────────────────────────────────────────────────────────

def hippocampus_node(
    state: BrainState,
    *,
    k: int = DEFAULT_K,
    search_type: str = DEFAULT_SEARCH_TYPE,
    collection: str = DEFAULT_COLLECTION,
    embedding_fn: Embeddings | None = None,
    chroma_client: chromadb.ClientAPI | None = None,
) -> BrainState:
    """
    LangGraph node — retrieve relevant memories then encode the current input.

    Reads
        ``state["input"]``            — text to retrieve against and to store
        ``state["emotional_weight"]`` — attached as metadata on the stored doc

    Writes
        ``state["retrieved_memories"]`` — list of relevant memory strings

    Args:
        state:         Current BrainState.
        k:             Number of memories to retrieve (default 5).
        search_type:   "mmr" (diverse) or "similarity".
        collection:    ChromaDB collection name (default "episodic").
        embedding_fn:  Embeddings override for testing.
        chroma_client: ChromaDB client override for testing.

    Returns:
        Updated BrainState with ``retrieved_memories`` populated.
    """
    shared: dict[str, Any] = {
        "collection": collection,
        "embedding_fn": embedding_fn,
        "chroma_client": chroma_client,
    }

    # 1. Retrieve relevant memories BEFORE encoding (avoids self-matching)
    hits = search_ltm_with_metadata(
        state["input"],
        k=k,
        search_type=search_type,
        **shared,
    )
    retrieved: list[str] = [h["content"] for h in hits]

    # 2. Encode current input into LTM
    add_to_ltm(
        state["input"],
        metadata={
            "source": "user",
            "emotional_weight": float(state["emotional_weight"]),
            "type": collection,
        },
        **shared,
    )

    return {**state, "retrieved_memories": retrieved}
