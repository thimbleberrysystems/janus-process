"""
tests/test_phase5_hippocampus.py — Phase 5: Hippocampus Agent test suite.

Unit tests (default) use EphemeralClient + DeterministicFakeEmbedding so no
containers or LLM calls are needed.

Integration tests (``@pytest.mark.integration``) require janus-chroma on
localhost:8000 and the configured Ollama embedding model.

Run unit tests only:
    pytest tests/test_phase5_hippocampus.py -m "not integration"

Run all:
    pytest tests/test_phase5_hippocampus.py
"""
from __future__ import annotations

import chromadb
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from agents.hippocampus import hippocampus_node
from memory.long_term import add_to_ltm, search_ltm, search_ltm_with_metadata
from models.brain_state import default_brain_state

# ── Fixtures ───────────────────────────────────────────────────────────────────

EMBED_DIM = 64


@pytest.fixture()
def fake_embed() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=EMBED_DIM)


@pytest.fixture()
def mem_client() -> chromadb.ClientAPI:
    """Fresh in-memory ChromaDB; reset shared singleton before each test."""
    client = chromadb.EphemeralClient(settings=chromadb.Settings(allow_reset=True))
    client.reset()
    return client


@pytest.fixture()
def node_kwargs(mem_client, fake_embed) -> dict:
    """Convenience kwargs for hippocampus_node calls in unit tests."""
    return {"chroma_client": mem_client, "embedding_fn": fake_embed}


# ── Unit tests ─────────────────────────────────────────────────────────────────


class TestEncodeAndRetrieve:
    def test_encode_then_retrieve(self, mem_client, fake_embed, node_kwargs):
        """After the node runs, the input must be findable in LTM directly."""
        state = default_brain_state("The sky is blue today")
        hippocampus_node(state, **node_kwargs)

        # Query LTM directly — bypassing the node
        results = search_ltm(
            "sky",
            collection="episodic",
            k=5,
            search_type="similarity",
            chroma_client=mem_client,
            embedding_fn=fake_embed,
        )
        assert any("sky" in r for r in results)

    def test_retrieved_memories_in_state(self, mem_client, fake_embed, node_kwargs):
        """Pre-seed LTM; node must populate retrieved_memories."""
        # Seed two memories
        for text in ["Paris is the capital of France", "Rome is the capital of Italy"]:
            add_to_ltm(
                text,
                collection="episodic",
                chroma_client=mem_client,
                embedding_fn=fake_embed,
            )

        state = default_brain_state("What is the capital of France?")
        result = hippocampus_node(state, k=2, **node_kwargs)

        assert isinstance(result["retrieved_memories"], list)
        assert len(result["retrieved_memories"]) > 0

    def test_retrieved_memories_are_strings(self, mem_client, fake_embed, node_kwargs):
        """Every item in retrieved_memories must be a plain string."""
        add_to_ltm(
            "some memory",
            collection="episodic",
            chroma_client=mem_client,
            embedding_fn=fake_embed,
        )
        state = default_brain_state("some memory")
        result = hippocampus_node(state, **node_kwargs)

        for item in result["retrieved_memories"]:
            assert isinstance(item, str)


class TestMetadataStorage:
    def test_emotional_weight_stored_as_metadata(self, mem_client, fake_embed, node_kwargs):
        """emotional_weight from state must be persisted in LTM metadata."""
        state = default_brain_state("I am very upset right now")
        state["emotional_weight"] = 0.9

        hippocampus_node(state, **node_kwargs)

        hits = search_ltm_with_metadata(
            "upset",
            collection="episodic",
            k=5,
            search_type="similarity",
            chroma_client=mem_client,
            embedding_fn=fake_embed,
        )
        assert len(hits) > 0
        stored_weight = hits[0]["metadata"]["emotional_weight"]
        assert stored_weight == pytest.approx(0.9)

    def test_source_metadata_is_user(self, mem_client, fake_embed, node_kwargs):
        """Stored documents must have source='user'."""
        state = default_brain_state("checking source metadata")
        hippocampus_node(state, **node_kwargs)

        hits = search_ltm_with_metadata(
            "checking source",
            collection="episodic",
            k=1,
            search_type="similarity",
            chroma_client=mem_client,
            embedding_fn=fake_embed,
        )
        assert hits[0]["metadata"]["source"] == "user"


class TestStateImmutability:
    def test_state_fields_unchanged(self, mem_client, fake_embed, node_kwargs):
        """All BrainState fields except retrieved_memories must be unchanged."""
        state = default_brain_state("hello")
        state["emotional_weight"] = 0.4
        state["working_memory"] = ["prior message"]
        state["should_consolidate"] = False
        state["final_response"] = None

        result = hippocampus_node(state, **node_kwargs)

        assert result["input"] == "hello"
        assert result["emotional_weight"] == pytest.approx(0.4)
        assert result["working_memory"] == ["prior message"]
        assert result["should_consolidate"] is False
        assert result["final_response"] is None

    def test_only_retrieved_memories_changes(self, mem_client, fake_embed, node_kwargs):
        """No field other than retrieved_memories may change after the node runs."""
        state = default_brain_state("anything")
        result = hippocampus_node(state, **node_kwargs)

        diff = {k for k in state if result.get(k) != state.get(k)}
        # diff may be empty (when LTM is empty, retrieved_memories stays [])
        # but must never contain any field other than retrieved_memories
        assert diff.issubset({"retrieved_memories"})


class TestIrrelevantQuery:
    def test_irrelevant_query_returns_empty_or_low_count(
        self, mem_client, fake_embed, node_kwargs
    ):
        """
        Seed with 3 memories; run node with an orthogonal query.
        The retrieval should return at most k=3 items but we primarily
        verify it does not crash and returns a list.
        (With fake embeddings the relevance scores are random; we can only
        assert the shape, not the semantic quality.)
        """
        for text in ["cats like fish", "dogs enjoy walks", "birds sing at dawn"]:
            add_to_ltm(
                text,
                collection="episodic",
                chroma_client=mem_client,
                embedding_fn=fake_embed,
            )

        state = default_brain_state("quantum entanglement in photonics")
        result = hippocampus_node(state, k=3, **node_kwargs)

        assert isinstance(result["retrieved_memories"], list)
        assert len(result["retrieved_memories"]) <= 3

    def test_empty_ltm_returns_empty_list(self, mem_client, fake_embed, node_kwargs):
        """Node run against an empty collection must yield an empty list."""
        state = default_brain_state("anything at all")
        result = hippocampus_node(state, **node_kwargs)

        assert result["retrieved_memories"] == []


# ── Integration tests (require live ChromaDB + Ollama) ────────────────────────


@pytest.mark.integration
class TestHippocampusIntegration:
    """Requires `docker.exe compose up -d` with janus-chroma and janus-ollama."""

    def test_encode_and_retrieve_integration(self):
        """Round-trip encode + retrieve against the real ChromaDB container."""
        from providers.llm import get_embedding_model

        client = chromadb.HttpClient(host="localhost", port=8000)
        embed = get_embedding_model()

        unique = "phase5_integration_hippocampus_canary_delta"
        state = default_brain_state(unique)
        state["emotional_weight"] = 0.5

        hippocampus_node(
            state,
            chroma_client=client,
            embedding_fn=embed,
        )

        # retrieved_memories may be empty on first run (encoded after retrieval)
        # but the second run should find it
        result2 = hippocampus_node(
            default_brain_state(unique),
            chroma_client=client,
            embedding_fn=embed,
        )
        assert any(unique in r for r in result2["retrieved_memories"])
