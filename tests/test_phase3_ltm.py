"""
tests/test_phase3_ltm.py — Phase 3: Long-Term Memory test suite.

Unit tests use:
  - chromadb.EphemeralClient()  — in-memory, no container required
  - DeterministicFakeEmbedding  — fixed-dimension embeddings, no LLM calls

Integration tests (marked @pytest.mark.integration) require:
  - Running ChromaDB container on localhost:8000
  - Running Ollama container with the configured embedding model

Run unit tests only:
    pytest tests/test_phase3_ltm.py -m "not integration"

Run all:
    pytest tests/test_phase3_ltm.py
"""
from __future__ import annotations

import chromadb
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

import memory.long_term as ltm_mod
from memory.long_term import (
    COLLECTIONS,
    add_batch_to_ltm,
    add_to_ltm,
    search_ltm,
    search_ltm_with_metadata,
)

# ── Shared fixtures ────────────────────────────────────────────────────────────

EMBED_DIM = 64  # arbitrary; must be consistent within a client


@pytest.fixture()
def fake_embed() -> DeterministicFakeEmbedding:
    """Deterministic fake embeddings — no LLM required."""
    return DeterministicFakeEmbedding(size=EMBED_DIM)


@pytest.fixture()
def mem_client() -> chromadb.ClientAPI:
    """In-memory ChromaDB client.

    EphemeralClient reuses a module-level System singleton, so we call
    reset() before each test to clear any data left by previous tests.
    """
    client = chromadb.EphemeralClient(settings=chromadb.Settings(allow_reset=True))
    client.reset()
    return client


# ── Unit tests (no container) ──────────────────────────────────────────────────


class TestStoreAndRetrieve:
    def test_store_and_retrieve_single(self, mem_client, fake_embed):
        """Store one text; similarity query must return that text."""
        add_to_ltm(
            "The mitochondria is the powerhouse of the cell",
            collection="semantic",
            chroma_client=mem_client,
            embedding_fn=fake_embed,
        )
        results = search_ltm(
            "powerhouse of the cell",
            collection="semantic",
            k=1,
            search_type="similarity",
            chroma_client=mem_client,
            embedding_fn=fake_embed,
        )
        assert len(results) == 1
        assert "mitochondria" in results[0]

    def test_empty_collection_returns_empty_list(self, mem_client, fake_embed):
        """Querying an empty collection must return an empty list."""
        results = search_ltm(
            "anything",
            collection="episodic",
            chroma_client=mem_client,
            embedding_fn=fake_embed,
        )
        assert results == []


class TestMetadata:
    def test_metadata_persisted(self, mem_client, fake_embed):
        """All supplied metadata fields must survive a round-trip."""
        meta = {
            "source": "user",
            "emotional_weight": 0.9,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "type": "episodic",
        }
        add_to_ltm(
            "Today I felt deeply moved",
            metadata=meta,
            collection="episodic",
            chroma_client=mem_client,
            embedding_fn=fake_embed,
        )
        results = search_ltm_with_metadata(
            "felt deeply moved",
            collection="episodic",
            k=1,
            search_type="similarity",
            chroma_client=mem_client,
            embedding_fn=fake_embed,
        )
        assert len(results) == 1
        stored_meta = results[0]["metadata"]
        assert stored_meta["source"] == "user"
        assert stored_meta["emotional_weight"] == pytest.approx(0.9)
        assert stored_meta["timestamp"] == "2026-01-01T00:00:00+00:00"
        assert stored_meta["type"] == "episodic"

    def test_default_metadata_applied(self, mem_client, fake_embed):
        """When no metadata is passed, sensible defaults must be set."""
        add_to_ltm(
            "No metadata provided",
            collection="episodic",
            chroma_client=mem_client,
            embedding_fn=fake_embed,
        )
        results = search_ltm_with_metadata(
            "No metadata provided",
            collection="episodic",
            k=1,
            search_type="similarity",
            chroma_client=mem_client,
            embedding_fn=fake_embed,
        )
        meta = results[0]["metadata"]
        assert meta["source"] == "unknown"
        assert meta["emotional_weight"] == 0.0
        assert meta["timestamp"] != ""  # auto-filled with current time
        assert meta["type"] == "episodic"


class TestBatchIngestion:
    def test_batch_ingestion(self, mem_client, fake_embed):
        """Ingesting 50 texts must store all 50 documents."""
        texts = [f"Memory fragment number {i}" for i in range(50)]
        add_batch_to_ltm(
            texts,
            collection="procedural",
            chroma_client=mem_client,
            embedding_fn=fake_embed,
        )
        # Retrieve with a high k to verify count
        results = search_ltm(
            "Memory fragment",
            collection="procedural",
            k=50,
            search_type="similarity",
            chroma_client=mem_client,
            embedding_fn=fake_embed,
        )
        assert len(results) == 50

    def test_empty_batch_is_no_op(self, mem_client, fake_embed):
        """Passing an empty list must not raise and must not change the store."""
        add_batch_to_ltm(
            [],
            collection="episodic",
            chroma_client=mem_client,
            embedding_fn=fake_embed,
        )
        results = search_ltm(
            "anything",
            collection="episodic",
            chroma_client=mem_client,
            embedding_fn=fake_embed,
        )
        assert results == []


class TestCollectionIsolation:
    def test_collection_isolation(self, mem_client, fake_embed):
        """Writing to 'episodic' must not appear in 'semantic' queries."""
        add_to_ltm(
            "Episodic memory entry",
            collection="episodic",
            chroma_client=mem_client,
            embedding_fn=fake_embed,
        )
        results = search_ltm(
            "Episodic memory entry",
            collection="semantic",
            k=5,
            search_type="similarity",
            chroma_client=mem_client,
            embedding_fn=fake_embed,
        )
        assert results == []

    def test_unknown_collection_raises(self, mem_client, fake_embed):
        """Passing an unknown collection name must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown collection"):
            add_to_ltm(
                "bad collection",
                collection="nonexistent",
                chroma_client=mem_client,
                embedding_fn=fake_embed,
            )


class TestMmrDiversity:
    def test_mmr_returns_diverse_results(self, mem_client, fake_embed):
        """
        MMR should return k results from a diverse corpus.
        We store 10 distinct texts and expect 3 unique results.
        """
        topics = [
            "apple fruit",
            "car engine",
            "ocean wave",
            "mountain peak",
            "jazz music",
            "quantum physics",
            "ancient history",
            "cooking pasta",
            "cloud computing",
            "flower garden",
        ]
        add_batch_to_ltm(
            topics,
            collection="semantic",
            chroma_client=mem_client,
            embedding_fn=fake_embed,
        )
        results = search_ltm(
            "nature and science",
            collection="semantic",
            k=3,
            search_type="mmr",
            chroma_client=mem_client,
            embedding_fn=fake_embed,
        )
        assert len(results) == 3
        assert len(set(results)) == 3  # all distinct


# ── Integration tests (require live ChromaDB + Ollama) ────────────────────────


@pytest.mark.integration
class TestIntegrationChromaDB:
    """Requires `docker.exe compose up -d` with janus-chroma running."""

    def test_store_and_retrieve_integration(self):
        """Round-trip store + retrieve against the real ChromaDB container."""
        from providers.llm import get_embedding_model

        client = chromadb.HttpClient(host="localhost", port=8000)
        embed = get_embedding_model()

        # Use a unique text to avoid collisions with existing data
        unique_text = "phase3_integration_test_canary_alpha"

        add_to_ltm(
            unique_text,
            metadata={"source": "integration_test", "type": "episodic"},
            collection="episodic",
            chroma_client=client,
            embedding_fn=embed,
        )
        results = search_ltm(
            unique_text,
            collection="episodic",
            k=1,
            search_type="similarity",
            chroma_client=client,
            embedding_fn=embed,
        )
        assert any(unique_text in r for r in results)
