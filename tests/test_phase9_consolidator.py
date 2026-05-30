"""
tests/test_phase9_consolidator.py — Unit and integration tests for the Memory
Consolidator.

Unit tests
----------
- Redis (STM) is mocked via ``unittest.mock.patch`` so no Redis container is
  needed.
- ChromaDB is provided by an in-memory ``EphemeralClient`` that is reset
  before every test.
- Embeddings use ``DeterministicFakeEmbedding`` (no LLM calls).

Integration tests
-----------------
Marked ``@pytest.mark.integration`` — require live Redis + ChromaDB + Ollama.
"""

from __future__ import annotations

from unittest.mock import ANY, patch

import chromadb
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from memory.consolidator import (
    DEFAULT_COLLECTION,
    _is_duplicate,
    _text_hash,
    consolidate,
    maybe_consolidate,
)
from memory.long_term import search_ltm_with_metadata
from models.brain_state import default_brain_state

EMBED_DIM = 64


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_embed() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=EMBED_DIM)


@pytest.fixture()
def chroma():
    """Fresh in-memory ChromaDB client — reset before each test."""
    client = chromadb.EphemeralClient(settings=chromadb.Settings(allow_reset=True))
    client.reset()
    yield client


# ---------------------------------------------------------------------------
# _text_hash unit tests
# ---------------------------------------------------------------------------

class TestTextHash:
    def test_returns_32_char_hex(self):
        h = _text_hash("hello")
        assert len(h) == 32
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        assert _text_hash("same text") == _text_hash("same text")

    def test_different_inputs_different_hashes(self):
        assert _text_hash("foo") != _text_hash("bar")


# ---------------------------------------------------------------------------
# _is_duplicate unit tests
# ---------------------------------------------------------------------------

class TestIsDuplicate:
    def test_empty_collection_is_not_duplicate(self, chroma):
        assert not _is_duplicate("abc123", DEFAULT_COLLECTION, chroma)

    def test_after_adding_document_detects_duplicate(self, chroma, fake_embed):
        text = "unique message"
        h = _text_hash(text)
        # Seed directly via the raw client to simulate a prior consolidation.
        col = chroma.get_or_create_collection(DEFAULT_COLLECTION)
        col.add(ids=[h], documents=[text], metadatas=[{"hash": h}])
        assert _is_duplicate(h, DEFAULT_COLLECTION, chroma)

    def test_different_hash_not_duplicate(self, chroma, fake_embed):
        col = chroma.get_or_create_collection(DEFAULT_COLLECTION)
        col.add(ids=["aaa"], documents=["doc"], metadatas=[{"hash": "aaa"}])
        assert not _is_duplicate("bbb", DEFAULT_COLLECTION, chroma)


# ---------------------------------------------------------------------------
# consolidate(summarize=False, ) unit tests
# ---------------------------------------------------------------------------

FIVE_MESSAGES = [f"memory item {i}" for i in range(5)]


class TestConsolidate:
    def test_consolidate_writes_to_ltm(self, chroma, fake_embed):
        """5 new STM messages → all 5 written to LTM."""
        with patch("memory.consolidator.get_short_term_memory", return_value=FIVE_MESSAGES):
            count = consolidate(summarize=False, chroma_client=chroma, embedding_fn=fake_embed)
        assert count == 5

    def test_consolidate_writes_correct_count(self, chroma, fake_embed):
        """Returned count matches actual documents in ChromaDB."""
        with patch("memory.consolidator.get_short_term_memory", return_value=FIVE_MESSAGES):
            count = consolidate(summarize=False, chroma_client=chroma, embedding_fn=fake_embed)
        col = chroma.get_or_create_collection(DEFAULT_COLLECTION)
        assert col.count() == count

    def test_deduplication_skips_existing(self, chroma, fake_embed):
        """Running consolidation twice: second pass writes 0 new documents."""
        with patch("memory.consolidator.get_short_term_memory", return_value=FIVE_MESSAGES):
            first = consolidate(summarize=False, chroma_client=chroma, embedding_fn=fake_embed)
            second = consolidate(summarize=False, chroma_client=chroma, embedding_fn=fake_embed)
        assert first == 5
        assert second == 0

    def test_deduplication_collection_count_does_not_double(self, chroma, fake_embed):
        """ChromaDB collection count stays the same after re-consolidation."""
        with patch("memory.consolidator.get_short_term_memory", return_value=FIVE_MESSAGES):
            consolidate(summarize=False, chroma_client=chroma, embedding_fn=fake_embed)
            before = chroma.get_or_create_collection(DEFAULT_COLLECTION).count()
            consolidate(summarize=False, chroma_client=chroma, embedding_fn=fake_embed)
            after = chroma.get_or_create_collection(DEFAULT_COLLECTION).count()
        assert before == after == 5

    def test_replay_window_respected(self, chroma, fake_embed):
        """consolidate(summarize=False, ) passes *window* as n to get_short_term_memory."""
        ten_messages = [f"msg {i}" for i in range(10)]
        with patch(
            "memory.consolidator.get_short_term_memory", return_value=ten_messages
        ) as mock_stm:
            count = consolidate(summarize=False, window=10, chroma_client=chroma, embedding_fn=fake_embed)
        mock_stm.assert_called_once_with(10, session_id=ANY, ttl=ANY)
        assert count == 10

    def test_metadata_source_is_consolidated(self, chroma, fake_embed):
        """Every stored document must carry source='consolidated'."""
        with patch(
            "memory.consolidator.get_short_term_memory",
            return_value=["the test message"],
        ):
            consolidate(summarize=False, chroma_client=chroma, embedding_fn=fake_embed)
        results = search_ltm_with_metadata(
            "test message",
            DEFAULT_COLLECTION,
            k=1,
            search_type="similarity",
            chroma_client=chroma,
            embedding_fn=fake_embed,
        )
        assert len(results) == 1
        assert results[0]["metadata"]["source"] == "consolidated"

    def test_metadata_hash_is_stored(self, chroma, fake_embed):
        """Stored documents must include a 'hash' metadata field."""
        text = "hash metadata test"
        with patch(
            "memory.consolidator.get_short_term_memory", return_value=[text]
        ):
            consolidate(summarize=False, chroma_client=chroma, embedding_fn=fake_embed)
        results = search_ltm_with_metadata(
            text,
            DEFAULT_COLLECTION,
            k=1,
            search_type="similarity",
            chroma_client=chroma,
            embedding_fn=fake_embed,
        )
        assert "hash" in results[0]["metadata"]
        assert results[0]["metadata"]["hash"] == _text_hash(text)

    def test_empty_stm_does_nothing(self, chroma, fake_embed):
        """Empty STM → 0 written, ChromaDB collection stays empty."""
        with patch("memory.consolidator.get_short_term_memory", return_value=[]):
            count = consolidate(summarize=False, chroma_client=chroma, embedding_fn=fake_embed)
        assert count == 0
        col = chroma.get_or_create_collection(DEFAULT_COLLECTION)
        assert col.count() == 0

    def test_empty_stm_does_not_raise(self, chroma, fake_embed):
        """Consolidating an empty STM must never raise an exception."""
        with patch("memory.consolidator.get_short_term_memory", return_value=[]):
            consolidate(summarize=False, chroma_client=chroma, embedding_fn=fake_embed)  # no exception

    def test_partial_deduplication(self, chroma, fake_embed):
        """3 old + 2 new messages → only 2 written on second pass."""
        first_batch = [f"old msg {i}" for i in range(3)]
        combined = first_batch + ["new msg A", "new msg B"]
        with patch("memory.consolidator.get_short_term_memory", return_value=first_batch):
            consolidate(summarize=False, chroma_client=chroma, embedding_fn=fake_embed)
        with patch("memory.consolidator.get_short_term_memory", return_value=combined):
            count = consolidate(summarize=False, chroma_client=chroma, embedding_fn=fake_embed)
        assert count == 2


# ---------------------------------------------------------------------------
# maybe_consolidate() tests
# ---------------------------------------------------------------------------

class TestMaybeConsolidate:
    def setup_method(self):
        """Reset the on-demand cooldown clock before every test."""
        import memory.consolidator as _m
        _m._last_consolidation_time = 0.0

    def test_cooldown_prevents_rapid_refire(self, chroma, fake_embed):
        """Second call within CONSOLIDATION_INTERVAL_SECONDS must be skipped."""
        state = {**default_brain_state("test"), "should_consolidate": True}
        with patch("memory.consolidator.consolidate"):
            maybe_consolidate(state, chroma_client=chroma, embedding_fn=fake_embed)
        # _last_consolidation_time is now ~now; second call should be skipped
        with patch("memory.consolidator.consolidate") as mock2:
            maybe_consolidate(state, chroma_client=chroma, embedding_fn=fake_embed)
        mock2.assert_not_called()

    def test_consolidator_triggered_by_flag(self, chroma, fake_embed):
        """should_consolidate=True → consolidate() is called once."""
        state = {**default_brain_state("test"), "should_consolidate": True}
        with patch("memory.consolidator.consolidate") as mock_consolidate:
            maybe_consolidate(state, chroma_client=chroma, embedding_fn=fake_embed)
        mock_consolidate.assert_called_once()

    def test_consolidator_not_triggered_without_flag(self, chroma, fake_embed):
        """should_consolidate=False → consolidate() is NOT called."""
        state = {**default_brain_state("test"), "should_consolidate": False}
        with patch("memory.consolidator.consolidate") as mock_consolidate:
            maybe_consolidate(state, chroma_client=chroma, embedding_fn=fake_embed)
        mock_consolidate.assert_not_called()

    def test_consolidator_kwargs_forwarded(self, chroma, fake_embed):
        """Keyword args passed to maybe_consolidate are forwarded to consolidate."""
        state = {**default_brain_state("test"), "should_consolidate": True}
        with patch("memory.consolidator.consolidate") as mock_consolidate:
            maybe_consolidate(
                state,
                chroma_client=chroma,
                embedding_fn=fake_embed,
                window=10,
                collection="semantic",
            )
        mock_consolidate.assert_called_once_with(
            chroma_client=chroma,
            embedding_fn=fake_embed,
            window=10,
            collection="semantic",
        )


# ---------------------------------------------------------------------------
# Integration tests (require live Redis + ChromaDB)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestConsolidatorIntegration:
    def test_consolidate_live_stm_to_ltm(self):
        """Seed real Redis, run consolidation, verify ChromaDB is populated."""
        from memory.long_term import search_ltm
        from memory.short_term import add_to_short_term, clear_short_term

        session_id = "test-consolidator-integration"
        clear_short_term(session_id=session_id, ttl=60)

        messages = ["integration test memory alpha", "integration test memory beta"]
        for msg in messages:
            add_to_short_term(msg, session_id=session_id, ttl=60)

        written = consolidate(session_id=session_id, ttl=60, collection="episodic")
        assert written == len(messages)

        results = search_ltm(
            "integration test memory", "episodic", k=5, search_type="similarity"
        )
        assert len(results) >= 1

    def test_consolidate_integration_deduplication(self):
        """Running consolidation twice with live services writes 0 on second pass."""
        from memory.short_term import add_to_short_term, clear_short_term

        session_id = "test-consolidator-dedup-integration"
        clear_short_term(session_id=session_id, ttl=60)
        add_to_short_term("dedup check message", session_id=session_id, ttl=60)

        first = consolidate(session_id=session_id, ttl=60, collection="episodic")
        second = consolidate(session_id=session_id, ttl=60, collection="episodic")
        assert first == 1
        assert second == 0
