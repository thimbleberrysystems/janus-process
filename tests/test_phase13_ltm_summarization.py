"""
tests/test_phase13_ltm_summarization.py — Tests for LTM summarization consolidation.

Covers
------
- _summarize_messages() calls LLM and returns its content string.
- consolidate(summarize=True) stores exactly 1 summary document.
- Summary metadata includes source="summarized", message_count, and hash.
- A second consolidate(summarize=True) with the same batch is deduplicated (0 written).
- A changed batch produces a new summary (1 written).
- consolidate(summarize=True) on empty STM returns 0 without calling LLM.
- start_scheduler() returns a running APScheduler instance and can be shut down.
- CONSOLIDATION_INTERVAL_SECONDS and CONSOLIDATION_WINDOW are exposed in config.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import chromadb
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from memory.consolidator import (
    DEFAULT_COLLECTION,
    _summarize_messages,
    _text_hash,
    consolidate,
    start_scheduler,
)
from memory.long_term import search_ltm_with_metadata

EMBED_DIM = 64
MESSAGES = ["the user asked about sleep", "the assistant explained circadian rhythm", "user said thanks"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_embed() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=EMBED_DIM)


@pytest.fixture()
def chroma():
    client = chromadb.EphemeralClient(settings=chromadb.Settings(allow_reset=True))
    client.reset()
    yield client


def _mock_llm(summary_text: str = "This is a summary.") -> MagicMock:
    """Return a mock LLM whose invoke() returns a response with .content."""
    llm = MagicMock()
    response = MagicMock()
    response.content = summary_text
    llm.invoke.return_value = response
    return llm


# ---------------------------------------------------------------------------
# _summarize_messages()
# ---------------------------------------------------------------------------

class TestSummarizeMessages:
    def test_returns_llm_content_string(self):
        llm = _mock_llm("Summary result.")
        result = _summarize_messages(MESSAGES, llm=llm)
        assert result == "Summary result."

    def test_calls_llm_invoke_once(self):
        llm = _mock_llm()
        _summarize_messages(MESSAGES, llm=llm)
        llm.invoke.assert_called_once()

    def test_prompt_contains_all_messages(self):
        llm = _mock_llm()
        _summarize_messages(MESSAGES, llm=llm)
        call_args = llm.invoke.call_args[0][0]  # list[HumanMessage]
        prompt_text = call_args[0].content
        for msg in MESSAGES:
            assert msg in prompt_text

    def test_strips_whitespace(self):
        llm = _mock_llm("  padded summary  ")
        result = _summarize_messages(["msg"], llm=llm)
        assert result == "padded summary"


# ---------------------------------------------------------------------------
# consolidate(summarize=True) — happy path
# ---------------------------------------------------------------------------

class TestConsolidateSummarize:
    def test_writes_exactly_one_document(self, chroma, fake_embed):
        """Summarising 3 messages → exactly 1 LTM document."""
        llm = _mock_llm("Concise summary.")
        with patch("memory.consolidator.get_short_term_memory", return_value=MESSAGES):
            count = consolidate(summarize=True, chroma_client=chroma, embedding_fn=fake_embed, llm=llm)
        assert count == 1

    def test_collection_contains_one_document(self, chroma, fake_embed):
        llm = _mock_llm("Concise summary.")
        with patch("memory.consolidator.get_short_term_memory", return_value=MESSAGES):
            consolidate(summarize=True, chroma_client=chroma, embedding_fn=fake_embed, llm=llm)
        col = chroma.get_or_create_collection(DEFAULT_COLLECTION)
        assert col.count() == 1

    def test_metadata_source_is_summarized(self, chroma, fake_embed):
        llm = _mock_llm("Summary of the conversation.")
        with patch("memory.consolidator.get_short_term_memory", return_value=MESSAGES):
            consolidate(summarize=True, chroma_client=chroma, embedding_fn=fake_embed, llm=llm)
        results = search_ltm_with_metadata(
            "conversation",
            DEFAULT_COLLECTION,
            k=1,
            search_type="similarity",
            chroma_client=chroma,
            embedding_fn=fake_embed,
        )
        assert results[0]["metadata"]["source"] == "summarized"

    def test_metadata_message_count(self, chroma, fake_embed):
        llm = _mock_llm("summary")
        with patch("memory.consolidator.get_short_term_memory", return_value=MESSAGES):
            consolidate(summarize=True, chroma_client=chroma, embedding_fn=fake_embed, llm=llm)
        results = search_ltm_with_metadata(
            "summary",
            DEFAULT_COLLECTION,
            k=1,
            search_type="similarity",
            chroma_client=chroma,
            embedding_fn=fake_embed,
        )
        assert results[0]["metadata"]["message_count"] == len(MESSAGES)

    def test_metadata_hash_present(self, chroma, fake_embed):
        llm = _mock_llm("summary")
        with patch("memory.consolidator.get_short_term_memory", return_value=MESSAGES):
            consolidate(summarize=True, chroma_client=chroma, embedding_fn=fake_embed, llm=llm)
        results = search_ltm_with_metadata(
            "summary",
            DEFAULT_COLLECTION,
            k=1,
            search_type="similarity",
            chroma_client=chroma,
            embedding_fn=fake_embed,
        )
        expected_hash = _text_hash("\n".join(MESSAGES))
        assert results[0]["metadata"]["hash"] == expected_hash

    def test_stored_text_is_summary_not_raw(self, chroma, fake_embed):
        """The document stored in LTM must be the summary, not a raw message."""
        summary_text = "Distinct summary text XYZ."
        llm = _mock_llm(summary_text)
        with patch("memory.consolidator.get_short_term_memory", return_value=MESSAGES):
            consolidate(summarize=True, chroma_client=chroma, embedding_fn=fake_embed, llm=llm)
        col = chroma.get_or_create_collection(DEFAULT_COLLECTION)
        docs = col.get(include=["documents"])
        assert docs["documents"] == [summary_text]


# ---------------------------------------------------------------------------
# consolidate(summarize=True) — deduplication
# ---------------------------------------------------------------------------

class TestConsolidateSummarizeDedup:
    def test_same_batch_twice_returns_zero_second_time(self, chroma, fake_embed):
        llm = _mock_llm("summary")
        with patch("memory.consolidator.get_short_term_memory", return_value=MESSAGES):
            first = consolidate(summarize=True, chroma_client=chroma, embedding_fn=fake_embed, llm=llm)
            second = consolidate(summarize=True, chroma_client=chroma, embedding_fn=fake_embed, llm=llm)
        assert first == 1
        assert second == 0

    def test_same_batch_collection_count_does_not_double(self, chroma, fake_embed):
        llm = _mock_llm("summary")
        with patch("memory.consolidator.get_short_term_memory", return_value=MESSAGES):
            consolidate(summarize=True, chroma_client=chroma, embedding_fn=fake_embed, llm=llm)
            before = chroma.get_or_create_collection(DEFAULT_COLLECTION).count()
            consolidate(summarize=True, chroma_client=chroma, embedding_fn=fake_embed, llm=llm)
            after = chroma.get_or_create_collection(DEFAULT_COLLECTION).count()
        assert before == after == 1

    def test_different_batch_writes_new_summary(self, chroma, fake_embed):
        different_messages = ["completely different context", "new topic entirely"]
        llm = _mock_llm("summary A")
        with patch("memory.consolidator.get_short_term_memory", return_value=MESSAGES):
            consolidate(summarize=True, chroma_client=chroma, embedding_fn=fake_embed, llm=llm)
        llm2 = _mock_llm("summary B")
        with patch("memory.consolidator.get_short_term_memory", return_value=different_messages):
            second = consolidate(summarize=True, chroma_client=chroma, embedding_fn=fake_embed, llm=llm2)
        assert second == 1
        assert chroma.get_or_create_collection(DEFAULT_COLLECTION).count() == 2


# ---------------------------------------------------------------------------
# consolidate(summarize=True) — edge cases
# ---------------------------------------------------------------------------

class TestConsolidateSummarizeEdgeCases:
    def test_empty_stm_returns_zero_no_llm_call(self, chroma, fake_embed):
        """Empty STM → 0 written, LLM never called."""
        llm = _mock_llm()
        with patch("memory.consolidator.get_short_term_memory", return_value=[]):
            count = consolidate(summarize=True, chroma_client=chroma, embedding_fn=fake_embed, llm=llm)
        assert count == 0
        llm.invoke.assert_not_called()

    def test_empty_stm_collection_stays_empty(self, chroma, fake_embed):
        llm = _mock_llm()
        with patch("memory.consolidator.get_short_term_memory", return_value=[]):
            consolidate(summarize=True, chroma_client=chroma, embedding_fn=fake_embed, llm=llm)
        assert chroma.get_or_create_collection(DEFAULT_COLLECTION).count() == 0


# ---------------------------------------------------------------------------
# start_scheduler()
# ---------------------------------------------------------------------------

class TestStartScheduler:
    def test_returns_running_scheduler(self):
        from apscheduler.schedulers.background import BackgroundScheduler

        scheduler = start_scheduler(interval_seconds=9999)
        try:
            assert isinstance(scheduler, BackgroundScheduler)
            assert scheduler.running
        finally:
            scheduler.shutdown(wait=False)

    def test_scheduler_has_consolidator_job(self):
        scheduler = start_scheduler(interval_seconds=9999)
        try:
            job_ids = [j.id for j in scheduler.get_jobs()]
            assert "memory_consolidator" in job_ids
        finally:
            scheduler.shutdown(wait=False)

    def test_scheduler_can_be_shut_down(self):
        scheduler = start_scheduler(interval_seconds=9999)
        scheduler.shutdown(wait=False)
        assert not scheduler.running


# ---------------------------------------------------------------------------
# Config exports
# ---------------------------------------------------------------------------

class TestConfig:
    def test_consolidation_interval_is_int(self):
        from config import CONSOLIDATION_INTERVAL_SECONDS
        assert isinstance(CONSOLIDATION_INTERVAL_SECONDS, int)
        assert CONSOLIDATION_INTERVAL_SECONDS > 0

    def test_consolidation_window_is_int(self):
        from config import CONSOLIDATION_WINDOW
        assert isinstance(CONSOLIDATION_WINDOW, int)
        assert CONSOLIDATION_WINDOW > 0


# ---------------------------------------------------------------------------
# consolidate_ltm() — LTM-to-LTM compression
# ---------------------------------------------------------------------------

from memory.consolidator import consolidate_ltm  # noqa: E402 — after other imports

MANY_DOCS = [f"episode {i}: the user discussed topic {i}" for i in range(25)]


class TestConsolidateLtm:
    def _seed_docs(self, chroma, fake_embed, texts):
        """Insert texts directly into the ephemeral ChromaDB collection."""
        from memory.long_term import add_batch_to_ltm
        add_batch_to_ltm(
            texts,
            [{"source": "summarized", "hash": f"h{i}"} for i in range(len(texts))],
            DEFAULT_COLLECTION,
            embedding_fn=fake_embed,
            chroma_client=chroma,
        )

    def test_below_min_docs_skips(self, chroma, fake_embed):
        """Fewer docs than min_docs → 0 returned, collection unchanged."""
        self._seed_docs(chroma, fake_embed, MANY_DOCS[:5])
        llm = _mock_llm("meta summary")
        count = consolidate_ltm(
            collection=DEFAULT_COLLECTION,
            min_docs=20,
            embedding_fn=fake_embed,
            chroma_client=chroma,
            llm=llm,
        )
        assert count == 0
        assert chroma.get_or_create_collection(DEFAULT_COLLECTION).count() == 5
        llm.invoke.assert_not_called()

    def test_above_min_docs_compresses(self, chroma, fake_embed):
        """25 docs ≥ min_docs=20 → returns 25, collection has exactly 1 doc."""
        self._seed_docs(chroma, fake_embed, MANY_DOCS)
        llm = _mock_llm("Compressed meta-summary of 25 episodes.")
        count = consolidate_ltm(
            collection=DEFAULT_COLLECTION,
            min_docs=20,
            embedding_fn=fake_embed,
            chroma_client=chroma,
            llm=llm,
        )
        assert count == 25
        assert chroma.get_or_create_collection(DEFAULT_COLLECTION).count() == 1

    def test_stored_doc_is_meta_summary(self, chroma, fake_embed):
        """The stored document must be the LLM output, not a raw doc."""
        self._seed_docs(chroma, fake_embed, MANY_DOCS)
        llm = _mock_llm("Unique meta-summary XYZ.")
        consolidate_ltm(
            collection=DEFAULT_COLLECTION,
            min_docs=20,
            embedding_fn=fake_embed,
            chroma_client=chroma,
            llm=llm,
        )
        col = chroma.get_or_create_collection(DEFAULT_COLLECTION)
        remaining = col.get(include=["documents"])
        assert remaining["documents"] == ["Unique meta-summary XYZ."]

    def test_metadata_source_is_ltm_compressed(self, chroma, fake_embed):
        """Meta-summary must carry source='ltm_compressed'."""
        self._seed_docs(chroma, fake_embed, MANY_DOCS)
        llm = _mock_llm("summary")
        consolidate_ltm(
            collection=DEFAULT_COLLECTION,
            min_docs=20,
            embedding_fn=fake_embed,
            chroma_client=chroma,
            llm=llm,
        )
        results = search_ltm_with_metadata(
            "summary",
            DEFAULT_COLLECTION,
            k=1,
            search_type="similarity",
            chroma_client=chroma,
            embedding_fn=fake_embed,
        )
        assert results[0]["metadata"]["source"] == "ltm_compressed"

    def test_metadata_doc_count(self, chroma, fake_embed):
        """Meta-summary metadata must record the number of compressed docs."""
        self._seed_docs(chroma, fake_embed, MANY_DOCS)
        llm = _mock_llm("summary")
        consolidate_ltm(
            collection=DEFAULT_COLLECTION,
            min_docs=20,
            embedding_fn=fake_embed,
            chroma_client=chroma,
            llm=llm,
        )
        results = search_ltm_with_metadata(
            "summary",
            DEFAULT_COLLECTION,
            k=1,
            search_type="similarity",
            chroma_client=chroma,
            embedding_fn=fake_embed,
        )
        assert results[0]["metadata"]["doc_count"] == 25

    def test_empty_collection_skips(self, chroma, fake_embed):
        """Empty LTM → 0 returned, no LLM call."""
        llm = _mock_llm("summary")
        count = consolidate_ltm(
            collection=DEFAULT_COLLECTION,
            min_docs=1,
            embedding_fn=fake_embed,
            chroma_client=chroma,
            llm=llm,
        )
        assert count == 0
        llm.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# start_scheduler — ltm_consolidator job
# ---------------------------------------------------------------------------

class TestStartSchedulerLtmJob:
    def test_scheduler_has_ltm_consolidator_job(self):
        scheduler = start_scheduler(interval_seconds=9999, ltm_interval_seconds=9999)
        try:
            job_ids = [j.id for j in scheduler.get_jobs()]
            assert "ltm_consolidator" in job_ids
        finally:
            scheduler.shutdown(wait=False)

    def test_both_jobs_registered(self):
        scheduler = start_scheduler(interval_seconds=9999, ltm_interval_seconds=9999)
        try:
            job_ids = [j.id for j in scheduler.get_jobs()]
            assert "memory_consolidator" in job_ids
            assert "ltm_consolidator" in job_ids
        finally:
            scheduler.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Config — new constants
# ---------------------------------------------------------------------------

class TestNewConfig:
    def test_ltm_summarization_interval_is_int(self):
        from config import LTM_SUMMARIZATION_INTERVAL_SECONDS
        assert isinstance(LTM_SUMMARIZATION_INTERVAL_SECONDS, int)
        assert LTM_SUMMARIZATION_INTERVAL_SECONDS > 0

    def test_ltm_summarization_min_docs_is_int(self):
        from config import LTM_SUMMARIZATION_MIN_DOCS
        assert isinstance(LTM_SUMMARIZATION_MIN_DOCS, int)
        assert LTM_SUMMARIZATION_MIN_DOCS > 0

    def test_stm_ttl_greater_than_consolidation_interval(self):
        from config import CONSOLIDATION_INTERVAL_SECONDS, STM_TTL
        assert STM_TTL > CONSOLIDATION_INTERVAL_SECONDS, (
            f"STM_TTL ({STM_TTL}s) must exceed CONSOLIDATION_INTERVAL_SECONDS "
            f"({CONSOLIDATION_INTERVAL_SECONDS}s) to avoid race condition"
        )
