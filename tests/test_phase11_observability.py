"""
tests/test_phase11_observability.py — Unit and integration tests for the
Observability & Evaluation layer.

Unit tests
----------
- ``instrument_node`` wrapper behaviour (no LLM calls).
- ``Metrics`` counters.
- Evaluation dataset file structure.

Integration tests
-----------------
Marked ``@pytest.mark.integration`` — require live LLM + LangSmith API key.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from models.brain_state import default_brain_state
from observability import Metrics, _extract_tokens, instrument_node, metrics

EVAL_DATASET_PATH = Path(__file__).parent.parent / "eval" / "dataset.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_node(state: dict) -> dict:
    """Trivial node that returns state unchanged."""
    return state


def _slow_node(state: dict) -> dict:
    """Node that takes at least 5 ms so latency is measurable."""
    time.sleep(0.005)
    return state


def _token_node(state: dict) -> dict:
    """Node that stashes a token count in state."""
    return {**state, "_token_usage": 42}


# ---------------------------------------------------------------------------
# instrument_node — behaviour tests
# ---------------------------------------------------------------------------

class TestInstrumentNode:
    def test_returns_result_unchanged(self):
        """Instrumented node must return the same state the inner fn returns."""
        wrapped = instrument_node("test", _dummy_node)
        state = default_brain_state("hello")
        result = wrapped(state)
        assert result["input"] == "hello"

    def test_calls_inner_function(self):
        """The original node fn must be invoked exactly once."""
        mock_fn = MagicMock(side_effect=lambda s: s)
        wrapped = instrument_node("mock", mock_fn)
        state = default_brain_state("hi")
        wrapped(state)
        mock_fn.assert_called_once_with(state)

    def test_preserves_inner_function_name(self):
        """functools.wraps must propagate the original __name__."""
        wrapped = instrument_node("test", _dummy_node)
        assert wrapped.__name__ == "_dummy_node"

    def test_log_contains_agent_name_and_latency(self, caplog):
        """Log record must carry 'agent' and 'latency_ms' extra fields."""
        wrapped = instrument_node("amygdala", _dummy_node)
        with caplog.at_level(logging.INFO, logger="janus.observability"):
            wrapped(default_brain_state("test"))

        assert caplog.records, "Expected at least one log record"
        record = caplog.records[-1]
        assert hasattr(record, "agent"), "Log record missing 'agent' field"
        assert record.agent == "amygdala"
        assert hasattr(record, "latency_ms"), "Log record missing 'latency_ms' field"
        assert isinstance(record.latency_ms, float)

    def test_log_contains_tokens_field(self, caplog):
        """Log record must carry a 'tokens' extra field."""
        wrapped = instrument_node("pfc", _dummy_node)
        with caplog.at_level(logging.INFO, logger="janus.observability"):
            wrapped(default_brain_state("test"))

        record = caplog.records[-1]
        assert hasattr(record, "tokens"), "Log record missing 'tokens' field"
        assert isinstance(record.tokens, int)

    def test_latency_ms_is_positive(self, caplog):
        """Measured latency must be ≥ 0."""
        wrapped = instrument_node("basal_ganglia", _dummy_node)
        with caplog.at_level(logging.INFO, logger="janus.observability"):
            wrapped(default_brain_state("hi"))
        assert caplog.records[-1].latency_ms >= 0.0

    def test_latency_reflects_actual_duration(self, caplog):
        """Slow node must record latency ≥ 5 ms."""
        wrapped = instrument_node("slow", _slow_node)
        with caplog.at_level(logging.INFO, logger="janus.observability"):
            wrapped(default_brain_state("test"))
        assert caplog.records[-1].latency_ms >= 5.0

    def test_tokens_extracted_from_state(self, caplog):
        """When node returns _token_usage=42 the log record shows tokens=42."""
        wrapped = instrument_node("llm_node", _token_node)
        with caplog.at_level(logging.INFO, logger="janus.observability"):
            wrapped(default_brain_state("test"))
        assert caplog.records[-1].tokens == 42

    def test_zero_tokens_when_not_set(self, caplog):
        """When _token_usage is absent the log record shows tokens=0."""
        wrapped = instrument_node("no_llm", _dummy_node)
        with caplog.at_level(logging.INFO, logger="janus.observability"):
            wrapped(default_brain_state("test"))
        assert caplog.records[-1].tokens == 0

    def test_multiple_nodes_each_emit_one_record(self, caplog):
        """Each wrapped node emits exactly one log record per call."""
        nodes = [instrument_node(f"node_{i}", _dummy_node) for i in range(3)]
        with caplog.at_level(logging.INFO, logger="janus.observability"):
            for node in nodes:
                node(default_brain_state("test"))
        obs_records = [r for r in caplog.records if r.name == "janus.observability"]
        assert len(obs_records) == 3


# ---------------------------------------------------------------------------
# _extract_tokens helper
# ---------------------------------------------------------------------------

class TestExtractTokens:
    def test_returns_zero_when_absent(self):
        assert _extract_tokens({"input": "hi"}) == 0

    def test_returns_value_when_present(self):
        assert _extract_tokens({"_token_usage": 99}) == 99

    def test_returns_zero_for_non_dict(self):
        assert _extract_tokens(None) == 0
        assert _extract_tokens("string") == 0


# ---------------------------------------------------------------------------
# Metrics singleton
# ---------------------------------------------------------------------------

class TestMetrics:
    @pytest.fixture(autouse=True)
    def reset_metrics(self):
        """Use a fresh Metrics instance for every test."""
        m = Metrics()
        yield m

    def test_initial_state_is_zero(self, reset_metrics):
        m = reset_metrics
        assert m.memory_hits == 0
        assert m.consolidation_count == 0
        assert m.total_latency_ms == 0.0
        assert m.node_invocations == {}

    def test_record_memory_hit(self, reset_metrics):
        m = reset_metrics
        m.record_memory_hit()
        m.record_memory_hit()
        assert m.memory_hits == 2

    def test_record_consolidation(self, reset_metrics):
        m = reset_metrics
        m.record_consolidation()
        assert m.consolidation_count == 1

    def test_record_latency_updates_total(self, reset_metrics):
        m = reset_metrics
        m.record_latency("amygdala", 10.0)
        m.record_latency("pfc", 20.0)
        assert m.total_latency_ms == pytest.approx(30.0)

    def test_record_latency_updates_node_invocations(self, reset_metrics):
        m = reset_metrics
        m.record_latency("amygdala", 5.0)
        m.record_latency("amygdala", 7.0)
        assert m.node_invocations["amygdala"] == 2

    def test_avg_latency_correct(self, reset_metrics):
        m = reset_metrics
        m.record_latency("amygdala", 10.0)
        m.record_latency("pfc", 20.0)
        assert m.avg_latency_ms() == pytest.approx(15.0)

    def test_avg_latency_zero_when_no_records(self, reset_metrics):
        m = reset_metrics
        assert m.avg_latency_ms() == 0.0

    def test_reset_clears_all_fields(self, reset_metrics):
        m = reset_metrics
        m.record_memory_hit()
        m.record_consolidation()
        m.record_latency("pfc", 50.0)
        m.reset()
        assert m.memory_hits == 0
        assert m.consolidation_count == 0
        assert m.total_latency_ms == 0.0
        assert m.node_invocations == {}

    def test_instrument_node_updates_module_metrics(self, caplog):
        """instrument_node() must update the module-level metrics singleton."""
        metrics.reset()
        wrapped = instrument_node("tracked_node", _dummy_node)
        with caplog.at_level(logging.INFO, logger="janus.observability"):
            wrapped(default_brain_state("hi"))
        assert "tracked_node" in metrics.node_invocations
        assert metrics.node_invocations["tracked_node"] >= 1


# ---------------------------------------------------------------------------
# Evaluation dataset
# ---------------------------------------------------------------------------

class TestEvalDataset:
    @pytest.fixture(scope="class")
    def dataset(self):
        assert EVAL_DATASET_PATH.exists(), f"Eval dataset not found at {EVAL_DATASET_PATH}"
        return json.loads(EVAL_DATASET_PATH.read_text(encoding="utf-8"))

    def test_eval_dataset_minimum_size(self, dataset):
        """Dataset must contain at least 20 entries."""
        assert len(dataset) >= 20, f"Expected ≥20 rows, got {len(dataset)}"

    def test_eval_dataset_has_required_fields(self, dataset):
        """Every entry must have 'input' and 'expected_output' keys."""
        for i, item in enumerate(dataset):
            assert "input" in item, f"Row {i} missing 'input'"
            assert "expected_output" in item, f"Row {i} missing 'expected_output'"

    def test_eval_dataset_inputs_are_non_empty_strings(self, dataset):
        """'input' values must be non-empty strings."""
        for i, item in enumerate(dataset):
            assert isinstance(item["input"], str) and item["input"].strip(), (
                f"Row {i} has empty or non-string 'input'"
            )

    def test_eval_dataset_has_emotional_labels(self, dataset):
        """Every entry should carry an 'emotional_label' float in [0, 1]."""
        for i, item in enumerate(dataset):
            assert "emotional_label" in item, f"Row {i} missing 'emotional_label'"
            label = item["emotional_label"]
            assert isinstance(label, (int, float)), f"Row {i} label is not numeric"
            assert 0.0 <= label <= 1.0, f"Row {i} label {label} out of [0, 1]"

    def test_eval_dataset_has_diverse_emotional_range(self, dataset):
        """Dataset must contain both low-emotion (≤0.2) and high-emotion (≥0.7) examples."""
        labels = [item["emotional_label"] for item in dataset]
        assert any(v <= 0.2 for v in labels), "No low-emotion examples in dataset"
        assert any(v >= 0.7 for v in labels), "No high-emotion examples in dataset"

    def test_eval_dataset_is_valid_json(self):
        """The dataset file must be parseable JSON."""
        raw = EVAL_DATASET_PATH.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        assert isinstance(parsed, list)


# ---------------------------------------------------------------------------
# Integration tests (require live LLM + LangSmith)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestObservabilityIntegration:
    def test_langsmith_run_created_per_node(self):
        """Run the brain graph; verify LangSmith creates named runs per node."""
        import time as _time

        from langsmith import Client

        from agents.thalamus import build_graph
        from config import LANGSMITH_PROJECT

        graph = build_graph()
        state = default_brain_state("what is the capital of France?")
        graph.invoke(state)

        # Allow time for async trace upload
        _time.sleep(3)

        client = Client()
        runs = list(client.list_runs(project_name=LANGSMITH_PROJECT, limit=100))
        run_names = " ".join(r.name or "" for r in runs).lower()

        for node in ("amygdala", "basal_ganglia", "pfc"):
            assert node in run_names, (
                f"No LangSmith run found containing '{node}'. "
                f"Check LANGCHAIN_TRACING_V2 and LANGCHAIN_PROJECT env vars."
            )

    def test_log_emitted_by_instrumented_graph(self, caplog):
        """Running an instrumented graph produces observability log records."""
        import agents.amygdala as _amygdala_mod
        from agents.thalamus import build_graph

        graph = build_graph(
            amygdala=instrument_node("amygdala", _amygdala_mod.amygdala_node),
        )
        with caplog.at_level(logging.INFO, logger="janus.observability"):
            graph.invoke(default_brain_state("hello"))

        obs_records = [r for r in caplog.records if r.name == "janus.observability"]
        assert len(obs_records) >= 1

    def test_emotional_score_accuracy(self):
        """Amygdala must achieve ≥ 80% accuracy on the labelled eval set.

        'Accuracy' here means |predicted − label| ≤ 0.30 for each item.
        """
        from agents.amygdala import amygdala_node

        dataset = json.loads(EVAL_DATASET_PATH.read_text(encoding="utf-8"))
        threshold = 0.30
        correct = 0
        for item in dataset:
            state = default_brain_state(item["input"])
            result = amygdala_node(state)
            predicted = float(result["emotional_weight"])
            expected = float(item["emotional_label"])
            if abs(predicted - expected) <= threshold:
                correct += 1

        accuracy = correct / len(dataset)
        assert accuracy >= 0.80, (
            f"Emotional score accuracy {accuracy:.1%} < 80% "
            f"({correct}/{len(dataset)} within ±{threshold})"
        )

    def test_retrieval_relevance_score(self):
        """Hippocampus retrieved_memories list is non-empty after seeding LTM."""
        import chromadb
        from langchain_core.embeddings import DeterministicFakeEmbedding

        from agents.hippocampus import hippocampus_node
        from memory.long_term import add_to_ltm

        client = chromadb.EphemeralClient(settings=chromadb.Settings(allow_reset=True))
        client.reset()
        embed = DeterministicFakeEmbedding(size=64)

        add_to_ltm(
            "Paris is the capital of France",
            {"source": "test"},
            "episodic",
            embedding_fn=embed,
            chroma_client=client,
        )

        state = default_brain_state("capital of France")
        state["emotional_weight"] = 0.5
        result = hippocampus_node(
            state, embedding_fn=embed, chroma_client=client, k=3
        )
        assert isinstance(result["retrieved_memories"], list)
        assert len(result["retrieved_memories"]) >= 1, (
            "Hippocampus should retrieve at least one memory after seeding LTM"
        )
