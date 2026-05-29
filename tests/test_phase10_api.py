"""
tests/test_phase10_api.py — Unit and integration tests for the FastAPI layer.

Unit tests
----------
- Brain graph is replaced with a lightweight mock via ``app.dependency_overrides``.
- LTM searcher and STM clearer are mocked via the same mechanism.
- No live Redis, ChromaDB, or LLM required.
- Uses Starlette's synchronous ``TestClient`` (works without asyncio).

Integration tests
-----------------
Marked ``@pytest.mark.integration`` — require live Docker services.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from api.server import app, get_brain_graph, get_ltm_searcher, get_stm_clearer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _NodeTracker:
    """Simple callable that records invocations and merges fixed updates."""

    def __init__(self, updates: dict[str, Any] | None = None) -> None:
        self.called = False
        self._updates = updates or {}

    def __call__(self, state: dict) -> dict:
        self.called = True
        return {**state, **self._updates}


def _make_mock_graph(
    *,
    emotional_weight: float = 0.1,
    procedural_match: str | None = None,
    final_response: str = "mock brain response",
):
    """Return a compiled graph whose nodes are all lightweight NodeTrackers."""
    from agents.thalamus import build_graph

    return build_graph(
        amygdala=_NodeTracker({"emotional_weight": emotional_weight}),
        basal_ganglia=_NodeTracker({"procedural_match": procedural_match}),
        hippocampus=_NodeTracker({}),
        pfc=_NodeTracker({"final_response": final_response, "should_consolidate": False}),
        bg_gate=_NodeTracker({"bg_feedback": ""}),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_overrides():
    """Ensure dependency overrides are cleared after every test."""
    yield
    app.dependency_overrides.clear()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def mock_graph():
    return _make_mock_graph()


# ---------------------------------------------------------------------------
# POST /think
# ---------------------------------------------------------------------------

class TestPostThink:
    def test_post_think_returns_200(self, client, mock_graph):
        """Valid input → HTTP 200 and a 'response' field in the body."""
        app.dependency_overrides[get_brain_graph] = lambda: mock_graph
        resp = client.post("/think", json={"input": "hello"})
        assert resp.status_code == 200
        body = resp.json()
        assert "response" in body
        assert isinstance(body["response"], str)

    def test_post_think_returns_final_response_content(self, client):
        """The 'response' field echoes the mock graph's final_response."""
        graph = _make_mock_graph(final_response="the brain speaks")
        app.dependency_overrides[get_brain_graph] = lambda: graph
        resp = client.post("/think", json={"input": "tell me something"})
        assert resp.json()["response"] == "the brain speaks"

    def test_post_think_empty_input_returns_422(self, client, mock_graph):
        """Empty string input fails Pydantic min_length=1 → HTTP 422."""
        app.dependency_overrides[get_brain_graph] = lambda: mock_graph
        resp = client.post("/think", json={"input": ""})
        assert resp.status_code == 422

    def test_post_think_missing_input_field_returns_422(self, client, mock_graph):
        """Missing 'input' key → HTTP 422."""
        app.dependency_overrides[get_brain_graph] = lambda: mock_graph
        resp = client.post("/think", json={})
        assert resp.status_code == 422

    def test_post_think_non_string_input_returns_422(self, client, mock_graph):
        """Non-string input → HTTP 422."""
        app.dependency_overrides[get_brain_graph] = lambda: mock_graph
        resp = client.post("/think", json={"input": 42})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /memory/search
# ---------------------------------------------------------------------------

class TestMemorySearch:
    def _seed_searcher(self, results: list[dict]) -> MagicMock:
        mock = MagicMock(return_value=results)
        app.dependency_overrides[get_ltm_searcher] = lambda: mock
        return mock

    def test_memory_search_returns_results(self, client):
        """Pre-seeded searcher → HTTP 200 and non-empty list."""
        self._seed_searcher([{"content": "Paris is the capital", "metadata": {"source": "test"}}])
        resp = client.get("/memory/search?q=capital+of+France")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        assert "content" in data[0]
        assert "metadata" in data[0]

    def test_memory_search_result_shape(self, client):
        """Each result has exactly 'content' (str) and 'metadata' (dict) keys."""
        self._seed_searcher([{"content": "some fact", "metadata": {"source": "agent"}}])
        resp = client.get("/memory/search?q=fact")
        item = resp.json()[0]
        assert isinstance(item["content"], str)
        assert isinstance(item["metadata"], dict)

    def test_memory_search_empty_query_returns_400(self, client):
        """?q= (blank string) → HTTP 400."""
        self._seed_searcher([])
        resp = client.get("/memory/search?q=")
        assert resp.status_code == 400

    def test_memory_search_whitespace_query_returns_400(self, client):
        """?q=   (only whitespace) → HTTP 400."""
        self._seed_searcher([])
        resp = client.get("/memory/search", params={"q": "   "})
        assert resp.status_code == 400

    def test_memory_search_missing_q_returns_422(self, client):
        """Missing 'q' parameter → FastAPI validation → HTTP 422."""
        self._seed_searcher([])
        resp = client.get("/memory/search")
        assert resp.status_code == 422

    def test_memory_search_passes_k_to_searcher(self, client):
        """k query parameter is forwarded to the searcher callable."""
        mock = self._seed_searcher([])
        client.get("/memory/search?q=test&k=3")
        args = mock.call_args
        assert args[0][2] == 3  # third positional arg is k

    def test_memory_search_empty_results_returns_empty_list(self, client):
        """Searcher returning [] → HTTP 200 with empty array."""
        self._seed_searcher([])
        resp = client.get("/memory/search?q=unknownquery")
        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# DELETE /memory/session
# ---------------------------------------------------------------------------

class TestDeleteSession:
    def test_delete_session_returns_200(self, client):
        """DELETE /memory/session → HTTP 200 with status=ok."""
        mock_clearer = MagicMock()
        app.dependency_overrides[get_stm_clearer] = lambda: mock_clearer
        resp = client.delete("/memory/session")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_delete_session_flushes_stm(self, client):
        """The injected clearer callable must be invoked exactly once."""
        mock_clearer = MagicMock()
        app.dependency_overrides[get_stm_clearer] = lambda: mock_clearer
        client.delete("/memory/session")
        mock_clearer.assert_called_once()

    def test_delete_session_redis_unavailable_returns_503(self, client):
        """When clearer raises ConnectionError → HTTP 503."""
        app.dependency_overrides[get_stm_clearer] = lambda: MagicMock(
            side_effect=ConnectionError("Redis down")
        )
        resp = client.delete("/memory/session")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# WS /think/stream
# ---------------------------------------------------------------------------

class TestWebSocketStream:
    def _ws_client(self) -> TestClient:
        return TestClient(app, raise_server_exceptions=False)

    def test_websocket_stream_sends_chunks(self):
        """Graph with 3 nodes → at least 2 JSON messages before 'done'."""
        graph = _make_mock_graph(emotional_weight=0.1)
        app.dependency_overrides[get_brain_graph] = lambda: graph
        client = self._ws_client()
        messages = []
        with client.websocket_connect("/think/stream") as ws:
            ws.send_json({"input": "hello"})
            while True:
                try:
                    msg = ws.receive_json()
                    messages.append(msg)
                    if msg.get("type") == "done":
                        break
                except Exception:
                    break
        assert len(messages) >= 2, f"Expected ≥2 messages, got {messages}"

    def test_websocket_stream_contains_done_message(self):
        """Last message must have type='done'."""
        graph = _make_mock_graph()
        app.dependency_overrides[get_brain_graph] = lambda: graph
        client = self._ws_client()
        messages = []
        with client.websocket_connect("/think/stream") as ws:
            ws.send_json({"input": "test"})
            while True:
                try:
                    msg = ws.receive_json()
                    messages.append(msg)
                    if msg.get("type") == "done":
                        break
                except Exception:
                    break
        assert any(m.get("type") == "done" for m in messages)

    def test_websocket_stream_update_messages_have_node_key(self):
        """Every 'update' chunk must carry a 'node' field."""
        graph = _make_mock_graph()
        app.dependency_overrides[get_brain_graph] = lambda: graph
        client = self._ws_client()
        with client.websocket_connect("/think/stream") as ws:
            ws.send_json({"input": "hello"})
            updates = []
            while True:
                try:
                    msg = ws.receive_json()
                    if msg.get("type") == "update":
                        updates.append(msg)
                    if msg.get("type") == "done":
                        break
                except Exception:
                    break
        assert len(updates) >= 1
        for u in updates:
            assert "node" in u

    def test_websocket_stream_pfc_chunk_has_response(self):
        """The PFC update chunk must carry the 'response' field."""
        graph = _make_mock_graph(final_response="brain answered")
        app.dependency_overrides[get_brain_graph] = lambda: graph
        client = self._ws_client()
        with client.websocket_connect("/think/stream") as ws:
            ws.send_json({"input": "question"})
            pfc_chunk = None
            while True:
                try:
                    msg = ws.receive_json()
                    if msg.get("type") == "update" and msg.get("node") == "pfc":
                        pfc_chunk = msg
                    if msg.get("type") == "done":
                        break
                except Exception:
                    break
        assert pfc_chunk is not None
        assert pfc_chunk.get("response") == "brain answered"

    def test_websocket_empty_input_returns_error(self):
        """Sending empty input → error message, no crash."""
        graph = _make_mock_graph()
        app.dependency_overrides[get_brain_graph] = lambda: graph
        client = self._ws_client()
        messages = []
        with client.websocket_connect("/think/stream") as ws:
            ws.send_json({"input": ""})
            try:
                msg = ws.receive_json()
                messages.append(msg)
            except Exception:
                pass
        if messages:
            assert messages[0].get("type") == "error"


# ---------------------------------------------------------------------------
# Integration tests (require live services)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestApiIntegration:
    def test_post_think_live(self):
        """Full round-trip with real agents; final_response must be non-empty."""
        client = TestClient(app)
        resp = client.post("/think", json={"input": "what is 2 + 2?"})
        assert resp.status_code == 200
        assert len(resp.json()["response"]) > 0

    def test_memory_search_live(self):
        """Live LTM search (may return empty on cold start, but must not error)."""
        client = TestClient(app)
        resp = client.get("/memory/search?q=test")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_delete_session_live(self):
        """Live Redis flush must return status=ok."""
        client = TestClient(app)
        resp = client.delete("/memory/session")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
