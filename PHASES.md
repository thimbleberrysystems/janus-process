# Janus Process — Brain-Inspired Multi-Agent System
## Project Phases

---

## Phase 1 — Foundation & Scaffolding
**Goal:** Establish the project skeleton, tooling, and shared contracts before any agent logic.

### Deliverables
- [ ] Python project setup (`pyproject.toml` / `requirements.txt`, virtual env)
- [ ] Folder structure: `agents/`, `memory/`, `models/`, `tests/`
- [ ] `BrainState` TypedDict schema (`models/brain_state.py`)
- [ ] Environment config (`.env`, `config.py` — API keys, Redis URL, DB paths)
- [ ] Docker Compose: Redis + ChromaDB services
- [ ] Logging setup (LangSmith tracing integration)
- [ ] `main.py` entry point with stub graph (input → output, no logic yet)

### Tests — `tests/test_phase1_foundation.py`
- [ ] `test_brain_state_schema` — instantiate `BrainState` with all fields; assert types and defaults
- [ ] `test_brain_state_missing_required_field` — omitting a required field raises `TypeError`
- [ ] `test_config_loads_env_vars` — mock `.env`; assert `config.py` reads keys correctly
- [ ] `test_docker_services_reachable` — ping Redis and ChromaDB ports (integration, requires Docker)
- [ ] `test_stub_graph_returns_state` — invoke `main.py` stub with dummy input; assert state passes through unchanged

---

## Phase 2 — Short-Term Memory (STM)
**Goal:** Working memory layer that all agents can read from and write to.

### Deliverables
- [ ] `memory/short_term.py` — Redis-backed `RedisChatMessageHistory`
- [ ] TTL-based decay (configurable, default 1 hour)
- [ ] `get_short_term_memory(n)` and `add_to_short_term(text)` helpers
- [ ] Verify Redis container connectivity

### Tests — `tests/test_phase2_stm.py`
- [ ] `test_add_and_read_message` — write one message; assert it is returned by `get_short_term_memory()`
- [ ] `test_window_limit` — write 20 messages; assert `get_short_term_memory(n=5)` returns only the last 5
- [ ] `test_ttl_expiry` — write message with TTL=1s; sleep 2s; assert buffer is empty (integration)
- [ ] `test_empty_buffer_returns_empty_list` — fresh session; assert `get_short_term_memory()` returns `[]`
- [ ] `test_redis_unavailable_raises_gracefully` — point to bad Redis URL; assert a clear `ConnectionError` is raised

---

## Phase 3 — Long-Term Memory (LTM) & RAG Layer
**Goal:** Persistent vector store that acts as the declarative cortex.

### Deliverables
- [ ] `memory/long_term.py` — ChromaDB vector store with `OpenAIEmbeddings`
- [ ] Collections: `episodic`, `semantic`, `procedural`
- [ ] Metadata schema: `{ source, emotional_weight, timestamp, type }`
- [ ] MMR retrieval helper (`search_type="mmr"`, configurable `k`)
- [ ] Ingestion utility: add single text or batch with metadata

### Tests — `tests/test_phase3_ltm.py`
- [ ] `test_store_and_retrieve_single` — store one text; retrieve by similar query; assert original text in results
- [ ] `test_metadata_persisted` — store with full metadata; retrieve; assert all metadata fields are intact
- [ ] `test_mmr_returns_diverse_results` — store 10 near-duplicate texts; MMR retrieve k=3; assert results are not identical
- [ ] `test_batch_ingestion` — ingest 50 texts at once; assert collection count increments correctly
- [ ] `test_collection_isolation` — write to `episodic`; query `semantic`; assert no cross-collection leakage
- [ ] `test_empty_collection_returns_empty_list` — query fresh collection; assert `[]` returned

---

## Phase 4 — Amygdala Agent
**Goal:** Emotional salience scorer — first node all inputs pass through.

### Deliverables
- [ ] `agents/amygdala.py` — lightweight LLM classifier (0.0–1.0 score)
- [ ] Prompt: rate emotional/urgency salience of input
- [ ] Output written to `BrainState.emotional_weight`
- [ ] Use `gpt-4o-mini` (cost-efficient, fast)

### Tests — `tests/test_phase4_amygdala.py`
- [ ] `test_neutral_input_scores_low` — input: "what time is it?"; assert score < 0.3
- [ ] `test_emotional_input_scores_high` — input: "I just lost my job"; assert score > 0.7
- [ ] `test_score_is_float_in_range` — assert output is `float` and `0.0 <= score <= 1.0` for 10 varied inputs
- [ ] `test_state_updated_correctly` — pass full `BrainState`; assert `emotional_weight` field is set, all other fields unchanged
- [ ] `test_llm_mocked_returns_valid_score` — mock LLM response as `"0.85"`; assert node parses and stores correctly without live API call

---

## Phase 5 — Hippocampus Agent
**Goal:** Memory encoding and retrieval — connects LTM to the processing pipeline.

### Deliverables
- [ ] `agents/hippocampus.py` — RAG retrieval + encoding node
- [ ] Retrieve top-k relevant memories from ChromaDB on each input
- [ ] Encode current input into LTM with `emotional_weight` metadata
- [ ] Retrieval result written to `BrainState.retrieved_memories`

### Tests — `tests/test_phase5_hippocampus.py`
- [ ] `test_encode_then_retrieve` — run node with input; re-query LTM directly; assert input was stored
- [ ] `test_retrieved_memories_in_state` — pre-seed LTM; run node; assert `retrieved_memories` list is non-empty
- [ ] `test_emotional_weight_stored_as_metadata` — run node with `emotional_weight=0.9`; fetch stored doc; assert metadata matches
- [ ] `test_irrelevant_query_returns_empty` — seed LTM with unrelated topics; query with orthogonal input; assert no high-similarity results
- [ ] `test_state_fields_unchanged` — assert all `BrainState` fields except `retrieved_memories` are unmodified after node runs

---

## Phase 6 — PFC Agent (Working Memory + Synthesis)
**Goal:** Core reasoning agent that synthesizes STM, LTM, and current input.

### Deliverables
- [ ] `agents/pfc.py` — `gpt-4o` with structured prompt
- [ ] Prompt: working memory + retrieved memories + procedural match + input
- [ ] Output written to `BrainState.final_response`
- [ ] Sets `BrainState.should_consolidate` flag based on emotional weight

### Tests — `tests/test_phase6_pfc.py`
- [ ] `test_final_response_is_populated` — mock LLM; run node; assert `final_response` is a non-empty string
- [ ] `test_should_consolidate_true_when_high_emotion` — set `emotional_weight=0.8`; assert `should_consolidate=True`
- [ ] `test_should_consolidate_false_when_low_emotion` — set `emotional_weight=0.2`; assert `should_consolidate=False`
- [ ] `test_all_context_injected_into_prompt` — capture prompt via mock; assert working memory, retrieved memories, and procedural match all appear in prompt text
- [ ] `test_empty_context_does_not_crash` — run node with all context fields empty/None; assert graceful response
- [ ] `test_response_references_retrieved_memory` — seed retrieved memory with key fact; assert LLM response (live) mentions or uses that fact

---

## Phase 7 — Basal Ganglia Agent
**Goal:** Procedural/habit layer — bypasses deliberate reasoning for known patterns.

### Deliverables
- [ ] `agents/basal_ganglia.py` — pattern matcher against SQL store
- [ ] SQLite (dev) / Postgres (prod) table: `habits(pattern, response, hit_count)`
- [ ] LLM step: classify input as matching a known habit or not
- [ ] On match: populate `BrainState.procedural_match`, skip deep LTM retrieval
- [ ] On miss: pass through to PFC

### Tests — `tests/test_phase7_basal_ganglia.py`
- [ ] `test_register_habit` — insert habit into DB; assert row exists with `hit_count=0`
- [ ] `test_known_pattern_match` — register habit "greet user"; input "hello there"; assert `procedural_match` is populated
- [ ] `test_unknown_pattern_miss` — empty habits table; run node; assert `procedural_match` is `None`
- [ ] `test_hit_count_increments` — match same habit twice; assert `hit_count=2` in DB
- [ ] `test_match_bypasses_ltm` — mock Hippocampus; on habit match assert it is NOT called
- [ ] `test_miss_passes_through_to_pfc` — on miss assert node returns state unchanged (routing handled by Thalamus)

---

## Phase 8 — Thalamus Router (LangGraph Orchestration)
**Goal:** Wire all agents into a conditional LangGraph `StateGraph`.

### Deliverables
- [ ] `agents/thalamus.py` — `build_graph()` with all nodes and edges
- [ ] Entry point: Amygdala → conditional route → Hippocampus / Basal Ganglia / PFC
- [ ] Hippocampus and Basal Ganglia both feed into PFC before END
- [ ] LangSmith tracing enabled on the compiled graph

### Tests — `tests/test_phase8_thalamus.py`
- [ ] `test_graph_compiles` — call `build_graph()`; assert no exception and returns a runnable
- [ ] `test_low_emotion_routes_to_pfc_direct` — mock Amygdala score=0.2, no habit match; assert Hippocampus node is NOT invoked
- [ ] `test_high_emotion_routes_via_hippocampus` — mock score=0.9; assert Hippocampus node IS invoked before PFC
- [ ] `test_habit_match_routes_via_basal_ganglia` — mock procedural match present; assert Basal Ganglia node IS invoked
- [ ] `test_full_graph_end_to_end` — real inputs, all agents mocked; assert `final_response` is set and graph reaches END
- [ ] `test_graph_state_immutable_between_nodes` — assert each node receives the state output of the previous node (no state leakage)

---

## Phase 9 — Memory Consolidator
**Goal:** Async STM → LTM replay (systems consolidation, mimicking sleep).

### Deliverables
- [ ] `memory/consolidator.py` — reads Redis buffer, writes to ChromaDB
- [ ] Triggered by: `should_consolidate` flag OR scheduled interval (APScheduler)
- [ ] Deduplication: skip texts already present (hash check)
- [ ] Configurable replay window (default: last 50 messages)

### Tests — `tests/test_phase9_consolidator.py`
- [ ] `test_consolidate_writes_to_ltm` — seed Redis with 5 messages; run consolidator; assert all 5 appear in ChromaDB
- [ ] `test_deduplication_skips_existing` — consolidate same messages twice; assert ChromaDB count does not double
- [ ] `test_replay_window_respected` — seed 100 messages, window=10; assert only last 10 are written
- [ ] `test_metadata_source_is_consolidated` — after consolidation, assert stored docs have `source="consolidated"`
- [ ] `test_empty_stm_does_nothing` — run consolidator with empty Redis; assert ChromaDB count unchanged and no exception
- [ ] `test_consolidator_triggered_by_flag` — set `should_consolidate=True` in state; assert consolidator is called by the post-PFC hook

---

## Phase 10 — API Layer
**Goal:** Expose the brain as a REST / WebSocket service.

### Deliverables
- [ ] `api/server.py` — FastAPI app
- [ ] `POST /think` — synchronous single-turn input → response
- [ ] `WS /think/stream` — streaming response via LangChain streaming callbacks
- [ ] `GET /memory/search?q=...` — expose LTM retrieval endpoint
- [ ] `DELETE /memory/session` — flush STM (Redis)
- [ ] OpenAPI docs auto-generated

### Tests — `tests/test_phase10_api.py`
- [ ] `test_post_think_returns_200` — POST `{"input": "hello"}` with mocked brain graph; assert HTTP 200 and `response` field present
- [ ] `test_post_think_empty_input_returns_422` — POST `{"input": ""}` ; assert HTTP 422 validation error
- [ ] `test_memory_search_returns_results` — pre-seed LTM; GET `/memory/search?q=test`; assert non-empty list returned
- [ ] `test_memory_search_empty_query_returns_400` — GET `/memory/search?q=`; assert HTTP 400
- [ ] `test_delete_session_flushes_stm` — write to Redis; DELETE `/memory/session`; assert STM is empty
- [ ] `test_websocket_stream_sends_chunks` — connect WS `/think/stream`; send input; assert multiple partial message chunks received before close

---

## Phase 11 — Observability & Evaluation
**Goal:** Make the system inspectable, traceable, and measurable.

### Deliverables
- [ ] LangSmith project configured with named runs per agent
- [ ] Structured logging: agent name, latency, token usage per node
- [ ] Evaluation dataset: 20+ input/expected-output pairs
- [ ] LangSmith eval: relevance, faithfulness, emotional score accuracy
- [ ] Dashboard: memory hit rate, STM→LTM consolidation count, avg latency

### Tests — `tests/test_phase11_observability.py`
- [ ] `test_langsmith_run_created_per_node` — run full graph; query LangSmith API; assert one named run exists per agent node
- [ ] `test_log_contains_agent_name_and_latency` — capture log output; assert each log line includes `agent`, `latency_ms`, `tokens` fields
- [ ] `test_eval_dataset_minimum_size` — load eval dataset file; assert at least 20 rows with `input` and `expected_output`
- [ ] `test_emotional_score_accuracy` — run Amygdala on eval set; assert accuracy >= 80% against human-labelled scores
- [ ] `test_retrieval_relevance_score` — run Hippocampus on eval set; assert mean relevance score >= 0.7 (LangSmith evaluator)

---

## Phase 12 — Hardening & Production Readiness
**Goal:** Security, resilience, and deployment.

### Deliverables
- [ ] Secrets management (no keys in code — use `.env` + secret manager)
- [ ] Rate limiting on API endpoints
- [ ] Retry/fallback logic for LLM calls (`tenacity`)
- [ ] ChromaDB → Weaviate or Pinecone migration path (swappable interface)
- [ ] Docker multi-stage build for production image
- [ ] CI pipeline: lint (`ruff`), type check (`mypy`), tests (`pytest`)

### Tests — `tests/test_phase12_hardening.py`
- [ ] `test_no_secrets_in_source_code` — grep all `.py` files for hardcoded API key patterns; assert zero matches
- [ ] `test_rate_limit_enforced` — send 20 rapid requests to `POST /think`; assert HTTP 429 returned after limit
- [ ] `test_llm_retry_on_transient_failure` — mock LLM to fail twice then succeed; assert final response is returned (not an exception)
- [ ] `test_vectorstore_interface_swappable` — replace ChromaDB with in-memory stub implementing same interface; assert all LTM tests still pass
- [ ] `test_docker_image_builds` — run `docker build`; assert exit code 0 and image size within threshold
- [ ] `test_ci_lint_passes` — run `ruff check .`; assert zero lint errors
- [ ] `test_ci_type_check_passes` — run `mypy .`; assert zero type errors

---

## Dependency Map

```
Phase 1 (Foundation)
  └─► Phase 2 (STM)
        └─► Phase 6 (PFC)
  └─► Phase 3 (LTM)
        └─► Phase 5 (Hippocampus)
              └─► Phase 6 (PFC)
  └─► Phase 4 (Amygdala)
        └─► Phase 5 (Hippocampus)
        └─► Phase 7 (Basal Ganglia)
              └─► Phase 6 (PFC)
  Phases 2–7 ──► Phase 8 (Thalamus / LangGraph)
  Phase 8 ──► Phase 9 (Consolidator)
  Phase 8 ──► Phase 10 (API)
  Phase 10 ──► Phase 11 (Observability)
  Phase 11 ──► Phase 12 (Hardening)
```

---

## Tech Stack Summary

| Concern | Technology |
|---|---|
| Agent orchestration | LangGraph (StateGraph) |
| LLM calls | LangChain + OpenAI (`gpt-4o`, `gpt-4o-mini`) |
| Short-term memory | Redis (`RedisChatMessageHistory`) |
| Long-term memory | ChromaDB (dev) → Weaviate/Pinecone (prod) |
| Embeddings | `text-embedding-3-small` |
| Procedural store | SQLite (dev) / Postgres (prod) |
| API | FastAPI + WebSockets |
| Observability | LangSmith |
| Testing | pytest |
| Containerisation | Docker Compose |
