<div align="center">

# 🧠 Janus Process

### *A brain-inspired multi-agent cognitive architecture for next-generation AI reasoning*

[![Python 3.14](https://img.shields.io/badge/python-3.14-3776AB?logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2-FF6B35?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?logo=langchain&logoColor=white)](https://langchain.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20%2B%20WebSocket-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Redis](https://img.shields.io/badge/Redis-STM-DC382D?logo=redis&logoColor=white)](https://redis.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20LTM-F97316)](https://trychroma.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?logo=openai&logoColor=white)](https://openai.com)
[![Tests](https://img.shields.io/badge/tests-150%2B%20passing-22C55E?logo=pytest&logoColor=white)](tests/)
[![LangSmith](https://img.shields.io/badge/LangSmith-Observability-F59E0B)](https://smith.langchain.com)

</div>

---

## Why Janus?

Most AI systems are **stateless pipelines** — they receive a prompt and emit a response with no sense of emotional context, no persistent memory, and no ability to evaluate or revise their own output before it reaches the user.

**Janus Process is different.**

Modelled directly on the functional regions of the human brain, Janus routes every request through a multi-agent graph that mirrors real cognition: emotional threat detection, memory consolidation, habit recall, deliberative reasoning, and self-correcting quality gating — all before a single word reaches the user.

The result is an AI reasoning engine that is:

| Property | Janus | Typical RAG/LLM Pipeline |
|---|---|---|
| Emotionally-aware | ✅ Dynamic temperature + directive injection | ❌ Static prompts |
| Self-correcting | ✅ Iterative PFC ↔ BG refinement loop | ❌ Single-shot generation |
| Persistent memory | ✅ Two-tier STM (Redis) + LTM (ChromaDB) | ❌ Stateless per request |
| Habit/procedural fast-path | ✅ SQLite habit store, bypasses LLM | ❌ Every query hits the LLM |
| Production observable | ✅ LangSmith tracing per agent node | ❌ Black-box inference |

---

## Cognitive Architecture

```mermaid
flowchart TD
    User(["👤 User Input"])
    API["🌐 FastAPI\n(REST · WebSocket)"]

    subgraph BRAIN ["🧠  Janus Brain  —  LangGraph StateGraph"]
        direction TB

        AMY["💢 Amygdala\n─────────────\nEmotional salience scorer\nDerives llm_temperature\n& emotional_directive\ngpt-4o-mini"]

        BG["⚙️ Basal Ganglia\n─────────────\nHabit pattern matcher\nSQLite habit store\ngpt-4o-mini"]

        HIPP["🔍 Hippocampus\n─────────────\nRAG retrieval\nChromaDB vector store\ntext-embedding-3-small"]

        PFC["🧩 PFC  (Prefrontal Cortex)\n─────────────\nCore reasoning agent\nSTM + LTM + directive + feedback\ngpt-4o  ·  dynamic temperature"]

        GATE["🚦 Basal Ganglia Gate\n─────────────\nPost-PFC quality evaluator\nACCEPT  /  REFINE + feedback\ngpt-4o-mini"]
    end

    subgraph MEMORY ["💾  Memory Layer"]
        STM["⚡ Short-Term Memory\nRedis  ·  TTL-decayed\nLast-N conversation turns"]
        LTM["🗄️ Long-Term Memory\nChromaDB  ·  Vector search\nEpisodic · Semantic · Procedural"]
        HABITS["📋 Habit Store\nSQLite  ·  pattern + response\nhit_count tracking"]
    end

    subgraph OBSERVE ["📡  Observability"]
        LS["LangSmith\nPer-node traces\nLatency · Token usage"]
    end

    User --> API --> AMY
    AMY -->|"emotional_weight\nllm_temperature\nemotional_directive"| BG

    BG -->|"habit match\nprocedural_match set"| PFC
    BG -->|"no habit\nhigh emotion"| HIPP
    BG -->|"no habit\nlow emotion"| PFC
    HIPP -->|"retrieved_memories"| PFC

    PFC -->|"habit path → skip gate"| OUT(["✅ Response"])
    PFC -->|"deliberate path"| GATE

    GATE -->|"ACCEPT"| OUT
    GATE -->|"REFINE  &  loop_count < MAX"| PFC
    GATE -->|"budget exhausted"| OUT

    PFC <-->|"read / write"| STM
    HIPP <-->|"retrieve / encode"| LTM
    BG <-->|"lookup"| HABITS
    BRAIN -.->|"trace every node"| LS

    style BRAIN fill:#0f172a,stroke:#334155,color:#f1f5f9
    style MEMORY fill:#1e1b4b,stroke:#4338ca,color:#e0e7ff
    style OBSERVE fill:#1a1a2e,stroke:#6d28d9,color:#ede9fe
    style AMY fill:#7f1d1d,stroke:#ef4444,color:#fef2f2
    style BG fill:#1c1917,stroke:#a8a29e,color:#f5f5f4
    style HIPP fill:#052e16,stroke:#22c55e,color:#f0fdf4
    style PFC fill:#1e3a5f,stroke:#3b82f6,color:#eff6ff
    style GATE fill:#451a03,stroke:#f97316,color:#fff7ed
    style STM fill:#450a0a,stroke:#dc2626,color:#fef2f2
    style LTM fill:#1e3a5f,stroke:#6366f1,color:#eef2ff
    style HABITS fill:#1c1917,stroke:#78716c,color:#fafaf9
    style LS fill:#2d1b69,stroke:#a78bfa,color:#f5f3ff
```

---

## The Five Cognitive Agents

### 💢 Amygdala — Emotional Threat Detector
The first node every input passes through. Rates emotional salience (0.0–1.0) and dynamically adjusts downstream behaviour:

| Emotional Weight | LLM Temperature | PFC Directive |
|---|---|---|
| < 0.2 (neutral) | 0.7 (creative) | — |
| 0.2–0.5 (mild) | 0.5 | *Be warm and supportive* |
| 0.5–0.7 (stressed) | 0.3 | *Acknowledge feelings before responding* |
| > 0.7 (crisis) | 0.1 (precise) | *Prioritise emotional safety and calm compassion* |

### ⚙️ Basal Ganglia — Habit Fast-Path + Response Gate
Two roles in one region. **Pre-PFC:** matches input against learned habits in SQLite — known patterns bypass expensive LLM reasoning entirely. **Post-PFC:** a quality gate that evaluates every deliberate response before release, issuing `ACCEPT` or `REFINE: <instruction>`.

### 🔍 Hippocampus — Long-Term Memory Retrieval
On high-emotion inputs, retrieves semantically relevant past experiences from ChromaDB using MMR (Maximum Marginal Relevance) for diverse, non-redundant context. Simultaneously encodes the current exchange into persistent memory with emotional weight as metadata.

### 🧩 PFC — Prefrontal Cortex (Core Reasoning)
The primary synthesis agent. Combines short-term conversation history (Redis), long-term retrieved memories, the current emotional directive, and any refinement feedback from the gate into a coherent, context-aware response. Runs `gpt-4o` at a temperature set by the Amygdala's assessment.

### 🚦 PFC ↔ Basal Ganglia Recurrent Loop
Biologically, deliberate cognition is iterative — the PFC and BG cycle through competing plans. The LangGraph topology makes this explicit:

```
PFC proposal → Gate evaluation
    ACCEPT  ──────────────────────────────────────► response
    REFINE  ──► (loop_count < MAX_PFC_LOOPS) ──► PFC
    REFINE  ──► (budget exhausted) ──────────────► best-effort response
```

---

## Technology Stack

| Layer | Technology | Role |
|---|---|---|
| **Orchestration** | [LangGraph 1.2](https://langchain-ai.github.io/langgraph/) | Stateful multi-agent `StateGraph` with conditional routing and loop edges |
| **LLM (reasoning)** | OpenAI `gpt-4o` | PFC synthesis — dynamic temperature per turn |
| **LLM (fast)** | OpenAI `gpt-4o-mini` | Amygdala scoring · BG habit matching · BG gate evaluation |
| **Embeddings** | OpenAI `text-embedding-3-small` | ChromaDB ingestion and retrieval |
| **Short-Term Memory** | Redis + `RedisChatMessageHistory` | TTL-decayed sliding window conversation buffer |
| **Long-Term Memory** | ChromaDB (vector store) | Episodic, semantic, and procedural memory with MMR retrieval |
| **Habit Store** | SQLite / Postgres | Structured habit patterns with hit-count tracking |
| **API** | FastAPI | `POST /think` · `WS /think/stream` · `GET /memory/search` |
| **Tracing** | LangSmith | Per-node latency, token usage, full run history |
| **Containerisation** | Docker Compose | Redis + ChromaDB services, multi-stage production image |
| **Testing** | pytest + pytest-asyncio | 150+ unit and integration tests across 16 phases |
| **Lint / Types** | ruff + mypy | CI-enforced code quality |

---

## Quick Start

```bash
# 1. Clone and create virtual environment
git clone https://github.com/your-org/janus-process.git
cd janus-process
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -U pip setuptools wheel
pip install -e '.[dev]'

# 3. Configure environment
cp .env.example .env
# Add: OPENAI_API_KEY, LANGSMITH_API_KEY

# 4. Start infrastructure
docker compose up -d          # Redis + ChromaDB

# 5. Run the interactive CLI
python main.py
```

```
Janus Process  (Ctrl-C to exit)

You: I'm really anxious about my presentation tomorrow
Brain: I can hear that you're feeling a lot of pressure right now...
```

---

## API

```bash
# Single-turn reasoning
curl -X POST http://localhost:8000/think \
  -H "Content-Type: application/json" \
  -d '{"input": "What is the capital of France?"}'

# Semantic memory search
curl "http://localhost:8000/memory/search?q=Paris"

# WebSocket streaming
wscat -c ws://localhost:8000/think/stream
```

---

## Test Suite

```bash
python -m pytest -q                          # all tests
python -m pytest -m "not integration" -q    # unit tests only (no Docker required)
python -m pytest tests/test_phase16_recurrent_loop.py -v
```

The test suite spans 16 implementation phases and covers every agent node, routing decision, memory layer, and recurrent loop behaviour with mocked LLM calls for deterministic unit testing.

---

## Roadmap

| Phase | Description | Status |
|---|---|---|
| 1–13 | Foundation · STM · LTM · Amygdala · Hippocampus · PFC · Basal Ganglia · Thalamus · Consolidator · API · Observability · Hardening · LTM Summarisation | ✅ Complete |
| 14 | Amygdala as Global State Modifier (dynamic temperature + directive injection) | ✅ Complete |
| 15 | Basal Ganglia Response Gate (post-PFC inhibitory gating) | ✅ Complete |
| 16 | PFC ↔ Basal Ganglia Recurrent Loop (iterative self-refinement) | ✅ Complete |
| 17+ | Multi-modal inputs · Long-horizon planning · Agent-to-agent delegation | 🔜 Planned |

---

## Local Development Notes

- The `.venv/` directory is excluded from git via `.gitignore`.
- Integration tests require Docker (`docker compose up -d`).
- LangSmith tracing is enabled automatically when `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` are set in `.env`.
- `MAX_PFC_LOOPS` (default `2`) is configurable via environment variable to control the PFC ↔ BG loop budget.
