"""
observability.py — Node instrumentation, structured logging, and in-process metrics.

Public surface
--------------
instrument_node(name, fn)  — wrap a LangGraph node with timing + structured logging.
metrics                    — module-level Metrics singleton (counters + averages).

Structured log records
----------------------
Each instrumented node emits an INFO record to the ``janus.observability`` logger
with the following ``extra`` fields available on the LogRecord:

    agent       str   — node name (e.g. "amygdala")
    latency_ms  float — wall-clock time for one node invocation, milliseconds
    tokens      int   — total LLM token count (input + output) for the invocation;
                        0 when no LLM was called or when token data is unavailable

LangSmith tracing
-----------------
Named runs per agent are created automatically by LangGraph because each node
is added to the StateGraph under its agent name (e.g. "amygdala", "pfc").
No additional code is needed — enable tracing by setting the env vars:

    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_PROJECT=<project>
    LANGCHAIN_API_KEY=<key>

``logging_config.py`` sets these variables when a LangSmith API key is present.

Dashboard metrics
-----------------
``metrics`` is a process-level singleton.  Integrators may reset it between
requests or export it to Prometheus / StatsD at a higher layer.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

from models.brain_state import BrainState

_logger = logging.getLogger("janus.observability")


# ---------------------------------------------------------------------------
# Metrics counters
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    """In-process counters for key observability events.

    All fields are plain integers / floats — no thread locking.  For
    production use behind a multi-worker server, export these to a shared
    store (Redis, Prometheus push-gateway, etc.).
    """

    memory_hits: int = 0
    """Incremented by the Hippocampus node whenever ≥1 memory is retrieved."""

    consolidation_count: int = 0
    """Incremented each time the Memory Consolidator writes at least one doc."""

    total_latency_ms: float = 0.0
    """Running sum of all instrumented node durations."""

    node_invocations: dict[str, int] = field(default_factory=dict)
    """Per-node invocation counters — keys are agent names."""

    def record_memory_hit(self) -> None:
        """Register a successful LTM retrieval."""
        self.memory_hits += 1

    def record_consolidation(self) -> None:
        """Register one consolidation pass."""
        self.consolidation_count += 1

    def record_latency(self, node_name: str, latency_ms: float) -> None:
        """Accumulate timing for *node_name*."""
        self.total_latency_ms += latency_ms
        self.node_invocations[node_name] = self.node_invocations.get(node_name, 0) + 1

    def avg_latency_ms(self) -> float:
        """Mean latency across all recorded node invocations."""
        total = sum(self.node_invocations.values())
        return self.total_latency_ms / total if total > 0 else 0.0

    def reset(self) -> None:
        """Zero all counters (useful between test runs or request batches)."""
        self.memory_hits = 0
        self.consolidation_count = 0
        self.total_latency_ms = 0.0
        self.node_invocations.clear()


# Module-level singleton.
metrics: Metrics = Metrics()


# ---------------------------------------------------------------------------
# Token extraction helper
# ---------------------------------------------------------------------------

def _extract_tokens(state: Any) -> int:
    """Return the token count stored in ``state["_token_usage"]``.

    Agent nodes that call an LLM can optionally stash the total token count
    in this field.  Falls back to ``0`` when the field is absent (e.g. for
    the Basal Ganglia on a cache hit, or any mocked node).
    """
    if isinstance(state, dict):
        return int(state.get("_token_usage", 0))
    return 0


# ---------------------------------------------------------------------------
# Node instrumentation
# ---------------------------------------------------------------------------

def instrument_node(node_name: str, fn: Callable) -> Callable:
    """Wrap a LangGraph node callable with timing and structured logging.

    Parameters
    ----------
    node_name:
        Human-readable agent name written to every log record.
    fn:
        The original node callable ``(BrainState) -> BrainState``.

    Returns
    -------
    A wrapper with the same calling convention that:
    1. Invokes *fn* with the incoming state.
    2. Measures wall-clock duration.
    3. Emits one ``INFO`` record to ``janus.observability`` with extra fields
       ``agent``, ``latency_ms``, and ``tokens``.
    4. Records the invocation in the module-level ``metrics`` singleton.
    5. Returns the result unchanged.
    """

    @wraps(fn)
    def wrapper(state: BrainState) -> BrainState:
        start = time.perf_counter()
        result = fn(state)
        latency_ms = (time.perf_counter() - start) * 1000.0
        tokens = _extract_tokens(result)

        metrics.record_latency(node_name, latency_ms)

        _logger.info(
            "node completed: agent=%s latency_ms=%.1f tokens=%d",
            node_name,
            round(latency_ms, 1),
            tokens,
            extra={
                "agent": node_name,
                "latency_ms": round(latency_ms, 1),
                "tokens": tokens,
            },
        )
        return result

    return wrapper
