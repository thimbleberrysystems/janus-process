"""
agents/thalamus.py — Thalamus Router: LangGraph Orchestration.

Wires all brain-region agents into a conditional ``StateGraph``.

Graph topology
--------------

    START
      └─► amygdala
            └─► basal_ganglia
                  ├─[procedural_match is not None]──► pfc ──► END
                  ├─[emotional_weight >= 0.5]────────► hippocampus ──► pfc ──► END
                  └─[otherwise]────────────────────────────────► pfc ──► END

Routing rules (evaluated after basal_ganglia runs):
  1. If a habit was matched (``procedural_match`` is not ``None``) → skip
     Hippocampus; proceed directly to PFC.
  2. Else if emotional weight is high (≥ HIPPOCAMPUS_THRESHOLD) → run
     Hippocampus for memory retrieval before PFC.
  3. Otherwise → proceed directly to PFC.

LangSmith tracing is enabled automatically when the ``LANGCHAIN_TRACING_V2``
and ``LANGCHAIN_API_KEY`` environment variables are set.  No additional
code is required.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from langgraph.graph import END, StateGraph

from agents.amygdala import amygdala_node
from agents.basal_ganglia import basal_ganglia_gate_node, basal_ganglia_node
from agents.hippocampus import hippocampus_node
from agents.pfc import pfc_node
from config import HIPPOCAMPUS_THRESHOLD, MAX_PFC_LOOPS
from models.brain_state import BrainState

_log = logging.getLogger("janus.trace")


# ---------------------------------------------------------------------------
# Per-agent trace wrapper
# ---------------------------------------------------------------------------

_DIVIDER = "─" * 60

_INPUT_FIELDS: dict[str, list[str]] = {
    "amygdala":      ["input"],
    "basal_ganglia": ["input", "emotional_weight"],
    "hippocampus":   ["input", "emotional_weight"],
    "pfc":           ["input", "emotional_weight", "procedural_match", "bg_feedback", "loop_count"],
    "bg_gate":       ["input", "final_response"],
}
_OUTPUT_FIELDS: dict[str, list[str]] = {
    "amygdala":      ["emotional_weight", "llm_temperature", "emotional_directive"],
    "basal_ganglia": ["procedural_match"],
    "hippocampus":   ["retrieved_memories"],
    "pfc":           ["working_memory", "retrieved_memories", "final_response",
                      "should_consolidate", "loop_count", "pfc_proposals"],
    "bg_gate":       ["bg_feedback", "pfc_proposals"],
}


def _fmt(value: object) -> str:
    """Compact but readable representation of a state value."""
    if isinstance(value, list):
        n = len(value)
        if n == 0:
            return "[]"
        preview = str(value[0])[:60] + ("…" if len(str(value[0])) > 60 else "")
        return f"[{n} item(s): {preview!r}{'…' if n > 1 else ''}]"
    if isinstance(value, str) and len(value) > 120:
        return repr(value[:120] + "…")
    return repr(value)


def _traced(name: str, fn: Callable) -> Callable:
    """Wrap an agent node with enter/exit trace logging."""
    in_fields  = _INPUT_FIELDS.get(name, ["input"])
    out_fields = _OUTPUT_FIELDS.get(name, [])

    def wrapper(state: BrainState) -> BrainState:
        _log.info("%s  ▶ %s", _DIVIDER, name.upper())
        for f in in_fields:
            _log.info("  IN  %-20s %s", f, _fmt(state.get(f)))  # type: ignore[arg-type]

        t0 = time.perf_counter()
        result = fn(state)
        ms = (time.perf_counter() - t0) * 1000

        for f in out_fields:
            _log.info("  OUT %-20s %s", f, _fmt(result.get(f)))  # type: ignore[arg-type]
        _log.info("  %-24s %.0f ms", "latency", ms)
        return result

    wrapper.__name__ = getattr(fn, '__name__', type(fn).__name__)
    return wrapper


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def _route_after_basal_ganglia(state: BrainState) -> str:
    """Determine the next node after Basal Ganglia runs.

    Returns
    -------
    "hippocampus" — high-emotion path; retrieve memories before reasoning.
    "pfc"         — direct path; habit already found or low-salience input.
    """
    if state.get("procedural_match") is not None:
        # A habit was matched — no need for expensive RAG retrieval.
        return "pfc"
    if float(state.get("emotional_weight", 0.0)) >= HIPPOCAMPUS_THRESHOLD:
        return "hippocampus"
    return "pfc"


def _route_after_pfc(state: BrainState) -> str:
    """Determine the next node after PFC runs.

    Returns
    -------
    "bg_gate" — deliberate path; evaluate the PFC proposal before committing.
    "end"     — habit path; proposal was driven by a matched habit, skip gate.
    """
    if state.get("procedural_match") is not None:
        return "end"
    return "bg_gate"


def _route_after_bg_gate(state: BrainState) -> str:
    """Determine whether to loop PFC again or exit.

    Returns
    -------
    "pfc" — gate requested refinement and loop budget remains.
    "end" — gate accepted, or loop budget exhausted (best-effort response).
    """
    if state.get("bg_feedback") and int(state.get("loop_count") or 0) < MAX_PFC_LOOPS:
        return "pfc"
    return "end"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(
    *,
    amygdala: Callable | None = None,
    basal_ganglia: Callable | None = None,
    hippocampus: Callable | None = None,
    pfc: Callable | None = None,
    bg_gate: Callable | None = None,
    trace: bool = True,
):
    """Build and compile the Janus brain ``StateGraph``.

    Parameters
    ----------
    amygdala, basal_ganglia, hippocampus, pfc:
        Optional replacement callables for each node.  Defaults to the real
        agent implementations.  Pass lightweight mock callables in unit tests
        to isolate routing logic without making any LLM or service calls.

    Returns
    -------
    A compiled LangGraph ``CompiledStateGraph`` that accepts a ``BrainState``
    dict via ``.invoke()`` and returns the final ``BrainState``.
    """
    _wrap = _traced if trace else (lambda name, fn: fn)
    _amygdala      = _wrap("amygdala",      amygdala      or amygdala_node)
    _basal_ganglia = _wrap("basal_ganglia", basal_ganglia or basal_ganglia_node)
    _hippocampus   = _wrap("hippocampus",   hippocampus   or hippocampus_node)
    _pfc           = _wrap("pfc",           pfc           or pfc_node)
    _bg_gate       = _wrap("bg_gate",       bg_gate       or basal_ganglia_gate_node)

    graph = StateGraph(BrainState)

    graph.add_node("amygdala", _amygdala)
    graph.add_node("basal_ganglia", _basal_ganglia)
    graph.add_node("hippocampus", _hippocampus)
    graph.add_node("pfc", _pfc)
    graph.add_node("bg_gate", _bg_gate)

    graph.set_entry_point("amygdala")
    graph.add_edge("amygdala", "basal_ganglia")
    graph.add_conditional_edges(
        "basal_ganglia",
        _route_after_basal_ganglia,
        {"hippocampus": "hippocampus", "pfc": "pfc"},
    )
    graph.add_edge("hippocampus", "pfc")
    graph.add_conditional_edges(
        "pfc",
        _route_after_pfc,
        {"bg_gate": "bg_gate", "end": END},
    )
    graph.add_conditional_edges(
        "bg_gate",
        _route_after_bg_gate,
        {"pfc": "pfc", "end": END},
    )

    return graph.compile()
