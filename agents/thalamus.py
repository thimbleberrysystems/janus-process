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
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from langgraph.graph import END, StateGraph

from agents.amygdala import amygdala_node
from agents.basal_ganglia import basal_ganglia_node
from agents.hippocampus import hippocampus_node
from agents.pfc import pfc_node
from models.brain_state import BrainState

_log = logging.getLogger("janus.trace")

HIPPOCAMPUS_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# Per-agent trace wrapper
# ---------------------------------------------------------------------------

_DIVIDER = "─" * 56

_INPUT_FIELDS: dict[str, list[str]] = {
    "amygdala":      ["input"],
    "basal_ganglia": ["input", "emotional_weight"],
    "hippocampus":   ["input", "emotional_weight"],
    "pfc":           ["input", "emotional_weight", "procedural_match",
                      "working_memory", "retrieved_memories"],
}
_OUTPUT_FIELDS: dict[str, list[str]] = {
    "amygdala":      ["emotional_weight"],
    "basal_ganglia": ["procedural_match"],
    "hippocampus":   ["retrieved_memories"],
    "pfc":           ["final_response", "should_consolidate"],
}


def _fmt(value: object) -> str:
    if isinstance(value, list):
        n = len(value)
        if n == 0:
            return "[]"
        preview = str(value[0])[:70] + ("…" if len(str(value[0])) > 70 else "")
        return f"[{n} item(s)  {preview!r}{'…' if n > 1 else ''}]"
    if isinstance(value, str) and len(value) > 120:
        return repr(value[:120] + "…")
    return repr(value)


def _traced(name: str, fn: Callable) -> Callable:
    in_fields  = _INPUT_FIELDS.get(name, ["input"])
    out_fields = _OUTPUT_FIELDS.get(name, [])

    def wrapper(state: BrainState) -> BrainState:
        _log.info("%s  ▶ %s", _DIVIDER, name.upper())
        for f in in_fields:
            _log.info("  IN  %-22s %s", f, _fmt(state.get(f)))  # type: ignore[arg-type]

        t0 = time.perf_counter()
        result = fn(state)
        ms = (time.perf_counter() - t0) * 1000

        for f in out_fields:
            _log.info("  OUT %-22s %s", f, _fmt(result.get(f)))  # type: ignore[arg-type]
        _log.info("  latency: %.0f ms", ms)
        return result

    wrapper.__name__ = getattr(fn, "__name__", name)
    return wrapper


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def _route_after_basal_ganglia(state: BrainState) -> str:
    if state.get("procedural_match") is not None:
        return "pfc"
    if float(state.get("emotional_weight", 0.0)) >= HIPPOCAMPUS_THRESHOLD:
        return "hippocampus"
    return "pfc"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(
    *,
    amygdala: Callable | None = None,
    basal_ganglia: Callable | None = None,
    hippocampus: Callable | None = None,
    pfc: Callable | None = None,
    trace: bool = True,
):
    """Build and compile the Janus brain StateGraph.

    Pass trace=False (or mock callables) in unit tests to skip logging.
    """
    _wrap = _traced if trace else (lambda name, fn: fn)
    _amygdala      = _wrap("amygdala",      amygdala      or amygdala_node)
    _basal_ganglia = _wrap("basal_ganglia", basal_ganglia or basal_ganglia_node)
    _hippocampus   = _wrap("hippocampus",   hippocampus   or hippocampus_node)
    _pfc           = _wrap("pfc",           pfc           or pfc_node)

    graph = StateGraph(BrainState)
    graph.add_node("amygdala",      _amygdala)
    graph.add_node("basal_ganglia", _basal_ganglia)
    graph.add_node("hippocampus",   _hippocampus)
    graph.add_node("pfc",           _pfc)

    graph.set_entry_point("amygdala")
    graph.add_edge("amygdala", "basal_ganglia")
    graph.add_conditional_edges(
        "basal_ganglia",
        _route_after_basal_ganglia,
        {"hippocampus": "hippocampus", "pfc": "pfc"},
    )
    graph.add_edge("hippocampus", "pfc")
    graph.add_edge("pfc", END)

    return graph.compile()
