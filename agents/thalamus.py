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

from typing import Callable, Optional

from langgraph.graph import END, StateGraph

from agents.amygdala import amygdala_node
from agents.basal_ganglia import basal_ganglia_node
from agents.hippocampus import hippocampus_node
from agents.pfc import pfc_node
from models.brain_state import BrainState

# Emotional-weight threshold above which the Hippocampus (RAG) path is taken.
HIPPOCAMPUS_THRESHOLD: float = 0.5


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


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(
    *,
    amygdala: Optional[Callable] = None,
    basal_ganglia: Optional[Callable] = None,
    hippocampus: Optional[Callable] = None,
    pfc: Optional[Callable] = None,
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
    _amygdala = amygdala or amygdala_node
    _basal_ganglia = basal_ganglia or basal_ganglia_node
    _hippocampus = hippocampus or hippocampus_node
    _pfc = pfc or pfc_node

    graph = StateGraph(BrainState)

    graph.add_node("amygdala", _amygdala)
    graph.add_node("basal_ganglia", _basal_ganglia)
    graph.add_node("hippocampus", _hippocampus)
    graph.add_node("pfc", _pfc)

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
