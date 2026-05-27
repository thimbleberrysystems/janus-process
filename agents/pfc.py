"""
agents/pfc.py — Prefrontal Cortex: Working Memory Synthesis Node.

The PFC is the core reasoning agent.  It receives the full BrainState —
working memory from Redis (STM), relevant memories from ChromaDB (LTM),
an optional procedural match from the Basal Ganglia, and the raw input —
and produces a synthesised ``final_response``.

It also decides whether the current turn is significant enough to warrant
consolidating STM into LTM by setting ``should_consolidate``.

Consolidation threshold: ``emotional_weight >= 0.5``

Usage as a LangGraph node::

    from agents.pfc import pfc_node
    graph.add_node("pfc", pfc_node)
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from models.brain_state import BrainState
from providers.llm import get_llm

# ── Thresholds ─────────────────────────────────────────────────────────────────

CONSOLIDATION_THRESHOLD: float = 0.5

# ── Prompts ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are the Prefrontal Cortex (PFC) — the core reasoning and synthesis centre \
of a brain-inspired AI system.

You receive:
  • The user's current input
  • Working memory: the most recent conversation turns (short-term memory)
  • Retrieved memories: relevant long-term memories retrieved from a vector store
  • Procedural match: a pre-learned habitual response if one was found (may be absent)

Your task: synthesise all of the above into a single, coherent, helpful response \
to the user's input. Use working memory for conversational continuity, retrieved \
memories for relevant background knowledge, and the procedural match (if present) \
as a strong starting signal.

Be concise and direct. Do not mention that you are an AI or reference these \
internal memory systems in your response."""


def _build_context_block(state: BrainState) -> str:
    """Format the full context payload as a structured text block."""
    parts: list[str] = []

    # Working memory (STM)
    wm = state.get("working_memory") or []
    if wm:
        parts.append("## Working Memory (recent turns)\n" + "\n".join(f"- {m}" for m in wm))
    else:
        parts.append("## Working Memory (recent turns)\n(empty)")

    # Retrieved memories (LTM)
    rm = state.get("retrieved_memories") or []
    if rm:
        parts.append("## Retrieved Memories\n" + "\n".join(f"- {m}" for m in rm))
    else:
        parts.append("## Retrieved Memories\n(none)")

    # Procedural match (Basal Ganglia)
    pm = state.get("procedural_match")
    if pm:
        parts.append(f"## Procedural Match\n{pm}")
    else:
        parts.append("## Procedural Match\n(none)")

    # Current input
    parts.append(f"## Current Input\n{state['input']}")

    return "\n\n".join(parts)


# ── LangGraph node ─────────────────────────────────────────────────────────────

def pfc_node(
    state: BrainState,
    *,
    llm: BaseChatModel | None = None,
) -> BrainState:
    """
    LangGraph node — synthesise all context into a final response.

    Reads
        ``state["input"]``             — current user input
        ``state["working_memory"]``    — STM messages
        ``state["retrieved_memories"]``— LTM hits from Hippocampus
        ``state["procedural_match"]``  — habit response from Basal Ganglia
        ``state["emotional_weight"]``  — used to set consolidation flag

    Writes
        ``state["final_response"]``    — synthesised answer
        ``state["should_consolidate"]``— True when emotional_weight >= threshold

    Args:
        state: Current BrainState.
        llm:   Optional LLM override (defaults to ``get_llm("gpt-4o")``).

    Returns:
        Updated BrainState with ``final_response`` and ``should_consolidate`` set.
    """
    _llm = llm or get_llm(model_name="gpt-4o", temperature=0.3)

    context_block = _build_context_block(state)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=context_block),
    ]

    response = _llm.invoke(messages)
    final_response: str = response.content.strip()

    should_consolidate: bool = float(state["emotional_weight"]) >= CONSOLIDATION_THRESHOLD

    return {
        **state,
        "final_response": final_response,
        "should_consolidate": should_consolidate,
    }
