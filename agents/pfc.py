"""
agents/pfc.py — Prefrontal Cortex: Working Memory Synthesis Node.

The PFC is the core reasoning agent.  It receives the full BrainState —
working memory from Redis (STM), relevant memories from ChromaDB (LTM),
an optional procedural match from the Basal Ganglia, and the raw input —
and produces a synthesised ``final_response``.

It also decides whether the current turn is significant enough to warrant
consolidating STM into LTM by setting ``should_consolidate``.

Consolidation threshold: ``emotional_weight >= 0.5``
"""
from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from memory.short_term import add_to_short_term, get_short_term_memory
from models.brain_state import BrainState
from providers.llm import get_llm

_log = logging.getLogger("janus.pfc")

CONSOLIDATION_THRESHOLD: float = 0.5
STM_WINDOW: int = 10  # last N messages to load as working memory

_SYSTEM_PROMPT = (
    "You are the Prefrontal Cortex (PFC) — the core reasoning and synthesis "
    "centre of a brain-inspired AI system.\n\n"
    "You receive:\n"
    "  • The user's current input\n"
    "  • Working memory: the most recent conversation turns (short-term memory)\n"
    "  • Retrieved memories: relevant long-term memories retrieved from a vector "
    "store\n"
    "  • Procedural match: a pre-learned habitual response if one was found "
    "(may be absent)\n\n"
    "Your task: synthesise all of the above into a single, coherent, helpful "
    "response to the user's input. Use working memory for conversational "
    "continuity, retrieved memories for relevant background knowledge, and the "
    "procedural match (if present) as a strong starting signal.\n\n"
    "Be concise and direct. Do not mention that you are an AI or reference "
    "these internal memory systems in your response."
)


def _build_context_block(state: BrainState) -> str:
    parts: list[str] = []

    wm = state.get("working_memory") or []
    if wm:
        parts.append(
            "## Working Memory (recent turns)\n"
            + "\n".join(f"- {m}" for m in wm)
        )
    else:
        parts.append("## Working Memory (recent turns)\n(empty)")

    rm = state.get("retrieved_memories") or []
    if rm:
        parts.append(
            "## Retrieved Memories\n" + "\n".join(f"- {m}" for m in rm)
        )
    else:
        parts.append("## Retrieved Memories\n(none)")

    pm = state.get("procedural_match")
    if pm:
        parts.append(f"## Procedural Match\n{pm}")

    parts.append(f"## Current Input\n{state['input']}")
    return "\n\n".join(parts)


def pfc_node(
    state: BrainState,
    *,
    llm: BaseChatModel | None = None,
) -> BrainState:
    """
    LangGraph node — synthesise all context into a final response.

    Reads STM (Redis) at the start of each turn to populate working_memory.
    Writes the input + response back to STM after inference so the next
    turn can use them as conversational context.  Both STM operations fail
    gracefully when Redis is unavailable.
    """
    _llm = llm or get_llm(temperature=0.3)

    # ── 1. Load working memory from STM ───────────────────────────────────────
    working_memory: list[str] = state.get("working_memory") or []
    if not working_memory:
        try:
            working_memory = get_short_term_memory(n=STM_WINDOW)
        except ConnectionError as exc:
            _log.warning("STM unavailable (Redis down?): %s", exc)

    state = {**state, "working_memory": working_memory}

    # ── 2. Build prompt and run LLM ───────────────────────────────────────────
    context_block = _build_context_block(state)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=context_block),
    ]

    response = _llm.invoke(messages)
    final_response: str = str(response.content).strip()
    should_consolidate: bool = (
        float(state["emotional_weight"]) >= CONSOLIDATION_THRESHOLD
    )

    # ── 3. Write exchange to STM for next turn ────────────────────────────────
    try:
        add_to_short_term(f"User: {state['input']}")
        add_to_short_term(f"Brain: {final_response}")
    except ConnectionError as exc:
        _log.warning("Could not save to STM (Redis down?): %s", exc)

    return {
        **state,
        "working_memory": working_memory,
        "final_response": final_response,
        "should_consolidate": should_consolidate,
    }
