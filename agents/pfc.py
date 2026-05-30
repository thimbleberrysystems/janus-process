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

from config import PFC_CONSOLIDATION_THRESHOLD, PFC_DEFAULT_TEMPERATURE, PFC_MODEL, PFC_STM_WINDOW, PFC_SYSTEM_PROMPT
from memory.short_term import add_to_short_term, get_short_term_memory
from models.brain_state import BrainState
from providers.llm import get_llm

_log = logging.getLogger("janus.pfc")

CONSOLIDATION_THRESHOLD: float = PFC_CONSOLIDATION_THRESHOLD
STM_WINDOW: int = PFC_STM_WINDOW

# PFC reasoning prompt loaded from config.PFC_SYSTEM_PROMPT


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

    feedback = state.get("bg_feedback")
    if feedback:
        parts.append(f"## Refinement Instruction\n{feedback}")

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
    _llm = llm or get_llm(model_name=PFC_MODEL, temperature=float(state.get("llm_temperature", PFC_DEFAULT_TEMPERATURE)))

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
    directive = (state.get("emotional_directive") or "").strip()
    system_content = f"{directive}\n\n{PFC_SYSTEM_PROMPT}" if directive else PFC_SYSTEM_PROMPT
    messages = [
        SystemMessage(content=system_content),
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

    loop_count: int = int(state.get("loop_count") or 0) + 1

    return {
        **state,
        "working_memory": working_memory,
        "final_response": final_response,
        "should_consolidate": should_consolidate,
        "loop_count": loop_count,
        "bg_feedback": "",  # clear so gate evaluates this fresh proposal
    }
