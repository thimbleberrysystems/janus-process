"""
agents/amygdala.py — Emotional Salience Scorer.

The Amygdala is the first node every input passes through.  It scores the
emotional/urgency salience of ``state["input"]`` on a 0.0–1.0 scale and
writes the result to ``state["emotional_weight"]``.

Design mirrors the biological amygdala: fast, lightweight, runs before deep
reasoning, and gates how much attention subsequent agents pay to an input.

Usage as a LangGraph node::

    from agents.amygdala import amygdala_node
    graph.add_node("amygdala", amygdala_node)
"""
from __future__ import annotations

import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from models.brain_state import BrainState
from providers.llm import get_llm

# ── Prompt ─────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an emotional salience classifier.
Rate how emotionally charged or urgent the following input is on a scale \
from 0.0 to 1.0.

Calibration guide:
  0.0 — completely neutral (e.g., "What is 2 + 2?", "Set a reminder")
  0.3 — mildly personal or moderately important
  0.6 — clearly emotional or stressful
  1.0 — crisis-level or highly distressing (grief, fear, rage, emergency)

Reply with ONLY a single decimal number between 0.0 and 1.0. Nothing else."""

# ── Score parsing ──────────────────────────────────────────────────────────────

_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_score(raw: str) -> float:
    """
    Extract a float in [0.0, 1.0] from *raw* LLM output.

    Returns 0.0 if no number is found; clamps values outside [0.0, 1.0].
    """
    match = _FLOAT_RE.search(raw.strip())
    if not match:
        return 0.0
    return max(0.0, min(1.0, float(match.group())))


# ── LangGraph node ─────────────────────────────────────────────────────────────

def amygdala_node(
    state: BrainState,
    *,
    llm: BaseChatModel | None = None,
) -> BrainState:
    """
    LangGraph node — score the emotional salience of the current input.

    Reads  ``state["input"]`` and writes ``state["emotional_weight"]``.
    All other state fields are forwarded unchanged.

    Args:
        state: Current BrainState.
        llm:   Optional LLM override (defaults to ``get_llm("gpt-4o-mini")``).

    Returns:
        Updated BrainState with ``emotional_weight`` set.
    """
    _llm = llm or get_llm(temperature=0.0)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=state["input"]),
    ]

    response = _llm.invoke(messages)
    score = _parse_score(str(response.content))

    return {**state, "emotional_weight": score}
