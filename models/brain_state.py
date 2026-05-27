"""
models/brain_state.py — shared state schema for the LangGraph pipeline.

Every agent node receives a BrainState dict and returns an updated copy.
Use default_brain_state() to create a correctly initialised instance, and
validate_brain_state() for runtime field checks.
"""
from typing import TypedDict

# Fields that must always be present (non-optional with no default).
_REQUIRED_FIELDS: frozenset[str] = frozenset({
    "input",
    "emotional_weight",
    "working_memory",
    "retrieved_memories",
    "should_consolidate",
})


class BrainState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    input: str                        # raw user / sensor input

    # ── Amygdala ──────────────────────────────────────────────────────────────
    emotional_weight: float           # 0.0 (neutral) → 1.0 (highly emotional)

    # ── PFC ───────────────────────────────────────────────────────────────────
    working_memory: list[str]         # last-N messages from Redis STM

    # ── Hippocampus ───────────────────────────────────────────────────────────
    retrieved_memories: list[str]     # top-k docs from ChromaDB LTM

    # ── Basal Ganglia ─────────────────────────────────────────────────────────
    procedural_match: str | None   # matched habit response, or None

    # ── Output ────────────────────────────────────────────────────────────────
    final_response: str | None     # synthesised answer from PFC

    # ── Consolidation flag ────────────────────────────────────────────────────
    should_consolidate: bool          # True → trigger STM→LTM consolidation


def default_brain_state(user_input: str) -> BrainState:
    """Return a BrainState with safe defaults for a new turn."""
    return BrainState(
        input=user_input,
        emotional_weight=0.0,
        working_memory=[],
        retrieved_memories=[],
        procedural_match=None,
        final_response=None,
        should_consolidate=False,
    )


def validate_brain_state(state: dict) -> BrainState:
    """
    Runtime validation of a BrainState dict.

    Raises:
        TypeError: if any required field is absent.
        ValueError: if field values are out of expected range/type.
    """
    missing = _REQUIRED_FIELDS - set(state.keys())
    if missing:
        raise TypeError(f"BrainState missing required fields: {sorted(missing)}")

    if not isinstance(state["emotional_weight"], (int, float)):
        raise ValueError("emotional_weight must be a number")
    if not (0.0 <= float(state["emotional_weight"]) <= 1.0):
        raise ValueError("emotional_weight must be between 0.0 and 1.0")
    if not isinstance(state["working_memory"], list):
        raise ValueError("working_memory must be a list")
    if not isinstance(state["retrieved_memories"], list):
        raise ValueError("retrieved_memories must be a list")
    if not isinstance(state["should_consolidate"], bool):
        raise ValueError("should_consolidate must be a bool")

    return state  # type: ignore[return-value]
