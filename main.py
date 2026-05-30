"""
main.py — application entry point.

Wires all brain-region agents via the Thalamus router (Phase 8 graph).
Graph: amygdala → basal_ganglia → [hippocampus →] pfc → END
"""
import logging

from agents.thalamus import build_graph
from logging_config import logger  # noqa: F401 — configures logging on import
from memory.consolidator import maybe_consolidate
from models.brain_state import BrainState, default_brain_state

_log = logging.getLogger("janus.main")


# ── Public API ────────────────────────────────────────────────────────────────

brain = build_graph()


def think(user_input: str) -> BrainState:
    """Run the brain graph for a single turn and return the final state."""
    state = default_brain_state(user_input)
    result: BrainState = brain.invoke(state)  # type: ignore[attr-defined]
    maybe_consolidate(result)
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Janus Process  (Ctrl-C to exit)\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        result = think(user_input)
        response = result.get("final_response") or "(no response)"
        print(f"Brain: {response}\n")
