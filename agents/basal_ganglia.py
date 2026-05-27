"""
agents/basal_ganglia.py — Procedural / Habit Pattern Layer.

The Basal Ganglia checks each input against a table of learned habits before
handing off to the slower deliberate pathway (Hippocampus → PFC).

If the LLM classifies the input as matching a stored habit:
  • ``state["procedural_match"]`` is set to the habit's canned response.
  • ``hit_count`` for that habit is incremented in the DB.
  • The Thalamus router will then skip Hippocampus and send straight to PFC.

If no habit matches:
  • ``state["procedural_match"]`` remains ``None``.
  • Normal Hippocampus → PFC pathway proceeds.

Schema — ``habits`` table::

    CREATE TABLE IF NOT EXISTS habits (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern   TEXT NOT NULL,
        response  TEXT NOT NULL,
        hit_count INTEGER NOT NULL DEFAULT 0
    )

Usage as a LangGraph node::

    from agents.basal_ganglia import basal_ganglia_node
    graph.add_node("basal_ganglia", basal_ganglia_node)
"""
from __future__ import annotations

import re
import sqlite3
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from config import DATABASE_URL
from models.brain_state import BrainState
from providers.llm import get_llm

# ── Schema ─────────────────────────────────────────────────────────────────────

_HABITS_DDL = """
CREATE TABLE IF NOT EXISTS habits (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern   TEXT NOT NULL,
    response  TEXT NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0
)
"""

_SYSTEM_PROMPT = """\
You are a habit pattern classifier.

You will be given a list of known habit patterns (each with an ID and a description) \
and a user input.  Decide whether the user input has the same intent as any listed \
habit pattern.

Reply with ONLY the integer ID of the matching habit.
If no habit matches, reply with ONLY the word NONE.
Do not explain your answer."""

_INT_RE = re.compile(r"\b(\d+)\b")

# ── Database helpers ───────────────────────────────────────────────────────────


def _url_to_path(db_url: str) -> str:
    prefix = "sqlite:///"
    return db_url[len(prefix):] if db_url.startswith(prefix) else db_url


def get_connection(db_url: str | None = None) -> sqlite3.Connection:
    """Return a sqlite3 connection with the habits schema ensured."""
    path = _url_to_path(db_url or DATABASE_URL)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(_HABITS_DDL)
    conn.commit()


def register_habit(
    pattern: str,
    response: str,
    *,
    conn: sqlite3.Connection,
) -> int:
    """
    Insert a new habit into the DB.

    Args:
        pattern:  Natural-language description of the trigger pattern.
        response: Canned response to return on match.
        conn:     Database connection.

    Returns:
        The ``id`` of the inserted row.
    """
    cur = conn.execute(
        "INSERT INTO habits (pattern, response) VALUES (?, ?)",
        (pattern, response),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def _fetch_all_habits(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT id, pattern, response FROM habits").fetchall()


def _increment_hit(habit_id: int, conn: sqlite3.Connection) -> None:
    conn.execute(
        "UPDATE habits SET hit_count = hit_count + 1 WHERE id = ?",
        (habit_id,),
    )
    conn.commit()


# ── LLM classification ─────────────────────────────────────────────────────────


def _classify(
    user_input: str,
    habits: list[sqlite3.Row],
    llm: BaseChatModel,
) -> Optional[int]:
    """
    Ask the LLM whether *user_input* matches any habit in *habits*.

    Returns the matched habit ``id`` or ``None``.
    """
    habit_list = "\n".join(
        f'[{h["id"]}] "{h["pattern"]}"' for h in habits
    )
    human_content = f"Habits:\n{habit_list}\n\nUser input: {user_input}"
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]
    response = llm.invoke(messages).content.strip()

    match = _INT_RE.search(response)
    return int(match.group(1)) if match else None


# ── LangGraph node ─────────────────────────────────────────────────────────────


def basal_ganglia_node(
    state: BrainState,
    *,
    llm: BaseChatModel | None = None,
    conn: sqlite3.Connection | None = None,
) -> BrainState:
    """
    LangGraph node — check input against learned habits.

    Reads  ``state["input"]``
    Writes ``state["procedural_match"]`` — habit response string, or ``None``

    Args:
        state: Current BrainState.
        llm:   Optional LLM override (defaults to ``get_llm("gpt-4o-mini")``).
        conn:  Optional DB connection override (defaults to configured DATABASE_URL).

    Returns:
        Updated BrainState.  All fields except ``procedural_match`` are unchanged.
    """
    _conn = conn or get_connection()
    _llm = llm or get_llm(model_name="gpt-4o-mini", temperature=0.0)

    habits = _fetch_all_habits(_conn)

    if not habits:
        return {**state, "procedural_match": None}

    matched_id = _classify(state["input"], habits, _llm)

    if matched_id is None:
        return {**state, "procedural_match": None}

    # Fetch the habit row for the matched id (validate it exists)
    row = _conn.execute(
        "SELECT response FROM habits WHERE id = ?", (matched_id,)
    ).fetchone()

    if row is None:
        # LLM hallucinated a non-existent ID
        return {**state, "procedural_match": None}

    _increment_hit(matched_id, _conn)
    return {**state, "procedural_match": row["response"]}
