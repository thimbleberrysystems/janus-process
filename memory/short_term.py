"""
memory/short_term.py — Redis-backed short-term memory (STM).

Provides a thin wrapper around RedisChatMessageHistory that exposes two
helpers consumed by agent nodes:

    add_to_short_term(text, session_id, ttl)
    get_short_term_memory(n, session_id)

TTL is applied to the underlying Redis key so the buffer decays automatically.
All Redis connectivity errors are surfaced as ConnectionError so callers can
handle them consistently.
"""
import redis
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import HumanMessage

from config import REDIS_URL, SESSION_ID, STM_TTL


def _make_history(session_id: str, ttl: int) -> RedisChatMessageHistory:
    """
    Return a RedisChatMessageHistory for *session_id*.

    Raises ConnectionError immediately if Redis cannot be reached, so
    callers always get a clear error rather than a silent failure.
    """
    try:
        history = RedisChatMessageHistory(
            session_id=session_id,
            url=REDIS_URL,
            ttl=ttl,
        )
        # Eagerly verify connectivity by issuing a cheap command.
        history.redis_client.ping()
        return history
    except redis.exceptions.ConnectionError as exc:
        raise ConnectionError(
            f"Cannot connect to Redis at {REDIS_URL!r}: {exc}"
        ) from exc
    except Exception as exc:
        raise ConnectionError(
            f"Redis initialisation failed for session {session_id!r}: {exc}"
        ) from exc


def add_to_short_term(
    text: str,
    *,
    session_id: str = SESSION_ID,
    ttl: int = STM_TTL,
) -> None:
    """Append *text* to the STM buffer for *session_id*."""
    history = _make_history(session_id, ttl)
    history.add_message(HumanMessage(content=text))


def get_short_term_memory(
    n: int | None = None,
    *,
    session_id: str = SESSION_ID,
    ttl: int = STM_TTL,
) -> list[str]:
    """
    Return the last *n* messages from the STM buffer as plain strings.

    If *n* is None all stored messages are returned.
    Returns an empty list when the buffer is empty.
    """
    history = _make_history(session_id, ttl)
    messages = history.messages
    if n is not None:
        messages = messages[-n:]
    return [m.content for m in messages]


def clear_short_term(
    *,
    session_id: str = SESSION_ID,
    ttl: int = STM_TTL,
) -> None:
    """Flush all messages for *session_id* (used by the API layer and tests)."""
    history = _make_history(session_id, ttl)
    history.clear()
