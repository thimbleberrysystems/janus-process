"""
logging_config.py — structured logging and LangSmith tracing setup.

Import and call setup_logging() once at application startup (main.py does this).
All other modules should use:  logging.getLogger("janus.<module_name>")
"""
import logging
import os
import sys


def setup_logging() -> logging.Logger:
    """
    Configure root logger with a structured format and enable LangSmith
    tracing if LANGSMITH_API_KEY is present in the environment.

    Returns:
        The top-level "janus" logger.
    """
    # Avoid importing config at module level so this file can be imported
    # before config is initialised during tests.
    from config import LANGSMITH_API_KEY, LANGSMITH_PROJECT  # noqa: PLC0415

    # ── LangSmith tracing ─────────────────────────────────────────────────────
    if LANGSMITH_API_KEY:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
        os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY

    # ── Standard logging ──────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-24s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,  # override any previously installed handlers
    )

    logger = logging.getLogger("janus")
    logger.info(
        "Logging initialised. LangSmith tracing: %s | project: %s",
        "enabled" if LANGSMITH_API_KEY else "disabled",
        LANGSMITH_PROJECT,
    )
    return logger


# Module-level logger — available as `from logging_config import logger`
logger = setup_logging()
