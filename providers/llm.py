"""
providers/llm.py — LLM and embedding provider abstraction.

Controls which backend (Ollama or OpenAI) is used across all agents.
All agents import get_llm() / get_embedding_model() from here — never
instantiate a model directly in agent code.

Usage:
    from providers.llm import get_llm, get_embedding_model

    llm   = get_llm(temperature=0.3)
    embed = get_embedding_model()
"""
from __future__ import annotations

from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from config import (
    LLM_PROVIDER,
    OLLAMA_EMBEDDING_MODEL,
    OLLAMA_MODEL,
    OLLAMA_URL,
    OPENAI_API_KEY,
)


def get_llm(model_name: Optional[str] = None, temperature: float = 0.0) -> BaseChatModel:
    """
    Return a LangChain chat model for the configured provider.

    Args:
        model_name:  Override the default model set in .env.
        temperature: Sampling temperature (0.0 = deterministic).

    Returns:
        BaseChatModel — ChatOllama or ChatOpenAI depending on LLM_PROVIDER.
    """
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama  # langchain-ollama package

        return ChatOllama(
            model=model_name or OLLAMA_MODEL,
            base_url=OLLAMA_URL,
            temperature=temperature,
        )

    from langchain_openai import ChatOpenAI  # langchain-openai package

    return ChatOpenAI(
        model=model_name or "gpt-4o",
        temperature=temperature,
        api_key=OPENAI_API_KEY,
    )


def get_embedding_model(model_name: Optional[str] = None) -> Embeddings:
    """
    Return a LangChain embeddings model for the configured provider.

    Args:
        model_name: Override the default embedding model set in .env.

    Returns:
        Embeddings — OllamaEmbeddings or OpenAIEmbeddings.
    """
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings  # langchain-ollama package

        return OllamaEmbeddings(
            model=model_name or OLLAMA_EMBEDDING_MODEL,
            base_url=OLLAMA_URL,
        )

    from langchain_openai import OpenAIEmbeddings  # langchain-openai package

    return OpenAIEmbeddings(api_key=OPENAI_API_KEY)
