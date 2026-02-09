"""
Model and embedder configuration.
Groq and Gemini via model objects; SentenceTransformer for RAG.
Automatic fallback when rate limits hit.
Env: GEMINI_API_KEY, GROQ_API_KEY.
"""

import os
from typing import Any, List, Optional

GROQ_MODELS = [
    "groq:llama-3.1-8b-instant",
    "groq:llama-3.3-70b-versatile",
    "groq:gpt-oss-20b",
]

AUDIO_MODEL = "whisper-large-v3"

GEMINI_MODELS = [
    "gemini:gemini-2.5-flash",
    "gemini:gemini-2.0-flash-lite",
    "gemini:gemini-3-flash-preview",
]

MODEL_DISPLAY_NAMES = {
    "groq:llama-3.1-8b-instant": "Llama 3.1 8B Instant (Groq)",
    "groq:llama-3.3-70b-versatile": "Llama 3.3 70B Versatile (Groq)",
    "groq:gpt-oss-20b": "GPT OSS 20B (Groq)",
    "gemini:gemini-2.5-flash": "Gemini 2.5 Flash (Google)",
    "gemini:gemini-2.0-flash-lite": "Gemini 2.0 Flash Lite (Google)",
    "gemini:gemini-3-flash-preview": "Gemini 3 Flash Preview (Google)",
}

DEFAULT_MODEL = os.getenv("AGNO_MODEL", GROQ_MODELS[1])
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_embedder_instance: Optional[Any] = None


def get_available_models() -> dict:
    """Return dict of provider -> models based on available API keys."""
    available = {}
    if os.getenv("GROQ_API_KEY"):
        available["Groq"] = GROQ_MODELS
    if os.getenv("GEMINI_API_KEY"):
        available["Gemini"] = GEMINI_MODELS
    return available


def get_all_models() -> List[str]:
    """Return all models across all providers."""
    all_models = []
    all_models.extend(GROQ_MODELS)
    all_models.extend(GEMINI_MODELS)
    return all_models


def get_fallback_models() -> List[str]:
    """
    Return ordered list of models to try when rate limits are hit.
    Only includes providers with API keys set.
    """
    fallback = []
    if os.getenv("GROQ_API_KEY"):
        fallback.extend(GROQ_MODELS)
    if os.getenv("GEMINI_API_KEY"):
        fallback.extend(GEMINI_MODELS)
    if not fallback:
        fallback = GROQ_MODELS
    return fallback


def get_model_string(
    prefer: Optional[str] = None, selected: Optional[str] = None
) -> str:
    """
    Return model ID string (provider:model_id).
    selected: specific model ID if user selected one.
    prefer: "google" | "groq" | None.
    """
    if selected:
        return selected
    if prefer == "google" and os.getenv("GEMINI_API_KEY"):
        return GEMINI_MODELS[0]
    if prefer == "groq" and os.getenv("GROQ_API_KEY"):
        return GROQ_MODELS[0]
    if os.getenv("AGNO_MODEL"):
        return os.getenv("AGNO_MODEL")
    fallback = get_fallback_models()
    return fallback[0] if fallback else DEFAULT_MODEL


def get_model_display_name(model_id: str) -> str:
    """Return human-readable name for model ID."""
    return MODEL_DISPLAY_NAMES.get(model_id, model_id)


def get_embedder(model: Optional[str] = None) -> Any:
    """
    Return a single SentenceTransformer embedder instance for RAG.
    Reused across LanceDB and chunking.
    """
    global _embedder_instance
    if _embedder_instance is None:
        from agno.knowledge.embedder.sentence_transformer import (
            SentenceTransformerEmbedder,
        )

        model_id = model or DEFAULT_EMBED_MODEL
        _embedder_instance = SentenceTransformerEmbedder(id=model_id)
    return _embedder_instance


def get_agent_model_string(selected: Optional[str] = None) -> str:
    """Return model string for agents. selected: user-selected model ID."""
    return get_model_string(selected=selected)


def get_agent_model(selected: Optional[str] = None) -> Any:
    """
    Return Agno model object (Groq or Gemini) based on selection.
    selected: model ID string (e.g., "groq:llama-3.3-70b-versatile" or "gemini:gemini-2.5-flash").
    Returns: Model object instance (Groq or Gemini).
    """
    model_id = get_model_string(selected=selected)

    # Groq models
    if model_id.startswith("groq:"):
        try:
            from agno.models.groq import Groq

            model_name = model_id.replace("groq:", "")
            # Handle GPT OSS model name mapping
            if model_name == "gpt-oss-20b":
                model_name = "openai/gpt-oss-20b"
            return Groq(id=model_name)
        except ImportError:
            raise ImportError("Groq models require: pip install agno[groq]")

    # Gemini models
    elif model_id.startswith("gemini:"):
        try:
            from agno.models.google import Gemini

            model_name = model_id.replace("gemini:", "")
            return Gemini(id=model_name)
        except ImportError:
            raise ImportError("Gemini models require: pip install agno[google]")

    # Default: try Groq
    else:
        try:
            from agno.models.groq import Groq

            default_model = GROQ_MODELS[1].replace("groq:", "")
            return Groq(id=default_model)
        except ImportError:
            raise ImportError("Default model requires: pip install agno[groq]")


def is_rate_limit_error(error: Exception) -> bool:
    """Return True if error indicates rate limit from any provider."""
    error_str = str(error).lower()
    indicators = [
        "rate limit",
        "rate_limit",
        "ratelimit",
        "tokens per day",
        "tokens per minute",
        "tpm",
        "tpd",
        "quota",
        "too many requests",
        "429",
    ]
    return any(ind in error_str for ind in indicators)
