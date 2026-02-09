# Config: model selection (Gemini / Groq), embedder, env.

from config.models import get_embedder, get_model_string

__all__ = ["get_model_string", "get_embedder"]
