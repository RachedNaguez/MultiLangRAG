"""Embedding model with singleton caching to avoid reloading."""

from __future__ import annotations

import threading

from langchain_huggingface import HuggingFaceEmbeddings

_instance: HuggingFaceEmbeddings | None = None
_lock = threading.Lock()

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_embeddings(model_name: str = DEFAULT_MODEL) -> HuggingFaceEmbeddings:
    """Return a shared HuggingFaceEmbeddings instance (loaded once)."""
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is None:
            _instance = HuggingFaceEmbeddings(model_name=model_name)
    return _instance
