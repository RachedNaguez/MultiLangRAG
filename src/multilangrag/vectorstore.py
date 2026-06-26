"""ChromaDB vector store with persistence and lazy initialization."""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import RAGConfig
from .embeddings import get_embeddings

_store: Chroma | None = None


def get_vectorstore(
    documents: list[Document] | None = None,
    config: RAGConfig | None = None,
) -> Chroma:
    """Return the Chroma vector store, creating it on first call.

    Uses a persistent directory so embeddings survive restarts.
    If the directory already exists, existing embeddings are reused
    and `documents` is ignored.
    """
    global _store
    if _store is not None:
        return _store

    if config is None:
        config = RAGConfig()

    persist_path = Path(config.persist_directory)
    embeddings = get_embeddings(config.embedding_model)

    if persist_path.exists() and any(persist_path.iterdir()):
        _store = Chroma(
            collection_name=config.collection_name,
            embedding_function=embeddings,
            persist_directory=config.persist_directory,
        )
    else:
        if documents is None:
            raise ValueError(
                "No persisted data found. Pass documents to initialize the store."
            )
        persist_path.mkdir(parents=True, exist_ok=True)
        _store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=config.collection_name,
            persist_directory=config.persist_directory,
        )
    return _store


def reset_store() -> None:
    """Reset the global store reference (useful for testing)."""
    global _store
    _store = None
