"""Configuration for MultiLangRAG."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class RAGConfig:
    """Immutable configuration for the RAG pipeline."""

    # Embedding model
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # LLM settings
    llm_model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0
    max_tokens: int = 1024

    # Retriever settings
    top_k: int = 2

    # ChromaDB persistence
    collection_name: str = "multilingual_hr_docs"
    persist_directory: str = "data/chroma"

    # Groq API
    groq_api_key: str = field(
        default_factory=lambda: os.environ.get("GROQ_API_KEY", "")
    )

    def validate(self) -> None:
        """Raise if required config is missing."""
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Set it in .env or as an environment variable."
            )
