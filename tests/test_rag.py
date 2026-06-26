"""Tests for MultiLangRAG."""

from __future__ import annotations

import pytest

from multilangrag.config import RAGConfig
from multilangrag.documents import DOCUMENTS
from multilangrag.embeddings import get_embeddings
from multilangrag.rag import _format_docs


class TestConfig:
    def test_defaults(self):
        cfg = RAGConfig()
        assert cfg.llm_model == "llama-3.3-70b-versatile"
        assert cfg.temperature == 0.0
        assert cfg.top_k == 2
        assert cfg.persist_directory == "data/chroma"

    def test_frozen(self):
        cfg = RAGConfig()
        with pytest.raises(AttributeError):
            cfg.temperature = 0.5  # type: ignore[misc]

    def test_validate_missing_key(self):
        cfg = RAGConfig(groq_api_key="")
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            cfg.validate()


class TestDocuments:
    def test_document_count(self):
        assert len(DOCUMENTS) == 3

    def test_languages_present(self):
        languages = {doc.metadata["language"] for doc in DOCUMENTS}
        assert languages == {"English", "French", "Spanish"}

    def test_documents_have_content(self):
        for doc in DOCUMENTS:
            assert len(doc.page_content) > 0


class TestFormatDocs:
    def test_single_doc(self):
        from langchain_core.documents import Document

        docs = [Document(page_content="hello", metadata={})]
        assert _format_docs(docs) == "hello"

    def test_multiple_docs(self):
        from langchain_core.documents import Document

        docs = [
            Document(page_content="first", metadata={}),
            Document(page_content="second", metadata={}),
        ]
        result = _format_docs(docs)
        assert "first" in result
        assert "second" in result
        assert "\n\n" in result


class TestEmbeddings:
    def test_singleton(self):
        e1 = get_embeddings()
        e2 = get_embeddings()
        assert e1 is e2
