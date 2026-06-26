"""RAG pipeline: retriever + LLM chain."""

from __future__ import annotations

from collections.abc import Iterable

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from .config import RAGConfig
from .documents import DOCUMENTS
from .vectorstore import get_vectorstore

_SYSTEM_PROMPT = """\
You are a helpful multilingual assistant.
Use the retrieved context to answer the user's question.
The context may be in a different language than the question.
Answer in the same language as the user's question.

Context:
{context}

Question:
{question}

Answer:"""


def _format_docs(docs: Iterable[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _retrieve_docs(query: str, retriever) -> list[Document]:
    return retriever.invoke(query)


def build_chain(config: RAGConfig | None = None) -> dict:
    """Build and return the RAG chain, retriever, vectorstore, and LLM.

    Returns a dict with keys: chain, retriever, vectorstore, llm.
    """
    if config is None:
        config = RAGConfig()
    config.validate()

    vectorstore = get_vectorstore(documents=DOCUMENTS, config=config)
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.top_k})

    llm = ChatGroq(
        model=config.llm_model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        api_key=config.groq_api_key,
    )

    prompt = ChatPromptTemplate.from_template(_SYSTEM_PROMPT)

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return {
        "chain": chain,
        "retriever": retriever,
        "vectorstore": vectorstore,
        "llm": llm,
    }
