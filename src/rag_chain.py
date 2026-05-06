"""
RAG question-answering pipeline.

This module is owned by Member 3. It connects the retriever layer with a
simple language-model interface and returns document-grounded answers.
"""

from __future__ import annotations

import os
import re
import sys
import importlib
from pathlib import Path
from typing import Any


INSUFFICIENT_INFORMATION_MESSAGE = (
    "The provided documents do not contain enough information to answer this question."
)

REPORT_NOTES = {
    "rag_chain_implementation": (
        "The RAG chain connects the semantic retriever with the language model to "
        "generate document-grounded answers. For each user query, the retriever "
        "returns the top-k most relevant document chunks from the vector store. "
        "These chunks are inserted into a structured prompt that instructs the "
        "language model to answer only using the provided context. If the retrieved "
        "documents do not contain enough information, the model is instructed to "
        "state that the information is unavailable. This design reduces "
        "hallucination and improves the traceability of generated answers."
    )
}

_RETRIEVER_INSTANCE = None


def _import_retriever_module():
    """Import retriever.py without hiding dependency errors from that module."""
    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return importlib.import_module("retriever")


def _doc_text(doc: Any) -> str:
    """Extract readable text from a retrieved document."""
    if isinstance(doc, dict):
        return str(
            doc.get("text")
            or doc.get("content")
            or doc.get("page_content")
            or doc.get("document")
            or ""
        ).strip()
    return str(doc).strip()


def _doc_source(doc: Any) -> str:
    """Extract source metadata when available."""
    if isinstance(doc, dict):
        return str(doc.get("source") or doc.get("id") or doc.get("metadata", "")).strip()
    return ""


def _doc_score(doc: Any) -> str:
    """Extract retrieval score when available."""
    if isinstance(doc, dict) and doc.get("score") is not None:
        try:
            return f"{float(doc['score']):.4f}"
        except (TypeError, ValueError):
            return str(doc["score"])
    return ""


def format_retrieved_docs(retrieved_docs: list[Any] | None) -> str:
    """
    Convert retrieved documents into a readable context block.

    Supports both list[dict] with text/source/score fields and list[str].
    """
    if not retrieved_docs:
        return ""

    context_parts = []
    for idx, doc in enumerate(retrieved_docs, start=1):
        text = _doc_text(doc)
        if not text:
            continue

        source = _doc_source(doc)
        score = _doc_score(doc)
        metadata = []
        if source:
            metadata.append(f"Source: {source}")
        if score:
            metadata.append(f"Score: {score}")

        heading = f"[Document {idx}]"
        if metadata:
            heading += " " + " | ".join(metadata)
        context_parts.append(f"{heading}\n{text}")

    return "\n\n".join(context_parts)


def build_prompt(question: str, retrieved_docs: list[Any] | None) -> str:
    """
    Build a grounded RAG prompt using the user question and retrieved documents.

    The prompt instructs the LLM to answer only using the provided context.
    """
    context = format_retrieved_docs(retrieved_docs)
    if not context:
        context = "No relevant documents were retrieved."

    return f"""You are a document-based question-answering assistant.

Answer the user's question using only the provided context.
Do not use external knowledge.
If the context does not contain enough information, say:
"{INSUFFICIENT_INFORMATION_MESSAGE}"

Provide a concise and factual answer.
Mention the relevant source document if available.

Context:
{context}

User question:
{question}

Answer:"""


def _simple_context_answer(prompt: str) -> str:
    """
    Lightweight local placeholder for answer generation.

    It intentionally stays conservative: when no context is available it refuses
    to answer, otherwise it returns the most relevant context sentences.
    """
    context_match = re.search(
        r"Context:\n(?P<context>.*?)\n\nUser question:", prompt, flags=re.DOTALL
    )
    question_match = re.search(
        r"User question:\n(?P<question>.*?)\n\nAnswer:", prompt, flags=re.DOTALL
    )

    context = context_match.group("context").strip() if context_match else ""
    question = question_match.group("question").strip() if question_match else ""

    if not context or context == "No relevant documents were retrieved.":
        return INSUFFICIENT_INFORMATION_MESSAGE

    question_terms = {
        term
        for term in re.findall(r"[A-Za-z0-9]+", question.lower())
        if len(term) > 2
    }
    sentences = re.split(r"(?<=[.!?])\s+", context.replace("\n", " "))
    ranked_sentences = sorted(
        (sentence.strip() for sentence in sentences if sentence.strip()),
        key=lambda sentence: sum(
            1 for term in question_terms if term in sentence.lower()
        ),
        reverse=True,
    )
    selected = [sentence for sentence in ranked_sentences[:3] if sentence]

    if not selected:
        return INSUFFICIENT_INFORMATION_MESSAGE

    return " ".join(selected)


def call_llm(prompt: str) -> str:
    """
    Call the selected LLM.

    This project currently uses a conservative local placeholder so the pipeline
    can run without API keys. Later this function can be replaced with OpenAI,
    Hugging Face, or another LLM provider.
    """
    provider = os.getenv("RAG_LLM_PROVIDER", "placeholder").lower()

    if provider != "placeholder":
        return (
            "LLM provider is not configured in this project yet. "
            "Set RAG_LLM_PROVIDER=placeholder or replace call_llm() with an API call."
        )

    try:
        return _simple_context_answer(prompt)
    except Exception as exc:
        return f"LLM call failed: {exc}"


def _retrieve_with_available_interface(question: str, top_k: int) -> tuple[list[Any], str | None]:
    """
    Retrieve documents while supporting multiple possible Member 2 interfaces.
    """
    try:
        retriever_module = _import_retriever_module()
        retrieve_top_k = getattr(retriever_module, "retrieve_top_k")
        return retrieve_top_k(question, top_k=top_k), None
    except (ImportError, AttributeError):
        pass
    except Exception as exc:
        return [], f"retrieve_top_k() failed: {exc}"

    try:
        retriever_module = _import_retriever_module()
        ProductRetriever = getattr(retriever_module, "ProductRetriever")

        global _RETRIEVER_INSTANCE
        if _RETRIEVER_INSTANCE is None:
            index_dir = os.getenv("VECTOR_INDEX_DIR", "artifacts/vector_index")
            _RETRIEVER_INSTANCE = ProductRetriever(index_dir=index_dir)
        return _RETRIEVER_INSTANCE.retrieve(question, top_k=top_k), None
    except Exception as exc:
        return [], (
            "Retriever is not available or could not be loaded. "
            f"Details: {exc}"
        )


def rag_answer(question: str, top_k: int = 3) -> dict[str, Any]:
    """
    Main RAG pipeline:
    1. Retrieve top-k relevant documents
    2. Build prompt
    3. Call LLM
    4. Return answer and supporting context
    """
    if question is None or str(question).strip() == "":
        prompt = build_prompt("", [])
        return {
            "question": question,
            "answer": "Question cannot be empty.",
            "retrieved_docs": [],
            "prompt": prompt,
            "error": "Question cannot be empty.",
        }

    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)

    retrieved_docs, retrieval_error = _retrieve_with_available_interface(question, top_k)
    if retrieval_error:
        prompt = build_prompt(question, retrieved_docs)
        return {
            "question": question,
            "answer": retrieval_error,
            "retrieved_docs": retrieved_docs,
            "prompt": prompt,
            "error": retrieval_error,
        }

    if not retrieved_docs:
        prompt = build_prompt(question, retrieved_docs)
        return {
            "question": question,
            "answer": INSUFFICIENT_INFORMATION_MESSAGE,
            "retrieved_docs": [],
            "prompt": prompt,
            "error": "No documents were retrieved.",
        }

    prompt = build_prompt(question, retrieved_docs)
    answer = call_llm(prompt)

    return {
        "question": question,
        "answer": answer,
        "retrieved_docs": retrieved_docs,
        "prompt": prompt,
    }


if __name__ == "__main__":
    example = rag_answer("What products are included in the dataset?")
    print(example["answer"])
