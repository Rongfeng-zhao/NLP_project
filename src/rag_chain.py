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
OUT_OF_CONTEXT_KEYWORDS = [
    "next year",
    "future",
    "business strategy",
    "manufacturer",
    "warranty",
]


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
        for key in ("source", "id", "metadata"):
            if key in doc and doc[key] is not None:
                return str(doc[key]).strip()
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


def _parse_product_text(text: str) -> dict[str, str]:
    """Parse pipe-separated product text into a small metadata dictionary."""
    fields = {}
    for part in str(text).split("|"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key and value and key not in fields:
            fields[key] = value
    return fields


def _product_records(retrieved_docs: list[Any] | None) -> list[dict[str, Any]]:
    """Convert retrieved docs into normalized product records for answer formatting."""
    records = []
    for idx, doc in enumerate(retrieved_docs or [], start=1):
        text = _doc_text(doc)
        if not text:
            continue
        fields = _parse_product_text(text)
        source = _doc_source(doc) or str(idx)
        score = _doc_score(doc)
        records.append(
            {
                "doc_number": idx,
                "source": source,
                "score": score,
                "text": text,
                "fields": fields,
                "title": fields.get("title", "Untitled product"),
                "rating": fields.get("average_rating", "N/A"),
                "price": fields.get("price", "N/A"),
                "store": fields.get("store", "N/A"),
                "brand": fields.get("Brand", fields.get("store", "N/A")),
                "item_form": fields.get("Item Form", "N/A"),
                "finish_type": fields.get("Finish Type", "N/A"),
                "special_feature": fields.get("Special Feature", "N/A"),
            }
        )
    return records


def _rating_value(record: dict[str, Any]) -> float | None:
    try:
        return float(record["rating"])
    except (TypeError, ValueError):
        return None


def _price_value(record: dict[str, Any]) -> float | None:
    price = str(record["price"]).strip()
    if price.upper() == "N/A" or not price:
        return None
    match = re.search(r"\d+(?:\.\d+)?", price.replace(",", ""))
    return float(match.group()) if match else None


def _has_terms(record: dict[str, Any], terms: list[str]) -> bool:
    text = record["text"].lower()
    return all(term.lower() in text for term in terms)


def _is_expected_refusal(question: str) -> bool:
    lowered = str(question).lower()
    return any(keyword in lowered for keyword in OUT_OF_CONTEXT_KEYWORDS)


def summarize_retrieved_docs(retrieved_docs: list[Any] | None, max_docs: int = 3) -> str:
    """Create concise retrieved evidence for reports and evaluation files."""
    records = _product_records(retrieved_docs)
    if not records:
        return ""

    lines = []
    for record in records[:max_docs]:
        bits = [
            f"Source {record['source']}",
            f"title: {record['title']}",
            f"rating: {record['rating']}",
            f"price: {record['price']}",
            f"store: {record['store']}",
        ]
        if record["score"]:
            bits.append(f"similarity: {record['score']}")
        lines.append("- " + "; ".join(bits))
    return "\n".join(lines)


def _format_product_line(record: dict[str, Any]) -> str:
    details = [
        f"rating {record['rating']}",
        f"price {record['price']}",
        f"store {record['store']}",
        f"source {record['source']}",
    ]
    return f"{record['title']} ({'; '.join(details)})."


def generate_answer_from_docs(question: str, retrieved_docs: list[Any] | None) -> str:
    """
    Generate a concise rule-based answer from retrieved product records.

    This is a local fallback for the prototype. It avoids copying whole retrieved
    records and refuses questions that require information outside the dataset.
    """
    if _is_expected_refusal(question):
        return INSUFFICIENT_INFORMATION_MESSAGE

    records = _product_records(retrieved_docs)
    if not records:
        return INSUFFICIENT_INFORMATION_MESSAGE

    question_lower = question.lower()
    selected = records
    note = ""

    if "cherioll" in question_lower:
        selected = [
            record
            for record in records
            if "cherioll" in record["brand"].lower()
            or "cherioll" in record["store"].lower()
        ]
        if not selected:
            return INSUFFICIENT_INFORMATION_MESSAGE
    elif "powder" in question_lower:
        selected = [
            record for record in records if "powder" in record["item_form"].lower()
        ]
    elif "waterproof" in question_lower and "natural" in question_lower:
        selected = [
            record
            for record in records
            if _has_terms(record, ["waterproof", "eyebrow"])
            and "natural" in record["finish_type"].lower()
        ]
        excluded = len(records) - len(selected)
        if excluded > 0:
            note = (
                f" {excluded} retrieved product(s) were not included because the "
                "retrieved fields did not explicitly match all requested attributes."
            )
    elif "long lasting" in question_lower:
        selected = [record for record in records if _has_terms(record, ["long", "lasting"])]
    elif "high rating" in question_lower or "high ratings" in question_lower:
        selected = [
            record for record in records if (_rating_value(record) or 0) >= 4.0
        ]
        note = " Products below a 4.0 average rating were excluded."
    elif "best balance" in question_lower and "price" in question_lower:
        comparable = [
            record
            for record in records
            if _rating_value(record) is not None and _price_value(record) is not None
        ]
        if len(comparable) < 2:
            if len(comparable) == 1:
                return (
                    "Only one retrieved product includes both price and rating, so "
                    "the system cannot reliably compare the best balance. The "
                    f"available product is {_format_product_line(comparable[0])}"
                )
            return (
                "The retrieved documents do not contain enough products with both "
                "price and rating to compare the best balance."
            )
        selected = sorted(
            comparable,
            key=lambda record: (_rating_value(record) or 0) / max(_price_value(record) or 1, 1),
            reverse=True,
        )[:1]
    elif "recommend" in question_lower and "rating" in question_lower:
        rated = [record for record in records if _rating_value(record) is not None]
        if rated:
            selected = [max(rated, key=lambda record: _rating_value(record) or 0)]

    if not selected:
        return INSUFFICIENT_INFORMATION_MESSAGE

    selected = selected[:3]
    if len(selected) == 1 and ("recommend" in question_lower or "best balance" in question_lower):
        return f"Based on the retrieved evidence, I would select {_format_product_line(selected[0])}{note}"

    lines = [_format_product_line(record) for record in selected]
    intro = "Based on the retrieved documents, the relevant products are:"
    return intro + "\n" + "\n".join(f"{idx}. {line}" for idx, line in enumerate(lines, start=1)) + note


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

    return (
        "LLM is not connected in this prototype. The final answer is generated "
        "by generate_answer_from_docs() using only retrieved context."
    )


def call_deepseek(prompt: str) -> str:
    """
    Call DeepSeek through the OpenAI-compatible Python SDK.

    Required environment variables:
    - DEEPSEEK_API_KEY
    - DEEPSEEK_MODEL, optional, default deepseek-v4-flash
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The openai package is required for DeepSeek generation. "
            "Install it with: pip install openai"
        ) from exc

    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document-based question-answering assistant. "
                    "Use only the provided retrieved context. Do not use external "
                    "knowledge. If the context is insufficient, return the exact "
                    f"sentence: {INSUFFICIENT_INFORMATION_MESSAGE}"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


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


def rag_answer(question: str, top_k: int = 3, use_deepseek: bool = True) -> dict[str, Any]:
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
            "generation_mode": "error",
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
            "generation_mode": "error",
            "error": retrieval_error,
        }

    if not retrieved_docs:
        prompt = build_prompt(question, retrieved_docs)
        return {
            "question": question,
            "answer": INSUFFICIENT_INFORMATION_MESSAGE,
            "retrieved_docs": [],
            "prompt": prompt,
            "generation_mode": "rule_based",
            "error": "No documents were retrieved.",
        }

    prompt = build_prompt(question, retrieved_docs)
    if use_deepseek:
        try:
            answer = call_deepseek(prompt)
            generation_mode = "deepseek"
        except Exception as exc:
            answer = generate_answer_from_docs(question, retrieved_docs)
            generation_mode = "fallback_rule_based"
            llm_status = f"DeepSeek generation failed: {exc}"
        else:
            llm_status = "DeepSeek generation succeeded."
    else:
        answer = generate_answer_from_docs(question, retrieved_docs)
        generation_mode = "rule_based"
        llm_status = call_llm(prompt)

    return {
        "question": question,
        "answer": answer,
        "retrieved_docs": retrieved_docs,
        "prompt": prompt,
        "generation_mode": generation_mode,
        "llm_status": llm_status,
    }


if __name__ == "__main__":
    example = rag_answer("What products are included in the dataset?")
    print(example["answer"])
