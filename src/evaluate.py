"""
Simple evaluation pipeline for the RAG system.

This module is owned by Member 3. It runs a fixed set of test questions,
records answers and retrieved context, and saves structured and readable
evaluation outputs.
"""

from __future__ import annotations

import re
import csv
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from rag_chain import (
        INSUFFICIENT_INFORMATION_MESSAGE,
        OUT_OF_CONTEXT_KEYWORDS,
        REPORT_NOTES,
        rag_answer,
        summarize_retrieved_docs,
    )
except ImportError:
    from src.rag_chain import (
        INSUFFICIENT_INFORMATION_MESSAGE,
        OUT_OF_CONTEXT_KEYWORDS,
        REPORT_NOTES,
        rag_answer,
        summarize_retrieved_docs,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
EVALUATION_CSV = RESULTS_DIR / "evaluation_results.csv"
SAMPLE_OUTPUTS_MD = RESULTS_DIR / "sample_outputs.md"


TEST_QUESTIONS = [
    "What waterproof eyebrow products have a natural finish?",
    "Which beauty products have powder as the item form?",
    "What products are from the Cherioll brand?",
    "Which makeup products are described as long lasting?",
    "Which all beauty products have high ratings?",
    "Which waterproof eyebrow product would you recommend based on the retrieved ratings?",
    "Which products are suitable for customers looking for long lasting makeup?",
    "Which retrieved product has the best balance between price and rating?",
    "What will the product price be next year?",
    "What is the manufacturer's future business strategy?",
    "Does the dataset contain warranty information?",
]


REPORT_TEXT = {
    **REPORT_NOTES,
    "evaluation_design": (
        "The evaluation was designed to assess both retrieval quality and answer "
        "generation quality. A set of test questions was created to cover direct "
        "factual queries, comparison-based queries, recommendation-style queries, "
        "and out-of-context queries. The current prototype uses a 20,000-row "
        "sampled product dataset and retrieval examples from the project notebook, "
        "including queries about waterproof eyebrow products, powder item form, "
        "Cherioll brand products, long lasting makeup, and highly rated beauty "
        "products. For each question, the system records the generated answer, "
        "retrieved context, and evaluation scores. The outputs are saved in a CSV "
        "file for structured analysis and in a Markdown file for qualitative "
        "inspection."
    ),
    "evaluation_metrics": (
        "Three evaluation criteria were used: relevance, faithfulness, and "
        "completeness. Relevance measures whether the answer directly addresses "
        "the user question. Faithfulness measures whether the answer is supported "
        "by the retrieved context and avoids unsupported claims. Completeness "
        "measures whether the answer provides sufficient detail based on the "
        "available evidence. These criteria are scored on a five-point scale, "
        "where a higher score indicates better RAG performance."
    ),
}


def _tokens(text: str) -> set[str]:
    """Return simple lowercase tokens for rule-based scoring."""
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", str(text).lower())
        if len(token) > 2
    }


def _contains_error(answer: str) -> bool:
    """Detect pipeline errors or unavailable components."""
    lowered = str(answer).lower()
    error_phrases = [
        "failed",
        "not available",
        "could not be loaded",
        "not configured",
        "cannot be empty",
    ]
    return any(phrase in lowered for phrase in error_phrases)


def _expected_refusal(question: str) -> bool:
    """Return True when the question asks for information outside product records."""
    lowered = str(question).lower()
    return any(keyword in lowered for keyword in OUT_OF_CONTEXT_KEYWORDS)


def _refusal_detected(answer: str) -> bool:
    """Detect the standard insufficient-information answer."""
    return INSUFFICIENT_INFORMATION_MESSAGE.lower() in str(answer).lower()


def _looks_like_raw_context(answer: str) -> bool:
    """Detect answers that mostly copy raw product records."""
    answer = str(answer)
    raw_markers = ["main_category:", "average_rating:", "Item Form:", "Brand:"]
    return answer.count("|") >= 8 or sum(marker in answer for marker in raw_markers) >= 2


def evaluate_answer(
    question: str,
    answer: str,
    retrieved_context: str,
    error: str | None = None,
) -> dict[str, Any]:
    """
    Rule-based placeholder evaluation.

    Scores are intentionally simple and transparent so they can be replaced with
    model-based or human evaluation later.
    """
    if error or _contains_error(answer):
        return {
            "relevance_score": 1,
            "faithfulness_score": 1,
            "completeness_score": 1,
            "notes": error or "Pipeline component unavailable during evaluation.",
        }

    if _expected_refusal(question):
        if _refusal_detected(answer):
            return {
                "relevance_score": 5,
                "faithfulness_score": 5,
                "completeness_score": 5,
                "notes": "Correct refusal for an out-of-context question.",
            }
        return {
            "relevance_score": 1,
            "faithfulness_score": 2,
            "completeness_score": 1,
            "notes": "Expected refusal, but the system attempted to answer from insufficient context.",
        }

    if not retrieved_context.strip():
        return {
            "relevance_score": 1,
            "faithfulness_score": 3
            if _refusal_detected(answer)
            else 1,
            "completeness_score": 1,
            "notes": "No retrieved context was available.",
        }

    if _refusal_detected(answer):
        return {
            "relevance_score": 2,
            "faithfulness_score": 5,
            "completeness_score": 2,
            "notes": "The system refused to answer; this is faithful but may indicate weak retrieval for this in-context question.",
        }

    if _looks_like_raw_context(answer):
        return {
            "relevance_score": 2,
            "faithfulness_score": 3,
            "completeness_score": 2,
            "notes": "The answer appears to copy raw retrieved records instead of producing a concise response.",
        }

    question_terms = _tokens(question)
    answer_terms = _tokens(answer)
    context_terms = _tokens(retrieved_context)

    overlap_with_question = len(question_terms & answer_terms)
    answer_supported_terms = len(answer_terms & context_terms)
    support_ratio = answer_supported_terms / max(len(answer_terms), 1)

    relevance_score = min(5, max(2, 2 + overlap_with_question))
    faithfulness_score = min(5, max(1, round(1 + support_ratio * 4)))

    if len(answer.split()) >= 20:
        completeness_score = 5
    elif len(answer.split()) >= 10:
        completeness_score = 4
    else:
        completeness_score = 3

    notes = (
        "Rule-based scores based on question-answer overlap, answer-evidence "
        "overlap, answer formatting, and refusal behavior."
    )

    return {
        "relevance_score": relevance_score,
        "faithfulness_score": faithfulness_score,
        "completeness_score": completeness_score,
        "notes": notes,
    }


def _save_csv(rows: list[dict[str, Any]]) -> None:
    """Save evaluation rows, using pandas when available."""
    if pd is not None:
        pd.DataFrame(rows).to_csv(EVALUATION_CSV, index=False)
        return

    fieldnames = [
        "question",
        "answer",
        "retrieved_context",
        "relevance_score",
        "faithfulness_score",
        "completeness_score",
        "notes",
    ]
    with open(EVALUATION_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_evaluation(top_k: int = 3) -> Any:
    """
    Run the RAG pipeline over the test questions and save evaluation outputs.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for question in TEST_QUESTIONS:
        result = rag_answer(question, top_k=top_k)
        answer = result.get("answer", "")
        retrieved_docs = result.get("retrieved_docs", [])
        retrieved_context = summarize_retrieved_docs(retrieved_docs)

        scores = evaluate_answer(
            question=question,
            answer=answer,
            retrieved_context=retrieved_context,
            error=result.get("error"),
        )

        rows.append(
            {
                "question": question,
                "answer": answer,
                "retrieved_context": retrieved_context,
                "relevance_score": scores["relevance_score"],
                "faithfulness_score": scores["faithfulness_score"],
                "completeness_score": scores["completeness_score"],
                "notes": scores["notes"],
            }
        )

    _save_csv(rows)
    write_sample_outputs(rows)
    return pd.DataFrame(rows) if pd is not None else rows


def write_sample_outputs(rows: list[dict[str, Any]]) -> None:
    """Save readable qualitative examples to Markdown."""
    sections = [
        "# RAG Evaluation Sample Outputs",
        "",
        "## Report Notes",
        "",
        "### RAG Chain Implementation",
        REPORT_TEXT["rag_chain_implementation"],
        "",
        "### Evaluation Design",
        REPORT_TEXT["evaluation_design"],
        "",
        "### Evaluation Metrics",
        REPORT_TEXT["evaluation_metrics"],
        "",
        "---",
        "",
    ]

    for row in rows:
        sections.extend(
            [
                "## Question",
                str(row["question"]),
                "",
                "## Answer",
                str(row["answer"]),
                "",
                "## Retrieved Evidence",
                str(row["retrieved_context"]) if row["retrieved_context"] else "No context retrieved.",
                "",
                "## Scores",
                f"- Relevance: {row['relevance_score']}",
                f"- Faithfulness: {row['faithfulness_score']}",
                f"- Completeness: {row['completeness_score']}",
                f"- Notes: {row['notes']}",
                "",
                "---",
                "",
            ]
        )

    SAMPLE_OUTPUTS_MD.write_text("\n".join(sections), encoding="utf-8")


def main() -> None:
    """Command-line entry point."""
    results = run_evaluation(top_k=3)
    print(f"Saved {len(results)} rows to {EVALUATION_CSV}")
    print(f"Saved readable outputs to {SAMPLE_OUTPUTS_MD}")


if __name__ == "__main__":
    main()
