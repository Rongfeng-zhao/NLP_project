"""
Simple Streamlit frontend for the RAG demo.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from src.rag_chain import rag_answer, summarize_retrieved_docs


load_dotenv(dotenv_path=Path(".env"), override=True)


def _doc_text(doc: Any) -> str:
    if isinstance(doc, dict):
        return str(doc.get("text") or doc.get("content") or doc.get("page_content") or doc)
    return str(doc)


def _doc_title(doc: Any, index: int) -> str:
    if isinstance(doc, dict):
        source = doc.get("source") or doc.get("id") or index
        score = doc.get("score")
        if score is not None:
            try:
                return f"Document {index} | Source {source} | Score {float(score):.4f}"
            except (TypeError, ValueError):
                return f"Document {index} | Source {source} | Score {score}"
        return f"Document {index} | Source {source}"
    return f"Document {index}"


st.set_page_config(page_title="RAG Product QA Demo", layout="wide")

st.title("RAG Product QA Demo")
st.caption("Retrieval-grounded question answering with optional DeepSeek generation.")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-k retrieved documents", min_value=1, max_value=10, value=3)
    use_deepseek = st.toggle("Use DeepSeek generation", value=True)
    st.markdown("DeepSeek requires `DEEPSEEK_API_KEY` in your environment or `.env` file.")

default_question = "What waterproof eyebrow products have a natural finish?"
question = st.text_area("Question", value=default_question, height=90)

if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            result = rag_answer(question.strip(), top_k=top_k, use_deepseek=use_deepseek)

        st.subheader("Answer")
        st.write(result.get("answer", "No answer generated."))

        generation_mode = result.get("generation_mode", "unknown")
        st.info(f"Generation mode: `{generation_mode}`")
        if result.get("llm_status"):
            st.caption(result["llm_status"])
        if result.get("error"):
            st.error(result["error"])

        retrieved_docs = result.get("retrieved_docs", [])
        st.subheader("Retrieved Evidence")
        evidence_summary = summarize_retrieved_docs(retrieved_docs, max_docs=top_k)
        if evidence_summary:
            st.markdown(evidence_summary)
        else:
            st.write("No retrieved evidence available.")

        for idx, doc in enumerate(retrieved_docs, start=1):
            with st.expander(_doc_title(doc, idx)):
                st.text(_doc_text(doc))

        with st.expander("Prompt"):
            st.code(result.get("prompt", ""), language="text")
else:
    st.write("Enter a product question and click Ask.")

st.divider()
st.caption(f"Project folder: {Path.cwd()}")
