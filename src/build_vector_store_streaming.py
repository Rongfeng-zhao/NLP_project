import argparse
import json
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def detect_text_column(parquet_file: pq.ParquetFile, user_text_column: Optional[str] = None) -> str:
    """
    Detect the text column from parquet schema.
    If user provides --text_column, use it directly.
    Otherwise, try common column names.
    If the parquet has only one column, use that column.
    """

    schema = parquet_file.schema_arrow
    columns = schema.names

    print(f"[INFO] Parquet columns: {columns}")

    if user_text_column is not None:
        if user_text_column not in columns:
            raise ValueError(
                f"Provided text column '{user_text_column}' not found in parquet columns: {columns}"
            )
        print(f"[INFO] Using user-provided text column: {user_text_column}")
        return user_text_column

    possible_columns = [
        "text",
        "document",
        "document_text",
        "product_text",
        "content",
        "combined_text",
        "combined",
        "product_info",
    ]

    for col in possible_columns:
        if col in columns:
            print(f"[INFO] Detected text column: {col}")
            return col

    if len(columns) == 1:
        print(f"[INFO] Only one column found. Using it as text column: {columns[0]}")
        return columns[0]

    raise ValueError(
        "Could not automatically detect text column. "
        "Please run the column check command and pass --text_column manually."
    )


def clean_texts(raw_texts: List) -> List[str]:
    """
    Clean a batch of raw text values.
    This function does not do heavy NLP preprocessing because product records are already formatted.
    It only removes empty or invalid rows.
    """

    cleaned = []

    for text in raw_texts:
        if text is None:
            continue

        text = str(text).strip()

        if text == "":
            continue

        cleaned.append(text)

    return cleaned


def create_faiss_index(vector_dimension: int):
    """
    Create FAISS index.
    We normalize embeddings, so inner product is equivalent to cosine similarity.
    """

    index = faiss.IndexFlatIP(vector_dimension)
    return index


def save_config(
    output_dir: Path,
    model_name: str,
    text_column: str,
    total_documents: int,
    batch_size: int,
):
    config = {
        "embedding_model": model_name,
        "text_column": text_column,
        "index_file": "products.faiss",
        "metadata_file": "metadata.jsonl",
        "num_documents": total_documents,
        "batch_size": batch_size,
        "similarity": "cosine_similarity_with_normalized_embeddings",
    }

    config_path = output_dir / "config.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Config saved to: {config_path}")


def build_vector_store_streaming(
    parquet_path: str,
    output_dir: str,
    model_name: str,
    text_column: Optional[str],
    parquet_batch_size: int,
    embedding_batch_size: int,
    max_rows: int,
):
    """
    Stream parquet data batch by batch.
    For each batch:
    1. Load only one batch into memory.
    2. Extract product text.
    3. Generate embeddings.
    4. Add embeddings to FAISS index.
    5. Append metadata to metadata.jsonl.
    """

    parquet_path = Path(parquet_path)
    output_dir = Path(output_dir)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "products.faiss"
    metadata_path = output_dir / "metadata.jsonl"

    if metadata_path.exists():
        metadata_path.unlink()

    print(f"[INFO] Loading parquet file in streaming mode: {parquet_path}")

    parquet_file = pq.ParquetFile(parquet_path)
    detected_text_column = detect_text_column(parquet_file, text_column)

    print(f"[INFO] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    faiss_index = None
    total_documents = 0
    batch_id = 0

    print("[INFO] Start streaming parquet batches...")

    with open(metadata_path, "a", encoding="utf-8") as metadata_file:
        for record_batch in tqdm(
            parquet_file.iter_batches(
                batch_size=parquet_batch_size,
                columns=[detected_text_column],
            ),
            desc="Streaming parquet batches",
        ):
            batch_id += 1

            batch_dict = record_batch.to_pydict()
            raw_texts = batch_dict[detected_text_column]
            texts = clean_texts(raw_texts)

            if len(texts) == 0:
                continue

            if max_rows > 0:
                remaining = max_rows - total_documents
                if remaining <= 0:
                    break
                texts = texts[:remaining]

            embeddings = model.encode(
                texts,
                batch_size=embedding_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")

            if faiss_index is None:
                vector_dimension = embeddings.shape[1]
                faiss_index = create_faiss_index(vector_dimension)
                print(f"[INFO] Created FAISS index with dimension: {vector_dimension}")

            faiss_index.add(embeddings)

            for text in texts:
                record = {
                    "id": total_documents,
                    "text": text,
                }
                metadata_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_documents += 1

            if batch_id % 10 == 0:
                print(
                    f"[INFO] Processed batches: {batch_id}, "
                    f"documents indexed: {total_documents}"
                )

            if max_rows > 0 and total_documents >= max_rows:
                break

    if faiss_index is None or total_documents == 0:
        raise RuntimeError("No valid documents were indexed. Please check the parquet file.")

    print(f"[INFO] Total indexed documents: {total_documents}")
    print(f"[INFO] Saving FAISS index to: {index_path}")

    faiss.write_index(faiss_index, str(index_path))

    save_config(
        output_dir=output_dir,
        model_name=model_name,
        text_column=detected_text_column,
        total_documents=total_documents,
        batch_size=parquet_batch_size,
    )

    print("[INFO] Streaming vector store building completed successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS vector store with streaming parquet loading."
    )
    
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    parser.add_argument(
        "--parquet_path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "sample_products.parquet"),
        help="Path to the parquet file.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/vector_index",
        help="Directory to save FAISS index and metadata.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name.",
    )

    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="Text column name. If not provided, the script will try to detect it.",
    )

    parser.add_argument(
        "--parquet_batch_size",
        type=int,
        default=2048,
        help="Number of parquet rows loaded per streaming batch.",
    )

    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=64,
        help="Batch size for embedding model.",
    )

    parser.add_argument(
        "--max_rows",
        type=int,
        default=20000,
        help=(
            "Maximum number of rows to process. "
            "Default is 20000 for the prototype. "
            "Use 5000 for quick testing or 50000 for a larger prototype. "
            "Use -1 to process all rows, but this is not recommended for local machines."
        ),
    )

    args = parser.parse_args()

    build_vector_store_streaming(
        parquet_path=args.parquet_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        text_column=args.text_column,
        parquet_batch_size=args.parquet_batch_size,
        embedding_batch_size=args.embedding_batch_size,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
