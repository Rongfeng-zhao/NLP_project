import argparse
import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


class ProductRetriever:
    """
    Semantic retriever for product RAG system.

    It loads:
    1. FAISS index
    2. metadata.jsonl
    3. embedding model

    Then it converts a query into embedding and retrieves top-k similar product records.
    """

    def __init__(self, index_dir: str = "artifacts/vector_index"):
        self.index_dir = Path(index_dir)

        config_path = self.index_dir / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}. "
                "Please run src/build_vector_store_streaming.py first."
            )

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.model_name = self.config["embedding_model"]
        self.index_path = self.index_dir / self.config["index_file"]
        self.metadata_path = self.index_dir / self.config["metadata_file"]

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        print(f"[INFO] Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        print(f"[INFO] Loading FAISS index from: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

        print(f"[INFO] Loading metadata from: {self.metadata_path}")
        self.metadata = self._load_metadata(self.metadata_path)

        print("[INFO] Retriever loaded successfully.")
        print(f"[INFO] Number of indexed documents: {len(self.metadata)}")

    def _load_metadata(self, metadata_path: Path):
        metadata = []

        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                metadata.append(json.loads(line))

        return metadata

    def retrieve(self, query: str, top_k: int = 5):
        if query is None or query.strip() == "":
            raise ValueError("Query cannot be empty.")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for rank, idx in enumerate(indices[0]):
            if idx == -1:
                continue

            metadata_item = self.metadata[int(idx)]

            results.append(
                {
                    "rank": rank + 1,
                    "score": float(scores[0][rank]),
                    "id": metadata_item["id"],
                    "text": metadata_item["text"],
                }
            )

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Test product semantic retriever."
    )

    parser.add_argument(
        "--index_dir",
        type=str,
        default="artifacts/vector_index",
        help="Directory of FAISS index and metadata.",
    )

    parser.add_argument(
        "--query",
        type=str,
        default="waterproof eyebrow products with natural finish",
        help="Search query.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of retrieval results.",
    )

    args = parser.parse_args()

    retriever = ProductRetriever(index_dir=args.index_dir)

    results = retriever.retrieve(
        query=args.query,
        top_k=args.top_k,
    )

    print("\n========== Retrieval Results ==========\n")

    for item in results:
        print(f"Rank: {item['rank']}")
        print(f"Score: {item['score']:.4f}")
        print(f"ID: {item['id']}")
        print(f"Text: {item['text']}")
        print("-" * 100)


if __name__ == "__main__":
    main()
