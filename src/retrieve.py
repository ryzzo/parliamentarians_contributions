"""
Query the ChromaDB collection for relevant Hansard chunks.

Usage (CLI):
    python retrieve.py "What was said about the finance bill?"
    python retrieve.py "housing policy" --top-k 10 --chunk-type speaker_turn

Usage (as a module):
    from retrieve import HansardRetriever
    r = HansardRetriever()
    results = r.query("affordable housing", top_k=5)
"""

import argparse
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL      = "all-MiniLM-L6-v2"
DEFAULT_DB_DIR     = "data/chroma"
DEFAULT_COLLECTION = "hansard"
DEFAULT_TOP_K      = 5

# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class HansardRetriever:
    def __init__(
        self,
        db_dir: str = DEFAULT_DB_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_MODEL,
    ):
        client = chromadb.PersistentClient(
            path=db_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = client.get_collection(collection_name)
        self.model = SentenceTransformer(model_name)

    def query(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        chunk_type: str | None = None,    # "full_topic" | "speaker_turn" | None
        where: dict | None = None,        # any extra ChromaDB metadata filter
    ) -> list[dict]:
        """Return top-k chunks most relevant to the question."""
        embedding = self.model.encode([question]).tolist()

        filters = dict(where) if where else {}
        if chunk_type:
            filters["chunk_type"] = chunk_type

        results = self.collection.query(
            query_embeddings=embedding,
            n_results=top_k,
            where=filters if filters else None,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text":       doc,
                "metadata":   meta,
                "score":      round(1 - dist, 4),   # cosine similarity
            })
        return hits

    def format_context(self, hits: list[dict]) -> str:
        """Format retrieved chunks as a prompt context block."""
        parts = []
        for i, h in enumerate(hits, 1):
            meta = h["metadata"]
            parts.append(
                f"[{i}] {meta.get('date', '')} | {meta.get('topic', '')} "
                f"(score: {h['score']})\n{h['text']}"
            )
        return "\n\n---\n\n".join(parts)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the Hansard ChromaDB collection.")
    parser.add_argument("question", help="Natural-language question to search for")
    parser.add_argument("--top-k",      type=int, default=DEFAULT_TOP_K, help="Number of results (default: 5)")
    parser.add_argument("--chunk-type", choices=["full_topic", "speaker_turn"],  help="Filter by chunk type")
    parser.add_argument("--db-dir",     default=DEFAULT_DB_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    args = parser.parse_args()

    retriever = HansardRetriever(
        db_dir=args.db_dir,
        collection_name=args.collection,
        model_name=args.model,
    )

    hits = retriever.query(args.question, top_k=args.top_k, chunk_type=args.chunk_type)

    if not hits:
        print("No results found.")
    else:
        print(f"\nTop {len(hits)} results for: \"{args.question}\"\n")
        print(retriever.format_context(hits))
