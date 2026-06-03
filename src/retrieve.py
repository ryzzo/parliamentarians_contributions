"""
Query the ChromaDB collection for relevant Hansard chunks.

Uses hybrid search — combines dense vector search (semantic) with BM25 keyword
search (exact names, bill numbers, MP names), then merges with Reciprocal Rank
Fusion so neither signal dominates.

Usage (CLI):
    python retrieve.py "What was said about the finance bill?"
    python retrieve.py "Hon. Kamau housing policy" --top-k 10
    python retrieve.py "Temporary Speaker" --chunk-type speaker_turn --alpha 0.3

Usage (as a module):
    from retrieve import HansardRetriever
    r = HansardRetriever()
    results = r.query("Hon. Kamau affordable housing", top_k=5)
"""

import argparse
import math
import re
from collections import defaultdict

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL      = "all-MiniLM-L6-v2"
DEFAULT_DB_DIR     = "data/chroma"
DEFAULT_COLLECTION = "hansard"
DEFAULT_TOP_K      = 5

# Reciprocal Rank Fusion constant — higher k reduces the impact of top ranks
_RRF_K = 60

# ---------------------------------------------------------------------------
# BM25 — lightweight in-memory implementation (no extra dependencies)
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


class BM25:
    """BM25 over a list of documents loaded at query time from ChromaDB."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.tokenised = [_tokenise(d) for d in docs]
        self.n = len(docs)
        avg_len = sum(len(t) for t in self.tokenised) / max(self.n, 1)
        self.avg_len = avg_len

        # term → document frequency
        df: dict[str, int] = defaultdict(int)
        for tokens in self.tokenised:
            for term in set(tokens):
                df[term] += 1
        self.df = df

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        tokens = self.tokenised[doc_idx]
        doc_len = len(tokens)
        tf_map: dict[str, int] = defaultdict(int)
        for t in tokens:
            tf_map[t] += 1

        score = 0.0
        for term in query_tokens:
            tf = tf_map.get(term, 0)
            idf = self.idf(term)
            numerator   = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_len)
            score += idf * (numerator / denominator)
        return score

    def rank(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return (doc_index, bm25_score) sorted descending."""
        q_tokens = _tokenise(query)
        scores = [(i, self.score(q_tokens, i)) for i in range(self.n)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _rrf_merge(
    dense_hits: list[dict],
    sparse_hits: list[dict],
    alpha: float,
    top_k: int,
) -> list[dict]:
    """
    Merge two ranked lists with RRF.
    alpha controls the blend: 1.0 = dense only, 0.0 = sparse only, 0.5 = equal.
    """
    scores: dict[str, float] = defaultdict(float)
    docs:   dict[str, dict]  = {}

    for rank, hit in enumerate(dense_hits):
        key = hit["text"][:120]          # stable identity key
        scores[key] += alpha * (1.0 / (_RRF_K + rank + 1))
        docs[key] = hit

    for rank, hit in enumerate(sparse_hits):
        key = hit["text"][:120]
        scores[key] += (1 - alpha) * (1.0 / (_RRF_K + rank + 1))
        if key not in docs:
            docs[key] = hit

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for key, rrf_score in ranked:
        hit = dict(docs[key])
        hit["score"] = round(rrf_score, 6)
        results.append(hit)
    return results


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
        self.collection = client.get_or_create_collection(collection_name)
        self.model = SentenceTransformer(model_name)

    def query(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        chunk_type: str | None = None,
        where: dict | None = None,
        alpha: float = 0.5,
    ) -> list[dict]:
        """
        Hybrid search: dense (semantic) + BM25 (keyword) merged with RRF.

        Parameters
        ----------
        question    : natural-language query (works for names, bill numbers, topics)
        top_k       : number of results to return
        chunk_type  : optional filter — "full_topic" | "speaker_turn"
        where       : additional ChromaDB metadata filters
        alpha       : blend weight — 1.0 = pure dense, 0.0 = pure BM25, 0.5 = equal
        """
        filters = dict(where) if where else {}
        if chunk_type:
            filters["chunk_type"] = chunk_type
        chroma_filter = filters if filters else None

        # ── 1. Dense retrieval (semantic) ────────────────────────────────────
        embedding = self.model.encode([question]).tolist()
        # Fetch a wider pool so BM25 re-ranking has enough candidates
        pool_size = min(top_k * 6, 100)

        dense_results = self.collection.query(
            query_embeddings=embedding,
            n_results=pool_size,
            where=chroma_filter,
            include=["documents", "metadatas", "distances"],
        )
        dense_hits = [
            {
                "text":     doc,
                "metadata": meta,
                "score":    round(1 - dist, 4),
            }
            for doc, meta, dist in zip(
                dense_results["documents"][0],
                dense_results["metadatas"][0],
                dense_results["distances"][0],
            )
        ]

        # ── 2. BM25 keyword retrieval over the same candidate pool ───────────
        bm25 = BM25([h["text"] for h in dense_hits])
        bm25_ranked = bm25.rank(question, top_k=pool_size)
        sparse_hits = [
            {**dense_hits[idx], "score": round(bm25_score, 4)}
            for idx, bm25_score in bm25_ranked
            if bm25_score > 0
        ]

        # ── 3. Merge with Reciprocal Rank Fusion ─────────────────────────────
        return _rrf_merge(dense_hits, sparse_hits, alpha=alpha, top_k=top_k)

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
    parser.add_argument("question",     help="Natural-language question or keyword search")
    parser.add_argument("--top-k",      type=int,   default=DEFAULT_TOP_K,
                        help="Number of results (default: 5)")
    parser.add_argument("--chunk-type", choices=["full_topic", "speaker_turn"],
                        help="Filter by chunk type")
    parser.add_argument("--alpha",      type=float, default=0.5,
                        help="Blend weight: 1.0=dense only, 0.0=BM25 only (default: 0.5)")
    parser.add_argument("--db-dir",     default=DEFAULT_DB_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    args = parser.parse_args()

    retriever = HansardRetriever(
        db_dir=args.db_dir,
        collection_name=args.collection,
        model_name=args.model,
    )

    hits = retriever.query(
        args.question,
        top_k=args.top_k,
        chunk_type=args.chunk_type,
        alpha=args.alpha,
    )

    if not hits:
        print("No results found.")
    else:
        print(f"\nTop {len(hits)} results for: \"{args.question}\"\n")
        print(retriever.format_context(hits))
