"""
Embed Kenya Hansard RAG chunks into a local ChromaDB collection.

Usage:
    # Embed all .jsonl files in a folder
    python embed.py --chunks-dir out/chunks/

    # Embed a single .jsonl file
    python embed.py --file out/chunks/hansard_2026_04_30.jsonl

    # Use a custom DB path or collection name
    python embed.py --chunks-dir out/chunks/ --db-dir data/chroma --collection hansard
"""

import json
import argparse
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL      = "all-MiniLM-L6-v2"   # fast, 384-dim, good for English
DEFAULT_DB_DIR     = "data/chroma"
DEFAULT_COLLECTION = "hansard"
BATCH_SIZE         = 128                   # chunks per embedding call

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def get_collection(db_dir: str, collection_name: str) -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=db_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def load_model(model_name: str) -> SentenceTransformer:
    print(f"Loading embedding model '{model_name}' ...")
    return SentenceTransformer(model_name)

# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def read_chunks(jsonl_path: Path) -> list[dict]:
    chunks = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def make_doc_id(jsonl_path: Path, index: int) -> str:
    return f"{jsonl_path.stem}__{index}"


def embed_file(
    jsonl_path: Path,
    collection: chromadb.Collection,
    model: SentenceTransformer,
) -> int:
    chunks = read_chunks(jsonl_path)
    if not chunks:
        print(f"  Skipping {jsonl_path.name} (empty)")
        return 0

    # De-duplicate against already-stored IDs
    all_ids = [make_doc_id(jsonl_path, i) for i in range(len(chunks))]
    existing = set(collection.get(ids=all_ids)["ids"])
    new_chunks = [(id_, c) for id_, c in zip(all_ids, chunks) if id_ not in existing]

    if not new_chunks:
        print(f"  {jsonl_path.name}: all {len(chunks)} chunks already embedded, skipping.")
        return 0

    print(f"  {jsonl_path.name}: embedding {len(new_chunks)} new chunks "
          f"({len(existing)} already stored) ...")

    for batch_start in range(0, len(new_chunks), BATCH_SIZE):
        batch = new_chunks[batch_start : batch_start + BATCH_SIZE]
        ids        = [id_ for id_, _ in batch]
        texts      = [c["text"] for _, c in batch]
        metadatas  = [c.get("metadata", {}) for _, c in batch]

        # Flatten any list values in metadata (ChromaDB only stores scalars)
        flat_metas = []
        for meta in metadatas:
            flat_metas.append({
                k: ", ".join(v) if isinstance(v, list) else (v if v is not None else "")
                for k, v in meta.items()
            })

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=flat_metas,
        )

    return len(new_chunks)


def embed_folder(
    chunks_dir: Path,
    collection: chromadb.Collection,
    model: SentenceTransformer,
) -> None:
    files = sorted(chunks_dir.glob("*.jsonl"))
    if not files:
        print(f"No .jsonl files found in {chunks_dir}")
        return

    total = 0
    failed = []
    for i, path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {path.name}")
        try:
            total += embed_file(path, collection, model)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failed.append(path.name)

    print(f"\nDone. {len(files) - len(failed)}/{len(files)} files processed, "
          f"{total} new chunks embedded.")
    print(f"Collection '{collection.name}' now has {collection.count()} total chunks.")
    if failed:
        print("Failed:", ", ".join(failed))

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed Hansard chunks into ChromaDB.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--chunks-dir", help="Folder of .jsonl chunk files")
    group.add_argument("--file",        help="Single .jsonl chunk file")

    parser.add_argument("--db-dir",     default=DEFAULT_DB_DIR,     help=f"ChromaDB storage path (default: {DEFAULT_DB_DIR})")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help=f"Collection name (default: {DEFAULT_COLLECTION})")
    parser.add_argument("--model",      default=DEFAULT_MODEL,      help=f"Sentence-transformers model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    Path(args.db_dir).mkdir(parents=True, exist_ok=True)

    model      = load_model(args.model)
    collection = get_collection(args.db_dir, args.collection)

    if args.file:
        path = Path(args.file)
        if not path.exists():
            raise FileNotFoundError(path)
        embed_file(path, collection, model)
    else:
        chunks_dir = Path(args.chunks_dir)
        if not chunks_dir.is_dir():
            raise NotADirectoryError(chunks_dir)
        embed_folder(chunks_dir, collection, model)
