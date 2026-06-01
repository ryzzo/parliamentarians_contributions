"""
Kenya Hansard RAG — retrieve chunks then synthesize with a local Llama model via Ollama.

Usage (CLI):
    python rag.py "What was debated about the housing bill?"
    python rag.py "Who spoke about education funding?" --top-k 10 --chunk-type speaker_turn
    python rag.py "What did members say about taxation?" --model llama3.2

Usage (as a module):
    from rag import HansardRAG
    rag = HansardRAG()
    answer = rag.ask("What did members say about taxation?")
    print(answer)
"""

import argparse

import os

import ollama

from retrieve import HansardRetriever, DEFAULT_DB_DIR, DEFAULT_COLLECTION, DEFAULT_MODEL

# Allow docker-compose to point at the ollama service; fall back to localhost
_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_ollama_client = ollama.Client(host=_OLLAMA_HOST)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_LLAMA_MODEL = "llama3.2"
DEFAULT_TOP_K       = 5

SYSTEM_PROMPT = """\
You are an expert analyst of Kenya's National Assembly Hansard debates.
Answer questions based solely on the provided parliamentary debate excerpts.
When answering:
- Reference specific speakers and dates when relevant.
- Quote directly from the excerpts to support your points.
- If the excerpts do not contain enough information to answer, say so clearly.
- Be concise and factual."""

# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------

class HansardRAG:
    def __init__(
        self,
        db_dir: str = DEFAULT_DB_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embed_model: str = DEFAULT_MODEL,
        llm_model: str = DEFAULT_LLAMA_MODEL,
    ):
        self.retriever = HansardRetriever(db_dir, collection_name, embed_model)
        self.llm_model = llm_model

    def ask(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        chunk_type: str | None = None,
        stream: bool = True,
    ) -> str:
        # 1. Retrieve relevant chunks
        hits = self.retriever.query(question, top_k=top_k, chunk_type=chunk_type)

        if not hits:
            return "No relevant parliamentary debates found for your question."

        # 2. Build prompt
        context_text = self.retriever.format_context(hits)
        user_message  = (
            f"PARLIAMENTARY DEBATE EXCERPTS:\n\n{context_text}"
            f"\n\nQuestion: {question}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]

        # 3. Call Llama via Ollama
        if stream:
            print()
            response = _ollama_client.chat(model=self.llm_model, messages=messages, stream=True)
            full_text = []
            for chunk in response:
                token = chunk["message"]["content"]
                print(token, end="", flush=True)
                full_text.append(token)
            print()
            return "".join(full_text)
        else:
            response = _ollama_client.chat(model=self.llm_model, messages=messages)
            return response["message"]["content"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ask a question about Kenya Hansard debates (powered by local Llama)."
    )
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--top-k",      type=int, default=DEFAULT_TOP_K,
                        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K})")
    parser.add_argument("--chunk-type", choices=["full_topic", "speaker_turn"],
                        help="Filter retrieved chunks by type")
    parser.add_argument("--model",      default=DEFAULT_LLAMA_MODEL,
                        help=f"Ollama model to use (default: {DEFAULT_LLAMA_MODEL})")
    parser.add_argument("--db-dir",     default=DEFAULT_DB_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--no-stream",  action="store_true",
                        help="Return full response at once instead of streaming")
    args = parser.parse_args()

    rag = HansardRAG(
        db_dir=args.db_dir,
        collection_name=args.collection,
        llm_model=args.model,
    )
    answer = rag.ask(
        args.question,
        top_k=args.top_k,
        chunk_type=args.chunk_type,
        stream=not args.no_stream,
    )
    if answer and args.no_stream:
        print(answer)
