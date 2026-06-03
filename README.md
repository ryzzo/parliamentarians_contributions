# Kenya Hansard RAG

A local, fully offline pipeline for parsing Kenya's National Assembly Hansard PDFs into searchable chunks, embedding them into a vector store, and querying them with a local LLM.

**Stack:** pdfplumber · sentence-transformers · ChromaDB · BM25 hybrid search · Ollama (Gemma 2) · FastAPI

---

## How it works

```
PDFs  ──parse──►  JSONL chunks  ──embed──►  ChromaDB
                                                │
                                      hybrid search (dense + BM25)
                                                │
                                           Gemma 2 (via Ollama)
                                                │
                                          FastAPI chat UI
```

1. **Parse** — extracts speaker turns and full topic blocks from Hansard PDFs
2. **Embed** — encodes chunks with `all-MiniLM-L6-v2` and stores them in ChromaDB
3. **Retrieve** — hybrid search combines semantic (dense) + keyword (BM25) results via Reciprocal Rank Fusion
4. **Answer** — retrieved context is fed to a local Gemma 2 model running in Ollama

---

## Project structure

```
parliament/
├── src/
│   ├── parse.py       # PDF → JSONL chunks
│   ├── embed.py       # JSONL → ChromaDB
│   ├── retrieve.py    # hybrid search (dense + BM25)
│   └── rag.py         # retriever + LLM synthesis
├── ui/
│   ├── Dockerfile     # FastAPI chat frontend image
│   └── ...
├── data/
│   ├── pdfs/          # ← drop your Hansard PDFs here
│   ├── chunks/        # parsed .jsonl output (auto-created)
│   └── chroma/        # ChromaDB vector store (auto-created)
├── Dockerfile         # pipeline image
├── docker-compose.yml
├── Makefile           # workflow shortcuts
└── requirements.txt
```

---

## Quickstart (Docker — recommended)

### 1. Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### 2. Build the pipeline image

```bash
docker compose build
```

First build takes ~5–10 minutes — it downloads Python packages and the embedding model.

### 3. Start Ollama and pull Gemma 2

```bash
docker compose up -d ollama ollama-pull
```

Watch the model download (~1.7 GB, one-time):

```bash
docker compose logs -f ollama-pull
```

### 4. Add your PDFs

Drop Hansard PDF files into `data/pdfs/`.

### 5. Run the full pipeline

```bash
# Parse PDFs → chunks
docker compose run --rm ingest

# Or step by step:
docker compose run --rm pipeline python src/parse.py --input-dir data/pdfs --output-dir data/chunks
docker compose run --rm pipeline python src/embed.py --chunks-dir data/chunks --db-dir data/chroma
```

### 6. Ask questions

```bash
docker compose run --rm pipeline python src/rag.py "What was debated about the housing bill?"
```

### 7. Start the chat UI

```bash
docker compose up ui ollama
```

Open [http://localhost:8000](http://localhost:8000)

---

## Quickstart (local — no Docker)

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com) installed and running

```bash
# Install dependencies
pip install -r requirements.txt

# Pull Gemma 2
ollama pull gemma2:2b

# Parse → embed → query
python src/parse.py --input-dir data/pdfs --output-dir data/chunks
python src/embed.py --chunks-dir data/chunks --db-dir data/chroma
python src/rag.py "What was said about the finance bill?"
```

---

## Makefile shortcuts

| Command | What it does |
|---|---|
| `make build` | Build the pipeline Docker image |
| `make up` | Start Ollama and pull the model |
| `make parse` | Parse all PDFs in `data/pdfs/` |
| `make embed` | Embed chunks into ChromaDB |
| `make ask Q="..."` | Ask a question |
| `make shell` | Open a shell in the pipeline container |
| `make logs` | Tail logs from all services |
| `make down` | Stop all services |

---

## Hybrid search

Retrieval uses **dense + BM25** merged with Reciprocal Rank Fusion — this handles both semantic queries and exact keyword lookups like MP names and bill numbers.

| Query type | Best `--alpha` | Example |
|---|---|---|
| MP name / bill number | `0.2` | `"Hon. Otieno"`, `"Finance Bill 2023"` |
| Balanced (default) | `0.5` | `"Kamau housing policy"` |
| Topic / semantic | `0.8` | `"what was said about drought relief"` |

```bash
# Lean on keyword matching for exact names
python src/retrieve.py "Hon. Kamau" --alpha 0.2

# Lean on semantic search for topics
python src/retrieve.py "affordable housing" --alpha 0.8
```

---

## Chunk types

Each Hansard topic produces two chunk types:

| Type | Description |
|---|---|
| `full_topic` | Entire topic conversation as one block |
| `speaker_turn` | Individual utterance with speaker metadata |

Filter by type:

```bash
python src/rag.py "education funding" --chunk-type speaker_turn
python src/retrieve.py "budget" --chunk-type full_topic
```

---

## Switching the LLM

The model is controlled by the `LLAMA_MODEL` environment variable (default: `gemma2:2b`).

```bash
# Use a larger model
LLAMA_MODEL=llama3.1 docker compose run --rm pipeline python src/rag.py "your question"

# Or set it permanently in a .env file
echo "LLAMA_MODEL=llama3.1" > .env
```

Available models: `gemma2:2b`, `llama3.1`, `llama3.2`, `mistral`, `phi4` — see [ollama.com/library](https://ollama.com/library).
