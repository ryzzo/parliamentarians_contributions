# Parliament Hansard pipeline — parse, embed, and query
FROM python:3.12-slim

# System deps for pdfplumber (needs poppler for PDF rendering)
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformers model so the image is self-contained
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy source
COPY src/ ./src/

# Bind-mount points (declared as volumes in compose):
#   /app/data/pdfs      — input PDFs
#   /app/data/chunks    — parsed .jsonl output
#   /app/data/chroma    — ChromaDB vector store

CMD ["python", "src/parse.py", "--help"]
