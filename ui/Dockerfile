FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so the image is self-contained
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy source code
COPY src/ ./src/
COPY ui/  ./ui/

EXPOSE 8000

CMD ["uvicorn", "ui.app:app", "--host", "0.0.0.0", "--port", "8000"]
