# 1. Parse PDFs → chunks
python src/parse.py --input-dir pdfs/ --output-dir out/chunks/

# 2. Embed chunks → ChromaDB
python src/embed.py --chunks-dir out/chunks/ --db-dir data/chroma

# 3. Query
python src/retrieve.py "What was said about the finance bill?"
python src/retrieve.py "housing policy" --top-k 10 --chunk-type speaker_turn

docker compose run --rm pipeline python src/parse.py --input-dir data/pdfs -o data/chunks 

# 1. Install Ollama: https://ollama.com
# 2. Pull the model (one-time download ~4.7GB)
ollama pull llama3.1

# 3. Install the Python client
pip install ollama


dcoker compose build

# Test if it works
docker compose run --rm pipeline python src/parse.py --help

docker compose run --rm pipeline python src/parse.py --input-dir data/pdfs --output-dir data/chunks
docker compose run --rm pipeline python src/embed.py --chunks-dir data/chunks --db-dir data/chroma
docker compose run --rm pipeline python src/rag.py "Who spoke about education funding?"