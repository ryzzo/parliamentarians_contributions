# Parliament Hansard — Docker workflow shortcuts

.PHONY: build up down pull parse embed ask shell logs

## Build the pipeline image
build:
	docker compose build

## Start Ollama (and pull llama3.1 on first run)
up:
	docker compose up -d ollama ollama-pull

## Stop all services
down:
	docker compose down

## Pull / re-pull the Llama model
pull:
	docker compose run --rm ollama-pull

## Parse PDFs in data/pdfs/ → data/chunks/
parse:
	docker compose run --rm pipeline \
		python src/parse.py --input-dir data/pdfs --output-dir data/chunks

## Embed data/chunks/ → data/chroma/
embed:
	docker compose run --rm pipeline \
		python src/embed.py --chunks-dir data/chunks --db-dir data/chroma

## Ask a question — usage: make ask Q="your question here"
ask:
	docker compose run --rm pipeline \
		python src/rag.py "$(Q)"

## Open a shell in the pipeline container
shell:
	docker compose run --rm pipeline bash

## Tail logs from all services
logs:
	docker compose logs -f
