"""
FastAPI web server for the Kenya Hansard RAG.

Endpoints:
  GET  /            — serves the frontend SPA
  POST /ask         — non-streaming JSON response
  GET  /ask/stream  — streaming SSE response
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from rag import HansardRAG  # noqa: E402

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Kenya Hansard RAG")

_rag: HansardRAG | None = None

def get_rag() -> HansardRAG:
    global _rag
    if _rag is None:
        _rag = HansardRAG()
    return _rag


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    chunk_type: str | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text())


@app.post("/ask")
async def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    try:
        answer = await asyncio.to_thread(
            get_rag().ask,
            req.question,
            top_k=req.top_k,
            chunk_type=req.chunk_type,
            stream=False,
        )
        return {"answer": answer}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/ask/stream")
async def ask_stream(question: str, top_k: int = 5, chunk_type: str | None = None):
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    rag = get_rag()
    hits = await asyncio.to_thread(
        rag.retriever.query, question, top_k, chunk_type
    )
    if not hits:
        async def no_results():
            yield f"data: {json.dumps({'token': 'No relevant parliamentary debates found for your question.'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(no_results(), media_type="text/event-stream")

    context_text = rag.retriever.format_context(hits)
    from rag import SYSTEM_PROMPT, _ollama_client  # noqa: PLC0415
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"PARLIAMENTARY DEBATE EXCERPTS:\n\n{context_text}"
                f"\n\nQuestion: {question}"
            ),
        },
    ]

    async def token_stream():
        response = await asyncio.to_thread(
            _ollama_client.chat,
            model=rag.llm_model,
            messages=messages,
            stream=True,
        )
        for chunk in response:
            token = chunk["message"]["content"]
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Static files (after routes so / doesn't get shadowed)
# ---------------------------------------------------------------------------

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
