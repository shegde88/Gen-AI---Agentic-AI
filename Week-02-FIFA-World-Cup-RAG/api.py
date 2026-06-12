"""
FastAPI backend for the vibe-coded chatbot UI.

WHY FASTAPI AND NOT STREAMLIT:
  The bonus deliverable requires an HTML/JS front-end. Streamlit renders
  server-side Python — a browser can't talk to it via fetch(). FastAPI
  exposes a proper REST endpoint that the HTML page can call with fetch().

  The same LangGraph chain, Pinecone index, and retriever are reused —
  this is purely a transport layer, not a duplicate pipeline.

Run:
    uvicorn api:app --reload --port 8000
Then open frontend/index.html in your browser.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from wc_rag.chain import ask
from wc_rag.indexing import load_vector_store
from wc_rag.ingestion import chunk_documents, load_csv_documents
from wc_rag.retriever import build_hybrid_retriever

app = FastAPI(title="FIFA World Cup 2026 RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # open for local dev — restrict in production
    allow_methods=["POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy-load the retriever once at startup (not on every request)
# ---------------------------------------------------------------------------

_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        vs = load_vector_store()
        csv_chunks = chunk_documents(load_csv_documents())
        _retriever = build_hybrid_retriever(vs, csv_chunks)
    return _retriever


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    chat_history: list[list[str]] = []  # [[human, ai], ...]


class SourceDoc(BaseModel):
    title: str
    source_type: str
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[str]
    routing: str
    sources: list[SourceDoc]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    history = [tuple(pair) for pair in req.chat_history]
    result = ask(
        question=req.question,
        retriever=_get_retriever(),
        chat_history=history,
    )

    sources = []
    for doc in result.get("documents", []):
        sources.append(
            SourceDoc(
                title=doc.metadata.get("title", "unknown"),
                source_type=doc.metadata.get("source_type", ""),
                snippet=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            )
        )

    return ChatResponse(
        answer=result["answer"],
        citations=result["citations"],
        routing=result.get("routing_decision", "answer"),
        sources=sources,
    )
