"""
Build and load the Pinecone vector store.

EMBEDDING ROUTING — WHY NEBIUS:
  The cohort mandates at least one model call through Nebius Token Factory.
  We satisfy this requirement at the embedding layer (which runs on every
  ingest AND every query), not just at generation.

  Nebius Token Factory is OpenAI-compatible. That means we use LangChain's
  standard OpenAIEmbeddings class but swap two parameters:
    api_key   → NEBIUS_API_KEY (instead of OPENAI_API_KEY)
    base_url  → https://api.studio.nebius.com/v1 (instead of api.openai.com)

  The model name "text-embedding-3-small" and the 1536-dim output are
  identical — Nebius proxies OpenAI's embedding model.

On first run, ingest.py calls build_vector_store() which:
  1. Creates the Pinecone index if it doesn't exist (serverless, AWS us-east-1).
  2. Embeds all chunks via Nebius → text-embedding-3-small.
  3. Upserts 1536-dim vectors + metadata into Pinecone.

Subsequent calls to load_vector_store() re-attach without re-embedding.
"""

import time

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

from wc_rag.config import (
    EMBEDDING_DIMENSION,
    NEBIUS_API_KEY,
    NEBIUS_BASE_URL,
    NEBIUS_EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)


def _get_embeddings() -> OpenAIEmbeddings:
    """
    Return an OpenAIEmbeddings instance pointed at Nebius Token Factory.
    LangChain's OpenAI wrapper works unchanged — only the URL and key differ.
    """
    if not NEBIUS_API_KEY:
        raise RuntimeError("NEBIUS_API_KEY is missing. Add it to .env.")
    return OpenAIEmbeddings(
        model=NEBIUS_EMBEDDING_MODEL,
        api_key=NEBIUS_API_KEY,
        base_url=NEBIUS_BASE_URL,
        check_embedding_ctx_length=False,  # Nebius doesn't accept pre-tokenized input
    )


def _get_pinecone_client() -> Pinecone:
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is missing. Add it to .env.")
    return Pinecone(api_key=PINECONE_API_KEY)


def ensure_index_exists() -> None:
    """Create the Pinecone index if it doesn't already exist."""
    pc = _get_pinecone_client()
    existing = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing:
        print(f"  Creating Pinecone index '{PINECONE_INDEX_NAME}' ...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait until the index is ready
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            print("  Waiting for index to be ready...")
            time.sleep(3)
        print(f"  Index '{PINECONE_INDEX_NAME}' created and ready.")
    else:
        print(f"  Pinecone index '{PINECONE_INDEX_NAME}' already exists.")


def build_vector_store(chunks: list[Document]) -> PineconeVectorStore:
    """Embed chunks and upsert into Pinecone. Returns the vector store handle."""
    ensure_index_exists()
    embeddings = _get_embeddings()

    print(f"  Embedding and upserting {len(chunks)} chunks into Pinecone ...")
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
    )
    print("  Done.")
    return vector_store


def load_vector_store() -> PineconeVectorStore:
    """Attach to an existing Pinecone index. Raises if the index doesn't exist."""
    pc = _get_pinecone_client()
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        raise RuntimeError(
            f"Pinecone index '{PINECONE_INDEX_NAME}' not found. "
            "Run `python scripts/ingest.py` first to build it."
        )
    embeddings = _get_embeddings()
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )


def index_stats() -> dict:
    """Return basic stats about the Pinecone index (vector count, etc.)."""
    pc = _get_pinecone_client()
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        return {"status": "not_created"}
    index = pc.Index(PINECONE_INDEX_NAME)
    return index.describe_index_stats()
