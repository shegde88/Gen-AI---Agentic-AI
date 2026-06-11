"""
Multi-source ingestion pipeline: web + PDF + CSV → Pinecone.

WHAT THIS SCRIPT DOES (step by step):

  0. Clear  — Deletes all existing vectors in the Pinecone index so the
             re-ingest is clean (no stale or duplicate vectors).

  1. Web   — Fetches 97 Wikipedia articles via WebBaseLoader.
             Polite 1.5s crawl delay between requests.
             Sources: main 2026 WC article, all 12 group articles,
             venues, squads, 22 edition histories, 49 teams, 8 players.

  2. PDF   — Loads any .pdf files in data/raw/ via PyPDFLoader.

  3. CSV   — Loads World Cup CSVs from data/raw/.

  4. Chunk — RecursiveCharacterTextSplitter: 512 tokens / 50 overlap.

  5. Embed — Qwen3-Embedding-8B via Nebius Token Factory → 4096-dim vectors.

  6. Upsert — Vectors + metadata into Pinecone serverless index.

Usage:
    cd "03. Fifa World Cup 2026 RAG Application"
    source .venv/bin/activate
    python scripts/ingest.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pinecone import Pinecone
from wc_rag.config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from wc_rag.ingestion import chunk_documents, load_all_documents
from wc_rag.indexing import build_vector_store


def clear_index() -> None:
    """Delete all vectors in the existing index for a clean re-ingest."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        print(f"  Index '{PINECONE_INDEX_NAME}' does not exist yet — will be created.")
        return
    print(f"  Clearing all vectors from '{PINECONE_INDEX_NAME}' ...")
    index = pc.Index(PINECONE_INDEX_NAME)
    index.delete(delete_all=True)
    # Brief pause to let the delete propagate before upserting
    time.sleep(5)
    print("  Index cleared.")


def main() -> None:
    print("=" * 60)
    print("FIFA World Cup 2026 RAG — Ingestion Pipeline")
    print("=" * 60)

    print("\n[0/3] Clearing existing Pinecone index ...")
    clear_index()

    print("\n[1/3] Loading all documents (web + PDF + CSV) ...")
    documents = load_all_documents(web_delay=1.5)

    if not documents:
        print("\nERROR: No documents loaded. Check network and data/raw/.")
        sys.exit(1)

    print("\n[2/3] Chunking into 512-token passages ...")
    chunks = chunk_documents(documents)

    print("\n[3/3] Embedding via Nebius and upserting to Pinecone ...")
    build_vector_store(chunks)

    print("\n✓ Ingestion complete.")
    print(f"  Total documents:  {len(documents)}")
    print(f"  Chunks upserted:  {len(chunks)}")
    print("\nNext steps:")
    print("  Streamlit UI:  streamlit run app.py")
    print("  FastAPI UI:    uvicorn api:app --reload --port 8000")
    print("  Evaluation:    python -m wc_rag.evaluation")


if __name__ == "__main__":
    main()
