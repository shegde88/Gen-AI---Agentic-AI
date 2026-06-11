"""
Multi-source ingestion pipeline: web + PDF + CSV → Pinecone.

WHAT THIS SCRIPT DOES (step by step):

  1. Web   — Fetches 26 Wikipedia articles via WebBaseLoader.
             Polite 1.5s crawl delay between requests.

  2. PDF   — Loads any .pdf files in data/raw/ via PyPDFLoader.
             Drop FIFA/ESPN match reports there before running.

  3. CSV   — Loads Kaggle World Cup CSVs from data/raw/.
             Download from: kaggle.com/datasets/abecklas/fifa-world-cup
             Files needed: WorldCupMatches.csv, WorldCups.csv, WorldCupPlayers.csv

  4. Chunk — RecursiveCharacterTextSplitter: 512 tokens / 50 overlap.

  5. Embed — text-embedding-3-small via Nebius Token Factory → 1536-dim vectors.

  6. Upsert — Vectors + metadata (team, player, year, stage) into Pinecone.

Re-running is safe (idempotent for the same content; upserts overwrite by vector ID).

Usage:
    cd "03. Fifa World Cup 2026 RAG Application"
    source .venv/bin/activate
    python scripts/ingest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wc_rag.ingestion import chunk_documents, load_all_documents
from wc_rag.indexing import build_vector_store


def main() -> None:
    print("=" * 60)
    print("FIFA World Cup 2026 RAG — Ingestion Pipeline")
    print("=" * 60)

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
