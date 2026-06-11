"""
Hybrid retrieval pipeline: EnsembleRetriever + Cohere reranker.

RETRIEVAL ARCHITECTURE (three-stage funnel):

  Stage 1 — Candidate generation (wide net)
  ┌─────────────────────────────────────────────────────────────┐
  │ Dense retriever (Pinecone cosine, k=10)                     │
  │   Encodes question as a 1536-dim vector via Nebius and      │
  │   finds the nearest chunks by cosine similarity.            │
  │   Strength: captures semantic intent and paraphrasing.      │
  │   Weakness: misses exact tokens ("1994", "Klose", "4-1").   │
  │                                                             │
  │ Sparse retriever (BM25, k=10)                               │
  │   Tokenises the question and scores chunks by term overlap. │
  │   Strength: exact matches on names, dates, scores.          │
  │   Weakness: misses synonyms and paraphrases.                │
  └─────────────────────────────────────────────────────────────┘
                          ↓ Reciprocal Rank Fusion (RRF)
  Stage 2 — Fusion (up to 20 merged candidates)
  ┌─────────────────────────────────────────────────────────────┐
  │ EnsembleRetriever merges both ranked lists using RRF.       │
  │ A chunk that ranks well in BOTH lists gets the highest      │
  │ combined score — rewarding both semantic and lexical match. │
  │ Weights: 60% dense, 40% BM25.                               │
  └─────────────────────────────────────────────────────────────┘
                          ↓ Cohere cross-encoder
  Stage 3 — Reranking (final top-5)
  ┌─────────────────────────────────────────────────────────────┐
  │ CohereRerank reads (question, chunk) pairs together.        │
  │ Unlike bi-encoders (embed separately), a cross-encoder sees │
  │ both at once — it can detect that "Who won in 2022?" and    │
  │ "Argentina beat France in the 2022 final" are highly        │
  │ relevant even if the embeddings don't align perfectly.      │
  │ This is the biggest faithfulness improvement in the stack.  │
  └─────────────────────────────────────────────────────────────┘
                          ↓ top-5 scored chunks to LangGraph
"""

from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from wc_rag.config import COHERE_API_KEY, RERANK_TOP_N, TOP_K

# Stage 1 asks for 2×TOP_K candidates from each leg so the reranker
# has a wider pool to work with before trimming to TOP_K final results.
_CANDIDATES_PER_LEG = TOP_K * 2


def _build_ensemble(
    vector_store: PineconeVectorStore,
    all_chunks: list[Document],
    dense_weight: float = 0.6,
    metadata_filter: dict | None = None,
) -> EnsembleRetriever:
    search_kwargs: dict = {"k": _CANDIDATES_PER_LEG}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    dense_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = _CANDIDATES_PER_LEG

    return EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[dense_weight, 1.0 - dense_weight],
    )


def build_hybrid_retriever(
    vector_store: PineconeVectorStore,
    all_chunks: list[Document],
    top_k: int = TOP_K,
    dense_weight: float = 0.6,
    metadata_filter: dict | None = None,
):
    """
    Full three-stage retriever: dense + BM25 fusion → Cohere reranker.

    If COHERE_API_KEY is not set, falls back to the two-stage ensemble
    without reranking so the app still works during development.
    """
    ensemble = _build_ensemble(vector_store, all_chunks, dense_weight, metadata_filter)

    if not COHERE_API_KEY:
        print("  WARNING: COHERE_API_KEY not set — reranker disabled, using ensemble only.")
        return ensemble

    reranker = CohereRerank(
        cohere_api_key=COHERE_API_KEY,
        model="rerank-english-v3.0",
        top_n=top_k,
    )
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble,
    )


def build_dense_retriever(
    vector_store: PineconeVectorStore,
    top_k: int = TOP_K,
    metadata_filter: dict | None = None,
):
    """Dense retriever with Cohere reranking when available, plain similarity otherwise."""
    search_kwargs: dict = {"k": _CANDIDATES_PER_LEG}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    base = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

    if not COHERE_API_KEY:
        base.search_kwargs["k"] = top_k
        return base

    reranker = CohereRerank(
        cohere_api_key=COHERE_API_KEY,
        model="rerank-english-v3.0",
        top_n=top_k,
    )
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base,
    )
