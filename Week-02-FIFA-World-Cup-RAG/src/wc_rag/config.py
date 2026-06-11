from pathlib import Path
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"

load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Nebius Token Factory — used for BOTH embeddings and LLM generation
# Nebius exposes an OpenAI-compatible API, so LangChain's OpenAI wrappers
# work as-is; we just point base_url at Nebius instead of api.openai.com.
# ---------------------------------------------------------------------------
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "")
NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1")

# Embedding — routed through Nebius Token Factory
NEBIUS_EMBEDDING_MODEL = os.getenv("NEBIUS_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")
EMBEDDING_DIMENSION = 4096  # Qwen3-Embedding-8B output dims

# LLM generation — also Nebius
NEBIUS_CHAT_MODEL = os.getenv("NEBIUS_CHAT_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

# ---------------------------------------------------------------------------
# Pinecone — vector store
# ---------------------------------------------------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "fifa-wc-2026")

# ---------------------------------------------------------------------------
# Cohere — reranker (cross-encoder applied after hybrid fusion)
# ---------------------------------------------------------------------------
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# ---------------------------------------------------------------------------
# RAG tuning constants
# 512-token chunks match text-embedding-3-small's effective signal range.
# Larger chunks (e.g. 800) cause the embedding to average over too much text,
# diluting the specific signal needed for precise fact retrieval.
# ---------------------------------------------------------------------------
TOP_K = 10                 # candidates returned by each retriever leg
RERANK_TOP_N = 5           # final docs kept after Cohere reranking
MIN_RELEVANCE_SCORE = 0.25 # below this → graceful refusal, no generation
CITATION_TOP_N = 3         # only the top-N ranked docs are candidates for citation
CITATION_MIN_SCORE = 0.40  # a candidate must also score above this to be cited
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
