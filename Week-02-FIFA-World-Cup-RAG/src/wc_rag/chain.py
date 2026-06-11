"""
LangGraph 4-node RAG chain with confidence grading, hallucination checking, and conversation memory.

GRAPH DESIGN:

  retrieve ──► grade_relevance ──► answer ──► check_hallucination ──► END
                      │                              │
                      └──────────► refuse            └──► answer  (retry, max 2)

NODE SUMMARY:
  Node 1 — retrieve          Fetch + deduplicate chunks, extract Cohere relevance scores.
  Node 2 — grade_relevance   Gate: if max score < 0.30 → refuse, else → answer.
  Node 3 — answer            Nebius Llama-3.3-70B generates a grounded answer.
  Node 4 — check_hallucination  LLM verifies answer is supported by retrieved chunks.
                                If not grounded → regenerate (capped at MAX_RETRIES=2).

WHY FOUR NODES:
  1. OBSERVABILITY: Each node appears as a distinct span in LangSmith traces.
  2. TESTABILITY: grade_relevance() is pure — testable without LLM calls.
  3. FAITHFULNESS: check_hallucination catches cases where the LLM added facts
     not present in any retrieved chunk, triggering regeneration before the
     answer reaches the user.
  4. DESIGN PRINCIPLE: "Your 'I don't know' path matters more than your
     happy path." Making refusal a first-class node enforces this.

CONVERSATION MEMORY:
  chat_history stores the last N (human, ai) message pairs in the LangGraph
  state. It's passed to the LLM prompt via MessagesPlaceholder so follow-up
  questions like "And what about France?" resolve correctly.
  We keep only the last 4 turns to bound context window usage.
"""

import json
import re
from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from wc_rag.config import MIN_RELEVANCE_SCORE, NEBIUS_API_KEY, NEBIUS_BASE_URL, NEBIUS_CHAT_MODEL

MAX_HISTORY_TURNS = 4
MAX_RETRIES = 2
REFUSAL_MESSAGE = "I couldn't find this in the World Cup knowledge base."


# ---------------------------------------------------------------------------
# State — the data object that flows through every graph node
# ---------------------------------------------------------------------------

class WcRagState(TypedDict):
    question: str
    chat_history: list[tuple[str, str]]   # (human_msg, ai_msg) pairs
    documents: list[Document]
    relevance_scores: list[float]          # per-chunk scores from Cohere reranker
    routing_decision: str                  # "answer" or "refuse"
    answer: str
    citations: list[str]
    hallucination_result: str              # "grounded" or "not_grounded"
    retry_count: int                       # regeneration attempts so far


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a knowledgeable FIFA World Cup expert and passionate football analyst. "
    "Answer the user's question directly and conversationally, as if you know this information yourself. "
    "Be specific, factual, and engaging. "
    "Never say phrases like 'according to the context', 'as stated in the passages', "
    "'based on the provided information', or any variation — just answer naturally. "
    "IMPORTANT: Only report facts that are explicitly stated in the context. "
    "Do NOT calculate, infer, or derive numbers from other numbers — if a specific stat is not directly stated, "
    "report what IS stated and clearly acknowledge what is not available. "
    "For example, if the context says '21 goal contributions' but not the goals-only count, say that. "
    "Do NOT add a Sources or References section — sources are shown automatically."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)

HALLUCINATION_CHECK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a hallucination detector for a FIFA World Cup knowledge base.\n"
            "Determine whether an AI-generated answer is fully supported by the provided source documents.\n\n"
            "Score 'yes' if the answer ONLY contains information present in the documents.\n"
            "Score 'no' if the answer contains facts, numbers, or claims NOT found in the documents.\n\n"
            'Respond with JSON only. Format: {"score": "yes"} or {"score": "no"}\n'
            "No explanation, no extra text.",
        ),
        ("human", "Source documents:\n{context}\n\nGenerated answer:\n{answer}"),
    ]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_context(documents: list[Document]) -> str:
    blocks = []
    for doc in documents:
        title = doc.metadata.get("title", doc.metadata.get("source_url", "unknown"))
        blocks.append(f"[Source: {title}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def _collect_citations(documents: list[Document]) -> list[str]:
    seen: list[str] = []
    for doc in documents:
        label = doc.metadata.get("title") or doc.metadata.get("source_url", "unknown")
        if label not in seen:
            seen.append(label)
    return seen


def _history_to_messages(history: list[tuple[str, str]]) -> list:
    messages = []
    for human, ai in history[-MAX_HISTORY_TURNS:]:
        messages.append(HumanMessage(content=human))
        messages.append(AIMessage(content=ai))
    return messages


def _extract_scores(docs: list[Document]) -> list[float]:
    """
    Pull relevance scores stored in document metadata by the Cohere reranker.
    Falls back to 1.0 (assume relevant) if no score metadata is present —
    this happens when Cohere is disabled and the ensemble runs without reranking.
    """
    scores = []
    for doc in docs:
        score = doc.metadata.get("relevance_score")
        if score is None:
            score = doc.metadata.get("score", 1.0)
        scores.append(float(score))
    return scores


def _get_llm() -> ChatOpenAI:
    if not NEBIUS_API_KEY:
        raise RuntimeError("NEBIUS_API_KEY is missing. Add it to .env.")
    return ChatOpenAI(
        api_key=NEBIUS_API_KEY,
        base_url=NEBIUS_BASE_URL,
        model=NEBIUS_CHAT_MODEL,
        temperature=0,
    )


# ---------------------------------------------------------------------------
# Query preprocessing
# ---------------------------------------------------------------------------

_MONTH_NUMS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
_DATE_PATTERN = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2})(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)


def _expand_date_in_query(question: str) -> str:
    """
    Append ISO date format alongside any natural language date in the query.
    "Which teams play on June 11th?" → "... June 11th (2026-06-11)?"
    Schedule CSV rows store dates as 2026-MM-DD; this bridges the semantic gap
    so Pinecone returns the right rows without requiring re-ingestion.
    """
    def _to_iso(m: re.Match) -> str:
        month = _MONTH_NUMS.get(m.group(1).lower(), 0)
        day = int(m.group(2))
        if month:
            return f"{m.group(0)} (2026-{month:02d}-{day:02d})"
        return m.group(0)

    return _DATE_PATTERN.sub(_to_iso, question)


# ---------------------------------------------------------------------------
# Node 1: retrieve
# ---------------------------------------------------------------------------

def _deduplicate_docs(docs: list[Document]) -> list[Document]:
    """Remove duplicate and reference-only chunks, keeping first occurrence of each."""
    seen: set[str] = set()
    unique = []
    for doc in docs:
        text = doc.page_content.strip()
        # Skip Wikipedia footnote/citation blocks (start with "^" or are mostly URLs)
        if text.startswith("^") or text.count("http") > 3:
            continue
        if text not in seen:
            seen.add(text)
            unique.append(doc)
    return unique


def _make_retrieve_node(retriever: BaseRetriever):
    def retrieve(state: WcRagState) -> WcRagState:
        query = _expand_date_in_query(state["question"])
        docs = _deduplicate_docs(retriever.invoke(query))
        scores = _extract_scores(docs)
        return {
            **state,
            "documents": docs,
            "relevance_scores": scores,
            "citations": _collect_citations(docs),
        }
    return retrieve


# ---------------------------------------------------------------------------
# Node 2: grade_relevance — explicit confidence gate
# ---------------------------------------------------------------------------

def grade_relevance(state: WcRagState) -> WcRagState:
    """
    Decide whether retrieved context is good enough to send to the LLM.

    Decision rule:
      - If no documents were retrieved → refuse
      - If max relevance score < MIN_RELEVANCE_SCORE → refuse
      - Otherwise → answer

    The threshold (0.30) is defined in config.py.
    """
    docs = state["documents"]
    scores = state["relevance_scores"]

    if not docs:
        decision = "refuse"
    elif not scores or max(scores, default=0.0) < MIN_RELEVANCE_SCORE:
        decision = "refuse"
    else:
        decision = "answer"

    return {**state, "routing_decision": decision}


def _route_after_grading(state: WcRagState) -> Literal["answer", "refuse"]:
    return state["routing_decision"]


# ---------------------------------------------------------------------------
# Node 3a: answer
# ---------------------------------------------------------------------------

def _answer_node(state: WcRagState) -> WcRagState:
    chain = PROMPT | _get_llm()
    response = chain.invoke(
        {
            "question": state["question"],
            "context": _format_context(state["documents"]),
            "chat_history": _history_to_messages(state.get("chat_history", [])),
        }
    )
    return {**state, "answer": response.content}


# ---------------------------------------------------------------------------
# Node 3b: refuse
# ---------------------------------------------------------------------------

def _refuse_node(state: WcRagState) -> WcRagState:
    return {
        **state,
        "answer": REFUSAL_MESSAGE,
        "citations": [],
    }


# ---------------------------------------------------------------------------
# Node 4: check_hallucination — verify answer is grounded in retrieved chunks
# ---------------------------------------------------------------------------

def _check_hallucination_node(state: WcRagState) -> WcRagState:
    """
    Ask the LLM whether the generated answer is fully supported by the
    retrieved documents. If not grounded, flag for regeneration.

    Fail-open: if the checker errors or returns unparseable output, we
    accept the answer to avoid blocking the user on a checker failure.
    """
    try:
        chain = HALLUCINATION_CHECK_PROMPT | _get_llm()
        response = chain.invoke(
            {
                "context": _format_context(state["documents"]),
                "answer": state["answer"],
            }
        )
        result = json.loads(response.content.strip())
        grounded = result.get("score", "yes") == "yes"
    except Exception:
        grounded = True  # fail-open

    return {
        **state,
        "hallucination_result": "grounded" if grounded else "not_grounded",
        "retry_count": state.get("retry_count", 0) + 1,
    }


def _route_after_checking(state: WcRagState) -> Literal["answer", "end"]:
    """
    Conditional edge after hallucination check.
      grounded                          → end (return answer to user)
      not_grounded + retries remaining  → answer (regenerate)
      not_grounded + retries exhausted  → end (accept imperfect answer)
    """
    if state["hallucination_result"] == "grounded":
        return "end"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "end"
    return "answer"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_graph(retriever: BaseRetriever):
    """
    Compile the 4-node LangGraph RAG graph.

    Flow:
      retrieve → grade_relevance → [answer → check_hallucination → END]
                                 → [refuse → END]
    """
    graph = StateGraph(WcRagState)

    graph.add_node("retrieve", _make_retrieve_node(retriever))
    graph.add_node("grade_relevance", grade_relevance)
    graph.add_node("answer", _answer_node)
    graph.add_node("refuse", _refuse_node)
    graph.add_node("check_hallucination", _check_hallucination_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_relevance")
    graph.add_conditional_edges(
        "grade_relevance",
        _route_after_grading,
        {"answer": "answer", "refuse": "refuse"},
    )
    graph.add_edge("answer", "check_hallucination")
    graph.add_conditional_edges(
        "check_hallucination",
        _route_after_checking,
        {"answer": "answer", "end": END},
    )
    graph.add_edge("refuse", END)

    return graph.compile()


def ask(
    question: str,
    retriever: BaseRetriever,
    chat_history: list[tuple[str, str]] | None = None,
) -> WcRagState:
    """Run a single question through the full RAG graph."""
    graph = build_graph(retriever)
    return graph.invoke(
        {
            "question": question,
            "chat_history": chat_history or [],
            "documents": [],
            "relevance_scores": [],
            "routing_decision": "",
            "answer": "",
            "citations": [],
            "hallucination_result": "",
            "retry_count": 0,
        }
    )
