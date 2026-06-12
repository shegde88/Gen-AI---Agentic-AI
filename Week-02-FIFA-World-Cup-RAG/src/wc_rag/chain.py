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

from wc_rag.config import CITATION_MIN_SCORE, CITATION_TOP_N, MIN_RELEVANCE_SCORE, NEBIUS_API_KEY, NEBIUS_BASE_URL, NEBIUS_CHAT_MODEL

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
    "WIN COUNTS: When stating how many times a team has won a competition, list the specific winning years "
    "you can name from the context first, then derive the count from those years. "
    "The total you state MUST exactly match the number of years you actually list. "
    "Never state a higher count than the years you can explicitly name. "
    "Example: 'France won in 1998 and 2018 — two titles.' NOT 'France won three times (1998, 2018, and another year).' "
    "WHEN CONTEXT IS INSUFFICIENT: If the retrieved context does not contain a clear answer, "
    "do NOT assert that the answer is 'none' or that the information 'does not exist'. "
    "Instead, acknowledge the gap and ask the user to refine their question — for example, "
    "suggest they mention a specific team name, add '2026 World Cup' to the query, or rephrase. "
    "Never confidently state a negative fact (e.g. 'No team has a base in X') unless the context explicitly says so. "
    "Exception: if the information genuinely does not yet exist because the event has not happened yet "
    "(e.g. attendance figures for a match not yet played, final scores of future matches, "
    "winners of rounds that haven't occurred), clearly state that this data is not yet available "
    "rather than asking the user to rephrase — the information simply cannot be looked up. "
    "MATCH TIMES: All 2026 World Cup match times are shown in the local time of the host city. "
    "Host city timezones: US Eastern venues (Boston/Foxborough, New York/East Rutherford, Philadelphia, Washington DC, Miami/Miami Gardens, Atlanta) and Toronto = UTC-4 (EDT). "
    "US Central venues (Dallas/Arlington, Houston, Kansas City) = UTC-5 (CDT). "
    "Mexico host cities (Mexico City/Estadio Azteca, Guadalajara/Zapopan/Estadio Akron, Monterrey/Guadalupe/Estadio BBVA) = UTC-6 (CST). "
    "US Pacific venues (Los Angeles/Inglewood/SoFi Stadium, Seattle/Lumen Field, San Francisco/Santa Clara/Levi's Stadium) and Vancouver = UTC-7 (PDT). "
    "When a user asks what time a match is, always state the local venue time with its timezone abbreviation "
    "and offer to convert to a different timezone if helpful. "
    "FOLLOW-UP TIMEZONE REQUESTS: If the user asks to convert a time you just provided to another timezone "
    "(e.g. 'show in PST', 'what time is that in UTC', 'convert to IST'), calculate directly from general knowledge — "
    "this is basic arithmetic, not a knowledge base lookup. "
    "When giving a timezone conversion, state only the result (e.g. '12:00 PM PDT'). "
    "Do NOT show UTC offsets, do NOT explain the arithmetic, do NOT say things like 'EDT is UTC-4 so …'. "
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
            "Pay special attention to NUMERIC CLAIMS. If the answer states a count (e.g. 'won 3 times', "
            "'scored 16 goals', 'four titles'), verify that exact number is explicitly stated in the documents "
            "OR is the direct count of items the documents list. "
            "If the answer states 'won 3 times' but the documents only show 2 winning years, score 'no'.\n\n"
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
    # Docs arrive sorted by Cohere score (highest first).
    # Only cite the top-N AND only when the score clears CITATION_MIN_SCORE.
    # Marginal docs (e.g. historical WC articles that share a place-name with
    # a 2026 query) can still feed context to the LLM without appearing as sources.
    seen: list[str] = []
    for doc in documents[:CITATION_TOP_N]:
        score = float(doc.metadata.get("relevance_score", doc.metadata.get("score", 0.0)))
        if score < CITATION_MIN_SCORE:
            continue
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
# Follow-up detection — bypass retrieval gate for short contextual questions
# ---------------------------------------------------------------------------

# Short phrases that almost always mean "continue from what you just said"
_FOLLOWUP_STARTERS = frozenset({
    "yes", "yep", "yup", "sure", "ok", "okay", "yeah",
})
# Timezone abbreviations — a question containing one of these is very likely
# asking for a conversion of a time mentioned in the previous turn
_TZ_TERMS = frozenset({
    "pst", "pdt", "mst", "mdt", "cst", "cdt", "est", "edt",
    "gmt", "utc", "ist", "jst", "cet", "bst", "aest", "aedt",
})
# Context-dependent pronouns — meaningful only if there's a prior exchange
_PRONOUN_REFS = frozenset({"this", "it", "that", "these", "those", "same"})


def _is_followup(question: str, chat_history: list) -> bool:
    """
    Return True when the question is almost certainly referring to the previous
    answer and needs no new Pinecone retrieval to be answered correctly.

    Heuristic: short (≤ 8 words) AND one of:
      • starts with a confirmatory word ("yes", "sure", "ok" …)
      • contains a timezone abbreviation ("PST", "UTC" …)
      • contains a bare pronoun reference ("this", "it", "that" …)
    """
    if not chat_history:
        return False
    words = question.strip().lower().split()
    if len(words) > 8:
        return False
    if words and words[0] in _FOLLOWUP_STARTERS:
        return True
    if any(w.rstrip("?.,!") in _TZ_TERMS for w in words):
        return True
    if any(w in _PRONOUN_REFS for w in words):
        return True
    return False


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
      - If no documents were retrieved → refuse (unless follow-up)
      - If max relevance score < MIN_RELEVANCE_SCORE → refuse (unless follow-up)
      - Otherwise → answer

    Follow-up bypass: short, context-dependent questions (timezone conversions,
    pronoun references like "show this in PST") are routed to the answer node
    with an empty document list so the LLM answers purely from chat_history.
    The hallucination check is skipped in that case (no docs to check against).
    """
    docs = state["documents"]
    scores = state["relevance_scores"]

    if not docs or not scores or max(scores, default=0.0) < MIN_RELEVANCE_SCORE:
        if _is_followup(state["question"], state.get("chat_history", [])):
            # Answer from chat_history; clear docs so hallucination check skips
            return {**state, "documents": [], "relevance_scores": [], "routing_decision": "answer"}
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

    Skip: when documents is empty the answer came from chat_history alone
    (a follow-up / timezone conversion). There's nothing to ground-check
    against, so we accept it directly.
    """
    if not state.get("documents"):
        return {
            **state,
            "hallucination_result": "grounded",
            "retry_count": state.get("retry_count", 0) + 1,
        }

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
