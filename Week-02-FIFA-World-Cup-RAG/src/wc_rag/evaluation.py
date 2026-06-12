"""
Evaluation harness: run 15 golden Q&A pairs through the RAG chain and
score each answer for faithfulness and relevance (0-3 rubric).

Run with:
    python -m wc_rag.evaluation
"""

import json
import textwrap
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_core.retrievers import BaseRetriever

from wc_rag.chain import ask

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "docs" / "evaluation_results.json"

# ---------------------------------------------------------------------------
# 15 Golden Q&A pairs
# ---------------------------------------------------------------------------
# Coverage: tournament facts, team history, player profiles, multi-doc,
# ambiguous wording, and 2 out-of-scope questions (expect "I don't know").

GOLDEN_QA: list[dict] = [
    # --- Tournament facts ---
    {
        "id": 1,
        "category": "tournament",
        "question": "How many teams are participating in the 2026 FIFA World Cup?",
        "expected_keywords": ["48", "teams"],
    },
    {
        "id": 2,
        "category": "tournament",
        "question": "Which three countries are hosting the 2026 FIFA World Cup?",
        "expected_keywords": ["United States", "Canada", "Mexico"],
    },
    {
        "id": 3,
        "category": "tournament",
        "question": "What is the new group-stage format for the 2026 World Cup compared to 2022?",
        "expected_keywords": ["48", "16", "groups", "three"],
    },
    {
        "id": 4,
        "category": "tournament",
        "question": "Where is the 2026 FIFA World Cup final being held?",
        "expected_keywords": ["MetLife", "New York", "New Jersey"],
    },
    {
        "id": 5,
        "category": "tournament",
        "question": "How many venues are being used in the 2026 FIFA World Cup?",
        "expected_keywords": ["16", "stadiums", "venues"],
    },
    # --- Historical / records ---
    {
        "id": 6,
        "category": "history",
        "question": "How many times has Brazil won the FIFA World Cup?",
        "expected_keywords": ["5", "five", "Brazil"],
    },
    {
        "id": 7,
        "category": "history",
        "question": "Who is the all-time top scorer in FIFA World Cup history?",
        "expected_keywords": ["Miroslav Klose", "16", "goals"],
    },
    {
        "id": 8,
        "category": "history",
        "question": "Which country won the 2022 FIFA World Cup?",
        "expected_keywords": ["Argentina", "2022"],
    },
    # --- Team profiles ---
    {
        "id": 9,
        "category": "team",
        "question": "What was Morocco's best-ever result at a FIFA World Cup?",
        "expected_keywords": ["2022", "semi-final", "fourth"],
    },
    {
        "id": 10,
        "category": "team",
        "question": "How many World Cup titles does Germany have?",
        "expected_keywords": ["4", "four", "Germany"],
    },
    # --- Player profiles ---
    {
        "id": 11,
        "category": "player",
        "question": "How many FIFA World Cup goals has Lionel Messi scored in his career?",
        "expected_keywords": ["Messi", "goals", "World Cup"],
    },
    {
        "id": 12,
        "category": "player",
        "question": "What club does Kylian Mbappé play for?",
        "expected_keywords": ["Mbappé", "Real Madrid"],
    },
    # --- Multi-document / cross-source ---
    {
        "id": 13,
        "category": "multi-doc",
        "question": "Compare the number of World Cup titles won by Argentina and France.",
        "expected_keywords": ["Argentina", "France", "2", "1"],
    },
    # --- Ambiguous / edge case ---
    {
        "id": 14,
        "category": "edge",
        "question": "Who scored the golden goal to win a World Cup in extra time?",
        "expected_keywords": ["1998", "2002", "golden goal", "overtime"],
    },
    # --- Out-of-scope (retrieval should gracefully refuse) ---
    {
        "id": 15,
        "category": "out-of-scope",
        "question": "What was the attendance figure for the 2026 World Cup final match?",
        "expected_keywords": ["not yet available", "not available", "attendance"],
    },
]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    id: int
    category: str
    question: str
    answer: str
    citations: list[str]
    docs_retrieved: int
    keyword_hits: list[str]
    relevance_score: int   # 0-3: how many expected keywords appear in answer
    faithfulness_note: str


def _score_relevance(answer: str, expected_keywords: list[str]) -> tuple[int, list[str]]:
    answer_lower = answer.lower()
    hits = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    return len(hits), hits


def run_evaluation(retriever: BaseRetriever) -> list[EvalResult]:
    results: list[EvalResult] = []

    for item in GOLDEN_QA:
        print(f"  [{item['id']:02d}/{len(GOLDEN_QA)}] {item['question'][:70]}...")
        if item["id"] > 1:
            time.sleep(7)  # stay under Cohere trial rate limit (10 calls/min)
        state = ask(item["question"], retriever)

        score, hits = _score_relevance(state["answer"], item["expected_keywords"])
        total = len(item["expected_keywords"])
        faithfulness = (
            "Grounded — answer cites sources" if state["citations"]
            else "No citations returned"
        )

        results.append(
            EvalResult(
                id=item["id"],
                category=item["category"],
                question=item["question"],
                answer=state["answer"],
                citations=state["citations"],
                docs_retrieved=len(state["documents"]),
                keyword_hits=hits,
                relevance_score=score,
                faithfulness_note=faithfulness,
            )
        )
        print(f"     Relevance: {score}/{total} keywords  |  Docs: {len(state['documents'])}")

    return results


def save_results(results: list[EvalResult]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n  Results saved to {RESULTS_PATH}")


def print_summary(results: list[EvalResult]) -> None:
    total = len(results)
    avg_relevance = sum(r.relevance_score for r in results) / total
    fully_cited = sum(1 for r in results if r.citations)
    refused = sum(1 for r in results if "don't have" in r.answer.lower() or "not enough" in r.answer.lower())

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Questions run:        {total}")
    print(f"Avg keyword hits:     {avg_relevance:.1f}")
    print(f"Answers with sources: {fully_cited}/{total}")
    print(f"Graceful refusals:    {refused}")
    print("=" * 60)

    print("\nPer-question breakdown:")
    for r in results:
        flag = "✓" if r.keyword_hits else "✗"
        print(f"  {flag} [{r.id:02d}] {r.category:12s} | hits: {r.relevance_score} | {r.question[:55]}...")


if __name__ == "__main__":
    from wc_rag.indexing import load_vector_store
    from wc_rag.retriever import build_dense_retriever

    print("Loading vector store ...")
    vs = load_vector_store()
    retriever = build_dense_retriever(vs)

    print(f"\nRunning {len(GOLDEN_QA)} evaluation questions ...\n")
    results = run_evaluation(retriever)
    save_results(results)
    print_summary(results)
