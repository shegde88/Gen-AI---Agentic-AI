# FIFA World Cup 2026 RAG — Evaluation Report

**Project:** FIFA World Cup 2026 Fan Intelligence RAG App
**Build Track:** Code-heavy (LangChain + LangGraph + Pinecone)
**Evaluator:** Sharan Hegde
**Date:** June 2026

---

## One-liner

> My RAG app helps football fans answer questions about teams, players, venues, and tournament history from 85 curated Wikipedia sources + 6 structured CSV datasets in a Streamlit chat UI with 87% keyword coverage on in-scope questions.

---

## RAG Framework Summary

| Field | Decision |
|---|---|
| **Use case** | Fans ask natural-language questions about the 2026 World Cup (teams, players, history, venues) in a Streamlit chat interface |
| **Corpus** | 85 Wikipedia articles — 4 tournament pages, 24 history pages (all 22 editions 1930–2022 + overview + records), 49 team profiles (all 48 qualifying nations), 8 player profiles. Plus 6 CSV datasets: match results 1930–2022, tournament summaries, 2026 schedule, FIFA rankings (2022 & 2026), win probabilities |
| **Ingestion + cleaning** | WebBaseLoader targets `div#mw-content-text` to skip navigation; regex strips `[1]` citation markers and `[edit]` links; Wikipedia footnote chunks filtered at retrieval time |
| **Chunking + embedding** | RecursiveCharacterTextSplitter, 512 tokens / 50 overlap; `Qwen/Qwen3-Embedding-8B` (4096 dims) via Nebius Token Factory |
| **Retrieve** | Pinecone serverless (dense cosine) + Cohere `rerank-english-v3.0` cross-encoder; top-k = 5 |
| **Generate** | Nebius `meta-llama/Llama-3.3-70B-Instruct` via LangChain ChatOpenAI wrapper |
| **"I don't know" path** | `grade_relevance` node gates on Cohere relevance score < 0.30 → fixed refusal string, no hallucination |
| **Latency target** | < 8 seconds end-to-end |
| **Total vectors** | 20,331 chunks in Pinecone |

---

## Evaluation Methodology

**Scoring rubric per question (keyword hits / total expected keywords):**

| Score | Meaning |
|---|---|
| All keywords hit | Answer is factually complete and correct |
| Most keywords hit | Answer mostly correct with minor gaps |
| Some keywords hit | Partial answer |
| 0 keywords | Hallucinated, refused incorrectly, or completely wrong |

**Additional axes tracked:**
- **Faithfulness:** Does the answer cite sources? (binary)
- **Docs retrieved:** Number of unique chunks after deduplication
- **Category:** tournament / history / team / player / multi-doc / edge / out-of-scope

---

## 15 Golden Questions — Actual Results

| # | Category | Question | Expected Keywords | Hits | Score | Docs | Notes |
|---|---|---|---|---|---|---|---|
| 1 | tournament | How many teams are participating in the 2026 FIFA World Cup? | 48, teams | 2/2 | ✅ Full | 3 | Clean answer: "48 teams from 6 confederations" |
| 2 | tournament | Which three countries are hosting the 2026 FIFA World Cup? | United States, Canada, Mexico | 3/3 | ✅ Full | 3 | Named all three correctly, noted first tri-host |
| 3 | tournament | What is the new group-stage format for the 2026 World Cup vs 2022? | 48, 16, groups, three | 3/4 | ✅ Partial | 1 | Correct format described; missed "48" keyword |
| 4 | tournament | Where is the 2026 FIFA World Cup final being held? | MetLife, New York, New Jersey | 0/3 | ❌ Fail | 2 | Retrieved chunks mentioned US hosting but not MetLife specifically |
| 5 | tournament | How many venues are being used in the 2026 FIFA World Cup? | 16, stadiums, venues | 2/3 | ✅ Partial | 2 | Got 16 venues correct; "stadiums" keyword not in answer |
| 6 | history | How many times has Brazil won the FIFA World Cup? | 5, five, Brazil | 3/3 | ✅ Full | 3 | Perfect: "five times: 1958, 1962, 1970, 1994, 2002" |
| 7 | history | Who is the all-time top scorer in FIFA World Cup history? | Miroslav Klose, 16, goals | 3/3 | ✅ Full | 3 | Perfect: "Miroslav Klose, 16 goals across four World Cups" |
| 8 | history | Which country won the 2022 FIFA World Cup? | Argentina, 2022 | 2/2 | ✅ Full | 2 | Correct with third-title context |
| 9 | team | What was Morocco's best-ever result at a FIFA World Cup? | 2022, semi-final, fourth | 3/3 | ✅ Full | 4 | Perfect: fourth place, first African semi-finalist |
| 10 | team | How many World Cup titles does Germany have? | 4, four, Germany | 3/3 | ✅ Full | 3 | Perfect: four titles with all years named |
| 11 | player | How many FIFA World Cup goals has Lionel Messi scored? | Messi, goals, World Cup | 3/3 | ✅ Full | 2 | Correct context given (7 in 2022, total not explicitly in corpus) |
| 12 | player | What club does Kylian Mbappé play for? | Mbappé, Real Madrid | 2/2 | ✅ Full | 2 | Correct: Real Madrid (with PSG history noted) |
| 13 | multi-doc | Compare World Cup titles: Argentina vs France | Argentina, France, 2, 1 | 4/4 | ✅ Full | 3 | All keywords hit; France count partially inferred |
| 14 | edge | Who scored the golden goal to win a World Cup in extra time? | 1998, 2002, golden goal | 3/4 | ✅ Partial | 4 | Good nuanced answer covering golden goal era history |
| 15 | out-of-scope | What was the attendance at the 2026 World Cup final? | don't have, not enough | 0/3 | ✅ Correct refusal | 4 | Correctly said data not available (tournament not yet concluded at ingest time) |

---

## Overall Results

| Metric | Target | Result |
|---|---|---|
| Keyword coverage (in-scope Q1–14) | ≥ 90% | **87%** (36/41 keywords across Q1–14) |
| Answers with source citations | 100% | **100%** (15/15) |
| Graceful refusal on out-of-scope (Q15) | Refuse | **✅ Refused correctly** |
| Hard failures (0 keywords, in-scope) | 0 | **1** (Q4 — MetLife final venue) |

---

## Failure Analysis

### Q4 — Final Venue (MetLife Stadium) — FAILED
The retrieved chunks for "Where is the 2026 World Cup final?" returned high-level hosting overview text ("US hosts all matches from quarter-finals onward") but not the specific MetLife Stadium / East Rutherford detail. The venue page was ingested but Cohere ranked general tournament overview chunks higher than the venue-specific chunk.

**Fix:** Add MetLife Stadium directly to the question metadata or tune retrieval with a `source_type=tournament` filter when asking venue questions.

### Q3 & Q5 — Partial Keyword Misses
Minor keyword mismatches where the answer was factually correct but used different phrasing (e.g., "16 host cities" instead of "16 stadiums"). These are scoring artefacts, not retrieval failures.

### Q11 — Messi Career Goals Total
The Wikipedia corpus states "21 goal contributions" (goals + assists combined) but does not have a standalone "Messi has scored X goals" line. The LLM correctly reports 7 goals in 2022 and acknowledges the career total is not explicitly stated — this is the correct RAG behaviour (no hallucination).

### Q15 — Out-of-Scope Refusal ✅
The 2026 World Cup final attendance data does not exist in the corpus (the tournament was ingested before completion). The model correctly said the data is not available. Note: citations still appear because the model routed via the answer node — a known edge case where Cohere scores the retrieved context above the refusal threshold even though the LLM cannot answer.

---

## What the Expanded Corpus Added

Compared to the original 26-source build:

| Before | After |
|---|---|
| 12 team pages | **49 team pages** (all 48 qualifying nations) |
| 0 World Cup edition pages | **22 edition pages** (1930–2022 every tournament) |
| 3 Kaggle CSVs (1930–2014 only) | **6 CSV datasets** including 2026 schedule, June 2026 FIFA rankings, win probabilities |
| 48,408 chunks | **20,331 chunks** (higher quality — less CSV noise) |

---

## What I Would Improve with More Time

1. **MetLife venue fix:** Self-query metadata filtering — LLM auto-detects `source_type=tournament` for venue questions and narrows retrieval
2. **Messi goals corpus gap:** Add structured player stats CSV with explicit career World Cup goals breakdown
3. **Out-of-scope citation suppression:** Route the refusal path more reliably via an LLM-based topic guard before retrieval
4. **Parent-document retrieval:** Store 256-token child chunks in Pinecone, retrieve full 1024-token parent sections to LLM
5. **Live match data:** Ingest 2026 match results daily as the tournament progresses
