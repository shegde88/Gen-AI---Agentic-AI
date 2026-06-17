from __future__ import annotations

import datetime
import os
import time
from typing import Any, Optional
from typing_extensions import TypedDict

import pandas as pd

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt


# ── State ─────────────────────────────────────────────────────────────────────

class TransferBrief(TypedDict):
    """Structured scouting brief entered by the user."""
    position: str           # e.g. "CM", "ST"
    max_age: int
    min_overall: int
    target_league: Optional[str]
    budget_eur: Optional[float]
    notes: str              # free-text requirements


class TransferScoutState(TypedDict):
    """Full graph state passed between nodes."""
    brief: TransferBrief
    longlist: list[dict[str, Any]]       # raw player rows from the dataset
    scored_players: list[dict[str, Any]] # players ranked with fit scores
    news_data: dict[str, Any]            # short_name -> list of {title, url, content} dicts
    draft_report: str                    # markdown transfer report
    approved: bool                       # set by human_review node


# ── Data helper ────────────────────────────────────────────────────────────────

def _load_df() -> pd.DataFrame:
    """Load the FIFA 23 snapshot without requiring a Streamlit session."""
    from data_loader import (
        _PARQUET_FIFA23, _post_process, TOP5_LEAGUES,
        DATA_PATH, TARGET_SNAPSHOT, _build_fifa23_parquet,
    )

    if os.path.exists(_PARQUET_FIFA23):
        df = pd.read_parquet(_PARQUET_FIFA23)
    else:
        df = _build_fifa23_parquet(DATA_PATH, TARGET_SNAPSHOT)

    if "league_name" in df.columns:
        df = df[df["league_name"].isin(TOP5_LEAGUES)]

    return _post_process(df)


# ── Node stubs ────────────────────────────────────────────────────────────────

def parse_brief(state: TransferScoutState) -> dict:
    """Validate and normalise the incoming TransferBrief."""
    brief = state["brief"]

    if not brief.get("position", "").strip():
        raise ValueError("brief.position is required (e.g. 'ST', 'CM').")
    if not isinstance(brief.get("notes"), str):
        raise ValueError("brief.notes must be a string (can be empty).")

    max_age = brief.get("max_age")
    if max_age is None:
        raise ValueError("brief.max_age is required.")
    if not (15 <= int(max_age) <= 45):
        raise ValueError(f"brief.max_age must be between 15 and 45, got {max_age}.")

    min_overall = brief.get("min_overall")
    if min_overall is None:
        raise ValueError("brief.min_overall is required.")
    if not (40 <= int(min_overall) <= 99):
        raise ValueError(f"brief.min_overall must be between 40 and 99, got {min_overall}.")

    budget = brief.get("budget_eur")
    if budget is not None and float(budget) <= 0:
        raise ValueError(f"brief.budget_eur must be a positive number, got {budget}.")

    return {
        "brief": {
            **brief,
            "position":    brief["position"].strip().upper(),
            "max_age":     int(max_age),
            "min_overall": int(min_overall),
            "budget_eur":  float(budget) if budget is not None else None,
            "notes":       brief["notes"].strip(),
        }
    }


def search_players(state: TransferScoutState) -> dict:
    """Query the FIFA dataset for candidates matching the brief."""
    brief = state["brief"]
    df = _load_df()

    # player_positions is comma-separated ("ST,CF") — str.contains matches
    # versatile players, consistent with how apply_filters in data_loader works
    mask = df["player_positions"].str.contains(brief["position"], na=False)
    mask &= df["age"] <= brief["max_age"]
    mask &= df["overall"] >= brief["min_overall"]

    if brief.get("budget_eur") is not None:
        mask &= df["value_eur"] <= brief["budget_eur"]

    if brief.get("target_league"):
        mask &= df["league_name"] == brief["target_league"]

    result = (
        df[mask]
        .sort_values("overall", ascending=False)
        .head(20)
    )

    return {"longlist": result.to_dict(orient="records")}


def score_players(state: TransferScoutState) -> dict:
    """Rank the longlist with a weighted fit score."""
    longlist = state["longlist"]
    if not longlist:
        return {"scored_players": []}

    brief = state["brief"]
    _MIN_AGE = 16  # youngest plausible FIFA player

    # First pass: compute the two non-relative dimensions and the raw VFM ratio
    intermediates = []
    for p in longlist:
        overall   = float(p.get("overall")   or 0)
        age       = float(p.get("age")       or brief["max_age"])
        value_eur = float(p.get("value_eur") or 0)

        overall_score = (overall / 99) * 100

        age_span = brief["max_age"] - _MIN_AGE
        age_score = ((brief["max_age"] - age) / age_span * 100) if age_span > 0 else 100.0
        age_score = max(0.0, min(100.0, age_score))

        vfm_raw = value_eur / max(overall, 1)

        intermediates.append((p, overall_score, age_score, vfm_raw))

    # Second pass: normalise VFM across the longlist so scores are relative
    vfm_values = [x[3] for x in intermediates]
    vfm_min, vfm_max = min(vfm_values), max(vfm_values)

    scored = []
    for p, overall_score, age_score, vfm_raw in intermediates:
        if vfm_max > vfm_min:
            vfm_score = (1 - (vfm_raw - vfm_min) / (vfm_max - vfm_min)) * 100
        else:
            vfm_score = 100.0

        fit_score = (
            0.50 * overall_score +
            0.30 * age_score +
            0.20 * vfm_score
        )
        scored.append({**p, "fit_score": round(max(0.0, min(100.0, fit_score)), 1)})

    scored.sort(key=lambda x: x["fit_score"], reverse=True)
    return {"scored_players": scored[:8]}


def fetch_news(state: TransferScoutState) -> dict:
    """Fetch recent transfer news for each candidate via Tavily search."""
    from tavily import TavilyClient

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not found. Add it to the .env file.")

    client = TavilyClient(api_key=api_key)
    news_data: dict[str, list[dict]] = {}

    for player in state["scored_players"]:
        # Key must match what generate_report uses for the lookup
        key = player.get("short_name") or player.get("long_name", "Unknown")
        search_name = player.get("long_name") or player.get("short_name", "Unknown")

        for attempt in range(2):
            try:
                response = client.search(
                    f"{search_name} transfer news",
                    search_depth="advanced",
                    max_results=3,
                )
                news_data[key] = [
                    {
                        "title":   r.get("title", ""),
                        "url":     r.get("url", ""),
                        "content": r.get("content", ""),
                    }
                    for r in response.get("results", [])
                ]
                break
            except Exception:
                if attempt == 1:
                    news_data[key] = []
                else:
                    time.sleep(1)

    return {"news_data": news_data}


def generate_report(state: TransferScoutState) -> dict:
    """Call Claude to produce a markdown transfer shortlist report."""
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found. Add it to the .env file in the project folder."
        )

    brief = state["brief"]
    players = state["scored_players"]
    news_data = state["news_data"]

    player_blocks = []
    for i, p in enumerate(players, 1):
        name = p.get("short_name") or p.get("long_name", "Unknown")
        articles = news_data.get(name, [])
        if articles:
            news_lines = []
            for a in articles:
                news_lines.append(f"  - {a['title']}")
                if a.get("content"):
                    news_lines.append(f"    {a['content']}")
            news_str = "\n".join(news_lines)
        else:
            news_str = "  - No recent news found"

        player_blocks.append(f"""
Player {i}: {p.get('long_name', name)}
  Fit Score:    {p.get('fit_score')}/100
  Position:     {p.get('primary_position', '—')}
  Club:         {p.get('club_name', '—')} ({p.get('league_name', '—')})
  Age:          {p.get('age', '—')}
  Overall:      {p.get('overall', '—')}/99
  Potential:    {p.get('potential', '—')}/99
  Pace:         {p.get('pace', '—')}  Shooting: {p.get('shooting', '—')}  Passing: {p.get('passing', '—')}
  Dribbling:    {p.get('dribbling', '—')}  Defending: {p.get('defending', '—')}  Physic: {p.get('physic', '—')}
  Market Value: €{int(p.get('value_eur') or 0):,}
  Weekly Wage:  €{int(p.get('wage_eur') or 0):,}
  Recent News:
{news_str}""")

    _SYSTEM_PROMPT = """You are a senior football scout compiling a transfer shortlist report for a sporting director.
Your reports are precise, data-driven, and written in present tense, third person.
Never use vague language — every claim must reference a specific stat from the data provided.
Output clean markdown only. No preamble, no sign-off."""

    user_message = f"""Compile a transfer shortlist report for this scouting brief:

BRIEF
Position: {brief['position']}
Max Age: {brief['max_age']}
Min Overall: {brief['min_overall']}
Target League: {brief.get('target_league') or 'Any'}
Budget: {'€{:,.0f}'.format(brief['budget_eur']) if brief.get('budget_eur') else 'Not specified'}
Scout Notes: {brief.get('notes') or 'None'}

CANDIDATES (ranked by fit score)
{''.join(player_blocks)}

OUTPUT FORMAT — follow exactly, including the blank lines between every section:

## Transfer Shortlist — {brief['position']}

| Rank | Player | Club | Age | OVR | Fit Score |
|------|--------|------|-----|-----|-----------|
(one row per player, ranked by fit score)

---

(Then for each player:)

### 1. [Player Name] — Fit Score: XX/100

**Club:** [Club Name] · [League]

**Key Stats:** Overall XX | Potential XX | Pace XX | Shooting XX | Passing XX | Dribbling XX | Defending XX | Physic XX

**Financials:** Value €XM | Wage €XK/wk

**Recent News:**
- [headline or snippet]
- [headline or snippet]
(write "No recent news available." if the list is empty)

**Scout's Verdict:** [Exactly 2 sentences referencing specific stats from the data.]

---"""

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )

    return {"draft_report": response.content[0].text}


def human_review(state: TransferScoutState) -> dict:
    """Pause for scout approval. Resume by passing {'approved': True/False}."""
    decision: dict = interrupt({
        "draft_report": state["draft_report"],
        "scored_players": state["scored_players"],
    })
    return {"approved": decision.get("approved", False)}


def save_report(state: TransferScoutState) -> dict:
    """Persist the approved report to disk."""
    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)

    position = state["brief"]["position"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"transfer_scout_{position}_{timestamp}.md"
    filepath = os.path.join(reports_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(state["draft_report"])

    return {"approved": True}


# ── Routing ───────────────────────────────────────────────────────────────────

def _route_after_scoring(state: TransferScoutState) -> str:
    return END if not state.get("scored_players") else "fetch_news"


def _route_after_review(state: TransferScoutState) -> str:
    return "save_report" if state.get("approved") else END


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    builder = StateGraph(TransferScoutState)

    builder.add_node("parse_brief",    parse_brief)
    builder.add_node("search_players", search_players)
    builder.add_node("score_players",  score_players)
    builder.add_node("fetch_news",     fetch_news)
    builder.add_node("generate_report", generate_report)
    builder.add_node("human_review",   human_review)
    builder.add_node("save_report",    save_report)

    builder.add_edge(START,             "parse_brief")
    builder.add_edge("parse_brief",     "search_players")
    builder.add_edge("search_players",   "score_players")
    builder.add_conditional_edges(
        "score_players",
        _route_after_scoring,
        {"fetch_news": "fetch_news", END: END},
    )
    builder.add_edge("fetch_news",      "generate_report")
    builder.add_edge("generate_report", "human_review")
    builder.add_conditional_edges(
        "human_review",
        _route_after_review,
        {"save_report": "save_report", END: END},
    )
    builder.add_edge("save_report", END)

    return builder


# ── Compiled graph (with in-memory checkpointing for interrupt support) ───────

checkpointer = MemorySaver()
graph = build_graph().compile(checkpointer=checkpointer)
