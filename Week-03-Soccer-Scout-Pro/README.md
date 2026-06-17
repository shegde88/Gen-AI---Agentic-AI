# Soccer Scout Pro ⚽

A data-driven football scouting application built for clubs competing in Europe's top five leagues — Premier League, La Liga, Bundesliga, Serie A, and Ligue 1.

Built with Claude AI (Anthropic) and LangGraph.

**Author:** Sharan Hegde

---

## What It Does

A browser-based Streamlit app with 11 tabs:

| Tab | Purpose |
|---|---|
| About | First-time user guide explaining every feature in plain language |
| Leaderboard | Ranked player table filterable by any metric |
| Player Profile | Full dossier — stats, radar chart, career arc, AI scouting report |
| Young Talent | Age vs potential scatter for high-ceiling young players |
| Compare Players | Side-by-side radar comparison of up to 3 players |
| Hidden Gems | Undervalued player finder using a composite scoring algorithm |
| Squad Analyzer | Interactive pitch layout showing positional gaps for any club |
| Contract Tracker | Expiring contracts and free agent targets by year |
| Budget Optimizer | Best-value players within a wage and transfer ceiling |
| Nationality Map | World choropleth showing player origins by league |
| Transfer Scout Agent | LangGraph agentic pipeline — enter a scouting brief, get a ranked shortlist with live transfer news and an AI-written report |

### Transfer Scout Agent

A 7-node LangGraph pipeline with a human-in-the-loop interrupt:

```
parse_brief → search_players → score_players → fetch_news → generate_report → [human review] → save_report
```

- **Fit scoring** — weighted composite: 50% overall rating, 30% age value (younger = higher), 20% value for money (lower market value relative to rating = higher)
- **Transfer news** — Tavily search API fetches up to 3 recent articles per player with full article content
- **AI report** — Claude Haiku writes a formatted markdown shortlist report with a ranked table and per-player scout verdict
- **Human interrupt** — scout reviews the draft report and scored players before approving or adjusting
- **Save report** — approved reports are saved to the `reports/` directory as timestamped markdown files

---

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Add the dataset

Download the FIFA 23 male players dataset (`male_players.csv`) and place it in the project root. The app will build a fast-loading Parquet cache on first run automatically.

> **First run:** The app will scan the CSV and build a Parquet cache (~2 minutes). Every subsequent run loads in under 3 seconds.

### 3. Add your API keys

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_anthropic_key_here
TAVILY_API_KEY=your_tavily_key_here
```

- `ANTHROPIC_API_KEY` — required for AI Scouting Reports and the Transfer Scout Agent report generation
- `TAVILY_API_KEY` — required for the Transfer Scout Agent's live transfer news search

### 4. Run the app

```bash
.venv/bin/python -m streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Project Structure

```
├── app.py                   # Streamlit app — layout, sidebar, 11 tabs
├── transfer_scout_agent.py  # LangGraph Transfer Scout Agent (7 nodes + graph)
├── data_loader.py           # CSV chunked reading, Parquet caching, filters
├── league_fit.py            # League fit scoring, hidden gem score, growth projection
├── charts.py                # All Plotly charts (radar, scatter, map, pitch layout)
├── ai_scout.py              # Claude Haiku API integration, 1000-report counter
├── test_brief.py            # Smoke test — runs the agent to interrupt, no approval
├── reports/                 # Saved Transfer Scout reports (generated, not committed)
├── pyproject.toml           # Project dependencies (managed by uv)
└── .env                     # API keys — not committed, create manually
```

---

## Built With

- [Streamlit](https://streamlit.io) — web app framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) — agentic pipeline with human-in-the-loop interrupt
- [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) — Claude Haiku for AI scouting reports
- [Tavily](https://tavily.com) — real-time transfer news search
- [Pandas](https://pandas.pydata.org) — data handling
- [Plotly](https://plotly.com) — interactive charts
- [uv](https://github.com/astral-sh/uv) — package manager
