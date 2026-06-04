# Soccer Scout Pro ⚽

A data-driven football scouting application built for clubs competing in Europe's top five leagues — Premier League, La Liga, Bundesliga, Serie A, and Ligue 1.

Built using **vibe coding** with Claude AI (Anthropic) as part of the Gen AI & Agentic AI Course.

**Author:** Sharan Hegde

---

## What It Does

A browser-based Streamlit app with 9 feature tabs:

| Tab | Purpose |
|---|---|
| Leaderboard | Ranked player table filterable by any metric |
| Player Profile | Full dossier — stats, radar chart, career arc, AI scouting report |
| Young Talent | Age vs potential scatter for high-ceiling young players |
| Compare Players | Side-by-side radar comparison of up to 3 players |
| Hidden Gems | Undervalued player finder using a composite scoring algorithm |
| Squad Analyzer | Interactive pitch layout showing positional gaps for any club |
| Contract Tracker | Expiring contracts and free agent targets by year |
| Budget Optimizer | Best-value players within a wage and transfer ceiling |
| Nationality Map | World choropleth showing player origins by league |

---

## Setup

### 1. Install dependencies

```bash
uv install
```

### 2. Add the dataset

Download the FIFA 23 male players dataset (male_players.csv) and place it in the project root. The app will build a fast-loading cache on first run automatically.

### 3. Add your Anthropic API key (optional — for AI Scouting Reports)

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_key_here
```

### 4. Run the app

```bash
uv run streamlit run app.py
```

Open **http://localhost:8501** in your browser.

> **First run:** The app will scan the CSV and build a Parquet cache (~2 minutes). Every subsequent run loads in under 3 seconds.

---

## Project Structure

```
├── app.py              # Streamlit app — layout, sidebar, 9 tabs
├── data_loader.py      # CSV chunked reading, Parquet caching, filters
├── league_fit.py       # League fit scoring, hidden gem score, growth projection
├── charts.py           # All Plotly charts (radar, scatter, map, pitch layout)
├── ai_scout.py         # Claude Haiku API integration, 1000-report counter
├── pyproject.toml      # Project dependencies (managed by uv)
└── .env                # API key — not committed, create manually
```

---

## Built With

- [Streamlit](https://streamlit.io) — web app framework
- [Pandas](https://pandas.pydata.org) — data handling
- [Plotly](https://plotly.com) — interactive charts
- [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) — Claude Haiku AI reports
- [uv](https://github.com/astral-sh/uv) — package manager
