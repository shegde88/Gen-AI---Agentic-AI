# Soccer Scout Pro ⚽

A data-driven football scouting application built for clubs competing in Europe's top five leagues — Premier League, La Liga, Bundesliga, Serie A, and Ligue 1.

Built using **vibe coding** with Claude AI (Anthropic) as part of the Gen AI & Agentic AI Course.

**Author:** Sharan Hegde

---

## What It Does

A browser-based Streamlit app with 9 feature tabs:

| Tab | Purpose |
|---|---|
| Leaderboard | Ranked player table sortable by any metric |
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

### Step 1 — Clone the repo

```bash
git clone https://github.com/shegde88/Gen-AI---Agentic-AI.git
cd Gen-AI---Agentic-AI/Week-01-Soccer-Scout-Pro
```

### Step 2 — Install dependencies

Make sure you have [uv](https://github.com/astral-sh/uv) installed, then run:

```bash
uv install
```

### Step 3 — Download the dataset

The dataset is not included in this repo because it is 5.37 GB. Download it from Kaggle:

**[FIFA 23 Complete Player Dataset on Kaggle](https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset)**

1. Download the file called **`male_players.csv`**
2. Place it in this folder (`Week-01-Soccer-Scout-Pro/`)

> **First run note:** The app will scan the CSV and build a fast-loading cache (~2 minutes). Every run after that loads in under 3 seconds.

### Step 4 — Add your Anthropic API key (optional)

The AI Scouting Report feature requires an Anthropic API key. Without it, the rest of the app works fully — only the AI report button will show an error.

Create a `.env` file in this folder:

```
ANTHROPIC_API_KEY=your_key_here
```

Get a free API key at [console.anthropic.com](https://console.anthropic.com).

### Step 5 — Run the app

```bash
uv run streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Project Structure

```
├── app.py              # Streamlit app — layout, sidebar, 9 tabs
├── data_loader.py      # CSV chunked reading, Parquet caching, filters
├── league_fit.py       # League fit scoring, hidden gem score, growth projection
├── charts.py           # All Plotly charts (radar, scatter, map, pitch layout)
├── ai_scout.py         # Claude Haiku API integration, 1,000-report counter
├── pyproject.toml      # Project dependencies (managed by uv)
└── .env                # API key — create this manually, not committed to git
```

---

## Built With

- [Streamlit](https://streamlit.io) — web app framework
- [Pandas](https://pandas.pydata.org) — data handling
- [Plotly](https://plotly.com) — interactive charts
- [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) — Claude Haiku AI scouting reports
- [uv](https://github.com/astral-sh/uv) — package manager
