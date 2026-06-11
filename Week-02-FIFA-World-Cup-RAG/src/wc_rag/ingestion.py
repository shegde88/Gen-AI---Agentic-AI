"""
Multi-source ingestion: Wikipedia (web), PDFs, and Kaggle CSVs.

Architecture:
  load_web_documents()  → Wikipedia via WebBaseLoader (97 articles)
  load_pdf_documents()  → any .pdf files placed in data/raw/
  load_csv_documents()  → WorldCup CSVs placed in data/raw/
  load_all_documents()  → combines all three for a full corpus ingest

WHY THREE LOADERS:
  - Web: Wikipedia is the richest source of narrative team/player history.
  - PDF: FIFA/ESPN match reports are often published as PDFs; PyPDFLoader
    extracts text page-by-page and preserves section breaks.
  - CSV: Kaggle's historical World Cup datasets have structured match stats
    (1930–2014) that Wikipedia narrates but doesn't tabulate per row.
    CSVLoader turns each row into a Document — useful for statistical queries
    like "Who scored the most goals at the 1994 World Cup?"

All documents are tagged with metadata so Pinecone can filter results
by source_type / team / player / tournament_year / stage.
"""

import csv
import re
import time
from datetime import datetime

import bs4
from langchain_community.document_loaders import CSVLoader, WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from wc_rag.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_RAW_DIR

# ---------------------------------------------------------------------------
# Source registry — Wikipedia articles with metadata labels
# ---------------------------------------------------------------------------

def _team(name: str, url_slug: str) -> dict:
    return {
        "url": f"https://en.wikipedia.org/wiki/{url_slug}",
        "metadata": {"source_type": "team", "team": name, "player": "", "year": "all", "title": f"{name} National Football Team"},
    }


def _player(name: str, url_slug: str, team: str) -> dict:
    return {
        "url": f"https://en.wikipedia.org/wiki/{url_slug}",
        "metadata": {"source_type": "player", "team": team, "player": name, "year": "all", "title": name},
    }


SOURCES: list[dict] = [
    # ── Tournament overview ───────────────────────────────────────────────
    {"url": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup",
     "metadata": {"source_type": "tournament", "team": "", "player": "", "year": "2026", "title": "2026 FIFA World Cup"}},
    {"url": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads",
     "metadata": {"source_type": "tournament", "team": "", "player": "", "year": "2026", "title": "2026 FIFA World Cup Squads"}},
    {"url": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_venues",
     "metadata": {"source_type": "tournament", "team": "", "player": "", "year": "2026", "title": "2026 FIFA World Cup Venues"}},
    # ── Group stage articles (schedule, venues, match results per group) ──
    {"url": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_A",
     "metadata": {"source_type": "tournament", "team": "", "player": "", "year": "2026", "title": "2026 FIFA World Cup Group A"}},
    {"url": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_B",
     "metadata": {"source_type": "tournament", "team": "", "player": "", "year": "2026", "title": "2026 FIFA World Cup Group B"}},
    {"url": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_C",
     "metadata": {"source_type": "tournament", "team": "", "player": "", "year": "2026", "title": "2026 FIFA World Cup Group C"}},
    {"url": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_D",
     "metadata": {"source_type": "tournament", "team": "", "player": "", "year": "2026", "title": "2026 FIFA World Cup Group D"}},
    {"url": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_E",
     "metadata": {"source_type": "tournament", "team": "", "player": "", "year": "2026", "title": "2026 FIFA World Cup Group E"}},
    {"url": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_F",
     "metadata": {"source_type": "tournament", "team": "", "player": "", "year": "2026", "title": "2026 FIFA World Cup Group F"}},
    {"url": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_G",
     "metadata": {"source_type": "tournament", "team": "", "player": "", "year": "2026", "title": "2026 FIFA World Cup Group G"}},
    {"url": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_H",
     "metadata": {"source_type": "tournament", "team": "", "player": "", "year": "2026", "title": "2026 FIFA World Cup Group H"}},
    {"url": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_I",
     "metadata": {"source_type": "tournament", "team": "", "player": "", "year": "2026", "title": "2026 FIFA World Cup Group I"}},
    {"url": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_J",
     "metadata": {"source_type": "tournament", "team": "", "player": "", "year": "2026", "title": "2026 FIFA World Cup Group J"}},
    {"url": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_K",
     "metadata": {"source_type": "tournament", "team": "", "player": "", "year": "2026", "title": "2026 FIFA World Cup Group K"}},
    {"url": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_L",
     "metadata": {"source_type": "tournament", "team": "", "player": "", "year": "2026", "title": "2026 FIFA World Cup Group L"}},
    # ── Historical context ────────────────────────────────────────────────
    {"url": "https://en.wikipedia.org/wiki/FIFA_World_Cup",
     "metadata": {"source_type": "history", "team": "", "player": "", "year": "all", "title": "FIFA World Cup"}},
    {"url": "https://en.wikipedia.org/wiki/FIFA_World_Cup_records_and_statistics",
     "metadata": {"source_type": "history", "team": "", "player": "", "year": "all", "title": "FIFA World Cup Records and Statistics"}},
    # ── Every World Cup edition (1930–2022) ───────────────────────────────
    {"url": "https://en.wikipedia.org/wiki/1930_FIFA_World_Cup",
     "metadata": {"source_type": "history", "team": "", "player": "", "year": "1930", "title": "1930 FIFA World Cup"}},
    {"url": "https://en.wikipedia.org/wiki/1934_FIFA_World_Cup",
     "metadata": {"source_type": "history", "team": "", "player": "", "year": "1934", "title": "1934 FIFA World Cup"}},
    {"url": "https://en.wikipedia.org/wiki/1938_FIFA_World_Cup",
     "metadata": {"source_type": "history", "team": "", "player": "", "year": "1938", "title": "1938 FIFA World Cup"}},
    {"url": "https://en.wikipedia.org/wiki/1950_FIFA_World_Cup",
     "metadata": {"source_type": "history", "team": "", "player": "", "year": "1950", "title": "1950 FIFA World Cup"}},
    {"url": "https://en.wikipedia.org/wiki/1954_FIFA_World_Cup",
     "metadata": {"source_type": "history", "team": "", "player": "", "year": "1954", "title": "1954 FIFA World Cup"}},
    {"url": "https://en.wikipedia.org/wiki/1958_FIFA_World_Cup",
     "metadata": {"source_type": "history", "team": "", "player": "", "year": "1958", "title": "1958 FIFA World Cup"}},
    {"url": "https://en.wikipedia.org/wiki/1962_FIFA_World_Cup",
     "metadata": {"source_type": "history", "team": "", "player": "", "year": "1962", "title": "1962 FIFA World Cup"}},
    {"url": "https://en.wikipedia.org/wiki/1966_FIFA_World_Cup",
     "metadata": {"source_type": "history", "team": "", "player": "", "year": "1966", "title": "1966 FIFA World Cup"}},
    {"url": "https://en.wikipedia.org/wiki/1970_FIFA_World_Cup",
     "metadata": {"source_type": "history", "team": "", "player": "", "year": "1970", "title": "1970 FIFA World Cup"}},
    {"url": "https://en.wikipedia.org/wiki/1974_FIFA_World_Cup",
     "metadata": {"source_type": "history", "team": "", "player": "", "year": "1974", "title": "1974 FIFA World Cup"}},
    {"url": "https://en.wikipedia.org/wiki/1978_FIFA_World_Cup",
     "metadata": {"source_type": "history", "team": "", "player": "", "year": "1978", "title": "1978 FIFA World Cup"}},
    {"url": "https://en.wikipedia.org/wiki/1982_FIFA_World_Cup",
     "metadata": {"source_type": "history", "team": "", "player": "", "year": "1982", "title": "1982 FIFA World Cup"}},
    {"url": "https://en.wikipedia.org/wiki/1986_FIFA_World_Cup",
     "metadata": {"source_type": "history", "team": "", "player": "", "year": "1986", "title": "1986 FIFA World Cup"}},
    {"url": "https://en.wikipedia.org/wiki/1990_FIFA_World_Cup",
     "metadata": {"source_type": "history", "team": "", "player": "", "year": "1990", "title": "1990 FIFA World Cup"}},
    {"url": "https://en.wikipedia.org/wiki/1994_FIFA_World_Cup",
     "metadata": {"source_type": "history", "team": "", "player": "", "year": "1994", "title": "1994 FIFA World Cup"}},
    {"url": "https://en.wikipedia.org/wiki/1998_FIFA_World_Cup",
     "metadata": {"source_type": "history", "team": "", "player": "", "year": "1998", "title": "1998 FIFA World Cup"}},
    {"url": "https://en.wikipedia.org/wiki/2002_FIFA_World_Cup",
     "metadata": {"source_type": "history", "team": "", "player": "", "year": "2002", "title": "2002 FIFA World Cup"}},
    {"url": "https://en.wikipedia.org/wiki/2006_FIFA_World_Cup",
     "metadata": {"source_type": "history", "team": "", "player": "", "year": "2006", "title": "2006 FIFA World Cup"}},
    {"url": "https://en.wikipedia.org/wiki/2010_FIFA_World_Cup",
     "metadata": {"source_type": "history", "team": "", "player": "", "year": "2010", "title": "2010 FIFA World Cup"}},
    {"url": "https://en.wikipedia.org/wiki/2014_FIFA_World_Cup",
     "metadata": {"source_type": "history", "team": "", "player": "", "year": "2014", "title": "2014 FIFA World Cup"}},
    {"url": "https://en.wikipedia.org/wiki/2018_FIFA_World_Cup",
     "metadata": {"source_type": "history", "team": "", "player": "", "year": "2018", "title": "2018 FIFA World Cup"}},
    {"url": "https://en.wikipedia.org/wiki/2022_FIFA_World_Cup",
     "metadata": {"source_type": "history", "team": "", "player": "", "year": "2022", "title": "2022 FIFA World Cup"}},
    # ── All 48 participating teams ────────────────────────────────────────
    # CONMEBOL
    _team("Argentina",  "Argentina_national_football_team"),
    _team("Brazil",     "Brazil_national_football_team"),
    _team("Colombia",   "Colombia_national_football_team"),
    _team("Ecuador",    "Ecuador_national_football_team"),
    _team("Paraguay",   "Paraguay_national_football_team"),
    _team("Uruguay",    "Uruguay_national_football_team"),
    # UEFA
    _team("France",       "France_national_football_team"),
    _team("England",      "England_national_football_team"),
    _team("Germany",      "Germany_national_football_team"),
    _team("Spain",        "Spain_national_football_team"),
    _team("Portugal",     "Portugal_national_football_team"),
    _team("Netherlands",  "Netherlands_national_football_team"),
    _team("Belgium",      "Belgium_national_football_team"),
    _team("Croatia",      "Croatia_national_football_team"),
    _team("Switzerland",  "Switzerland_national_football_team"),
    _team("Austria",      "Austria_national_football_team"),
    _team("Scotland",     "Scotland_national_football_team"),
    _team("Norway",       "Norway_national_football_team"),
    _team("Sweden",       "Sweden_national_football_team"),
    _team("Türkiye",      "Turkey_national_football_team"),
    _team("Czechia",      "Czech_Republic_national_football_team"),
    _team("Bosnia-Herzegovina", "Bosnia_and_Herzegovina_national_football_team"),
    _team("Serbia",       "Serbia_national_football_team"),
    # CONCACAF
    _team("United States", "United_States_men%27s_national_soccer_team"),
    _team("Mexico",        "Mexico_national_football_team"),
    _team("Canada",        "Canada_men%27s_national_soccer_team"),
    _team("Panama",        "Panama_national_football_team"),
    _team("Haiti",         "Haiti_national_football_team"),
    _team("Curaçao",       "Cura%C3%A7ao_national_football_team"),
    # CAF
    _team("Morocco",     "Morocco_national_football_team"),
    _team("Senegal",     "Senegal_national_football_team"),
    _team("Egypt",       "Egypt_national_football_team"),
    _team("Algeria",     "Algeria_national_football_team"),
    _team("South Africa","South_Africa_national_football_team"),
    _team("Tunisia",     "Tunisia_national_football_team"),
    _team("Ghana",       "Ghana_national_football_team"),
    _team("Congo DR",    "DR_Congo_national_football_team"),
    _team("Côte d'Ivoire", "Ivory_Coast_national_football_team"),
    _team("Cape Verde",  "Cape_Verde_national_football_team"),
    # AFC
    _team("Japan",        "Japan_national_football_team"),
    _team("Korea Republic","South_Korea_national_football_team"),
    _team("IR Iran",      "Iran_national_football_team"),
    _team("Australia",    "Australia_national_soccer_team"),
    _team("Saudi Arabia", "Saudi_Arabia_national_football_team"),
    _team("Qatar",        "Qatar_national_football_team"),
    _team("Uzbekistan",   "Uzbekistan_national_football_team"),
    _team("Jordan",       "Jordan_national_football_team"),
    _team("Iraq",         "Iraq_national_football_team"),
    # OFC
    _team("New Zealand",  "New_Zealand_national_football_team"),
    # ── Key players ───────────────────────────────────────────────────────
    _player("Lionel Messi",    "Lionel_Messi",          "Argentina"),
    _player("Kylian Mbappé",   "Kylian_Mbapp%C3%A9",   "France"),
    _player("Vinicius Junior", "Vinicius_Junior",        "Brazil"),
    _player("Jude Bellingham", "Jude_Bellingham",        "England"),
    _player("Cristiano Ronaldo","Cristiano_Ronaldo",     "Portugal"),
    _player("Lamine Yamal",    "Lamine_Yamal",           "Spain"),
    _player("Pedri",           "Pedri",                  "Spain"),
    _player("Rodri",           "Rodri",                  "Spain"),
]


def _clean_text(text: str) -> str:
    """Strip Wikipedia citation markers and normalize whitespace."""
    text = re.sub(r"\[\d+\]", "", text)        # [1], [23] citation refs
    text = re.sub(r"\[edit\]", "", text)        # section edit links
    text = re.sub(r"\[note \d+\]", "", text)   # [note 1] footnotes
    text = re.sub(r"\n{3,}", "\n\n", text)     # collapse blank lines
    return text.strip()


def load_web_documents(delay_seconds: float = 1.5) -> list[Document]:
    """
    Fetch each Wikipedia source, extract main content, and tag with metadata.
    delay_seconds: polite crawl delay between requests.
    """
    documents: list[Document] = []
    failed: list[str] = []

    for source in SOURCES:
        url = source["url"]
        meta = source["metadata"]
        print(f"  Fetching: {meta['title']} ...")

        try:
            loader = WebBaseLoader(
                web_paths=[url],
                bs_kwargs={
                    "parse_only": bs4.SoupStrainer(
                        "div", attrs={"id": "mw-content-text"}
                    )
                },
                requests_kwargs={"timeout": 15},
            )
            raw_docs = loader.load()

            for doc in raw_docs:
                cleaned = _clean_text(doc.page_content)
                if len(cleaned) < 200:
                    continue  # skip near-empty pages (disambiguation, redirects)
                documents.append(
                    Document(
                        page_content=cleaned,
                        metadata={
                            **meta,
                            "source_url": url,
                        },
                    )
                )
        except Exception as exc:
            print(f"  WARNING: Failed to load {url}: {exc}")
            failed.append(url)

        time.sleep(delay_seconds)

    if failed:
        print(f"\n  {len(failed)} source(s) failed to load: {failed}")

    print(f"\n  Loaded {len(documents)} documents from {len(SOURCES)} sources.")
    return documents


def load_pdf_documents() -> list[Document]:
    """
    Load every .pdf in data/raw/ using PyPDFLoader.

    PyPDFLoader produces one Document per page, preserving page breaks as
    natural chunk boundaries. Each doc is tagged source_type="pdf".

    HOW TO USE: Drop any FIFA/ESPN PDF match reports into data/raw/ and
    re-run scripts/ingest.py. No code changes needed.
    """
    documents: list[Document] = []
    pdf_files = list(DATA_RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        print("  No PDF files found in data/raw/ — skipping PDF loader.")
        return documents

    for pdf_path in sorted(pdf_files):
        print(f"  Loading PDF: {pdf_path.name} ...")
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            for page in pages:
                page.metadata.update({
                    "source_type": "pdf",
                    "team": "",
                    "player": "",
                    "year": "",
                    "title": pdf_path.stem,
                    "source_url": str(pdf_path),
                })
            documents.extend(pages)
        except Exception as exc:
            print(f"  WARNING: Failed to load {pdf_path.name}: {exc}")

    print(f"  Loaded {len(documents)} pages from {len(pdf_files)} PDF(s).")
    return documents


_CSV_SCHEMA: dict[str, dict] = {
    "matches_1930_2022.csv": {
        "source_type": "match",
        "title": "World Cup Match Results (1930–2022)",
    },
    "world_cup.csv": {
        "source_type": "history",
        "title": "World Cup Tournament Summaries",
    },
    "schedule_2026.csv": {
        "source_type": "tournament",
        "title": "2026 FIFA World Cup Schedule",
    },
    "fifa_ranking_2026-06-08.csv": {
        "source_type": "ranking",
        "title": "FIFA World Rankings (June 2026)",
    },
    "fifa_ranking_2022-10-06.csv": {
        "source_type": "ranking",
        "title": "FIFA World Rankings (October 2022)",
    },
    "future_match_probabilities_baseline.csv": {
        "source_type": "prediction",
        "title": "2026 World Cup Match Win Probabilities",
    },
}


_UTC_OFFSET_TO_TZ = {
    "UTC-4": "EDT (Eastern Daylight Time)",
    "UTC-5": "CDT (Central Daylight Time)",
    "UTC-6": "CST (Mexico Central Standard Time)",
    "UTC-7": "PDT (Pacific Daylight Time)",
}


def _enrich_schedule_row(doc: Document) -> Document:
    """
    Rewrite a schedule_2026.csv row into a natural language sentence.

    The enriched CSV includes venue, city, and timezone columns. This function
    produces a human-readable summary line that semantic search can match for
    natural language queries like "June 11th" or "What time does Mexico play?"
    """
    fields: dict[str, str] = {}
    for line in doc.page_content.split("\n"):
        if ": " in line:
            key, _, value = line.partition(": ")
            fields[key.strip()] = value.strip()

    date_str = fields.get("Date", "")
    human_date = date_str
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        human_date = dt.strftime("%B %-d, %Y")
    except ValueError:
        pass

    home = fields.get("home_team", "")
    away = fields.get("away_team", "")
    round_ = fields.get("Round", "match")
    group = fields.get("Group", "")
    day = fields.get("Day", "")
    time_local = fields.get("time_local", "") or fields.get("Time", "")
    utc_offset = fields.get("utc_offset", "")
    time_utc = fields.get("time_utc", "")
    venue = fields.get("venue", "")
    city = fields.get("city", "")

    tz_label = _UTC_OFFSET_TO_TZ.get(utc_offset, utc_offset)
    group_label = f"Group {group}" if group else round_

    time_part = (f"at {time_local} {tz_label} ({time_utc} UTC)" if time_local and tz_label
                 else f"at {time_local}" if time_local else "")
    venue_part = f"at {venue}, {city}" if venue and city else (f"in {city}" if city else "")
    day_part = f"({day})" if day else ""

    parts = [p for p in [f"{home} vs {away}", f"on {human_date}", day_part, time_part, venue_part] if p]
    summary = f"2026 FIFA World Cup {group_label}: {' '.join(parts)}"

    updated_content = doc.page_content.replace(f"Date: {date_str}", f"Date: {human_date}")
    return Document(page_content=f"{summary}\n\n{updated_content}", metadata=doc.metadata)


def load_csv_documents() -> list[Document]:
    """
    Load Kaggle World Cup CSVs from data/raw/.

    CSVLoader turns each row into a Document. Row fields are concatenated
    into page_content as "column: value\\n" strings so the LLM can read them.
    Metadata is enriched with source_type, year, and title from the schema map.

    WHY ROW-AS-DOCUMENT: A question like "Who won the 1994 World Cup?" is
    answered by a single row in WorldCups.csv. Embedding whole-file content
    would dilute that signal. One row = one retrievable fact unit.
    """
    documents: list[Document] = []
    csv_files = list(DATA_RAW_DIR.glob("*.csv"))

    if not csv_files:
        print("  No CSV files found in data/raw/ — skipping CSV loader.")
        print("  Download from Kaggle: https://www.kaggle.com/datasets/abecklas/fifa-world-cup")
        return documents

    for csv_path in sorted(csv_files):
        schema = _CSV_SCHEMA.get(csv_path.name, {
            "source_type": "csv",
            "title": csv_path.stem,
        })
        print(f"  Loading CSV: {csv_path.name} ({schema['title']}) ...")
        try:
            loader = CSVLoader(
                file_path=str(csv_path),
                csv_args={"delimiter": ","},
            )
            rows = loader.load()
            for row in rows:
                row.metadata.update({
                    "source_type": schema["source_type"],
                    "team": "",
                    "player": "",
                    "year": "",
                    "title": schema["title"],
                    "source_url": str(csv_path),
                })
            if csv_path.name == "schedule_2026.csv":
                rows = [_enrich_schedule_row(row) for row in rows]
            documents.extend(rows)
        except Exception as exc:
            print(f"  WARNING: Failed to load {csv_path.name}: {exc}")

    print(f"  Loaded {len(documents)} rows from {len(csv_files)} CSV file(s).")
    return documents


def load_all_documents(web_delay: float = 1.5) -> list[Document]:
    """
    Master loader: combines web + PDF + CSV sources into a single corpus.
    Call this from scripts/ingest.py for a full re-ingest.
    """
    print("\n── Web sources (Wikipedia) ──────────────────────")
    web_docs = load_web_documents(delay_seconds=web_delay)

    print("\n── PDF sources (data/raw/*.pdf) ─────────────────")
    pdf_docs = load_pdf_documents()

    print("\n── CSV sources (data/raw/*.csv) ─────────────────")
    csv_docs = load_csv_documents()

    all_docs = web_docs + pdf_docs + csv_docs
    print(f"\n  Total documents loaded: {len(all_docs)} "
          f"(web={len(web_docs)}, pdf={len(pdf_docs)}, csv={len(csv_docs)})")
    return all_docs


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split all documents into overlapping chunks for embedding.

    WHY 512/50:
    - 512 tokens sits in the sweet spot for text-embedding-3-small. The model
      produces coherent, query-relevant embeddings for passages up to ~600
      tokens. Beyond that, the vector starts averaging over too much text and
      loses specific signal (e.g. a single player's stats drowns in context).
    - 50-token overlap ensures that facts straddling a chunk boundary
      (e.g. a score mentioned at the end of one paragraph and elaborated in
      the next) appear in at least one complete chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"  Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunks
