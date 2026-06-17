import pandas as pd
import numpy as np
import streamlit as st
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "male_players.csv")
TARGET_SNAPSHOT = "2022-12-17"

# Parquet cache files — generated on first run, reused after that
_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".data_cache")
_PARQUET_FIFA23 = os.path.join(_CACHE_DIR, "fifa23_snapshot.parquet")
_PARQUET_CAREER = os.path.join(_CACHE_DIR, "career_data.parquet")

REQUIRED_COLUMNS = {
    "player_id", "short_name", "long_name", "player_positions", "overall",
    "potential", "value_eur", "wage_eur", "age", "league_name", "club_name",
    "nationality_name", "preferred_foot", "pace", "shooting", "passing",
    "dribbling", "defending", "physic", "work_rate",
}

NUMERIC_COLS = [
    "overall", "potential", "value_eur", "wage_eur", "age", "height_cm",
    "weight_kg", "pace", "shooting", "passing", "dribbling", "defending",
    "physic", "weak_foot", "skill_moves", "international_reputation",
    "release_clause_eur", "club_contract_valid_until_year",
    "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
    "attacking_short_passing", "attacking_volleys", "skill_dribbling",
    "skill_curve", "skill_fk_accuracy", "skill_long_passing", "skill_ball_control",
    "movement_acceleration", "movement_sprint_speed", "movement_agility",
    "movement_reactions", "movement_balance", "power_shot_power", "power_jumping",
    "power_stamina", "power_strength", "power_long_shots", "mentality_aggression",
    "mentality_interceptions", "mentality_positioning", "mentality_vision",
    "mentality_penalties", "mentality_composure", "defending_marking_awareness",
    "defending_standing_tackle", "defending_sliding_tackle",
    "goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking",
    "goalkeeping_positioning", "goalkeeping_reflexes",
]

POSITION_COLS = [
    "ls", "st", "rs", "lw", "lf", "cf", "rf", "rw",
    "lam", "cam", "ram", "lm", "lcm", "cm", "rcm", "rm",
    "lwb", "ldm", "cdm", "rdm", "rwb", "lb", "lcb", "cb", "rcb", "rb", "gk",
]

CAREER_COLS = [
    "player_id", "short_name", "fifa_version", "fifa_update_date",
    "overall", "potential", "pace", "shooting", "passing",
    "dribbling", "defending", "physic", "age",
]

TOP5_LEAGUES = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]

CHUNK_SIZE = 100_000


def _parse_position_rating(val):
    """Position rating cells look like '85+2' — take the base number."""
    if pd.isna(val):
        return np.nan
    s = str(val).split("+")[0].split("-")[0].strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def _ensure_cache_dir():
    os.makedirs(_CACHE_DIR, exist_ok=True)


def _build_fifa23_parquet(csv_path: str, snapshot: str) -> pd.DataFrame:
    """Scan CSV in chunks, keep only the target FIFA 23 snapshot, save Parquet."""
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=False):
        if "fifa_version" not in chunk.columns:
            break
        mask = chunk["fifa_version"] == 23
        if "fifa_update_date" in chunk.columns:
            mask &= chunk["fifa_update_date"] == snapshot
        filtered = chunk[mask]
        if not filtered.empty:
            chunks.append(filtered)

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    _ensure_cache_dir()
    df.to_parquet(_PARQUET_FIFA23, index=False)
    return df


def _build_career_parquet(csv_path: str) -> pd.DataFrame:
    """Scan CSV in chunks, keep career columns only, deduplicate per player/version."""
    available_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    use_cols = [c for c in CAREER_COLS if c in available_cols]

    chunks = []
    for chunk in pd.read_csv(csv_path, usecols=use_cols, chunksize=CHUNK_SIZE, low_memory=False):
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)

    for col in ["overall", "potential", "pace", "shooting", "passing",
                "dribbling", "defending", "physic", "age"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "fifa_update_date" in df.columns:
        df = df.sort_values("fifa_update_date")
        df = df.groupby(["player_id", "fifa_version"], as_index=False).last()

    _ensure_cache_dir()
    df.to_parquet(_PARQUET_CAREER, index=False)
    return df


def _post_process(df: pd.DataFrame) -> pd.DataFrame:
    """Apply numeric coercion, derived columns, and position parsing."""
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in POSITION_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_position_rating)

    df["growth_potential"] = df["potential"] - df["overall"]
    df["value_eur_m"] = df["value_eur"] / 1_000_000
    df["wage_eur_k"] = df["wage_eur"] / 1_000
    df["primary_position"] = df["player_positions"].str.split(",").str[0].str.strip()

    if "club_contract_valid_until_year" in df.columns:
        df["contract_year"] = pd.to_numeric(
            df["club_contract_valid_until_year"], errors="coerce"
        ).astype("Int64")

    return df.reset_index(drop=True)


@st.cache_data(show_spinner="Loading player data…")
def load_data(csv_path: str = DATA_PATH, snapshot: str = TARGET_SNAPSHOT) -> pd.DataFrame:
    """
    Load FIFA 23 snapshot. Uses a Parquet cache after the first run so
    subsequent loads take < 1 second instead of several minutes.
    """
    parquet_path = _PARQUET_FIFA23
    # If this is an uploaded file, use a separate cache key
    if csv_path != DATA_PATH:
        parquet_path = os.path.join(_CACHE_DIR, "uploaded_snapshot.parquet")

    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    else:
        df = _build_fifa23_parquet(csv_path, snapshot)

    # Restrict to top 5 European leagues only
    if "league_name" in df.columns:
        df = df[df["league_name"].isin(TOP5_LEAGUES)]

    return _post_process(df)


@st.cache_data(show_spinner="Loading career history… (first run only — this takes a few minutes)")
def load_career_data(csv_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load career arc data across all FIFA versions.
    Builds a Parquet cache on first run; instant on subsequent runs.
    """
    parquet_path = _PARQUET_CAREER
    if csv_path != DATA_PATH:
        parquet_path = os.path.join(_CACHE_DIR, "uploaded_career.parquet")

    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)

    return _build_career_parquet(csv_path)


def invalidate_cache():
    """Remove Parquet cache files so the next load rebuilds from CSV."""
    for path in [_PARQUET_FIFA23, _PARQUET_CAREER,
                 os.path.join(_CACHE_DIR, "uploaded_snapshot.parquet"),
                 os.path.join(_CACHE_DIR, "uploaded_career.parquet")]:
        if os.path.exists(path):
            os.remove(path)


def validate_uploaded_csv(df: pd.DataFrame) -> tuple[bool, str]:
    """Return (is_valid, error_message)."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        return False, f"Missing required columns: {', '.join(sorted(missing))}"
    return True, ""


def get_filter_options(df: pd.DataFrame) -> dict:
    """Return sorted unique values for each sidebar filter."""
    all_positions = sorted(
        {p.strip() for pos in df["player_positions"].dropna() for p in pos.split(",")}
    )
    leagues = [l for l in TOP5_LEAGUES if l in df["league_name"].values]

    return {
        "positions": ["All"] + all_positions,
        "leagues": ["All"] + leagues,
        "clubs": ["All"] + sorted(df["club_name"].dropna().unique().tolist()),
        "feet": ["All"] + sorted(df["preferred_foot"].dropna().unique().tolist()),
        "work_rates": ["All"] + sorted(df["work_rate"].dropna().unique().tolist()),
        "age_min": int(df["age"].min()),
        "age_max": int(df["age"].max()),
        "wage_min": int(df["wage_eur"].min()),
        "wage_max": int(df["wage_eur"].max()),
        "value_min": int(df["value_eur"].min()),
        "value_max": int(df["value_eur"].max()),
    }


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    fdf = df.copy()

    if filters.get("position") and filters["position"] != "All":
        fdf = fdf[fdf["player_positions"].str.contains(filters["position"], na=False)]

    if filters.get("league") and filters["league"] != "All":
        fdf = fdf[fdf["league_name"] == filters["league"]]

    if filters.get("club") and filters["club"] != "All":
        fdf = fdf[fdf["club_name"] == filters["club"]]

    if filters.get("foot") and filters["foot"] != "All":
        fdf = fdf[fdf["preferred_foot"] == filters["foot"]]

    if filters.get("work_rate") and filters["work_rate"] != "All":
        fdf = fdf[fdf["work_rate"] == filters["work_rate"]]

    age_range = filters.get("age_range")
    if age_range:
        fdf = fdf[(fdf["age"] >= age_range[0]) & (fdf["age"] <= age_range[1])]

    wage_range = filters.get("wage_range")
    if wage_range:
        fdf = fdf[(fdf["wage_eur"] >= wage_range[0]) & (fdf["wage_eur"] <= wage_range[1])]

    return fdf
