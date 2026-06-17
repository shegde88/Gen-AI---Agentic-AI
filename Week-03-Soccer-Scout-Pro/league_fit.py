import pandas as pd
import numpy as np

# Each league profile is a weighted dict of player attributes (0–100 scale).
# Weights reflect the known tactical fingerprint of that league.
LEAGUE_PROFILES = {
    "Premier League": {
        "movement_sprint_speed": 0.20,
        "movement_acceleration": 0.15,
        "power_strength": 0.20,
        "power_stamina": 0.15,
        "physic": 0.15,
        "mentality_aggression": 0.10,
        "mentality_composure": 0.05,
    },
    "La Liga": {
        "skill_ball_control": 0.20,
        "skill_dribbling": 0.15,
        "attacking_short_passing": 0.20,
        "skill_long_passing": 0.10,
        "passing": 0.15,
        "mentality_vision": 0.10,
        "mentality_composure": 0.10,
    },
    "Bundesliga": {
        "power_stamina": 0.20,
        "mentality_aggression": 0.20,
        "mentality_interceptions": 0.15,
        "movement_acceleration": 0.15,
        "pace": 0.15,
        "power_strength": 0.10,
        "defending": 0.05,
    },
    "Serie A": {
        "defending": 0.20,
        "defending_marking_awareness": 0.20,
        "defending_standing_tackle": 0.15,
        "mentality_composure": 0.15,
        "mentality_interceptions": 0.10,
        "mentality_positioning": 0.10,
        "power_strength": 0.10,
    },
    "Ligue 1": {
        "movement_acceleration": 0.20,
        "movement_sprint_speed": 0.20,
        "power_strength": 0.15,
        "pace": 0.15,
        "power_stamina": 0.10,
        "skill_dribbling": 0.10,
        "mentality_aggression": 0.10,
    },
}

LEAGUE_DESCRIPTIONS = {
    "Premier League": "Pace & Physicality — high-intensity, direct, physical battles.",
    "La Liga": "Technical & Passing — possession, creativity, close control.",
    "Bundesliga": "Pressing & Work Rate — high press, stamina, aggression.",
    "Serie A": "Defensive & Tactical — structured, disciplined, compact.",
    "Ligue 1": "Athletic & Pace — explosive, fast-transitioning, physical.",
}


def compute_league_fit(player: pd.Series) -> dict[str, float]:
    """Return a 0–100 fit score for each league for a single player row."""
    scores = {}
    for league, profile in LEAGUE_PROFILES.items():
        total_weight = 0.0
        weighted_score = 0.0
        for attr, weight in profile.items():
            val = player.get(attr)
            if pd.notna(val):
                weighted_score += float(val) * weight
                total_weight += weight
        scores[league] = round(weighted_score / total_weight, 1) if total_weight > 0 else 0.0
    return scores


def best_league_fit(player: pd.Series) -> str:
    scores = compute_league_fit(player)
    return max(scores, key=scores.get)


def compute_hidden_gem_score(df: pd.DataFrame) -> pd.Series:
    """
    Score = high potential + large growth gap + low wage + low reputation.
    Returns a 0–100 normalised series.
    """
    potential_norm = df["potential"] / 99.0
    growth_norm = df["growth_potential"].clip(0, 30) / 30.0

    max_wage = df["wage_eur"].replace(0, np.nan).quantile(0.95)
    wage_inv = 1 - (df["wage_eur"].clip(0, max_wage) / max_wage)

    rep = df.get("international_reputation", pd.Series(3, index=df.index))
    rep_inv = 1 - ((rep.fillna(3) - 1) / 4.0)

    raw = (
        potential_norm * 0.35
        + growth_norm * 0.35
        + wage_inv * 0.20
        + rep_inv * 0.10
    )
    return (raw * 100).round(1)


def growth_curve(age: int, overall: float, potential: float) -> pd.DataFrame:
    """
    Model a player's projected rating from their current age to 35.
    Peak age assumed at 27. Growth is logistic up to peak, then linear decline.
    """
    ages = list(range(age, 36))
    ratings = []
    gap = potential - overall
    peak_age = 27

    for a in ages:
        if a <= age:
            ratings.append(overall)
        elif a <= peak_age:
            progress = (a - age) / max(peak_age - age, 1)
            logistic = 1 / (1 + np.exp(-10 * (progress - 0.5)))
            ratings.append(overall + gap * logistic)
        else:
            years_past_peak = a - peak_age
            decline_per_year = 0.8
            peak_rating = overall + gap
            ratings.append(max(overall, peak_rating - decline_per_year * years_past_peak))

    return pd.DataFrame({"age": ages, "projected_overall": [round(r, 1) for r in ratings]})


def compute_value_score(df: pd.DataFrame) -> pd.Series:
    """Overall rating divided by weekly wage (per 1000 EUR). Higher = better value."""
    safe_wage = df["wage_eur"].replace(0, np.nan).fillna(1)
    return ((df["overall"] / (safe_wage / 1000)) * 10).round(1)
