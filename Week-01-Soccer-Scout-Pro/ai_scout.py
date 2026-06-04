import json
import os
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()

_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".data_cache")
_COUNTER_PATH = os.path.join(_CACHE_DIR, "report_counter.json")
_REPORT_LIMIT = 1000
_MODEL = "claude-haiku-4-5-20251001"

# System prompt is stable across all requests — marked for prompt caching.
# Cache_control is set as best practice; activates once the prefix exceeds
# the model's minimum cacheable size (4096 tokens for Haiku 4.5).
_SYSTEM_PROMPT = """You are an elite football scout with 20 years of experience assessing players
for top European clubs. Club directors rely on your reports to make fast, confident transfer decisions.

REPORT FORMAT — use exactly these five sections with markdown bold headers:

**Overview**
Two sentences. Summarise who this player is and their primary value to a buying club.

**Key Strengths**
Three bullet points. Each one must reference a specific attribute number from the data provided.
Be concrete — "Elite pace (97) makes him a consistent threat in behind" not "He is fast."

**Areas for Development**
Two bullet points. Identify genuine weaknesses visible in the data. Be honest — a scout who
glosses over weaknesses is useless.

**Best League Fit**
One sentence. Name the single best-fit league from: Premier League, La Liga, Bundesliga,
Serie A, Ligue 1. Explain why in terms of his specific attributes.

**Transfer Recommendation**
Start with exactly one of: BUY / MONITOR / PASS
Follow with one sentence of rationale referencing his age, overall, potential, and wage.

RULES:
- Total report must be under 220 words.
- Always reference specific numbers — never say "good" or "strong" without a stat.
- Write in present tense, third person.
- Do not add any text before **Overview** or after the recommendation sentence."""


def _load_counter() -> dict:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    if os.path.exists(_COUNTER_PATH):
        with open(_COUNTER_PATH) as f:
            return json.load(f)
    return {"count": 0, "limit": _REPORT_LIMIT}


def _save_counter(counter: dict) -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(_COUNTER_PATH, "w") as f:
        json.dump(counter, f, indent=2)


def get_report_usage() -> tuple[int, int]:
    """Return (reports_used, report_limit)."""
    c = _load_counter()
    return c["count"], c.get("limit", _REPORT_LIMIT)


def generate_scouting_report(player: dict) -> Optional[str]:
    """
    Call Claude Haiku to generate a scouting report for a player.
    Returns the report text, or None if the 1,000-report limit is reached.
    Raises ValueError if the API key is missing.
    """
    counter = _load_counter()
    if counter["count"] >= counter.get("limit", _REPORT_LIMIT):
        return None

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found. Add it to the .env file in the project folder."
        )

    def _safe_int(val, default=0):
        try:
            return int(val) if val is not None and str(val) != "nan" else default
        except (ValueError, TypeError):
            return default

    user_message = f"""Generate a scouting report for this player:

NAME: {player.get('long_name', player.get('short_name', 'Unknown'))}
POSITION: {player.get('player_positions', 'Unknown')}
AGE: {_safe_int(player.get('age'))}
CLUB: {player.get('club_name', 'Unknown')}
LEAGUE: {player.get('league_name', 'Unknown')}
NATIONALITY: {player.get('nationality_name', 'Unknown')}

RATINGS
Overall: {_safe_int(player.get('overall'))}/99
Potential: {_safe_int(player.get('potential'))}/99
Growth headroom: +{_safe_int(player.get('growth_potential'))}

MAIN ATTRIBUTES
Pace: {_safe_int(player.get('pace'))}/99
Shooting: {_safe_int(player.get('shooting'))}/99
Passing: {_safe_int(player.get('passing'))}/99
Dribbling: {_safe_int(player.get('dribbling'))}/99
Defending: {_safe_int(player.get('defending'))}/99
Physicality: {_safe_int(player.get('physic'))}/99

PROFILE
Preferred foot: {player.get('preferred_foot', 'Unknown')}
Work rate: {player.get('work_rate', 'Unknown')}
Skill moves: {_safe_int(player.get('skill_moves'))}★
Weak foot: {_safe_int(player.get('weak_foot'))}★

FINANCIALS
Weekly wage: €{_safe_int(player.get('wage_eur')):,}
Market value: €{_safe_int(player.get('value_eur')):,}
Contract until: {player.get('contract_year', 'Unknown')}"""

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=_MODEL,
        max_tokens=512,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )

    report = response.content[0].text

    counter["count"] += 1
    _save_counter(counter)

    return report
