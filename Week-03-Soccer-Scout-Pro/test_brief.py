"""
Smoke test — runs the Transfer Scout graph to the interrupt and prints results.
Does NOT resume (no approve/save). Safe to run repeatedly.

Usage:
    python test_brief.py
"""
import json
import uuid

from dotenv import load_dotenv
load_dotenv()

from transfer_scout_agent import graph

BRIEF = {
    "position":     "ST",
    "max_age":      26,
    "min_overall":  80,
    "budget_eur":   50_000_000.0,
    "target_league": None,
    "notes":        "",
}

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

print("=" * 60)
print("Running Transfer Scout Agent…")
print("=" * 60)

result = graph.invoke({"brief": BRIEF}, config)

interrupts = result.get("__interrupt__")
if not interrupts:
    print("\n⚠️  Graph finished without hitting the interrupt.")
    print("scored_players:", result.get("scored_players"))
else:
    payload = interrupts[0].value
    scored = payload.get("scored_players", [])

    print(f"\n✅  Interrupt reached — {len(scored)} player(s) scored.\n")

    print("-" * 60)
    print("SCORED PLAYERS")
    print("-" * 60)
    for p in scored:
        print(
            f"  {p.get('fit_score'):>5.1f}  "
            f"{p.get('short_name', '?'):<22} "
            f"OVR {p.get('overall')}  "
            f"Age {p.get('age')}  "
            f"€{int(p.get('value_eur') or 0) // 1_000_000}M  "
            f"{p.get('club_name', '—')}"
        )

    print("\n" + "-" * 60)
    print("DRAFT REPORT")
    print("-" * 60)
    print(payload.get("draft_report", "(no report)"))

print("\n" + "=" * 60)
print("Test complete. Graph paused at human_review — not resumed.")
print("=" * 60)
