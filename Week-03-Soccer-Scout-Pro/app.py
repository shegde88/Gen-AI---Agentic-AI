import streamlit as st
import pandas as pd
import numpy as np
import os
import shutil
import requests
from typing import Optional

from data_loader import (
    load_data, load_career_data, validate_uploaded_csv, invalidate_cache,
    get_filter_options, apply_filters, DATA_PATH, TARGET_SNAPSHOT, TOP5_LEAGUES,
)
from league_fit import (
    compute_league_fit, compute_hidden_gem_score,
    growth_curve, compute_value_score, LEAGUE_DESCRIPTIONS,
)
from charts import (
    radar_chart, league_fit_chart, growth_projection_chart,
    age_vs_potential_scatter, wage_vs_overall_scatter,
    versatility_chart, career_arc_chart, nationality_map,
    hidden_gems_scatter, squad_map_chart,
    BRAND_COLORS, RADAR_ATTRS,
)

import uuid
from langgraph.types import Command
from transfer_scout_agent import graph as ts_graph

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Soccer Scout Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Base dark theme */
    .stApp { background-color: #0E1117; }
    section[data-testid="stSidebar"] { background-color: #1A1F2E; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #1A1F2E;
        border: 1px solid #2A2F3E;
        border-radius: 10px;
        padding: 14px 18px;
    }
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; color: #00D4AA; }
    [data-testid="stMetricLabel"] { font-size: 0.75rem !important; color: #8892A4; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { background-color: #1A1F2E; border-radius: 10px; }
    .stTabs [data-baseweb="tab"] { color: #8892A4; font-size: 0.85rem; }
    .stTabs [aria-selected="true"] { color: #00D4AA !important; }

    /* Section headers */
    .section-title {
        font-size: 1.1rem; font-weight: 600;
        color: #00D4AA; margin: 1rem 0 0.5rem 0;
        border-left: 3px solid #00D4AA; padding-left: 10px;
    }

    /* Player card */
    .player-card {
        background-color: #1A1F2E; border-radius: 14px;
        border: 1px solid #2A2F3E; padding: 20px; margin-bottom: 16px;
    }

    /* Stat pill */
    .stat-pill {
        display: inline-block; background-color: #0E1117;
        border: 1px solid #2A2F3E; border-radius: 20px;
        padding: 4px 12px; margin: 3px; font-size: 0.82rem; color: #FAFAFA;
    }

    /* Badge */
    .badge {
        display: inline-block; border-radius: 6px;
        padding: 2px 10px; font-size: 0.75rem; font-weight: 600;
        margin-right: 6px;
    }
    .badge-green  { background-color: #00D4AA22; color: #00D4AA; border: 1px solid #00D4AA44; }
    .badge-orange { background-color: #FF6B3522; color: #FF6B35; border: 1px solid #FF6B3544; }
    .badge-blue   { background-color: #457B9D22; color: #A8DADC; border: 1px solid #457B9D44; }

    /* Info box */
    .info-box {
        background-color: #1A1F2E; border-radius: 10px;
        border: 1px solid #2A2F3E; padding: 16px; color: #8892A4;
        font-size: 0.85rem;
    }

    /* Divider */
    hr { border-color: #2A2F3E; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

if "data_path" not in st.session_state:
    st.session_state.data_path = DATA_PATH
if "upload_status" not in st.session_state:
    st.session_state.upload_status = None
if "ts_phase" not in st.session_state:
    st.session_state.ts_phase = "form"
if "ts_thread_id" not in st.session_state:
    st.session_state.ts_thread_id = None
if "ts_interrupt" not in st.session_state:
    st.session_state.ts_interrupt = None
if "ts_filename" not in st.session_state:
    st.session_state.ts_filename = None


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch_player_image(url: str) -> Optional[bytes]:
    """Fetch player image server-side to bypass browser CSP restrictions."""
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.content
    except Exception:
        pass
    return None


def fmt_eur(val):
    if pd.isna(val) or val == 0:
        return "—"
    if val >= 1_000_000:
        return f"€{val/1_000_000:.1f}M"
    if val >= 1_000:
        return f"€{val/1_000:.0f}K"
    return f"€{int(val)}"


def fmt_wage(val):
    if pd.isna(val) or val == 0:
        return "—"
    return f"€{val/1_000:.1f}K/wk" if val >= 1000 else f"€{int(val)}/wk"


def overall_color(val):
    if val >= 85:
        return "#00D4AA"
    if val >= 75:
        return "#F4A261"
    return "#E63946"


def rating_badge(val):
    color = overall_color(val)
    return f'<span style="background:{color}22;color:{color};border:1px solid {color}44;border-radius:6px;padding:2px 10px;font-weight:700;font-size:1rem;">{int(val)}</span>'


def _radar_colors():
    return [
        BRAND_COLORS["primary"], "#FF6B35", "#E63946",
        "#F4A261", "#457B9D", "#A8DADC",
    ]


# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_df(path: str) -> pd.DataFrame:
    return load_data(path)


@st.cache_data(show_spinner=False)
def get_career(path: str) -> pd.DataFrame:
    return load_career_data(path)


with st.spinner("Loading FIFA 23 player data…"):
    df_full = get_df(st.session_state.data_path)
    career_df_full = get_career(st.session_state.data_path)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0 8px 0;'>
        <div style='font-size:2rem;'>⚽</div>
        <div style='font-size:1.3rem; font-weight:700; color:#00D4AA;'>Soccer Scout Pro</div>
        <div style='font-size:0.75rem; color:#8892A4;'>FIFA 23 · Top 5 European Leagues</div>
    </div>
    <hr style='border-color:#2A2F3E;'>
    """, unsafe_allow_html=True)

    opts = get_filter_options(df_full)

    st.markdown("#### Filters")
    sel_league   = st.selectbox("League",         opts["leagues"], index=0)
    sel_position = st.selectbox("Position",       opts["positions"], index=0)

    # Clubs filtered to the selected league
    if sel_league and sel_league != "All":
        club_pool = sorted(
            df_full[df_full["league_name"] == sel_league]["club_name"].dropna().unique().tolist()
        )
    else:
        club_pool = sorted(df_full["club_name"].dropna().unique().tolist())
    sel_club = st.selectbox("Club", ["All"] + club_pool, index=0)

    sel_foot     = st.selectbox("Preferred Foot", opts["feet"], index=0)
    sel_wr       = st.selectbox("Work Rate (Attacking/Defensive)", opts["work_rates"], index=0)

    age_range = st.slider(
        "Age Range",
        min_value=opts["age_min"], max_value=opts["age_max"],
        value=(opts["age_min"], opts["age_max"]),
    )
    wage_range = st.slider(
        "Max Weekly Wage (€)",
        min_value=0, max_value=min(opts["wage_max"], 500_000),
        value=(0, min(opts["wage_max"], 500_000)),
        step=1_000,
        format="€%d",
    )

    st.markdown("<hr style='border-color:#2A2F3E;'>", unsafe_allow_html=True)
    st.markdown("#### Upload New Dataset")
    uploaded = st.file_uploader(
        "Drop a FIFA players CSV here",
        type=["csv"],
        help="Must contain the same columns as the FIFA 23 dataset.",
    )
    if uploaded is not None:
        with st.spinner("Validating file…"):
            try:
                preview = pd.read_csv(uploaded, nrows=5)
                valid, err = validate_uploaded_csv(preview)
                if valid:
                    save_path = os.path.join(os.path.dirname(DATA_PATH), "uploaded_players.csv")
                    uploaded.seek(0)
                    with open(save_path, "wb") as f:
                        shutil.copyfileobj(uploaded, f)
                    invalidate_cache()
                    st.session_state.data_path = save_path
                    st.session_state.upload_status = "success"
                    get_df.clear()
                    get_career.clear()
                    st.rerun()
                else:
                    st.session_state.upload_status = err
            except Exception as e:
                st.session_state.upload_status = str(e)

    if st.session_state.upload_status == "success":
        st.success("Dataset loaded successfully!")
        if st.button("Restore original dataset"):
            invalidate_cache()
            st.session_state.data_path = DATA_PATH
            st.session_state.upload_status = None
            get_df.clear()
            get_career.clear()
            st.rerun()
    elif st.session_state.upload_status:
        st.error(f"Upload failed: {st.session_state.upload_status}")

    st.markdown("<hr style='border-color:#2A2F3E;'>", unsafe_allow_html=True)
    st.caption(f"Dataset: {len(df_full):,} players · FIFA 23 snapshot {TARGET_SNAPSHOT}")


# ── Apply global filters ──────────────────────────────────────────────────────

filters = {
    "position":   sel_position,
    "league":     sel_league,
    "club":       sel_club,
    "foot":       sel_foot,
    "work_rate":  sel_wr,
    "age_range":  age_range,
    "wage_range": wage_range,
}
df = apply_filters(df_full, filters)

# ── Tabs ──────────────────────────────────────────────────────────────────────

TAB_NAMES = [
    "📖 About",
    "🏆 Leaderboard",
    "👤 Player Profile",
    "🌱 Young Talent",
    "⚖️ Compare Players",
    "💎 Hidden Gems",
    "🏟️ Squad Analyzer",
    "📋 Contract Tracker",
    "💰 Budget Optimizer",
    "🌍 Nationality Map",
    "🔍 Transfer Scout",
]
tabs = st.tabs(TAB_NAMES)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.markdown("## Welcome to Soccer Scout Pro ⚽")
    st.markdown(
        "Soccer Scout Pro is a scouting and analytics platform built on FIFA 23 data "
        "covering the **top 5 European leagues** — Premier League, La Liga, Bundesliga, "
        "Serie A, and Ligue 1. It gives you two things: a set of data explorer tabs for "
        "browsing and analysing the player database, and an AI-powered Transfer Scout agent "
        "that does the legwork of building a shortlist for you."
    )

    st.info(
        "📊 **Data source:** FIFA 23 snapshot (December 2022). Player ratings, values, wages, "
        "and contract details all reflect that point in time. The sidebar filters on the left "
        "apply globally across most tabs — use them to narrow down by league, position, club, "
        "age, or weekly wage before exploring."
    )

    st.markdown("---")
    st.markdown("### 📂 Tab Guide")
    st.markdown("Click any section below to see what that tab does and when to use it.")

    with st.expander("🏆 Leaderboard"):
        st.markdown(
            "Browse and sort the full player database by any attribute — overall rating, "
            "potential, pace, value, wage, and more. Apply the sidebar filters first to focus "
            "on a specific league or position, then use the sort controls inside the tab to "
            "find the best players within those constraints. Includes position group and "
            "league composition charts to give you a feel for the filtered pool."
        )

    with st.expander("👤 Player Profile"):
        st.markdown(
            "A full breakdown of a single player. Select any player from the filtered list and "
            "you'll see their attribute radar, how well they fit each of the five leagues, a "
            "projected growth curve showing where their rating is likely to peak, their career "
            "arc across multiple FIFA versions, and their contract and financial details. "
            "There's also an **AI Scouting Report** button that asks Claude to write a concise "
            "professional report covering strengths, weaknesses, best league fit, and a "
            "BUY / MONITOR / PASS recommendation."
        )

    with st.expander("🌱 Young Talent"):
        st.markdown(
            "Focused view for identifying high-ceiling young players. Set a maximum age and "
            "minimum potential rating, and the tab surfaces the best rising stars within those "
            "bounds — including a scatter of age vs potential and a ranked list of the top "
            "growth candidates. Select any player to see a projected growth curve showing when "
            "and how high their rating could peak."
        )

    with st.expander("⚖️ Compare Players"):
        st.markdown(
            "Pick up to three players and compare them side by side. You get an overlaid radar "
            "chart showing all six main attributes at once, a head-to-head stat table covering "
            "everything from pace and dribbling to market value and wage, and a league fit "
            "comparison bar chart showing which league suits each player best. Useful when "
            "you've narrowed down a shortlist and need to decide between two or three options."
        )

    with st.expander("💎 Hidden Gems"):
        st.markdown(
            "Finds undervalued players using a Gem Score that combines high potential, large "
            "growth headroom, low wages, and low international reputation. The lower a player's "
            "profile and the higher their ceiling, the better their gem score. Use this tab "
            "when you're looking for players other clubs may not have priced up yet — the kind "
            "of signing that looks obvious in hindsight."
        )

    with st.expander("🏟️ Squad Analyzer"):
        st.markdown(
            "Pick any club in the database and see their full squad laid out — average ratings "
            "by position group, a visual squad map, and a breakdown of positional depth and "
            "quality. Gaps are flagged automatically. Useful for identifying which areas of a "
            "squad are thin before making a targeted approach, or for scouting an opponent's "
            "weaknesses."
        )

    with st.expander("📋 Contract Tracker"):
        st.markdown(
            "Surfaces players whose contracts expire within a window you choose. Players whose "
            "deals run out in the next 12–18 months can often be signed at a significant "
            "discount, and those expiring within 6 months can be approached for pre-contract "
            "talks immediately. Filter by minimum overall rating to focus on quality targets "
            "rather than squad depth."
        )

    with st.expander("💰 Budget Optimizer"):
        st.markdown(
            "Set a maximum transfer value and a weekly wage cap, then see every player who "
            "fits within those limits. Players are ranked by a Value Score — overall rating "
            "divided by weekly wage — so the best bang-for-buck options float to the top. "
            "Useful when the budget is fixed and you need to find the highest quality player "
            "you can realistically afford."
        )

    with st.expander("🌍 Nationality Map"):
        st.markdown(
            "A world map showing where the players in your current filtered view come from. "
            "Hover over any country to see the number of players, their average overall rating, "
            "and their average potential. Use this alongside the sidebar filters to understand "
            "the nationality makeup of a specific league or position group — useful for "
            "international recruitment planning or understanding squad diversity."
        )

    st.markdown("---")
    st.markdown("### 🔍 Transfer Scout — AI Agent")
    st.info(
        "Transfer Scout works differently from the other tabs. Rather than browsing data "
        "yourself, you describe what you're looking for and the agent does the work — "
        "searching, scoring, researching, and drafting a report for your review."
    )

    st.markdown("**Here's how a typical run works:**")
    st.markdown(
        "1. **Fill in a scouting brief** — position, maximum age, minimum overall rating, "
        "transfer budget (optional), target league (optional), and any free-text notes "
        "like 'strong in the air' or 'two-footed preferred'.\n"
        "2. **The agent searches the FIFA 23 dataset** and scores every matching candidate "
        "against your brief using a weighted fit score (50% quality, 30% age, 20% value "
        "for money). The top 8 are shortlisted.\n"
        "3. **Live transfer news is fetched** for each candidate via Tavily — current "
        "availability, injury news, and transfer rumours are pulled in automatically.\n"
        "4. **Claude drafts a full shortlist report** with a ranked summary table and a "
        "scout's verdict for each player, drawing on both the FIFA stats and the live news.\n"
        "5. **You review the draft** and either approve it (which saves it to the "
        "`reports/` folder as a markdown file) or go back to adjust your filters."
    )

    st.markdown(
        "> **Note:** Transfer Scout uses the FIFA 23 database for player stats but fetches "
        "**live news** at the time you run it, so availability information and transfer "
        "rumours will reflect the current real-world situation — not December 2022."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LEADERBOARD
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.markdown('<div class="section-title">Player Leaderboard</div>', unsafe_allow_html=True)

    if df.empty:
        st.info("No players match the current filters. Try adjusting the sidebar.")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            sort_by = st.selectbox(
                "Sort by",
                ["overall", "potential", "growth_potential", "value_eur", "wage_eur",
                 "pace", "shooting", "passing", "dribbling", "defending", "physic"],
                index=0,
            )
        with c2:
            top_n = st.selectbox("Show top", [25, 50, 100, 250, "All"], index=0)

        display_cols = [
            "short_name", "age", "primary_position", "club_name", "league_name",
            "nationality_name", "overall", "potential", "growth_potential",
            "pace", "shooting", "passing", "dribbling", "defending", "physic",
            "value_eur_m", "wage_eur_k",
        ]
        display_cols = [c for c in display_cols if c in df.columns]
        ldf = df.sort_values(sort_by, ascending=False)
        if top_n != "All":
            ldf = ldf.head(int(top_n))

        rename_map = {
            "short_name": "Player", "age": "Age", "primary_position": "Pos",
            "club_name": "Club", "league_name": "League",
            "nationality_name": "Nation", "overall": "OVR", "potential": "POT",
            "growth_potential": "Growth", "pace": "PAC", "shooting": "SHO",
            "passing": "PAS", "dribbling": "DRI", "defending": "DEF",
            "physic": "PHY", "value_eur_m": "Value (€M)", "wage_eur_k": "Wage (€K/wk)",
        }

        st.dataframe(
            ldf[display_cols].rename(columns=rename_map),
            use_container_width=True,
            height=520,
            hide_index=True,
            column_config={
                "OVR": st.column_config.ProgressColumn("OVR", min_value=0, max_value=99, format="%d"),
                "POT": st.column_config.ProgressColumn("POT", min_value=0, max_value=99, format="%d"),
                "Growth": st.column_config.NumberColumn("Growth", format="+%d"),
                "Value (€M)": st.column_config.NumberColumn("Value (€M)", format="€%.1fM"),
                "Wage (€K/wk)": st.column_config.NumberColumn("Wage (€K/wk)", format="€%.1fK"),
            },
        )

        # Leaderboard composition charts
        st.markdown('<div class="section-title">Leaderboard Composition</div>', unsafe_allow_html=True)
        import plotly.express as px
        import plotly.graph_objects as go

        _LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="#FAFAFA"), margin=dict(l=10, r=10, t=40, b=10))

        # Map positions to broad groups
        POS_GROUPS = {
            "GK":  "Goalkeeper",
            "CB": "Defender", "LB": "Defender", "RB": "Defender",
            "LWB": "Defender", "RWB": "Defender", "LCB": "Defender", "RCB": "Defender",
            "CDM": "Midfielder", "CM": "Midfielder", "CAM": "Midfielder",
            "LM": "Midfielder", "RM": "Midfielder", "LCM": "Midfielder",
            "RCM": "Midfielder", "LDM": "Midfielder", "RDM": "Midfielder",
            "LAM": "Midfielder", "RAM": "Midfielder",
            "LW": "Forward", "RW": "Forward", "LF": "Forward", "RF": "Forward",
            "ST": "Forward", "CF": "Forward", "LS": "Forward", "RS": "Forward",
        }
        GROUP_COLORS = {
            "Goalkeeper": "#F4A261",
            "Defender":   "#2A9D8F",
            "Midfielder": "#457B9D",
            "Forward":    "#E63946",
        }

        comp_df = ldf.copy()
        comp_df["position_group"] = comp_df["primary_position"].map(POS_GROUPS).fillna("Other")

        chart_left, chart_right = st.columns(2)

        # Left — Position group breakdown (count + avg overall)
        with chart_left:
            grp = (comp_df.groupby("position_group", as_index=False)
                   .agg(count=("short_name", "count"),
                        avg_overall=("overall", "mean"))
                   .sort_values("count", ascending=False))
            grp["avg_overall"] = grp["avg_overall"].round(1)
            grp["color"] = grp["position_group"].map(GROUP_COLORS).fillna("#8892A4")

            fig_pos = go.Figure()
            fig_pos.add_trace(go.Bar(
                x=grp["position_group"], y=grp["count"],
                marker_color=grp["color"],
                text=grp["count"], textposition="outside",
                textfont=dict(color="#FAFAFA"),
                customdata=grp["avg_overall"],
                hovertemplate="%{x}<br>Players: %{y}<br>Avg Overall: %{customdata}<extra></extra>",
                name="",
            ))
            fig_pos.update_layout(
                **_LAYOUT,
                title="Players by Position Group",
                xaxis=dict(gridcolor="#2A2F3E"),
                yaxis=dict(gridcolor="#2A2F3E", title="Number of Players"),
                height=340,
            )
            st.plotly_chart(fig_pos, use_container_width=True)

        # Right — League distribution with avg overall per league
        with chart_right:
            if sel_league == "All":
                lgdist = (comp_df.groupby("league_name", as_index=False)
                          .agg(count=("short_name", "count"),
                               avg_overall=("overall", "mean"))
                          .sort_values("count", ascending=False)
                          .head(10))
                lgdist["avg_overall"] = lgdist["avg_overall"].round(1)
                LEAGUE_COLS = {
                    "Premier League": "#7B2FBE", "La Liga": "#E63946",
                    "Bundesliga": "#D20515", "Serie A": "#0066B2", "Ligue 1": "#3D6FBE",
                }
                lgdist["color"] = lgdist["league_name"].map(LEAGUE_COLS).fillna("#8892A4")

                fig_lg = go.Figure()
                fig_lg.add_trace(go.Bar(
                    x=lgdist["league_name"], y=lgdist["count"],
                    marker_color=lgdist["color"],
                    text=lgdist["count"], textposition="outside",
                    textfont=dict(color="#FAFAFA"),
                    customdata=lgdist["avg_overall"],
                    hovertemplate="%{x}<br>Players: %{y}<br>Avg Overall: %{customdata}<extra></extra>",
                    name="",
                ))
                fig_lg.update_layout(
                    **_LAYOUT,
                    title="Players by League",
                    xaxis=dict(gridcolor="#2A2F3E", tickangle=-20),
                    yaxis=dict(gridcolor="#2A2F3E", title="Number of Players"),
                    height=340,
                )
                st.plotly_chart(fig_lg, use_container_width=True)
            else:
                # Single league selected — show nationality breakdown instead
                nat = (comp_df.groupby("nationality_name", as_index=False)
                       .agg(count=("short_name","count"),
                            avg_overall=("overall","mean"))
                       .sort_values("count", ascending=False)
                       .head(10))
                nat["avg_overall"] = nat["avg_overall"].round(1)
                fig_nat = go.Figure(go.Bar(
                    x=nat["nationality_name"], y=nat["count"],
                    marker_color="#00D4AA",
                    text=nat["count"], textposition="outside",
                    textfont=dict(color="#FAFAFA"),
                    customdata=nat["avg_overall"],
                    hovertemplate="%{x}<br>Players: %{y}<br>Avg Overall: %{customdata}<extra></extra>",
                    name="",
                ))
                fig_nat.update_layout(
                    **_LAYOUT,
                    title=f"Top Nationalities in {sel_league}",
                    xaxis=dict(gridcolor="#2A2F3E", tickangle=-25),
                    yaxis=dict(gridcolor="#2A2F3E", title="Number of Players"),
                    height=340,
                )
                st.plotly_chart(fig_nat, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PLAYER PROFILE
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown('<div class="section-title">Player Profile</div>', unsafe_allow_html=True)

    if df.empty:
        st.info("No players match the current filters.")
    else:
        player_names = df.sort_values("overall", ascending=False)["short_name"].tolist()
        selected_name = st.selectbox("Search player", player_names, key="profile_player")
        player = df[df["short_name"] == selected_name].iloc[0]

        # ── Header card ──
        col_img, col_info = st.columns([1, 3])
        with col_img:
            face_url = player.get("player_face_url", "")
            img_bytes = fetch_player_image(str(face_url)) if face_url and str(face_url).startswith("http") else None
            if img_bytes:
                st.image(img_bytes, width=140)
            else:
                initials = "".join(w[0].upper() for w in str(player.get("short_name", "?")).split(".")[-1].split() if w)[:2]
                st.markdown(
                    f'<div style="width:140px;height:140px;background:#1A1F2E;border-radius:50%;'
                    f'border:2px solid #00D4AA;display:flex;align-items:center;justify-content:center;'
                    f'font-size:2.2rem;font-weight:700;color:#00D4AA;">{initials}</div>',
                    unsafe_allow_html=True,
                )

        with col_info:
            st.markdown(f"### {player['long_name']}")
            badges = ""
            if pd.notna(player.get("primary_position")):
                badges += f'<span class="badge badge-green">{player["primary_position"]}</span>'
            if pd.notna(player.get("league_name")):
                badges += f'<span class="badge badge-blue">{player["league_name"]}</span>'
            if pd.notna(player.get("club_name")):
                badges += f'<span class="badge badge-orange">{player["club_name"]}</span>'
            st.markdown(badges, unsafe_allow_html=True)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Overall", int(player["overall"]))
            m2.metric("Potential", int(player["potential"]))
            m3.metric("Growth", f"+{int(player['growth_potential'])}")
            m4.metric("Age", int(player["age"]))
            m5.metric("Nation", player.get("nationality_name", "—"))

            detail_cols = st.columns(4)
            detail_cols[0].metric("Height", f"{int(player['height_cm'])} cm" if pd.notna(player.get('height_cm')) else "—")
            detail_cols[1].metric("Weight", f"{int(player['weight_kg'])} kg" if pd.notna(player.get('weight_kg')) else "—")
            detail_cols[2].metric("Preferred Foot", player.get("preferred_foot", "—"))
            detail_cols[3].metric("Work Rate", player.get("work_rate", "—"))

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Attributes row ──
        a1, a2, a3, a4, a5, a6 = st.columns(6)
        for col_ui, attr, label in zip(
            [a1, a2, a3, a4, a5, a6],
            RADAR_ATTRS,
            ["Pace", "Shooting", "Passing", "Dribbling", "Defending", "Physicality"],
        ):
            val = player.get(attr)
            col_ui.metric(label, int(val) if pd.notna(val) else "—")

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Charts row ──
        ch1, ch2 = st.columns(2)
        with ch1:
            player_radar = [{
                "name": player["short_name"],
                "data": {a: player.get(a, 0) for a in RADAR_ATTRS},
                "color": BRAND_COLORS["primary"],
            }]
            st.plotly_chart(radar_chart(player_radar, "Attribute Radar"), use_container_width=True)

        with ch2:
            fit_scores = compute_league_fit(player)
            st.plotly_chart(league_fit_chart(fit_scores), use_container_width=True)
            best = max(fit_scores, key=fit_scores.get)
            st.markdown(
                f'<div class="info-box">🏆 Best league fit: <strong style="color:#00D4AA">{best}</strong>'
                f'<br>{LEAGUE_DESCRIPTIONS[best]}</div>',
                unsafe_allow_html=True,
            )

        # ── Growth + Versatility ──
        g1, g2 = st.columns(2)
        with g1:
            curve = growth_curve(int(player["age"]), float(player["overall"]), float(player["potential"]))
            st.plotly_chart(
                growth_projection_chart(curve, player["short_name"],
                                        int(player["age"]), float(player["overall"]), float(player["potential"])),
                use_container_width=True,
            )
        with g2:
            st.plotly_chart(versatility_chart(player), use_container_width=True)

        # ── Career arc ──
        st.markdown('<div class="section-title">Career Arc</div>', unsafe_allow_html=True)
        player_career = career_df_full[career_df_full["player_id"] == player["player_id"]]
        st.plotly_chart(career_arc_chart(player_career, player["short_name"]), use_container_width=True)

        # ── Contract & Financial ──
        st.markdown('<div class="section-title">Contract & Financials</div>', unsafe_allow_html=True)
        fin1, fin2, fin3, fin4 = st.columns(4)
        fin1.metric("Market Value", fmt_eur(player.get("value_eur", 0)))
        fin2.metric("Weekly Wage", fmt_wage(player.get("wage_eur", 0)))
        fin3.metric("Release Clause", fmt_eur(player.get("release_clause_eur", 0)))
        fin4.metric("Contract Until", str(int(player["contract_year"])) if pd.notna(player.get("contract_year")) else "—")

        # ── AI Scouting Report ──
        st.markdown('<div class="section-title">AI Scouting Report</div>', unsafe_allow_html=True)
        try:
            from ai_scout import generate_scouting_report, get_report_usage
            used, limit = get_report_usage()
            remaining = limit - used

            ui_left, ui_right = st.columns([3, 1])
            with ui_right:
                pct = used / limit
                bar_color = "#00D4AA" if pct < 0.8 else "#FF6B35" if pct < 0.95 else "#E63946"
                st.markdown(
                    f'<div style="text-align:right;font-size:0.8rem;color:#8892A4;margin-bottom:4px;">'
                    f'Reports used: <strong style="color:{bar_color}">{used:,} / {limit:,}</strong></div>'
                    f'<div style="background:#2A2F3E;border-radius:4px;height:6px;">'
                    f'<div style="background:{bar_color};width:{min(pct*100,100):.1f}%;height:6px;border-radius:4px;"></div></div>',
                    unsafe_allow_html=True,
                )

            with ui_left:
                report_key = f"report_{player['player_id']}"
                if remaining <= 0:
                    st.warning("The 1,000 AI report limit for this installation has been reached.")
                elif st.button("🤖 Generate AI Scouting Report", key="ai_report_btn"):
                    with st.spinner("Generating report with Claude AI…"):
                        player_dict = {k: (None if str(v) == "nan" else v) for k, v in player.items()}
                        report_text = generate_scouting_report(player_dict)
                        if report_text:
                            st.session_state[report_key] = report_text
                        else:
                            st.warning("Report limit reached.")

            if st.session_state.get(report_key):
                st.markdown(
                    f'<div class="player-card" style="white-space:pre-line;font-size:0.9rem;line-height:1.7;">'
                    f'{st.session_state[report_key]}</div>',
                    unsafe_allow_html=True,
                )

        except ImportError:
            st.info("Install the `anthropic` package to enable AI scouting reports.")
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Could not generate report: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — YOUNG TALENT
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.markdown('<div class="section-title">Young Talent & Growth Potential</div>', unsafe_allow_html=True)

    yt_max_age = st.slider("Maximum age for young talent view", 16, 30, 23, key="yt_age")
    yt_min_potential = st.slider("Minimum potential rating", 60, 95, 75, key="yt_pot")

    yt_df = df[(df["age"] <= yt_max_age) & (df["potential"] >= yt_min_potential)].copy()

    if yt_df.empty:
        st.info("No young players match the criteria. Try adjusting the sliders.")
    else:
        st.caption(f"Showing {len(yt_df):,} players aged ≤{yt_max_age} with potential ≥{yt_min_potential}")

        st.plotly_chart(age_vs_potential_scatter(yt_df), use_container_width=True)

        st.markdown('<div class="section-title">Top Rising Stars</div>', unsafe_allow_html=True)
        stars_df = yt_df.sort_values("growth_potential", ascending=False).head(20)

        for _, row in stars_df.iterrows():
            c1, c2, c3, c4, c5, c6 = st.columns([3, 1, 1, 1, 1, 2])
            c1.markdown(f"**{row['short_name']}**  \n{row.get('club_name','—')} · {row.get('league_name','—')}")
            c2.metric("Age", int(row["age"]))
            c3.metric("OVR", int(row["overall"]))
            c4.metric("POT", int(row["potential"]))
            c5.metric("Growth", f"+{int(row['growth_potential'])}")
            c6.markdown(f"*{row.get('primary_position','—')} · {row.get('nationality_name','—')}*")

        st.markdown('<div class="section-title">Growth Projection — Select a Player</div>', unsafe_allow_html=True)
        yt_names = yt_df.sort_values("growth_potential", ascending=False)["short_name"].tolist()
        yt_selected = st.selectbox("Choose player for growth projection", yt_names, key="yt_projection")
        yt_player = yt_df[yt_df["short_name"] == yt_selected].iloc[0]
        curve = growth_curve(int(yt_player["age"]), float(yt_player["overall"]), float(yt_player["potential"]))
        st.plotly_chart(
            growth_projection_chart(curve, yt_player["short_name"],
                                    int(yt_player["age"]), float(yt_player["overall"]), float(yt_player["potential"])),
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — COMPARE PLAYERS
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.markdown('<div class="section-title">Compare Players Side by Side</div>', unsafe_allow_html=True)

    if df.empty:
        st.info("No players match the current filters.")
    else:
        all_names = df.sort_values("overall", ascending=False)["short_name"].tolist()
        cmp_defaults = all_names[:3] if len(all_names) >= 3 else all_names

        cmp_cols = st.columns(3)
        selected_players = []
        colors = [BRAND_COLORS["primary"], "#FF6B35", "#E63946"]
        for i, (col_ui, color) in enumerate(zip(cmp_cols, colors)):
            with col_ui:
                default_idx = min(i, len(all_names) - 1)
                name = st.selectbox(f"Player {i+1}", all_names, index=default_idx, key=f"cmp_{i}")
                selected_players.append((name, color))

        st.markdown("<hr>", unsafe_allow_html=True)

        # Build radar data
        radar_data = []
        players_data = []
        for name, color in selected_players:
            rows = df[df["short_name"] == name]
            if not rows.empty:
                p = rows.iloc[0]
                radar_data.append({
                    "name": name,
                    "data": {a: p.get(a, 0) for a in RADAR_ATTRS},
                    "color": color,
                })
                players_data.append((name, color, p))

        if radar_data:
            st.plotly_chart(radar_chart(radar_data, "Attribute Comparison"), use_container_width=True)

        # Stat comparison table
        st.markdown('<div class="section-title">Head-to-Head Stats</div>', unsafe_allow_html=True)
        stat_rows = [
            ("Overall", "overall"), ("Potential", "potential"), ("Age", "age"),
            ("Growth", "growth_potential"),
            ("Pace", "pace"), ("Shooting", "shooting"), ("Passing", "passing"),
            ("Dribbling", "dribbling"), ("Defending", "defending"), ("Physicality", "physic"),
            ("Weak Foot ★", "weak_foot"), ("Skill Moves ★", "skill_moves"),
            ("Market Value", "value_eur"), ("Weekly Wage", "wage_eur"),
        ]
        headers = ["Attribute"] + [n for n, _ in selected_players]
        rows_out = []
        for label, attr in stat_rows:
            row = [label]
            vals = []
            for name, _ in selected_players:
                p_rows = df[df["short_name"] == name]
                v = p_rows.iloc[0].get(attr, None) if not p_rows.empty else None
                vals.append(v)
            for v in vals:
                if attr in ("value_eur", "wage_eur"):
                    row.append(fmt_eur(v) if attr == "value_eur" else fmt_wage(v))
                elif pd.isna(v) if not isinstance(v, str) else False:
                    row.append("—")
                else:
                    row.append(int(v) if isinstance(v, (int, float)) and not pd.isna(v) else str(v))
            rows_out.append(row)

        cmp_tbl = pd.DataFrame(rows_out, columns=headers)
        st.dataframe(cmp_tbl, use_container_width=True, hide_index=True)

        # League fit comparison
        st.markdown('<div class="section-title">League Fit Comparison</div>', unsafe_allow_html=True)
        if players_data:
            import plotly.graph_objects as go
            leagues = list(["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"])
            fig_lf = go.Figure()
            for name, color, p in players_data:
                scores = compute_league_fit(p)
                fig_lf.add_trace(go.Bar(
                    name=name, x=leagues,
                    y=[scores.get(l, 0) for l in leagues],
                    marker_color=color,
                ))
            fig_lf.update_layout(
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#FAFAFA"),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(gridcolor="#2A2F3E"),
                yaxis=dict(gridcolor="#2A2F3E", title="Fit Score"),
                margin=dict(l=20, r=20, t=20, b=20),
                height=320,
            )
            st.plotly_chart(fig_lf, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — HIDDEN GEMS
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.markdown('<div class="section-title">Hidden Gems — Undervalued Players</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Players with high potential, large growth headroom, '
        'low wages, and low international reputation. These are scouts\' best-kept secrets.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    gem_df = df.copy()
    gem_df["gem_score"] = compute_hidden_gem_score(gem_df)

    col_gs1, col_gs2, col_gs3 = st.columns(3)
    with col_gs1:
        min_gem = st.slider("Minimum Gem Score", 0, 100, 60, key="gem_min")
    with col_gs2:
        max_overall = st.slider("Maximum Current Overall", 60, 99, 82, key="gem_ovr")
    with col_gs3:
        min_pot = st.slider("Minimum Potential", 70, 99, 78, key="gem_pot")

    gem_filtered = gem_df[
        (gem_df["gem_score"] >= min_gem) &
        (gem_df["overall"] <= max_overall) &
        (gem_df["potential"] >= min_pot)
    ].sort_values("gem_score", ascending=False)

    if gem_filtered.empty:
        st.info("No hidden gems match those criteria. Try lowering the minimum gem score.")
    else:
        st.caption(f"{len(gem_filtered):,} gems found")
        st.plotly_chart(hidden_gems_scatter(gem_filtered), use_container_width=True)

        st.markdown('<div class="section-title">Top Hidden Gems</div>', unsafe_allow_html=True)
        gem_display = gem_filtered.head(30)[[
            "short_name", "age", "primary_position", "club_name", "league_name",
            "nationality_name", "overall", "potential", "growth_potential",
            "wage_eur_k", "international_reputation", "gem_score",
        ]].rename(columns={
            "short_name": "Player", "age": "Age", "primary_position": "Pos",
            "club_name": "Club", "league_name": "League", "nationality_name": "Nation",
            "overall": "OVR", "potential": "POT", "growth_potential": "Growth",
            "wage_eur_k": "Wage €K/wk", "international_reputation": "Rep ★",
            "gem_score": "Gem Score",
        })
        st.dataframe(gem_display, use_container_width=True, hide_index=True,
                     column_config={
                         "Gem Score": st.column_config.ProgressColumn("Gem Score", min_value=0, max_value=100, format="%.1f"),
                         "OVR": st.column_config.ProgressColumn("OVR", min_value=0, max_value=99, format="%d"),
                         "POT": st.column_config.ProgressColumn("POT", min_value=0, max_value=99, format="%d"),
                     })


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — SQUAD ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[6]:
    st.markdown('<div class="section-title">Squad Analyzer</div>', unsafe_allow_html=True)

    clubs_available = sorted(df_full["club_name"].dropna().unique().tolist())
    selected_club = st.selectbox("Select a club", clubs_available, key="squad_club")

    club_df = df_full[df_full["club_name"] == selected_club].copy()

    if club_df.empty:
        st.info("No players found for that club.")
    else:
        sq1, sq2, sq3, sq4 = st.columns(4)
        sq1.metric("Squad Size", len(club_df))
        sq2.metric("Avg Overall", f"{club_df['overall'].mean():.1f}")
        sq3.metric("Avg Age", f"{club_df['age'].mean():.1f}")
        sq4.metric("Avg Potential", f"{club_df['potential'].mean():.1f}")

        st.plotly_chart(squad_map_chart(club_df), use_container_width=True)

        # Positional gaps
        st.markdown('<div class="section-title">Positional Strength & Gaps</div>', unsafe_allow_html=True)
        pos_groups = {
            "Goalkeeper": ["GK"],
            "Defence": ["CB", "LB", "RB", "LWB", "RWB", "LCB", "RCB"],
            "Midfield": ["CDM", "CM", "CAM", "LDM", "RDM", "LCM", "RCM", "LM", "RM", "LAM", "RAM"],
            "Attack": ["ST", "CF", "LW", "RW", "LF", "RF", "LS", "RS"],
        }
        for group, positions in pos_groups.items():
            group_players = club_df[club_df["primary_position"].isin(positions)]
            if not group_players.empty:
                best_ovr = group_players["overall"].max()
                count = len(group_players)
                color = "#00D4AA" if best_ovr >= 75 else "#FF6B35"
                st.markdown(
                    f'<span class="badge" style="background:{color}22;color:{color};border:1px solid {color}44">'
                    f'{group}</span> {count} players · Best OVR: <strong>{int(best_ovr)}</strong>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<span class="badge badge-orange">{group}</span> ⚠️ No players found — potential gap',
                    unsafe_allow_html=True,
                )

        # Full squad table
        st.markdown('<div class="section-title">Full Squad</div>', unsafe_allow_html=True)
        squad_tbl = club_df.sort_values("overall", ascending=False)[[
            "short_name", "age", "primary_position", "nationality_name",
            "overall", "potential", "growth_potential", "wage_eur_k", "contract_year",
        ]].rename(columns={
            "short_name": "Player", "age": "Age", "primary_position": "Pos",
            "nationality_name": "Nation", "overall": "OVR", "potential": "POT",
            "growth_potential": "Growth", "wage_eur_k": "Wage €K/wk",
            "contract_year": "Contract Until",
        })
        st.dataframe(squad_tbl, use_container_width=True, hide_index=True,
                     column_config={
                         "OVR": st.column_config.ProgressColumn("OVR", min_value=0, max_value=99, format="%d"),
                         "POT": st.column_config.ProgressColumn("POT", min_value=0, max_value=99, format="%d"),
                     })


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — CONTRACT TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[7]:
    st.markdown('<div class="section-title">Contract Tracker & Free Agent Finder</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Players whose contracts expire soon can be signed at a discount '
        'or for free. Players expiring in 2023 can be approached for pre-contract talks now.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    ct_df = df.dropna(subset=["contract_year"]).copy()

    cc1, cc2 = st.columns(2)
    with cc1:
        max_contract_year = st.selectbox(
            "Show contracts expiring by end of",
            [2023, 2024, 2025, 2026],
            index=1,
            key="contract_year_filter",
        )
    with cc2:
        ct_min_overall = st.slider("Minimum Overall", 60, 90, 70, key="ct_ovr")

    ct_filtered = ct_df[
        (ct_df["contract_year"] <= max_contract_year) &
        (ct_df["overall"] >= ct_min_overall)
    ].sort_values(["contract_year", "overall"], ascending=[True, False])

    if ct_filtered.empty:
        st.info("No players match those contract criteria.")
    else:
        # Summary metrics
        cm1, cm2, cm3 = st.columns(3)
        cm1.metric("Expiring Contracts", len(ct_filtered))
        free_agents = ct_filtered[ct_filtered["contract_year"] <= 2023]
        cm2.metric("Free Agents (2023)", len(free_agents))
        cm3.metric("Avg Overall", f"{ct_filtered['overall'].mean():.1f}")

        # Chart by expiry year
        import plotly.express as px
        year_counts = ct_filtered.groupby("contract_year").size().reset_index(name="count")
        fig_ct = px.bar(
            year_counts, x="contract_year", y="count",
            color="count", color_continuous_scale="Teal",
            labels={"contract_year": "Contract Expires", "count": "Players"},
            title="Contracts Expiring by Year",
            text="count",
        )
        fig_ct.update_traces(textposition="outside")
        fig_ct.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FAFAFA"), showlegend=False,
            xaxis=dict(gridcolor="#2A2F3E", type="category"),
            yaxis=dict(gridcolor="#2A2F3E"),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_ct, use_container_width=True)

        ct_display = ct_filtered[[
            "short_name", "age", "primary_position", "club_name", "league_name",
            "nationality_name", "overall", "potential", "wage_eur_k",
            "value_eur_m", "contract_year",
        ]].rename(columns={
            "short_name": "Player", "age": "Age", "primary_position": "Pos",
            "club_name": "Club", "league_name": "League", "nationality_name": "Nation",
            "overall": "OVR", "potential": "POT",
            "wage_eur_k": "Wage €K/wk", "value_eur_m": "Value €M",
            "contract_year": "Contract Until",
        })
        st.dataframe(ct_display, use_container_width=True, hide_index=True,
                     column_config={
                         "OVR": st.column_config.ProgressColumn("OVR", min_value=0, max_value=99, format="%d"),
                         "POT": st.column_config.ProgressColumn("POT", min_value=0, max_value=99, format="%d"),
                         "Value €M": st.column_config.NumberColumn("Value €M", format="€%.1fM"),
                     })


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8 — BUDGET OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[8]:
    st.markdown('<div class="section-title">Budget Optimizer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Set your maximum wage or transfer budget and find the best-value '
        'players within it. Value Score = overall rating ÷ weekly wage × 10.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    bo1, bo2 = st.columns(2)
    with bo1:
        max_wage_budget = st.number_input(
            "Maximum weekly wage (€)", min_value=1000, max_value=500_000,
            value=50_000, step=1_000, key="bo_wage",
        )
    with bo2:
        max_value_budget = st.number_input(
            "Maximum transfer value (€M)", min_value=0.1, max_value=200.0,
            value=20.0, step=0.5, key="bo_value",
        )

    bo_min_ovr = st.slider("Minimum Overall", 60, 90, 70, key="bo_ovr")

    bo_df = df[
        (df["wage_eur"] <= max_wage_budget) &
        (df["value_eur"] <= max_value_budget * 1_000_000) &
        (df["overall"] >= bo_min_ovr) &
        (df["wage_eur"] > 0)
    ].copy()
    bo_df["value_score"] = compute_value_score(bo_df)

    if bo_df.empty:
        st.info("No players found within that budget. Try increasing the limits.")
    else:
        bm1, bm2, bm3 = st.columns(3)
        bm1.metric("Players in Budget", len(bo_df))
        bm2.metric("Best Value Score", f"{bo_df['value_score'].max():.1f}")
        bm3.metric("Highest Overall", int(bo_df["overall"].max()))

        # Scatter: value vs overall
        st.plotly_chart(wage_vs_overall_scatter(bo_df), use_container_width=True)

        # Table
        st.markdown('<div class="section-title">Best Value Players</div>', unsafe_allow_html=True)
        bo_display = bo_df.sort_values("value_score", ascending=False).head(50)[[
            "short_name", "age", "primary_position", "club_name", "league_name",
            "overall", "potential", "wage_eur_k", "value_eur_m", "value_score",
        ]].rename(columns={
            "short_name": "Player", "age": "Age", "primary_position": "Pos",
            "club_name": "Club", "league_name": "League",
            "overall": "OVR", "potential": "POT",
            "wage_eur_k": "Wage €K/wk", "value_eur_m": "Value €M",
            "value_score": "Value Score",
        })
        st.dataframe(bo_display, use_container_width=True, hide_index=True,
                     column_config={
                         "OVR": st.column_config.ProgressColumn("OVR", min_value=0, max_value=99, format="%d"),
                         "POT": st.column_config.ProgressColumn("POT", min_value=0, max_value=99, format="%d"),
                         "Value Score": st.column_config.ProgressColumn("Value Score", min_value=0, max_value=100, format="%.1f"),
                         "Value €M": st.column_config.NumberColumn("Value €M", format="€%.1fM"),
                     })


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 9 — NATIONALITY MAP
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[9]:
    st.markdown('<div class="section-title">Player Origins — World Map</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Hover over a country to see how many players in the current '
        'filtered view come from there, along with average overall and potential ratings. '
        'Use the league/position filters in the sidebar to explore scouting regions.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    map_df = df if not df.empty else df_full
    st.plotly_chart(nationality_map(map_df), use_container_width=True)

    # Top 20 nations table
    st.markdown('<div class="section-title">Top 20 Nations by Player Count</div>', unsafe_allow_html=True)
    top_nations = (
        map_df.groupby("nationality_name", as_index=False)
        .agg(
            Players=("short_name", "count"),
            Avg_Overall=("overall", "mean"),
            Avg_Potential=("potential", "mean"),
            Top_Player_Overall=("overall", "max"),
        )
        .rename(columns={
            "nationality_name": "Nation",
            "Avg_Overall": "Avg OVR",
            "Avg_Potential": "Avg POT",
            "Top_Player_Overall": "Best OVR",
        })
        .sort_values("Players", ascending=False)
        .head(20)
    )
    top_nations["Avg OVR"] = top_nations["Avg OVR"].round(1)
    top_nations["Avg POT"] = top_nations["Avg POT"].round(1)
    st.dataframe(top_nations, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 10 — TRANSFER SCOUT AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def _ts_reset():
    st.session_state.ts_phase = "form"
    st.session_state.ts_thread_id = None
    st.session_state.ts_interrupt = None
    st.session_state.ts_filename = None


with tabs[10]:
    st.markdown('<div class="section-title">Transfer Scout Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Describe what you\'re looking for. The agent searches the FIFA 23 '
        'dataset, scores candidates, fetches recent transfer news, and drafts a shortlist report '
        'for your review before saving.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Phase: FORM ───────────────────────────────────────────────────────────

    if st.session_state.ts_phase == "form":
        st.markdown('<div class="section-title">Scouting Brief</div>', unsafe_allow_html=True)

        with st.form("ts_brief_form"):
            ts_col1, ts_col2 = st.columns(2)

            with ts_col1:
                ts_position = st.text_input(
                    "Position",
                    placeholder="e.g. ST, CM, CB, LW",
                    help="Use FIFA position codes. The agent matches players who can play this position.",
                )
                ts_max_age = st.slider("Max Age", min_value=15, max_value=45, value=28)
                ts_min_overall = st.slider("Min Overall", min_value=40, max_value=99, value=75)

            with ts_col2:
                ts_budget = st.number_input(
                    "Transfer Budget (€) — leave 0 to ignore",
                    min_value=0,
                    max_value=250_000_000,
                    value=0,
                    step=1_000_000,
                    format="%d",
                )
                ts_league = st.selectbox("Target League", ["Any"] + TOP5_LEAGUES)
                ts_notes = st.text_area(
                    "Scout Notes",
                    placeholder="e.g. Must be strong in the air, two-footed preferred…",
                    height=108,
                )

            ts_submitted = st.form_submit_button("🔍 Run Scout", use_container_width=True)

        if ts_submitted:
            if not ts_position.strip():
                st.error("Position is required (e.g. ST, CM, CB).")
            else:
                brief = {
                    "position":     ts_position.strip().upper(),
                    "max_age":      int(ts_max_age),
                    "min_overall":  int(ts_min_overall),
                    "budget_eur":   float(ts_budget) if ts_budget > 0 else None,
                    "target_league": ts_league if ts_league != "Any" else None,
                    "notes":        ts_notes.strip(),
                }
                thread_id = str(uuid.uuid4())
                st.session_state.ts_thread_id = thread_id
                config = {"configurable": {"thread_id": thread_id}}

                with st.spinner("Running Transfer Scout Agent — this may take 20–30 seconds…"):
                    try:
                        result = ts_graph.invoke({"brief": brief}, config)
                        interrupts = result.get("__interrupt__")
                        if interrupts:
                            st.session_state.ts_interrupt = interrupts[0].value
                            st.session_state.ts_phase = "review"
                            st.rerun()
                        elif not result.get("scored_players"):
                            st.info("No players matched your brief. Try relaxing the filters.")
                        else:
                            st.warning("Agent completed without reaching the review stage.")
                    except ValueError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Agent error: {e}")

    # ── Phase: REVIEW ─────────────────────────────────────────────────────────

    elif st.session_state.ts_phase == "review":
        interrupt_val = st.session_state.ts_interrupt

        st.markdown('<div class="section-title">Draft Report — Pending Scout Review</div>', unsafe_allow_html=True)
        st.markdown(interrupt_val.get("draft_report", ""))

        st.markdown('<div class="section-title">Scored Players</div>', unsafe_allow_html=True)
        st.caption(
            "Fit Score (0–100): how well each player matches your specific brief. "
            "Weighted: 50% overall rating · 30% age (younger scores higher) · "
            "20% value for money relative to other candidates. "
            "Use it to rank candidates against each other — not as an absolute quality measure."
        )
        scored = interrupt_val.get("scored_players", [])
        if scored:
            scored_df = pd.DataFrame(scored)
            display_cols = ["short_name", "primary_position", "club_name", "league_name",
                            "age", "overall", "potential", "fit_score"]
            available = [c for c in display_cols if c in scored_df.columns]
            scored_df = scored_df[available].rename(columns={
                "short_name": "Player", "primary_position": "Pos",
                "club_name": "Club", "league_name": "League",
                "age": "Age", "overall": "OVR", "potential": "POT", "fit_score": "Fit Score",
            })
            st.dataframe(
                scored_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "OVR":       st.column_config.ProgressColumn("OVR", min_value=0, max_value=99, format="%d"),
                    "POT":       st.column_config.ProgressColumn("POT", min_value=0, max_value=99, format="%d"),
                    "Fit Score": st.column_config.ProgressColumn("Fit Score", min_value=0, max_value=100, format="%.1f"),
                },
            )

        st.markdown("<hr>", unsafe_allow_html=True)
        rev_col1, rev_col2 = st.columns(2)

        with rev_col1:
            if st.button("✅ Approve & Save", use_container_width=True, type="primary"):
                config = {"configurable": {"thread_id": st.session_state.ts_thread_id}}
                with st.spinner("Saving report…"):
                    try:
                        ts_graph.invoke(Command(resume={"approved": True}), config)
                        # Identify the saved file as the newest entry in reports/
                        reports_dir = os.path.join(os.path.dirname(__file__), "reports")
                        files = sorted(os.listdir(reports_dir)) if os.path.isdir(reports_dir) else []
                        st.session_state.ts_filename = files[-1] if files else "report.md"
                        st.session_state.ts_phase = "done"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Save failed: {e}")

        with rev_col2:
            if st.button("🔄 Adjust Filters", use_container_width=True):
                _ts_reset()
                st.rerun()

    # ── Phase: DONE ───────────────────────────────────────────────────────────

    elif st.session_state.ts_phase == "done":
        ts_filename = st.session_state.ts_filename or "report.md"
        st.success(f"Report saved: **reports/{ts_filename}**")
        st.markdown(
            '<div class="info-box">The approved report has been written to the '
            '<code>reports/</code> folder in the project directory.</div>',
            unsafe_allow_html=True,
        )
        if st.button("🔍 Scout Again"):
            _ts_reset()
            st.rerun()
