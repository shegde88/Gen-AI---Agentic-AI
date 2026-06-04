import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from league_fit import LEAGUE_DESCRIPTIONS


def _color_to_rgba(color: str, alpha: float = 0.15) -> str:
    """Convert any color string to rgba() for Plotly fillcolor."""
    color = color.strip()
    if color.startswith("rgba"):
        return color
    if color.startswith("rgb("):
        return color.replace("rgb(", "rgba(").replace(")", f",{alpha})")
    if color.startswith("#"):
        h = color.lstrip("#")
        if len(h) == 6:
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"
    return f"rgba(0,212,170,{alpha})"

RADAR_ATTRS = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
RADAR_LABELS = ["Pace", "Shooting", "Passing", "Dribbling", "Defending", "Physicality"]

BRAND_COLORS = {
    "primary": "#00D4AA",
    "secondary": "#1E3A5F",
    "accent": "#FF6B35",
    "bg": "#0E1117",
    "card": "#1A1F2E",
    "text": "#FAFAFA",
    "muted": "#8892A4",
}

LEAGUE_COLORS = {
    "Premier League": "#3D195B",
    "La Liga": "#EE2523",
    "Bundesliga": "#D20515",
    "Serie A": "#0066B2",
    "Ligue 1": "#003189",
}

POSITION_COLORS = {
    "GK": "#F4A261",
    "CB": "#2A9D8F", "LB": "#2A9D8F", "RB": "#2A9D8F",
    "LWB": "#2A9D8F", "RWB": "#2A9D8F",
    "CDM": "#457B9D", "CM": "#457B9D", "CAM": "#457B9D",
    "LM": "#457B9D", "RM": "#457B9D",
    "LW": "#E63946", "RW": "#E63946",
    "ST": "#E63946", "CF": "#E63946", "LF": "#E63946", "RF": "#E63946",
    "LS": "#E63946", "RS": "#E63946",
}

_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=BRAND_COLORS["text"], family="Inter, sans-serif"),
    margin=dict(l=20, r=20, t=40, b=20),
)


def _pos_color(pos: str) -> str:
    return POSITION_COLORS.get(str(pos).strip().upper(), BRAND_COLORS["primary"])


# ── Radar / Spider Chart ──────────────────────────────────────────────────────

def radar_chart(players: list[dict], title: str = "Player Attributes") -> go.Figure:
    """
    players: list of dicts with keys: name, data (dict of attr->value), color
    """
    fig = go.Figure()
    for p in players:
        vals = [float(p["data"].get(a, 0) or 0) for a in RADAR_ATTRS]
        vals_closed = vals + [vals[0]]
        labels_closed = RADAR_LABELS + [RADAR_LABELS[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=labels_closed,
            fill="toself",
            name=p["name"],
            line=dict(color=p.get("color", BRAND_COLORS["primary"]), width=2),
            fillcolor=_color_to_rgba(p.get("color", BRAND_COLORS["primary"])),
        ))
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(text=title, font=dict(size=16)),
        polar=dict(
            bgcolor=BRAND_COLORS["card"],
            radialaxis=dict(visible=True, range=[0, 100],
                            gridcolor="#2A2F3E", linecolor="#2A2F3E",
                            tickfont=dict(size=9, color=BRAND_COLORS["muted"])),
            angularaxis=dict(gridcolor="#2A2F3E", linecolor="#2A2F3E",
                             tickfont=dict(size=11)),
        ),
        showlegend=len(players) > 1,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ── League Fit Bar Chart ──────────────────────────────────────────────────────

def league_fit_chart(fit_scores: dict[str, float]) -> go.Figure:
    leagues = list(fit_scores.keys())
    scores = list(fit_scores.values())
    colors = [LEAGUE_COLORS.get(l, BRAND_COLORS["primary"]) for l in leagues]

    fig = go.Figure(go.Bar(
        x=scores, y=leagues, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{s:.1f}" for s in scores],
        textposition="outside",
        textfont=dict(color=BRAND_COLORS["text"]),
        hovertemplate="%{y}: %{x:.1f}<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title="League Fit Score",
        xaxis=dict(range=[0, 100], gridcolor="#2A2F3E", zeroline=False),
        yaxis=dict(gridcolor="#2A2F3E"),
        height=260,
    )
    return fig


# ── Growth Projection ─────────────────────────────────────────────────────────

def growth_projection_chart(curve_df: pd.DataFrame, player_name: str,
                             current_age: int, overall: float, potential: float) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=curve_df["age"], y=curve_df["projected_overall"],
        mode="lines+markers",
        name="Projected Rating",
        line=dict(color=BRAND_COLORS["primary"], width=3),
        marker=dict(size=6),
        hovertemplate="Age %{x}: %{y:.1f}<extra></extra>",
    ))

    fig.add_hline(y=potential, line_dash="dash",
                  line_color=BRAND_COLORS["accent"], opacity=0.7,
                  annotation_text=f"Potential ceiling: {int(potential)}",
                  annotation_position="top right")

    fig.add_vline(x=current_age, line_dash="dot",
                  line_color=BRAND_COLORS["muted"], opacity=0.6,
                  annotation_text="Now", annotation_position="top left")

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=f"{player_name} — Growth Projection",
        xaxis=dict(title="Age", gridcolor="#2A2F3E", dtick=1),
        yaxis=dict(title="Overall Rating", range=[
            max(40, overall - 5), min(99, potential + 5)
        ], gridcolor="#2A2F3E"),
        height=340,
    )
    return fig


# ── Age vs Potential Scatter ──────────────────────────────────────────────────

def age_vs_potential_scatter(df: pd.DataFrame) -> go.Figure:
    plot_df = df.dropna(subset=["age", "potential", "overall", "short_name"]).copy()
    plot_df["color_pos"] = plot_df["primary_position"].apply(_pos_color)
    plot_df["hover"] = (
        plot_df["short_name"] + "<br>"
        + plot_df["club_name"].fillna("") + " · " + plot_df["league_name"].fillna("") + "<br>"
        + "Overall: " + plot_df["overall"].astype(int).astype(str)
        + " | Potential: " + plot_df["potential"].astype(int).astype(str)
    )

    fig = px.scatter(
        plot_df, x="age", y="potential",
        color="primary_position",
        size="overall",
        size_max=18,
        hover_name="short_name",
        custom_data=["hover"],
        opacity=0.75,
        title="Age vs Potential",
        labels={"age": "Age", "potential": "Potential Rating", "primary_position": "Position"},
    )
    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        xaxis=dict(gridcolor="#2A2F3E"),
        yaxis=dict(gridcolor="#2A2F3E", range=[50, 100]),
        legend=dict(bgcolor="rgba(0,0,0,0)", title="Position"),
    )
    return fig


# ── Wage vs Overall Scatter ───────────────────────────────────────────────────

def wage_vs_overall_scatter(df: pd.DataFrame) -> go.Figure:
    plot_df = df[df["wage_eur"] > 0].dropna(subset=["wage_eur", "overall"]).copy()
    plot_df["hover"] = (
        plot_df["short_name"] + "<br>"
        + "Wage: €" + (plot_df["wage_eur"] / 1000).round(1).astype(str) + "k/wk<br>"
        + "Overall: " + plot_df["overall"].astype(int).astype(str)
    )

    fig = px.scatter(
        plot_df, x="wage_eur", y="overall",
        color="league_name",
        opacity=0.65,
        title="Wage vs Overall Rating",
        labels={"wage_eur": "Weekly Wage (EUR)", "overall": "Overall Rating", "league_name": "League"},
        hover_name="short_name",
        custom_data=["hover"],
        log_x=True,
    )
    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        xaxis=dict(gridcolor="#2A2F3E"),
        yaxis=dict(gridcolor="#2A2F3E"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ── Position Versatility Bar ──────────────────────────────────────────────────

POSITION_COL_MAP = {
    "GK": "gk", "ST": "st", "CF": "cf", "LW": "lw", "RW": "rw",
    "CAM": "cam", "CM": "cm", "CDM": "cdm", "LM": "lm", "RM": "rm",
    "LWB": "lwb", "RWB": "rwb", "LB": "lb", "RB": "rb", "CB": "cb",
    "LS": "ls", "RS": "rs", "LF": "lf", "RF": "rf",
    "LAM": "lam", "RAM": "ram", "LCM": "lcm", "RCM": "rcm",
    "LDM": "ldm", "RDM": "rdm", "LCB": "lcb", "RCB": "rcb",
}


def versatility_chart(player: pd.Series) -> go.Figure:
    positions_str = str(player.get("player_positions", ""))
    playable = [p.strip().upper() for p in positions_str.split(",") if p.strip()]

    rows = []
    for pos in playable:
        col = POSITION_COL_MAP.get(pos)
        if col and pd.notna(player.get(col)):
            rows.append({"position": pos, "rating": int(player[col])})

    if not rows:
        return go.Figure().update_layout(**_LAYOUT_DEFAULTS, title="Position Versatility")

    vdf = pd.DataFrame(rows).sort_values("rating", ascending=True)
    colors = [_pos_color(p) for p in vdf["position"]]

    fig = go.Figure(go.Bar(
        x=vdf["rating"], y=vdf["position"], orientation="h",
        marker=dict(color=colors),
        text=vdf["rating"], textposition="outside",
        textfont=dict(color=BRAND_COLORS["text"]),
    ))
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title="Position Versatility",
        xaxis=dict(range=[50, 100], gridcolor="#2A2F3E"),
        yaxis=dict(gridcolor="#2A2F3E"),
        height=max(180, len(rows) * 38),
    )
    return fig


# ── Career Arc Line Chart ─────────────────────────────────────────────────────

def career_arc_chart(career_df: pd.DataFrame, player_name: str) -> go.Figure:
    if career_df.empty:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT_DEFAULTS, title="No career history available")
        return fig

    attrs = [a for a in ["overall", "pace", "shooting", "passing", "dribbling", "defending", "physic"]
             if a in career_df.columns]
    attr_colors = {
        "overall": BRAND_COLORS["primary"],
        "pace": "#FF6B35",
        "shooting": "#E63946",
        "passing": "#2A9D8F",
        "dribbling": "#F4A261",
        "defending": "#457B9D",
        "physic": "#A8DADC",
    }

    fig = go.Figure()
    for attr in attrs:
        visible = True if attr == "overall" else "legendonly"
        fig.add_trace(go.Scatter(
            x=career_df["fifa_version"],
            y=career_df[attr],
            mode="lines+markers",
            name=attr.capitalize(),
            line=dict(color=attr_colors.get(attr, BRAND_COLORS["primary"]),
                      width=3 if attr == "overall" else 1.5),
            marker=dict(size=7 if attr == "overall" else 5),
            visible=visible,
            hovertemplate=f"FIFA %{{x}}: %{{y:.0f}}<extra>{attr}</extra>",
        ))

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=f"{player_name} — Career Arc (FIFA 15–23)",
        xaxis=dict(title="FIFA Version", tickmode="linear", dtick=1, gridcolor="#2A2F3E"),
        yaxis=dict(title="Rating", range=[40, 100], gridcolor="#2A2F3E"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=360,
    )
    return fig


# ── Nationality World Map ─────────────────────────────────────────────────────

def nationality_map(df: pd.DataFrame) -> go.Figure:
    map_df = (
        df.groupby("nationality_name", as_index=False)
        .agg(
            player_count=("short_name", "count"),
            avg_overall=("overall", "mean"),
            avg_potential=("potential", "mean"),
        )
    )
    map_df["avg_overall"] = map_df["avg_overall"].round(1)
    map_df["avg_potential"] = map_df["avg_potential"].round(1)
    map_df["hover"] = (
        map_df["nationality_name"] + "<br>"
        + "Players: " + map_df["player_count"].astype(str) + "<br>"
        + "Avg Overall: " + map_df["avg_overall"].astype(str) + "<br>"
        + "Avg Potential: " + map_df["avg_potential"].astype(str)
    )

    fig = px.choropleth(
        map_df,
        locations="nationality_name",
        locationmode="country names",
        color="player_count",
        color_continuous_scale="Teal",
        hover_name="nationality_name",
        custom_data=["hover"],
        title="Player Origins — World Map",
        labels={"player_count": "Players"},
    )
    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        geo=dict(
            bgcolor="rgba(0,0,0,0)",
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#2A2F3E",
            showland=True, landcolor="#1A1F2E",
            showocean=True, oceancolor="#0E1117",
            showlakes=False,
            showcountries=True, countrycolor="#2A2F3E",
            projection_type="natural earth",
        ),
        coloraxis_colorbar=dict(
            title=dict(text="Players", font=dict(color=BRAND_COLORS["text"])),
            tickfont=dict(color=BRAND_COLORS["text"]),
        ),
        height=480,
    )
    return fig


# ── Hidden Gems Scatter ───────────────────────────────────────────────────────

def hidden_gems_scatter(df: pd.DataFrame) -> go.Figure:
    plot_df = df.dropna(subset=["wage_eur", "potential", "gem_score"]).copy()
    plot_df = plot_df[plot_df["wage_eur"] > 0]
    plot_df["hover"] = (
        plot_df["short_name"] + "<br>"
        + plot_df["primary_position"].fillna("") + " · " + plot_df["club_name"].fillna("") + "<br>"
        + "Overall: " + plot_df["overall"].astype(int).astype(str)
        + " | Potential: " + plot_df["potential"].astype(int).astype(str) + "<br>"
        + "Wage: €" + (plot_df["wage_eur"] / 1000).round(1).astype(str) + "k/wk<br>"
        + "Gem Score: " + plot_df["gem_score"].astype(str)
    )

    fig = px.scatter(
        plot_df, x="wage_eur", y="potential",
        color="gem_score",
        color_continuous_scale="Teal",
        size="growth_potential",
        size_max=20,
        opacity=0.75,
        title="Hidden Gems — Potential vs Wage",
        labels={"wage_eur": "Weekly Wage (EUR)", "potential": "Potential",
                "gem_score": "Gem Score"},
        hover_name="short_name",
        custom_data=["hover"],
        log_x=True,
    )
    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        xaxis=dict(gridcolor="#2A2F3E"),
        yaxis=dict(gridcolor="#2A2F3E"),
    )
    return fig


# ── Squad Position Map ────────────────────────────────────────────────────────

PITCH_POSITIONS = {
    "GK":  (0.5, 0.05),
    "LB":  (0.15, 0.22), "LCB": (0.33, 0.18), "CB":  (0.5, 0.18),
    "RCB": (0.67, 0.18), "RB":  (0.85, 0.22),
    "LWB": (0.12, 0.38), "LDM": (0.33, 0.35), "CDM": (0.5, 0.35),
    "RDM": (0.67, 0.35), "RWB": (0.88, 0.38),
    "LM":  (0.12, 0.52), "LCM": (0.30, 0.50), "CM":  (0.5, 0.50),
    "RCM": (0.70, 0.50), "RM":  (0.88, 0.52),
    "LAM": (0.25, 0.65), "CAM": (0.5, 0.65), "RAM": (0.75, 0.65),
    "LW":  (0.15, 0.78), "LF":  (0.30, 0.80), "CF":  (0.5, 0.78),
    "RF":  (0.70, 0.80), "RW":  (0.85, 0.78),
    "LS":  (0.35, 0.90), "ST":  (0.5, 0.90), "RS":  (0.65, 0.90),
}

SQUAD_THRESHOLD = 75


def squad_map_chart(club_df: pd.DataFrame) -> go.Figure:
    """Show best player at each position for a club on a pitch layout."""
    fig = go.Figure()

    # Draw pitch outline
    for shape in [
        dict(type="rect", x0=0, y0=0, x1=1, y1=1,
             line=dict(color="#4CAF50", width=2), fillcolor="#2D5A1B"),
        dict(type="rect", x0=0.2, y0=0, x1=0.8, y1=0.15,
             line=dict(color="#4CAF50", width=1), fillcolor="rgba(0,0,0,0)"),
        dict(type="rect", x0=0.2, y0=0.85, x1=0.8, y1=1,
             line=dict(color="#4CAF50", width=1), fillcolor="rgba(0,0,0,0)"),
        dict(type="line", x0=0, y0=0.5, x1=1, y1=0.5,
             line=dict(color="#4CAF50", width=1, dash="dot")),
    ]:
        fig.add_shape(**shape)

    # Add circle
    fig.add_shape(type="circle", x0=0.42, y0=0.43, x1=0.58, y1=0.57,
                  line=dict(color="#4CAF50", width=1), fillcolor="rgba(0,0,0,0)")

    placed = {}
    for _, player in club_df.iterrows():
        pos = str(player.get("club_position", "")).strip().upper()
        if pos in PITCH_POSITIONS and pos not in placed:
            placed[pos] = player

    for pos, (px_coord, py_coord) in PITCH_POSITIONS.items():
        player = placed.get(pos)
        if player is not None:
            rating = int(player.get("overall", 0))
            color = BRAND_COLORS["primary"] if rating >= SQUAD_THRESHOLD else BRAND_COLORS["accent"]
            fig.add_trace(go.Scatter(
                x=[px_coord], y=[py_coord], mode="markers+text",
                marker=dict(size=32, color=color, opacity=0.9,
                            line=dict(color="white", width=1.5)),
                text=[str(rating)], textposition="middle center",
                textfont=dict(color="white", size=10, family="Inter, sans-serif"),
                hovertext=f"{player['short_name']}<br>{pos} | OVR {rating}",
                hoverinfo="text", name="",
                showlegend=False,
            ))
            fig.add_annotation(
                x=px_coord, y=py_coord - 0.055,
                text=player["short_name"].split()[-1] if " " in str(player["short_name"]) else str(player["short_name"]),
                showarrow=False,
                font=dict(color="white", size=8),
            )
        else:
            fig.add_trace(go.Scatter(
                x=[px_coord], y=[py_coord], mode="markers+text",
                marker=dict(size=32, color="#444", opacity=0.5,
                            line=dict(color="#666", width=1)),
                text=["?"], textposition="middle center",
                textfont=dict(color="#999", size=12),
                hovertext=f"{pos} — No player",
                hoverinfo="text", name="",
                showlegend=False,
            ))
            fig.add_annotation(
                x=[px_coord][0], y=py_coord - 0.055,
                text=pos, showarrow=False,
                font=dict(color="#666", size=8),
            )

    squad_layout = {
        **_LAYOUT_DEFAULTS,
        "title": "Squad Map",
        "xaxis": dict(range=[-0.05, 1.05], showgrid=False, zeroline=False, showticklabels=False),
        "yaxis": dict(range=[-0.05, 1.1], showgrid=False, zeroline=False,
                      showticklabels=False, scaleanchor="x", scaleratio=1.6),
        "height": 560,
        "plot_bgcolor": "#2D5A1B",  # overrides the transparent default from _LAYOUT_DEFAULTS
    }
    fig.update_layout(**squad_layout)
    return fig
