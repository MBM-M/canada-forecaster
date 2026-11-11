"""Canada Forecaster: Streamlit dashboard.

Forecasts Canadian wages, shelter costs and employment (2025–2027) from
Statistics Canada open data, comparing a linear baseline with a random forest.
"""

import json
import os
import re
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
sys.path.insert(0, os.path.join(ROOT, "src"))

st.set_page_config(
    page_title="Canada Forecaster",
    page_icon="🍁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design tokens (soot and pale lime, shared with the landing page in site/) ──
BG = "#0F0F0F"
SURFACE = "#1A1A1A"
INK = "#FFFFFF"
MUTED = "#8F8F8F"
SUBTLE = "#5F5F5F"
GRID = "#262626"
LINE = "#2A2A2A"
ACCENT = "#DAFA9C"
ACCENT_INK = "#0F0F0F"
ACCENT_SOFT = "rgba(218, 250, 156, 0.14)"
NEUTRAL_BAR = "#333333"
ALT_MODEL = "#8F8F8F"
FONT = "'Host Grotesk', system-ui, sans-serif"
MONO = "'JetBrains Mono', ui-monospace, monospace"
REPO = "https://github.com/MBM-M/canada-forecaster"

st.markdown(
    f"""
<style>
:root {{ --ease: cubic-bezier(.2, .7, .2, 1); }}
[data-testid="stMainBlockContainer"], .block-container {{
    max-width: 1180px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}}
[data-testid="stHeader"] {{ background: transparent; }}

.brand {{
    display: flex; align-items: center; gap: 0.6rem;
    font-weight: 500; font-size: 1.02rem; color: {INK}; letter-spacing: -0.01em;
}}
.brand .mark {{ display: inline-grid; grid-template-columns: repeat(2, 7px); gap: 3px; }}
.brand .mark i {{ width: 7px; height: 7px; border-radius: 2.5px; background: {ACCENT}; }}
.stMarkdown .masthead-title {{
    font-family: {FONT}; font-weight: 500;
    font-size: clamp(2.2rem, 4.4vw, 3.4rem); line-height: 1.02; letter-spacing: -0.04em;
    color: {INK}; margin: 2.2rem 0 0 0; max-width: 820px; padding: 0;
}}
.stMarkdown .masthead-lede {{
    color: {MUTED}; font-size: 1.05rem; line-height: 1.5; margin: 0.9rem 0 0 0; max-width: 620px;
}}

.stMarkdown .headline {{
    font-family: {FONT}; font-weight: 500;
    font-size: 1.75rem; line-height: 1.22; letter-spacing: -0.025em;
    color: {INK}; margin: 0.4rem 0 1.2rem 0; max-width: 900px;
}}
.stMarkdown .headline b {{ color: {ACCENT}; font-weight: 500; }}

.tiles {{
    display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.75rem; margin-bottom: 0.5rem;
}}
@media (max-width: 900px) {{ .tiles {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.tile {{
    background: {SURFACE}; border: 1px solid {LINE}; border-radius: 0.75rem;
    padding: 1rem 1.1rem;
    transition: transform .25s var(--ease), border-color .25s var(--ease);
}}
.tile:hover {{ transform: translateY(-2px); border-color: #3D3D3D; }}
.tile-label {{ font-family: {MONO}; font-size: 0.72rem; color: {MUTED}; letter-spacing: 0.01em; }}
.tile-value {{
    font-size: 1.7rem; font-weight: 500; color: {INK}; letter-spacing: -0.03em;
    font-variant-numeric: tabular-nums; margin-top: 0.35rem; line-height: 1.15;
}}
.tile-value.compact {{ font-size: 1.3rem; padding-top: 0.3rem; }}
.tile-note {{ font-size: 0.8rem; color: {MUTED}; margin-top: 0.2rem; }}
.tile.accent {{ background: {ACCENT}; border-color: {ACCENT}; }}
.tile.accent:hover {{ border-color: {ACCENT}; }}
.tile.accent .tile-label, .tile.accent .tile-note {{ color: rgba(15, 15, 15, 0.7); }}
.tile.accent .tile-value {{ color: {ACCENT_INK}; }}

.stMarkdown .section-title {{
    font-family: {FONT}; font-weight: 500; letter-spacing: -0.03em;
    font-size: 1.6rem; color: {INK}; margin: 0;
}}
.stMarkdown .section-sub {{ color: {MUTED}; font-size: 0.92rem; margin: 0.3rem 0 0.8rem 0; max-width: 760px; }}
.rule {{ height: 1px; background: {GRID}; margin: 2.75rem 0 2rem 0; }}

.scorecard {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
.scorecard th, .scorecard td {{
    padding: 0.6rem 0.4rem; border-bottom: 1px solid {GRID}; text-align: right;
}}
.scorecard th {{ font-family: {MONO}; font-size: 0.72rem; font-weight: 400; color: {MUTED}; }}
.scorecard td {{ font-family: {MONO}; font-size: 0.88rem; }}
.scorecard th:first-child, .scorecard td:first-child {{ text-align: left; font-family: {FONT}; font-size: 0.92rem; }}
.scorecard td.best {{ color: {ACCENT}; }}
.scorecard td.worse {{ color: {MUTED}; }}
.stMarkdown .verdict {{ font-size: 0.95rem; color: {INK}; line-height: 1.55; margin-top: 1rem; }}

/* quiet motion: lift on hover, settle on press */
.stButton button, .stDownloadButton button, [data-testid="stPopover"] button {{
    transition: transform .2s var(--ease), background-color .2s var(--ease),
                border-color .2s var(--ease), color .2s var(--ease);
}}
.stButton button:hover, .stDownloadButton button:hover, [data-testid="stPopover"] button:hover {{
    transform: translateY(-1px);
}}
.stButton button:active, .stDownloadButton button:active, [data-testid="stPopover"] button:active {{
    transform: translateY(0) scale(0.98);
}}
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primary"] p {{ color: {ACCENT_INK} !important; }}
@media (prefers-reduced-motion: reduce) {{
    .tile, .stButton button, .stDownloadButton button, [data-testid="stPopover"] button {{ transition: none; }}
    .tile:hover, .stButton button:hover, .stDownloadButton button:hover,
    [data-testid="stPopover"] button:hover {{ transform: none; }}
}}

.footer {{
    color: {MUTED}; font-size: 0.82rem; line-height: 1.7;
    border-top: 1px solid {GRID}; padding-top: 1.4rem; margin-top: 3rem;
}}
.footer a {{ color: {INK}; text-decoration-color: {SUBTLE}; text-underline-offset: 3px; }}
.footer a:hover {{ color: {ACCENT}; text-decoration-color: {ACCENT}; }}
</style>
""",
    unsafe_allow_html=True,
)

# ── Target definitions ────────────────────────────────────────────────────────
TARGETS = {
    "wages": dict(
        label="Wages",
        group_col="geo",
        group_label="Province",
        hist_key="hist_wages",
        hist_col="avg_hourly_wage",
        axis_title="Average hourly wage ($)",
        subject="{g} wages are",
        fmt=lambda v: f"${v:,.2f}",
        hover=":$.2f",
        tickprefix="$",
    ),
    "shelter": dict(
        label="Shelter costs",
        group_col="geo",
        group_label="Province",
        hist_key="hist_wages",
        hist_col="shelter_index",
        axis_title="Shelter price index (2014 = 100)",
        subject="{g} shelter costs are",
        prefix="an index of ",
        fmt=lambda v: f"{v:,.1f}",
        hover=":.1f",
    ),
    "employment": dict(
        label="Employment",
        group_col="industry",
        group_label="Industry",
        hist_key="hist_emp",
        hist_col="avg_employed_thousands",
        axis_title="Employed (thousands)",
        subject="Employment in {g} is",
        fmt=lambda v: f"{v:,.0f}k",
        hover=":,.0f",
        ticksuffix="k",
        tickformat=",.0f",
    ),
}

MODEL_NAMES = {"linear": "Linear regression", "rf": "Random forest"}
METRIC_KEYS = {"linear": "linear", "rf": "random_forest"}

PROVINCES = sorted([
    "Alberta", "British Columbia", "Manitoba", "New Brunswick",
    "Newfoundland and Labrador", "Nova Scotia", "Ontario",
    "Prince Edward Island", "Quebec", "Saskatchewan",
])

# Totals and overlapping groupings: selectable, but left out of the ranking so
# it compares like with like.
EMPLOYMENT_AGGREGATES = {
    "Total, all industries",
    "Goods-producing sector",
    "Services-producing sector",
    "Finance, insurance, real estate, rental and leasing",
    "Forestry, fishing, mining, quarrying, oil and gas",
    "Wholesale and retail trade",
    "Durables",
    "Non-durables",
}

FEATURE_LABELS = {
    "avg_hourly_wage_lag1": "Wage, previous year",
    "avg_hourly_wage_lag2": "Wage, two years back",
    "avg_hourly_wage_yoy": "Wage growth (YoY)",
    "shelter_index_lag1": "Shelter index, previous year",
    "shelter_index_lag2": "Shelter index, two years back",
    "shelter_index_yoy": "Shelter growth (YoY)",
    "cpi_index_lag1": "All-items CPI, previous year",
    "wage_index_lag1": "Wage index, previous year",
    "province_code": "Province",
    "industry_code": "Industry",
    "year": "Year (trend)",
    "covid": "COVID period",
    "avg_employed_thousands_lag1": "Employment, previous year",
    "avg_employed_thousands_lag2": "Employment, two years back",
    "avg_employed_thousands_yoy": "Employment growth (YoY)",
}

REQUIRED_FILES = [
    f"{kind}_{t}.csv" for kind in ("forecasts", "importance") for t in TARGETS
]


def clean_name(name: str) -> str:
    """Strip NAICS codes, e.g. 'Construction [23]' -> 'Construction'."""
    return re.sub(r"\s*\[.*?\]\s*$", "", name)


def short(name: str, n: int = 44) -> str:
    return name if len(name) <= n else name[: n - 1].rstrip() + "…"


# ── Data ──────────────────────────────────────────────────────────────────────
def outputs_ready() -> bool:
    return all(os.path.exists(os.path.join(DATA_DIR, f)) for f in REQUIRED_FILES)


@st.cache_data
def load_all() -> dict:
    read = lambda f: pd.read_csv(os.path.join(DATA_DIR, f))
    data = {
        "hist_wages": read("wages_vs_cpi.csv"),
        "hist_emp": read("employment_annual.csv"),
        "metrics": {},
    }
    for t in TARGETS:
        data[f"fc_{t}"] = read(f"forecasts_{t}.csv")
        data[f"imp_{t}"] = read(f"importance_{t}.csv")
        path = os.path.join(MODELS_DIR, f"metrics_{t}.json")
        if os.path.exists(path):
            with open(path) as fh:
                data["metrics"][t] = json.load(fh)
    return data


def run_pipeline():
    from build_features import (load_raw, make_employment_features,
                                make_shelter_features, make_wage_features)
    from train_models import (EMPLOYMENT_FEATURES, SHELTER_FEATURES,
                              WAGE_FEATURES, train_and_forecast)

    with st.status("Running the forecasting pipeline…", expanded=True) as status:
        st.write("Building features from the raw Statistics Canada tables")
        wc, emp = load_raw()
        make_wage_features(wc).to_csv(os.path.join(DATA_DIR, "features_wages.csv"), index=False)
        make_shelter_features(wc).to_csv(os.path.join(DATA_DIR, "features_shelter.csv"), index=False)
        make_employment_features(emp).to_csv(os.path.join(DATA_DIR, "features_employment.csv"), index=False)

        st.write("Training models and forecasting wages")
        train_and_forecast("wages", "features_wages.csv", WAGE_FEATURES, "avg_hourly_wage", "geo")
        st.write("Training models and forecasting shelter costs")
        train_and_forecast("shelter", "features_shelter.csv", SHELTER_FEATURES, "shelter_index", "geo")
        st.write("Training models and forecasting employment")
        train_and_forecast("employment", "features_employment.csv", EMPLOYMENT_FEATURES,
                           "avg_employed_thousands", "industry")
        status.update(label="Pipeline complete", state="complete")

    st.cache_data.clear()
    st.rerun()


def best_model(metrics: dict) -> str:
    if not metrics:
        return "rf"
    return "linear" if metrics["linear"]["RMSE"] <= metrics["random_forest"]["RMSE"] else "rf"


# ── Chart helpers ─────────────────────────────────────────────────────────────
def base_layout(**overrides) -> dict:
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Host Grotesk, sans-serif", color=MUTED, size=13),
        margin=dict(l=8, r=8, t=12, b=8),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=LINE, font=dict(color=INK, family="Host Grotesk")),
        xaxis=dict(gridcolor="rgba(0,0,0,0)", linecolor=LINE, ticks="outside",
                   tickcolor=GRID, zeroline=False, automargin=True,
                   tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor=GRID, zeroline=False, ticks="", automargin=True,
                   tickfont=dict(color=MUTED)),
        showlegend=False,
    )
    layout.update(overrides)
    return layout


PLOT_CONFIG = {"displayModeBar": False}


def forecast_chart(cfg, hist, fc, national, model, last_year, end_year):
    col = cfg["hist_col"]
    hv = cfg["hover"]
    other = "rf" if model == "linear" else "linear"
    fig = go.Figure()

    fig.add_vrect(x0=last_year + 0.5, x1=end_year + 0.5, fillcolor=SURFACE,
                  opacity=0.8, line_width=0, layer="below")
    fig.add_annotation(x=last_year + 0.6, y=1, yref="paper", xanchor="left", yanchor="top",
                       text="forecast", showarrow=False,
                       font=dict(size=11, color=MUTED, family="JetBrains Mono"))

    if national is not None and not national.empty:
        fig.add_trace(go.Scatter(
            x=national["year"], y=national[col], name="Canada", mode="lines",
            line=dict(color="#6B6B6B", width=1.5, dash="dot"),
            hovertemplate=f"Canada: %{{y{hv}}}<extra></extra>",
        ))
        fig.add_annotation(x=national["year"].iloc[-1], y=national[col].iloc[-1],
                           text="Canada", showarrow=False, xanchor="left", xshift=6,
                           font=dict(size=11, color=MUTED))

    fig.add_trace(go.Scatter(
        x=hist["year"], y=hist[col], name="Actual", mode="lines+markers",
        line=dict(color=INK, width=2.25), marker=dict(size=5, color=INK),
        hovertemplate=f"Actual: %{{y{hv}}}<extra></extra>",
    ))

    if not fc.empty:
        x0, y0 = hist["year"].iloc[-1], hist[col].iloc[-1]
        bx = [x0] + fc["year"].tolist()

        if model == "rf":
            hi = [y0] + fc["rf_upper"].tolist()
            lo = [y0] + fc["rf_lower"].tolist()
            fig.add_trace(go.Scatter(
                x=bx + bx[::-1], y=hi + lo[::-1], fill="toself", fillcolor=ACCENT_SOFT, mode="lines",
                line=dict(width=0), hoverinfo="skip", name="90% interval",
            ))

        other_col = "lr_forecast" if other == "linear" else "rf_forecast"
        fig.add_trace(go.Scatter(
            x=bx, y=[y0] + fc[other_col].tolist(), mode="lines",
            name=MODEL_NAMES[other], line=dict(color=ALT_MODEL, width=1.5, dash="dash"),
            hovertemplate=f"{MODEL_NAMES[other]}: %{{y{hv}}}<extra></extra>",
        ))

        main_col = "lr_forecast" if model == "linear" else "rf_forecast"
        fig.add_trace(go.Scatter(
            x=bx, y=[y0] + fc[main_col].tolist(), mode="lines+markers",
            name=MODEL_NAMES[model], line=dict(color=ACCENT, width=3),
            marker=dict(size=[0] + [9] * len(fc), color=ACCENT,
                        line=dict(color=BG, width=2)),
            hovertemplate=f"{MODEL_NAMES[model]}: %{{y{hv}}}<extra></extra>",
        ))

    fig.update_layout(**base_layout(
        height=420,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", x=0, y=1.08, xanchor="left", yanchor="bottom",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=MUTED)),
        margin=dict(l=8, r=60, t=36, b=8),
    ))
    fig.update_xaxes(dtick=1, range=[hist["year"].min() - 0.4, end_year + 0.6])
    fig.update_yaxes(tickprefix=cfg.get("tickprefix", ""), ticksuffix=cfg.get("ticksuffix", ""),
                     tickformat=cfg.get("tickformat", ""))
    return fig


def ranking_chart(rank_df, selected):
    rank_df = rank_df.sort_values("pct")
    colors = [ACCENT if g == selected else NEUTRAL_BAR for g in rank_df["group"]]
    text_colors = [ACCENT if g == selected else MUTED for g in rank_df["group"]]
    fig = go.Figure(go.Bar(
        x=rank_df["pct"], y=rank_df["label"], orientation="h",
        marker=dict(color=colors), customdata=rank_df["group"],
        text=[f"{v:+.1f}%" for v in rank_df["pct"]], textposition="outside",
        textfont=dict(color=text_colors, size=12), cliponaxis=False,
        hovertemplate="%{y}: %{x:+.1f}%<extra></extra>",
        selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1)),
    ))
    span = max(rank_df["pct"].abs().max(), 1)
    fig.update_layout(**base_layout(
        height=max(260, 30 * len(rank_df) + 40),
        bargap=0.35,
        margin=dict(l=8, r=48, t=8, b=8),
        clickmode="event+select",
        dragmode=False,
    ))
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=True, zerolinecolor=SUBTLE,
                     range=[min(0, rank_df["pct"].min()) - span * 0.18,
                            max(0, rank_df["pct"].max()) + span * 0.18],
                     ticks="", showline=False)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=12.5, color=INK), ticklabelstandoff=10)
    return fig


def importance_chart(imp_df):
    imp = imp_df.copy()
    imp["label"] = imp["feature"].map(FEATURE_LABELS).fillna(imp["feature"])
    imp = imp[imp["importance"] >= 0.005].sort_values("importance")
    top = imp["importance"].max()
    fig = go.Figure(go.Bar(
        x=imp["importance"], y=imp["label"], orientation="h",
        marker=dict(color=[INK if v == top else NEUTRAL_BAR for v in imp["importance"]]),
        text=[f"{v:.0%}" for v in imp["importance"]], textposition="outside",
        textfont=dict(color=MUTED, size=12), cliponaxis=False,
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))
    fig.update_layout(**base_layout(height=max(120, 34 * len(imp) + 30), bargap=0.45,
                                    margin=dict(l=8, r=44, t=4, b=4)))
    fig.update_xaxes(visible=False, range=[0, top * 1.18])
    fig.update_yaxes(showgrid=False, tickfont=dict(size=12.5, color=INK), ticklabelstandoff=10)
    return fig


# ── Masthead ──────────────────────────────────────────────────────────────────
head_l, head_r = st.columns([4, 1], vertical_alignment="center")
with head_l:
    st.markdown('<div class="brand"><span class="mark"><i></i><i></i><i></i><i></i></span>'
                "Canada Forecaster</div>", unsafe_allow_html=True)

with head_r:
    with st.popover("How it works", icon=":material/info:", width="stretch"):
        st.markdown(
            """
            **Sources**
            Statistics Canada tables 14-10-0023-01 (employment by industry) and
            18-10-0005-01 (CPI by province), annual, 2014 to 2024.

            **Models**
            Linear regression and a random forest (200 trees, max depth 6),
            trained on the years before 2022 and scored on 2022 to 2024.

            **Forecasts**
            Both models are refit on every year, then rolled forward one year at a time
            to 2027. The random forest's 90% interval runs from the 5th to the 95th
            percentile of its trees.
            """
        )
        if outputs_ready():
            st.divider()
            st.caption("Rebuilds the features and retrains both models. About 30 seconds.")
            if st.button("Retrain the models", icon=":material/refresh:", width="stretch"):
                run_pipeline()

st.markdown(
    """
    <h1 class="masthead-title">Wage, shelter and job forecasts for Canada</h1>
    <p class="masthead-lede">Two models trained on Statistics Canada data from 2014 to 2024
    project wages and shelter costs for 10 provinces, and jobs in 28 industries, to 2027.</p>
    """,
    unsafe_allow_html=True,
)

# ── First-run gate ────────────────────────────────────────────────────────────
if not outputs_ready():
    st.space("medium")
    with st.container(border=True):
        st.markdown('<p class="section-title">Build the forecasts</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-sub">No forecast files yet. This reads the raw tables in '
            "<code>data/</code>, trains both models and writes forecasts for 2025 to 2027. "
            "About 30 seconds.</p>",
            unsafe_allow_html=True,
        )
        if st.button("Build the forecasts", type="primary", icon=":material/play_arrow:"):
            run_pipeline()
    st.stop()

data = load_all()

# ── Controls ──────────────────────────────────────────────────────────────────
st.space("medium")
tname = st.segmented_control(
    "Indicator", list(TARGETS), format_func=lambda t: TARGETS[t]["label"],
    default="wages", key="target", label_visibility="collapsed",
) or "wages"
cfg = TARGETS[tname]
group_col = cfg["group_col"]
hist_all = data[cfg["hist_key"]]
fc_all = data[f"fc_{tname}"]
metrics = data["metrics"].get(tname, {})
best = best_model(metrics)

if group_col == "geo":
    groups = PROVINCES
    default_group = "Ontario"
    fmt_group = lambda g: g
else:
    names = sorted(fc_all["industry"].unique(), key=clean_name)
    totals = [g for g in names if clean_name(g) == "Total, all industries"]
    groups = totals + [g for g in names if g not in totals]
    default_group = groups[0]
    fmt_group = clean_name

group_key = f"group_{tname}"
if st.session_state.get(group_key) not in groups:
    st.session_state[group_key] = default_group

forecast_years = sorted(int(y) for y in fc_all["year"].unique())

with st.container(border=True):
    c1, c2, c3 = st.columns([1.9, 1.3, 1.9], gap="medium")
    with c1:
        group = st.selectbox(cfg["group_label"], groups, format_func=fmt_group, key=group_key)
    with c2:
        end_year = st.segmented_control(
            "Forecast to", forecast_years, default=forecast_years[-1],
            format_func=str, key="horizon",
        ) or forecast_years[-1]
    with c3:
        model = st.segmented_control(
            "Model", ["linear", "rf"], default=best, key=f"model_{tname}",
            format_func=lambda m: MODEL_NAMES[m] + (" ★" if m == best else ""),
            help="★ marks the model with the lower error on 2022 to 2024, years it never trained on.",
        ) or best

# ── Slice data ────────────────────────────────────────────────────────────────
col = cfg["hist_col"]
fc_col = "lr_forecast" if model == "linear" else "rf_forecast"
hist = hist_all[hist_all[group_col] == group].sort_values("year")
fc = fc_all[(fc_all[group_col] == group) & (fc_all["year"] <= end_year)].sort_values("year")
national = (hist_all[hist_all["geo"] == "Canada"].sort_values("year")
            if group_col == "geo" else None)

last_year = int(hist["year"].iloc[-1])
last_val = float(hist[col].iloc[-1])
proj_val = float(fc[fc_col].iloc[-1]) if not fc.empty else last_val
change = proj_val - last_val
pct = change / last_val * 100 if last_val else 0.0
fmt = cfg["fmt"]
display_group = fmt_group(group)

# ── Headline ──────────────────────────────────────────────────────────────────
st.space("small")
direction = "up" if change >= 0 else "down"
subject = ("Total employment in Canada is" if display_group == "Total, all industries"
           else cfg["subject"].format(g=display_group))
st.markdown(
    f'<p class="headline">{subject} projected to reach {cfg.get("prefix", "")}<b>{fmt(proj_val)}</b> by {end_year}, '
    f"{direction} {abs(pct):.1f}% from {last_year}.</p>",
    unsafe_allow_html=True,
)

mkey = METRIC_KEYS[model]
mae = metrics.get(mkey, {}).get("MAE")
if model == "rf" and not fc.empty:
    lo, hi = fc["rf_lower"].iloc[-1], fc["rf_upper"].iloc[-1]
    fourth = ("90% interval", f"{fmt(lo)} – {fmt(hi)}", f"Random forest, {end_year}", "compact")
else:
    fourth = ("Typical error", f"± {fmt(mae)}" if mae is not None else "—",
              "Average miss, 2022 to 2024", "")

tiles = [
    (f"Actual, {last_year}", fmt(last_val), "Latest Statistics Canada figure", ""),
    (f"Projected, {end_year}", fmt(proj_val), MODEL_NAMES[model], "accent"),
    ("Change", f"{'▲' if change >= 0 else '▼'} {abs(pct):.1f}%",
     f"{'+' if change >= 0 else '−'}{fmt(abs(change))} over {end_year - last_year} yr", ""),
    fourth,
]
st.markdown(
    '<div class="tiles">' + "".join(
        f'<div class="tile {cls}"><div class="tile-label">{lab}</div>'
        f'<div class="tile-value {cls}">{val}</div><div class="tile-note">{note}</div></div>'
        for lab, val, note, cls in tiles
    ) + "</div>",
    unsafe_allow_html=True,
)

# ── Main chart ────────────────────────────────────────────────────────────────
st.plotly_chart(
    forecast_chart(cfg, hist, fc, national, model, last_year, end_year),
    config=PLOT_CONFIG, theme=None, key=f"main_{tname}",
)

with st.expander("Forecast table", icon=":material/table_rows:"):
    table = fc[["year", "lr_forecast", "rf_forecast", "rf_lower", "rf_upper"]].rename(columns={
        "year": "Year", "lr_forecast": "Linear regression", "rf_forecast": "Random forest",
        "rf_lower": "RF lower (90%)", "rf_upper": "RF upper (90%)",
    })
    st.dataframe(table, hide_index=True,
                 column_config={"Year": st.column_config.NumberColumn(format="%d")})
    st.download_button(
        "Download the table", table.to_csv(index=False),
        file_name=f"forecast_{tname}_{re.sub(r'[^a-z0-9]+', '_', display_group.lower())}.csv",
        mime="text/csv", icon=":material/download:",
    )

# ── Comparison across groups ──────────────────────────────────────────────────
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
noun = "province" if group_col == "geo" else "industry"
plural = "provinces" if group_col == "geo" else "industries"

last_actuals = (hist_all[hist_all[group_col].isin(fc_all[group_col].unique())]
                .sort_values("year").groupby(group_col)[col].last())
end_fc = fc_all[fc_all["year"] == end_year].set_index(group_col)[fc_col]
rank = pd.DataFrame({"last": last_actuals, "end": end_fc}).dropna()
rank["pct"] = (rank["end"] / rank["last"] - 1) * 100
rank = rank.reset_index().rename(columns={group_col: "group"})
rank["label"] = rank["group"].map(lambda g: short(fmt_group(g)))
if group_col == "industry":
    rank = rank[~rank["group"].map(clean_name).isin(EMPLOYMENT_AGGREGATES)]

rank_sorted = rank.sort_values("pct", ascending=False).reset_index(drop=True)
position = rank_sorted.index[rank_sorted["group"] == group]

st.markdown(f'<p class="section-title">{plural.capitalize()} ranked by projected change</p>',
            unsafe_allow_html=True)
sub = (f"Change from {last_year} to {end_year}, {MODEL_NAMES[model].lower()} forecast. "
       f"Click a bar to open that {noun}.")
if len(position):
    sub = f"{display_group} ranks #{position[0] + 1} of {len(rank_sorted)}. " + sub
if group_col == "industry":
    sub += " Sector totals and overlapping groupings are left out."
st.markdown(f'<p class="section-sub">{sub}</p>', unsafe_allow_html=True)

rank_key = f"rank_{tname}"


def _on_rank_select():
    points = st.session_state.get(rank_key, {}).get("selection", {}).get("points", [])
    if points:
        picked = points[0].get("customdata")
        if isinstance(picked, list):
            picked = picked[0]
        if picked in groups:
            st.session_state[group_key] = picked


st.plotly_chart(
    ranking_chart(rank, group), config=PLOT_CONFIG, theme=None, key=rank_key,
    on_select=_on_rank_select, selection_mode="points",
)

# ── Model check ───────────────────────────────────────────────────────────────
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-title">How the two models scored</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-sub">Both models trained on the years before 2022, then forecast '
    "2022 to 2024, years they never saw.</p>",
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1], gap="large")
with left:
    with st.container(border=True):
        st.markdown("**Error on 2022 to 2024**")
        if metrics:
            lin, rf = metrics["linear"], metrics["random_forest"]
            rows = [("Mean absolute error", "MAE", True),
                    ("Root mean squared error", "RMSE", True),
                    ("R²", "R2", False)]
            body = ""
            for name, k, lower_better in rows:
                a, b = lin[k], rf[k]
                a_best = (a <= b) if lower_better else (a >= b)
                fa = f"{a:,.2f}" if k != "R2" else f"{a:.3f}"
                fb = f"{b:,.2f}" if k != "R2" else f"{b:.3f}"
                body += (f"<tr><td>{name}</td>"
                         f"<td class='{'best' if a_best else 'worse'}'>{fa}</td>"
                         f"<td class='{'worse' if a_best else 'best'}'>{fb}</td></tr>")
            ratio = max(lin["RMSE"], rf["RMSE"]) / max(min(lin["RMSE"], rf["RMSE"]), 1e-9)
            if best == "linear":
                verdict = (f"<b>Linear regression wins here.</b> Its error is "
                           f"{ratio:,.1f}× smaller. A random forest can't predict past the highest "
                           "value it trained on, so it flattens out on a series that keeps setting "
                           "new highs.")
            else:
                verdict = (f"<b>Random forest wins here.</b> Its error is "
                           f"{ratio:,.1f}× smaller: it picks up bends in the series a straight "
                           "line misses.")
            st.markdown(
                "<table class='scorecard'><tr><th>Metric</th><th>Linear</th>"
                f"<th>Random forest</th></tr>{body}</table>"
                f"<p class='verdict'>{verdict}</p>",
                unsafe_allow_html=True,
            )
            if tname != "shelter":
                st.caption("Errors are in the indicator's own units: "
                           + ("dollars per hour." if tname == "wages" else "thousands of jobs."))
            else:
                st.caption("Errors are in index points (2014 = 100).")
        else:
            st.caption("No scores yet. Retrain the models from How it works.")

with right:
    with st.container(border=True):
        st.markdown("**What the random forest relies on**")
        imp_df = data[f"imp_{tname}"]
        hidden = int((imp_df["importance"] < 0.005).sum())
        st.caption("Each input's share of the forest's splits (impurity importance)."
                   + (f" {hidden} inputs under 0.5% are hidden." if hidden else ""))
        st.plotly_chart(importance_chart(imp_df), config=PLOT_CONFIG,
                        theme=None, key=f"imp_{tname}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="footer">
    Data: Statistics Canada tables 14-10-0023-01 and 18-10-0005-01, annual, 2014 to 2024.
    Each province's shelter index is rebased to 2014 = 100.<br>
    These are model projections, not official Statistics Canada estimates.
    Built with pandas, scikit-learn, Plotly and Streamlit.
    <a href="{REPO}" target="_blank">Get the code</a>
    </div>
    """,
    unsafe_allow_html=True,
)
