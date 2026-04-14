import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
sys.path.insert(0, os.path.join(ROOT, "src"))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Canada Economic Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS (same dark theme as the StatCan dashboard) ────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
.stApp { background-color: #0e1117; color: #e8e8e8; }
[data-testid="stSidebar"] { background-color: #161b27; border-right: 1px solid #2a2f3e; }
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1a2035 0%, #1e2640 100%);
    border: 1px solid #2a3555; border-radius: 12px; padding: 16px 20px;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important; letter-spacing: 0.08em;
    text-transform: uppercase; color: #7a8aaa !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.8rem !important; color: #e8e8e8 !important;
}
[data-testid="stMetricDelta"] svg { display: none; }
.header-banner {
    background: linear-gradient(135deg, #0d1b3e 0%, #1a2d5a 50%, #0d2240 100%);
    border: 1px solid #2a4080; border-radius: 16px;
    padding: 2.5rem 3rem; margin-bottom: 2rem; position: relative; overflow: hidden;
}
.header-banner::before {
    content: ''; position: absolute; top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.header-title {
    font-family: 'Syne', sans-serif; font-size: 3.6rem; font-weight: 800;
    color: #ffffff; margin: 0 0 0.4rem 0; letter-spacing: -0.02em; line-height: 1.1;
}
.header-subtitle {
    font-family: 'Inter', sans-serif; font-size: 1rem;
    color: #7a9acc; font-weight: 300; margin: 0;
}
.header-badge {
    display: inline-block; background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.3); color: #60a5fa;
    font-size: 0.7rem; font-weight: 500; letter-spacing: 0.1em;
    text-transform: uppercase; padding: 4px 12px; border-radius: 20px; margin-bottom: 1rem;
}
.section-header {
    font-family: 'Syne', sans-serif; font-size: 1.3rem; font-weight: 700;
    color: #e8e8e8; border-left: 3px solid #3b82f6;
    padding-left: 0.75rem; margin: 1.5rem 0 1rem 0;
}
.metric-card {
    background: linear-gradient(135deg, #1a2035 0%, #1c2438 100%);
    border: 1px solid #2a3555; border-radius: 10px;
    padding: 1rem 1.25rem; margin-bottom: 0.75rem;
}
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2a3555, transparent);
    margin: 1.5rem 0;
}
.footer {
    font-size: 0.75rem; color: #4a5568; text-align: center;
    padding: 2rem 0 1rem 0; border-top: 1px solid #1e2535; margin-top: 3rem;
}
.setup-box {
    background: linear-gradient(135deg, #1a2035, #1e2640);
    border: 1px solid #3b82f6; border-radius: 12px; padding: 2rem;
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly base layout ─────────────────────────────────────────────────────────
LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#c8d4e8"),
    xaxis=dict(gridcolor="#1e2535", linecolor="#2a3555", tickcolor="#2a3555"),
    yaxis=dict(gridcolor="#1e2535", linecolor="#2a3555", tickcolor="#2a3555"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2a3555", borderwidth=1,
                orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    margin=dict(l=20, r=20, t=50, b=60),
)

COLORS = {
    "historical": "#3b82f6",
    "lr": "#f59e0b",
    "rf": "#22c55e",
    "band": "rgba(34,197,94,0.15)",
    "band_line": "rgba(34,197,94,0.4)",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _check_ready() -> bool:
    needed = [
        "forecasts_wages.csv", "forecasts_shelter.csv", "forecasts_employment.csv",
        "importance_wages.csv", "importance_shelter.csv", "importance_employment.csv",
    ]
    return all(os.path.exists(os.path.join(DATA_DIR, f)) for f in needed)


@st.cache_data
def load_all():
    hist_wages = pd.read_csv(os.path.join(DATA_DIR, "wages_vs_cpi.csv"))
    hist_emp = pd.read_csv(os.path.join(DATA_DIR, "employment_annual.csv"))

    fc_wages = pd.read_csv(os.path.join(DATA_DIR, "forecasts_wages.csv"))
    fc_shelter = pd.read_csv(os.path.join(DATA_DIR, "forecasts_shelter.csv"))
    fc_emp = pd.read_csv(os.path.join(DATA_DIR, "forecasts_employment.csv"))

    imp_wages = pd.read_csv(os.path.join(DATA_DIR, "importance_wages.csv"))
    imp_shelter = pd.read_csv(os.path.join(DATA_DIR, "importance_shelter.csv"))
    imp_emp = pd.read_csv(os.path.join(DATA_DIR, "importance_employment.csv"))

    metrics = {}
    for name in ("wages", "shelter", "employment"):
        path = os.path.join(MODELS_DIR, f"metrics_{name}.json")
        if os.path.exists(path):
            with open(path) as f:
                metrics[name] = json.load(f)

    return dict(
        hist_wages=hist_wages, hist_emp=hist_emp,
        fc_wages=fc_wages, fc_shelter=fc_shelter, fc_emp=fc_emp,
        imp_wages=imp_wages, imp_shelter=imp_shelter, imp_emp=imp_emp,
        metrics=metrics,
    )


def run_pipeline():
    from build_features import load_raw, make_wage_features, make_shelter_features, make_employment_features
    from train_models import train_and_forecast, WAGE_FEATURES, SHELTER_FEATURES, EMPLOYMENT_FEATURES

    with st.spinner("Building features…"):
        wc, emp = load_raw()
        make_wage_features(wc).to_csv(os.path.join(DATA_DIR, "features_wages.csv"), index=False)
        make_shelter_features(wc).to_csv(os.path.join(DATA_DIR, "features_shelter.csv"), index=False)
        make_employment_features(emp).to_csv(os.path.join(DATA_DIR, "features_employment.csv"), index=False)

    with st.spinner("Training models and generating forecasts (takes ~30s)…"):
        train_and_forecast("wages",      "features_wages.csv",      WAGE_FEATURES,       "avg_hourly_wage",          "geo")
        train_and_forecast("shelter",    "features_shelter.csv",    SHELTER_FEATURES,    "shelter_index",            "geo")
        train_and_forecast("employment", "features_employment.csv", EMPLOYMENT_FEATURES, "avg_employed_thousands",   "industry")

    st.cache_data.clear()
    st.success("Done! Reloading…")
    st.rerun()


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <div class="header-badge">Statistics Canada · ML Forecaster · 2025–2027</div>
    <div class="header-title">Canada Economic Forecaster</div>
    <div class="header-subtitle">
        Linear baseline vs. Random Forest — predicting wages, shelter costs &amp;
        employment by province and industry.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Gate: run pipeline if outputs don't exist yet ──────────────────────────────
if not _check_ready():
    st.markdown("""
    <div class="setup-box">
        <div class="section-header">First-time setup required</div>
        <p style="color:#c8d4e8;">
        The forecast files haven't been generated yet. Click below to run
        feature engineering + model training (~30 seconds).
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Run pipeline & generate forecasts", type="primary"):
        run_pipeline()
    st.stop()

data = load_all()

PROVINCES = sorted([
    "Alberta", "British Columbia", "Manitoba", "New Brunswick",
    "Newfoundland and Labrador", "Nova Scotia", "Ontario",
    "Prince Edward Island", "Quebec", "Saskatchewan",
])
INDUSTRIES = sorted(data["fc_emp"]["industry"].unique().tolist())

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-family:Syne;font-size:1.1rem;font-weight:700;color:#e8e8e8;">Controls</p>',
                unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    target = st.selectbox(
        "Forecast target",
        ["Wages ($/hr)", "Shelter CPI (index)", "Employment (thousands)"],
    )

    if target != "Employment (thousands)":
        selected_group = st.selectbox(
            "Province",
            PROVINCES,
            index=PROVINCES.index("Ontario"),
        )
        overview_groups = st.multiselect(
            "Provinces for overview chart",
            PROVINCES,
            default=["British Columbia", "Ontario", "Alberta", "Quebec"],
        )
    else:
        top8 = (
            data["hist_emp"][data["hist_emp"]["year"] == 2019]
            .nlargest(8, "avg_employed_thousands")["industry"].tolist()
        )
        selected_group = st.selectbox("Industry", INDUSTRIES,
                                      index=INDUSTRIES.index(top8[0]) if top8[0] in INDUSTRIES else 0)
        overview_groups = st.multiselect("Industries for overview", INDUSTRIES, default=top8[:5])

    horizon = st.slider("Forecast horizon (years)", 1, 3, 3)
    show_lr = st.checkbox("Show Linear baseline", value=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    if st.button("Re-run pipeline"):
        for f in ["forecasts_wages.csv", "forecasts_shelter.csv", "forecasts_employment.csv",
                  "importance_wages.csv", "importance_shelter.csv", "importance_employment.csv",
                  "features_wages.csv", "features_shelter.csv", "features_employment.csv"]:
            p = os.path.join(DATA_DIR, f)
            if os.path.exists(p):
                os.remove(p)
        st.cache_data.clear()
        st.rerun()

    st.markdown("""
    <p style="font-size:0.75rem;color:#4a5568;font-family:Inter;">
    Data: Statistics Canada<br>
    Linear + Random Forest (200 trees)<br>
    Train &lt;2022 · Test 2022–2024<br>
    Forecast: autoregressive, 90% PI
    </p>
    """, unsafe_allow_html=True)

# ── Resolve target-specific config ────────────────────────────────────────────
if target == "Wages ($/hr)":
    tname = "wages"
    group_col = "geo"
    hist_col = "avg_hourly_wage"
    y_label = "Avg Hourly Wage ($/hr)"
    hist_df_full = data["hist_wages"]
    fc_df = data["fc_wages"]
    imp_df = data["imp_wages"]

elif target == "Shelter CPI (index)":
    tname = "shelter"
    group_col = "geo"
    hist_col = "shelter_index"
    y_label = "Shelter CPI Index (2014=100)"
    hist_df_full = data["hist_wages"]
    fc_df = data["fc_shelter"]
    imp_df = data["imp_shelter"]

else:
    tname = "employment"
    group_col = "industry"
    hist_col = "avg_employed_thousands"
    y_label = "Avg Employed (thousands)"
    hist_df_full = data["hist_emp"]
    fc_df = data["fc_emp"]
    imp_df = data["imp_emp"]

fc_years = sorted(fc_df["year"].unique())[:horizon]
fc_group = fc_df[fc_df[group_col] == selected_group & fc_df["year"].isin(fc_years)
                 if False else (fc_df[group_col] == selected_group) & (fc_df["year"].isin(fc_years))]

if group_col == "geo":
    hist_group = hist_df_full[hist_df_full["geo"] == selected_group].sort_values("year")
else:
    hist_group = hist_df_full[hist_df_full["industry"] == selected_group].sort_values("year")

# ── KPI strip ─────────────────────────────────────────────────────────────────
last_actual = hist_group[hist_col].iloc[-1]
last_year = hist_group["year"].iloc[-1]
rf_end = fc_group["rf_forecast"].iloc[-1] if not fc_group.empty else last_actual
lr_end = fc_group["lr_forecast"].iloc[-1] if not fc_group.empty else last_actual
metrics_row = data["metrics"].get(tname, {})

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(f"Actual {last_year}", f"{last_actual:.2f}")
with col2:
    rf_delta = rf_end - last_actual
    st.metric(f"RF Forecast {fc_years[-1] if fc_years else ''}", f"{rf_end:.2f}",
              delta=f"{rf_delta:+.2f}")
with col3:
    lr_delta = lr_end - last_actual
    st.metric(f"LR Forecast {fc_years[-1] if fc_years else ''}", f"{lr_end:.2f}",
              delta=f"{lr_delta:+.2f}")
with col4:
    rf_r2 = metrics_row.get("random_forest", {}).get("R2", None)
    st.metric("RF R² (test set)", f"{rf_r2:.3f}" if rf_r2 is not None else "—")

st.markdown(
    "<div style='font-size:0.78rem; color:#7a8aaa; margin-top:0.4rem; line-height:1.7'>"
    "<b style='color:#a0aec0'>RF</b> — Random Forest: an ensemble of decision trees that averages many predictions for robust forecasts. &nbsp;|&nbsp; "
    "<b style='color:#a0aec0'>LR</b> — Linear Regression: a simple baseline that fits a straight-line trend through historical data. &nbsp;|&nbsp; "
    "<b style='color:#a0aec0'>RF R²</b> — Coefficient of determination on the 2022–2024 test set; 1.0 = perfect fit, 0.0 = no better than the mean."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Forecast", "Feature Importance", "Model Comparison", "Province / Industry Overview"
])

# ─── Tab 1: Forecast chart ─────────────────────────────────────────────────────
with tab1:
    st.markdown(f'<div class="section-header">{selected_group} — Historical + Forecast ({y_label})</div>',
                unsafe_allow_html=True)

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=hist_group["year"], y=hist_group[hist_col],
        name="Historical", mode="lines+markers",
        line=dict(color=COLORS["historical"], width=3),
        marker=dict(size=6),
    ))

    if not fc_group.empty:
        # Bridge: connect last historical point to forecasts
        bridge_x = [hist_group["year"].iloc[-1]] + fc_group["year"].tolist()
        bridge_rf = [hist_group[hist_col].iloc[-1]] + fc_group["rf_forecast"].tolist()
        bridge_lo = [hist_group[hist_col].iloc[-1]] + fc_group["rf_lower"].tolist()
        bridge_hi = [hist_group[hist_col].iloc[-1]] + fc_group["rf_upper"].tolist()
        bridge_lr = [hist_group[hist_col].iloc[-1]] + fc_group["lr_forecast"].tolist()

        # RF confidence band
        fig.add_trace(go.Scatter(
            x=bridge_x + bridge_x[::-1],
            y=bridge_hi + bridge_lo[::-1],
            fill="toself", fillcolor=COLORS["band"],
            line=dict(color="rgba(0,0,0,0)"),
            name="RF 90% interval", showlegend=True,
        ))

        # RF forecast line
        fig.add_trace(go.Scatter(
            x=bridge_x, y=bridge_rf,
            name="RF Forecast", mode="lines+markers",
            line=dict(color=COLORS["rf"], width=2.5, dash="dot"),
            marker=dict(size=7, symbol="diamond"),
        ))

        # LR forecast line
        if show_lr:
            fig.add_trace(go.Scatter(
                x=bridge_x, y=bridge_lr,
                name="LR Baseline", mode="lines+markers",
                line=dict(color=COLORS["lr"], width=2, dash="dash"),
                marker=dict(size=6),
            ))

        # Vertical divider at forecast start
        fig.add_vline(x=last_year + 0.5, line_dash="dash",
                      line_color="#2a3555", opacity=0.8)
        fig.add_annotation(
            x=last_year + 0.6, yref="paper", y=0.97,
            text="Forecast →", showarrow=False,
            font=dict(color="#4a5568", size=11),
        )

    fig.update_layout(**LAYOUT, height=420, yaxis_title=y_label, xaxis_title="Year",
                      hovermode="x unified", title=f"{selected_group} — {y_label}")
    st.plotly_chart(fig, use_container_width=True)

    if not fc_group.empty:
        st.markdown("**Forecast values**")
        display = fc_group[["year", "lr_forecast", "rf_forecast", "rf_lower", "rf_upper"]].copy()
        display.columns = ["Year", "LR Forecast", "RF Forecast", "RF Lower (90%)", "RF Upper (90%)"]
        st.dataframe(display.set_index("Year").round(3), use_container_width=True)

# ─── Tab 2: Feature importance ────────────────────────────────────────────────
with tab2:
    st.markdown(f'<div class="section-header">Random Forest Feature Importance — {target}</div>',
                unsafe_allow_html=True)

    FEATURE_LABELS = {
        "avg_hourly_wage_lag1": "Wage (t−1)",
        "avg_hourly_wage_lag2": "Wage (t−2)",
        "avg_hourly_wage_yoy": "Wage YoY %",
        "shelter_index_lag1": "Shelter CPI (t−1)",
        "cpi_index_lag1": "All-Items CPI (t−1)",
        "province_code": "Province",
        "year": "Year (trend)",
        "covid": "COVID dummy",
        "shelter_index_lag2": "Shelter CPI (t−2)",
        "shelter_index_yoy": "Shelter CPI YoY %",
        "wage_index_lag1": "Wage Index (t−1)",
        "industry_code": "Industry",
        "avg_employed_thousands_lag1": "Employment (t−1)",
        "avg_employed_thousands_lag2": "Employment (t−2)",
        "avg_employed_thousands_yoy": "Employment YoY %",
    }

    imp_plot = imp_df.copy()
    imp_plot["label"] = imp_plot["feature"].map(FEATURE_LABELS).fillna(imp_plot["feature"])
    imp_plot = imp_plot.sort_values("importance")

    fig_imp = go.Figure(go.Bar(
        x=imp_plot["importance"],
        y=imp_plot["label"],
        orientation="h",
        marker_color=[
            "#3b82f6" if imp_plot["importance"].iloc[i] == imp_plot["importance"].max()
            else "#1d4ed8"
            for i in range(len(imp_plot))
        ],
        text=imp_plot["importance"].round(3),
        textposition="outside",
        textfont=dict(color="#c8d4e8", size=11),
    ))
    fig_imp.update_layout(
        **LAYOUT, height=380,
        xaxis_title="Importance (mean decrease in impurity)",
        title="Which features drive predictions most?",
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    top_feat = imp_plot.iloc[-1]["label"]
    top_imp = imp_plot.iloc[-1]["importance"]
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size:0.7rem;color:#3b82f6;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px;">
            Top driver
        </div>
        <div style="color:#e8e8e8;font-size:0.95rem;">
            <strong>{top_feat}</strong> accounts for
            <strong>{top_imp*100:.1f}%</strong> of the Random Forest's
            predictive signal for {target.lower()}.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Tab 3: Model comparison ──────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Linear Regression vs. Random Forest — Test-Set Metrics</div>',
                unsafe_allow_html=True)

    if tname in data["metrics"]:
        m = data["metrics"][tname]
        comparison = pd.DataFrame({
            "Model": ["Linear Regression", "Random Forest"],
            "MAE": [m["linear"]["MAE"], m["random_forest"]["MAE"]],
            "RMSE": [m["linear"]["RMSE"], m["random_forest"]["RMSE"]],
            "R²": [m["linear"]["R2"], m["random_forest"]["R2"]],
        })

        col_a, col_b = st.columns([1, 2])

        with col_a:
            st.markdown("**Test-set performance**")
            st.dataframe(
                comparison.set_index("Model").style
                .highlight_min(subset=["MAE", "RMSE"], color="#0d2240", axis=0)
                .highlight_max(subset=["R²"], color="#0d2240", axis=0)
                .format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"}),
                use_container_width=True,
            )

            rf_improvement = (
                (m["linear"]["RMSE"] - m["random_forest"]["RMSE"])
                / m["linear"]["RMSE"] * 100
            )
            st.markdown(f"""
            <div class="metric-card" style="margin-top:1rem;">
                <div style="font-size:0.7rem;color:#22c55e;letter-spacing:0.1em;
                            text-transform:uppercase;margin-bottom:4px;">
                    RF improvement over baseline
                </div>
                <div style="color:#e8e8e8;font-size:0.95rem;">
                    Random Forest reduced RMSE by
                    <strong>{rf_improvement:.1f}%</strong> vs. Linear Regression.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            metrics_long = comparison.melt(id_vars="Model", var_name="Metric", value_name="Value")
            fig_m = px.bar(
                metrics_long[metrics_long["Metric"] != "R²"],
                x="Metric", y="Value", color="Model", barmode="group",
                color_discrete_map={"Linear Regression": COLORS["lr"], "Random Forest": COLORS["rf"]},
                template="plotly_dark", title="MAE & RMSE comparison",
            )
            fig_m.update_layout(**LAYOUT, height=340)
            st.plotly_chart(fig_m, use_container_width=True)

    # Show all-target metrics summary
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">All Targets — RF R² Summary</div>', unsafe_allow_html=True)

    all_metrics = []
    for n, label in [("wages", "Wages"), ("shelter", "Shelter CPI"), ("employment", "Employment")]:
        if n in data["metrics"]:
            all_metrics.append({
                "Target": label,
                "LR R²": data["metrics"][n]["linear"]["R2"],
                "RF R²": data["metrics"][n]["random_forest"]["R2"],
                "LR MAE": data["metrics"][n]["linear"]["MAE"],
                "RF MAE": data["metrics"][n]["random_forest"]["MAE"],
            })
    if all_metrics:
        st.dataframe(pd.DataFrame(all_metrics).set_index("Target").round(4),
                     use_container_width=True)

# ─── Tab 4: Overview ──────────────────────────────────────────────────────────
with tab4:
    label_4 = "Province" if group_col == "geo" else "Industry"
    st.markdown(f'<div class="section-header">RF Forecast Overlay — All {label_4}s ({y_label})</div>',
                unsafe_allow_html=True)

    if overview_groups:
        fig_ov = go.Figure()
        palette = px.colors.qualitative.Set2

        for i, grp in enumerate(overview_groups):
            hist_g = (
                hist_df_full[hist_df_full[group_col] == grp].sort_values("year")
                if group_col in hist_df_full.columns
                else pd.DataFrame()
            )
            fc_g = fc_df[(fc_df[group_col] == grp) & (fc_df["year"].isin(fc_years))]

            color = palette[i % len(palette)]

            if not hist_g.empty:
                fig_ov.add_trace(go.Scatter(
                    x=hist_g["year"], y=hist_g[hist_col],
                    name=grp, mode="lines",
                    line=dict(color=color, width=2),
                    legendgroup=grp,
                ))

            if not fc_g.empty:
                bridge_x = [hist_g["year"].iloc[-1]] + fc_g["year"].tolist() if not hist_g.empty else fc_g["year"].tolist()
                bridge_y = ([hist_g[hist_col].iloc[-1]] + fc_g["rf_forecast"].tolist()
                             if not hist_g.empty else fc_g["rf_forecast"].tolist())
                fig_ov.add_trace(go.Scatter(
                    x=bridge_x, y=bridge_y,
                    name=f"{grp} (forecast)", mode="lines+markers",
                    line=dict(color=color, width=2, dash="dot"),
                    marker=dict(size=7, symbol="diamond"),
                    legendgroup=grp, showlegend=False,
                ))

        if hist_g is not None and not hist_g.empty:
            fig_ov.add_vline(x=hist_g["year"].iloc[-1] + 0.5,
                             line_dash="dash", line_color="#2a3555", opacity=0.8)

        fig_ov.update_layout(
            **LAYOUT, height=460,
            yaxis_title=y_label, xaxis_title="Year",
            hovermode="x unified",
            title=f"All selected {label_4}s — historical + RF forecast",
        )
        st.plotly_chart(fig_ov, use_container_width=True)

        # Ranked forecast table
        if fc_years:
            final_yr = max(fc_years)
            fc_table = fc_df[
                (fc_df[group_col].isin(overview_groups)) & (fc_df["year"] == final_yr)
            ][[group_col, "rf_forecast", "rf_lower", "rf_upper"]].copy()

            # Add last actual
            if group_col == "geo":
                actuals = (
                    hist_df_full[hist_df_full["geo"].isin(overview_groups)]
                    .sort_values("year").groupby("geo")[hist_col].last().reset_index()
                    .rename(columns={hist_col: "last_actual", "geo": group_col})
                )
            else:
                actuals = (
                    hist_df_full[hist_df_full["industry"].isin(overview_groups)]
                    .sort_values("year").groupby("industry")[hist_col].last().reset_index()
                    .rename(columns={hist_col: "last_actual", "industry": group_col})
                )

            fc_table = fc_table.merge(actuals, on=group_col, how="left")
            fc_table["projected change"] = fc_table["rf_forecast"] - fc_table["last_actual"]
            fc_table = fc_table.sort_values("projected change", ascending=False)
            fc_table.columns = [label_4, f"RF Forecast {final_yr}", "Lower 90%", "Upper 90%",
                                 f"Last Actual", "Projected Δ"]
            st.markdown(f"**Ranked by projected change by {final_yr}**")
            st.dataframe(fc_table.set_index(label_4).round(3), use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Data: Statistics Canada open data · Tables 14-10-0023-01, 18-10-0005-01<br>
    Models: scikit-learn LinearRegression + RandomForestRegressor (200 trees, max_depth=6)<br>
    Train &lt; 2022 · Test 2022–2024 · Forecast 2025–2027 via autoregressive rollout · 90% prediction intervals from RF tree distribution<br>
    Built with Python, scikit-learn, Pandas, Plotly &amp; Streamlit
</div>
""", unsafe_allow_html=True)
