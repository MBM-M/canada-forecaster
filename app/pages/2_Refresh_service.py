"""Refresh service page: releases, promotion-gate decisions and the published nowcasts and briefs."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from service.cli import replay  # noqa: E402
from service.data import TARGETS, clean_name  # noqa: E402
from service.store import connect, db_path  # noqa: E402
from service.worker import Worker  # noqa: E402

st.set_page_config(page_title="Refresh service", layout="wide")
st.title("Refresh service")
st.caption(
    "Each simulated StatCan release refits the forecaster's models, promotes a new model only if it beats the current "
    "one on a rolling backtest, and publishes one nowcast and one brief. Replay mode on pinned data; see the README."
)

conn = connect()
n_runs = conn.execute("SELECT count(*) FROM runs").fetchone()[0]

c1, c2 = st.columns([1, 3])
with c1:
    if st.button("Replay and process releases", type="primary"):
        with st.spinner("Recording releases and running the worker..."):
            replay(conn)
            Worker(conn).run_forever(idle_sleep=0.05, exit_when_idle=True)
        st.rerun()
with c2:
    st.caption(f"Database: `{db_path()}`")

if n_runs == 0:
    st.info("No releases yet. Press the button, or run `python -m service replay` then `python -m service work --exit-when-idle`.")
    st.stop()

runs = pd.read_sql_query(
    """SELECT l.target, l.ref_year, l.ready_at, r.status, r.attempt,
              m.decision, m.spec, m.rmspe_candidate, m.rmspe_incumbent, m.rmspe_naive
         FROM releases l JOIN runs r ON r.release_id = l.id
         LEFT JOIN model_versions m ON m.created_from_run = r.id
        ORDER BY l.target, l.ref_year""", conn)

published = conn.execute("SELECT count(*) FROM nowcasts").fetchone()[0]
k = st.columns(5)
k[0].metric("Releases", len(runs))
k[1].metric("Published", published)
for col, d in zip(k[2:], ["promoted", "rejected", "retained"], strict=True):
    col.metric(d.capitalize(), int((runs["decision"] == d).sum()))

st.subheader("Latest nowcast per target")
for target, t in TARGETS.items():
    row = conn.execute(
        """SELECT n.id, n.forecast_year, n.data_through, n.produced_by, b.text FROM nowcasts n
             JOIN briefs b ON b.nowcast_id = n.id WHERE n.target = ? ORDER BY n.forecast_year DESC, n.id DESC LIMIT 1""",
        (target,)).fetchone()
    if row is None:
        continue
    with st.expander(f"{t.label}, {row['forecast_year']} (model {row['produced_by']})", expanded=target == "wages"):
        st.write(row["text"])
        vals = pd.read_sql_query("SELECT grp, last_actual, value, lo, hi FROM nowcast_values WHERE nowcast_id = ?",
                                 conn, params=(row["id"],))
        vals["name"] = vals["grp"].map(clean_name)
        vals["change %"] = 100 * (vals["value"] - vals["last_actual"]) / vals["last_actual"]
        st.dataframe(
            vals.sort_values("change %", ascending=False)[["name", "last_actual", "value", "lo", "hi", "change %"]]
            .rename(columns={"name": t.group_col.replace("geo", "province"), "last_actual": f"{row['data_through']} actual",
                             "value": f"{row['forecast_year']} nowcast", "lo": "5th pct", "hi": "95th pct"}),
            hide_index=True, use_container_width=True)

st.subheader("Promotion gate")
target = st.selectbox("Target", list(TARGETS), format_func=lambda x: TARGETS[x].label)
g = runs[(runs["target"] == target) & runs["decision"].notna()]
fig = go.Figure()
for col, name, colour in [("rmspe_candidate", "Candidate", "#DAFA9C"), ("rmspe_incumbent", "Incumbent", "#7F8C8D"),
                          ("rmspe_naive", "Naive (last year carried forward)", "#E74C3C")]:
    fig.add_bar(x=g["ref_year"].astype(str), y=g[col], name=name, marker_color=colour)
fig.update_layout(barmode="group", yaxis_title="Rolling RMSPE, % (lower is better)", xaxis_title="Data through",
                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#FFFFFF", height=360,
                  margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)
st.dataframe(g[["ref_year", "decision", "spec", "rmspe_candidate", "rmspe_incumbent", "rmspe_naive", "attempt"]],
             hide_index=True, use_container_width=True)
st.caption("A candidate is promoted only if its error is at or below both the incumbent's and the naive baseline's. "
           "'Retained' means the candidate had the same settings as the incumbent.")
