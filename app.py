"""Public Streamlit app for the German BESS dispatch research project."""

import json
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import ROOT
from src.optimize.lp_dispatch import BatteryParams
from src.webapp.dashboard import artifact_status, available_dates, compare_dispatch_day, forecast_ladder


REPOSITORY_URL = "https://github.com/Heesunjookr/bess-arbitrage"
ARTICLE_URL = "https://medium.com/@heesun.jookr/pricing-perfect-foresight-what-a-day-ahead-forecast-is-actually-worth-to-a-german-battery-3281c9d79ebe"


st.set_page_config(
    page_title="OpenBESS Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f7f7f2; color: #17211b; }
    [data-testid="stMetric"] { background: white; border: 1px solid #dde3dc;
        border-radius: 12px; padding: 16px; }
    .hero { padding: 1.1rem 0 .6rem 0; }
    .eyebrow { color: #197a54; font-weight: 700; letter-spacing: .08em;
        text-transform: uppercase; font-size: .78rem; }
    .hero h1 { font-size: 3rem; line-height: 1.05; margin: .35rem 0 .6rem; }
    .hero p { color: #4b5b52; max-width: 850px; font-size: 1.08rem; }
    .note { border-left: 4px solid #e2a93b; padding: .6rem 1rem;
        background: #fffaf0; border-radius: 4px; }
    .trust-row { display:flex; flex-wrap:wrap; gap:.45rem; margin:.8rem 0 .2rem; }
    .trust-chip { background:#e8f3ed; color:#155f43; border:1px solid #c8dfd2;
        border-radius:999px; padding:.28rem .65rem; font-size:.78rem; font-weight:650; }
    .brief { background:#17211b; color:#f5f7f3; border-radius:14px; padding:1.15rem 1.3rem;
        margin:.4rem 0 1rem; }
    .brief strong { color:#87ddb8; }
    .brief p { margin:.25rem 0; color:#e7eee9; }
    .status-ready { color:#197a54; font-weight:700; }
    </style>
    <div class="hero">
      <div class="eyebrow">German BESS Forecast Value Monitor · DE-LU</div>
      <h1>Forecast quality, measured in battery value.</h1>
      <p>A public decision tool that converts day-ahead forecast quality into dispatch value,
      compares executable model tiers, and exposes the assumptions behind every result.</p>
      <div class="trust-row">
        <span class="trust-chip">1,731 delivery days</span>
        <span class="trust-chip">No-lookahead schedules</span>
        <span class="trust-chip">Open methodology</span>
        <span class="trust-chip">Tested optimisation</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_artifacts():
    processed = ROOT / "data" / "processed"
    with open(processed / "kpi_summary.json") as handle:
        summary = json.load(handle)
    findings = pd.read_csv(processed / "forecast_gap_recovery.csv")
    prices = pd.read_csv(
        ROOT / "data" / "public" / "da_prices.csv.gz",
        parse_dates=["timestamp_utc"],
    ).set_index("timestamp_utc")
    prices.index = prices.index.tz_convert("Europe/Berlin")
    prices["date"] = prices.index.date
    return summary, forecast_ladder(findings), prices


summary, ladder, prices = load_artifacts()
headline = summary["full_sample_mean"]
status = artifact_status(
    [
        ROOT / "data" / "processed" / "kpi_summary.json",
        ROOT / "data" / "processed" / "forecast_gap_recovery.csv",
        ROOT / "data" / "public" / "da_prices.csv.gz",
    ]
)

with st.sidebar:
    st.markdown("### Research monitor")
    st.markdown('<span class="status-ready">● Evidence bundle ready</span>', unsafe_allow_html=True)
    if status["latest_modified"]:
        st.caption(
            "Versioned artifacts · latest build "
            + datetime.fromtimestamp(status["latest_modified"]).strftime("%d %b %Y")
        )
    st.markdown("**Market**  ")
    st.write("Germany / Luxembourg day-ahead")
    st.markdown("**Reference asset**  ")
    st.write("1 MW / 2 MWh · 85% round-trip efficiency")
    st.link_button("View source code", REPOSITORY_URL, width="stretch")
    st.link_button("Read the research note", ARTICLE_URL, width="stretch")
    st.divider()
    st.caption("Independent research. Not investment advice or realised trading performance.")

left, middle_left, middle_right, right = st.columns(4)
left.metric("Backtest decisions", f"{summary['sample']['n_days_total']:,} days")
middle_left.metric("D-1 capture", f"{headline['capture_persist_d1']:.1%}")
middle_right.metric("Best model capture", f"{ladder['capture'].max():.1%}")
right.metric(
    "Forecast opportunity gap",
    f"€{headline['forecast_value_gap_d1_eur_per_mw_yr']/1000:.1f}k/MW/yr",
)

st.caption(
    "All headline values are simulated backtest results, not realised trading revenue. "
    "The 94.5% result uses a common 1,640-day sample; full-sample KPIs use every eligible day."
)

st.markdown(
    """
    <div class="brief">
      <p><strong>Decision brief</strong></p>
      <p>A D-1 persistence schedule captures 82.3% of the common-sample ceiling. Adding a
      day-ahead residual-load forecast lifts capture to 94.5% and recovers 69.2% of the
      forecast-value gap. The practical edge came from the information set and objective,
      not from using the most complex model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_ladder, tab_explorer, tab_validation, tab_provenance = st.tabs(
    ["Executive results", "Dispatch explorer", "Validation", "Data & methodology"]
)

with tab_ladder:
    st.subheader("Model complexity was not the winning ingredient")
    st.write(
        "Each tier is evaluated through the same battery LP. Capture is settled "
        "revenue divided by the perfect-foresight ceiling on the common sample."
    )
    chart = px.bar(
        ladder,
        x="capture_pct",
        y="label",
        orientation="h",
        text=ladder["capture_pct"].map(lambda value: f"{value:.1f}%"),
        color="capture_pct",
        color_continuous_scale=["#dbe7df", "#197a54"],
        labels={"capture_pct": "Perfect-foresight capture (%)", "label": ""},
    )
    chart.update_layout(
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis={"categoryorder": "total ascending"},
        xaxis_range=[70, 100],
        height=430,
    )
    st.plotly_chart(chart, width="stretch")
    st.success(
        "The residual-load blend captured 94.5% of the ceiling and recovered "
        "69.2% of the D-1 persistence gap. A compact rolling OLS beat the GBM."
    )
    table = ladder[["label", "information_set", "n_days", "capture_pct", "gap_recovered_pct"]].copy()
    table.columns = ["Model tier", "Information available before auction", "Days", "Capture (%)", "Gap recovered (%)"]
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Capture (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Gap recovered (%)": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )

with tab_explorer:
    st.subheader("Run a transparent one-day dispatch comparison")
    st.write(
        "Change the public battery assumptions and inspect the schedule. The D-1 "
        "strategy uses yesterday's curve as its forecast; perfect foresight is a ceiling."
    )
    valid_dates = available_dates(prices)
    controls, output = st.columns([1, 3])
    with controls:
        selected = st.date_input(
            "Delivery day",
            value=valid_dates[-1],
            min_value=valid_dates[0],
            max_value=valid_dates[-1],
        )
        power = st.slider("Power (MW)", 0.5, 5.0, 1.0, 0.5)
        duration = st.slider("Duration (hours)", 1, 8, 2)
        efficiency = st.slider("Round-trip efficiency", 0.70, 0.95, 0.85, 0.01)
        throughput_cost = st.slider("Throughput cost (€/MWh)", 0.0, 20.0, 2.0, 1.0)
        st.caption("Start and end state of charge are fixed at zero for each day.")

    params = BatteryParams(
        e_max_mwh=power * duration,
        p_max_mw=power,
        eta_round_trip=efficiency,
        throughput_cost_eur_mwh=throughput_cost,
    )
    try:
        dispatch, day_metrics = compare_dispatch_day(prices, selected, params)
    except ValueError:
        st.warning("That day or its D-1 comparison has an incomplete 24-hour curve. Pick another date.")
    else:
        with output:
            a, b, c = st.columns(3)
            a.metric("Perfect-foresight ceiling", f"€{day_metrics['perfect_foresight_eur']:,.0f}")
            b.metric("D-1 settled result", f"€{day_metrics['persistence_eur']:,.0f}")
            capture = day_metrics["capture"]
            c.metric("D-1 capture", "n/a" if pd.isna(capture) else f"{capture:.1%}")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=dispatch["hour"], y=dispatch["realised_price"],
                           name="Realised price", line={"color": "#202a24", "width": 3})
            )
            fig.add_trace(
                go.Scatter(x=dispatch["hour"], y=dispatch["d1_price_curve"],
                           name="D-1 forecast curve", line={"color": "#a9b5ae", "dash": "dot"})
            )
            fig.add_trace(
                go.Bar(x=dispatch["hour"], y=dispatch["persistence_net_mw"],
                       name="D-1 dispatch (+ discharge)", marker_color="#20a06b", yaxis="y2", opacity=.7)
            )
            fig.update_layout(
                xaxis_title="Delivery hour",
                yaxis={"title": "Price (€/MWh)"},
                yaxis2={"title": "Dispatch (MW)", "overlaying": "y", "side": "right", "zeroline": True},
                legend={"orientation": "h", "y": 1.12},
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=470,
            )
            st.plotly_chart(fig, width="stretch")

with tab_validation:
    st.subheader("The credibility layer")
    st.markdown(
        """
        - **No lookahead:** executable schedules use only information intended to be available before the D-1 auction.
        - **Negative results remain visible:** the sibling DA-ID study found no useful out-of-sample predictive relationship (`OOS R² ≈ 0`).
        - **A mislabeled series was removed:** two supposed intraday series were traced to Belgian and Norwegian day-ahead prices.
        - **DST and granularity are tested:** 23/25-hour local days are excluded; the 2025 switch to 15-minute MTU is documented.
        - **Reproducible:** model code, tests, assumptions, and processed outputs live in the repository.
        """
    )
    st.markdown(
        '<div class="note"><strong>Main limitation:</strong> SMARD provides the latest snapshot of '
        'day-ahead forecast series without publication timestamps. Fundamental-tier capture is therefore '
        'treated as an upper bound until a point-in-time dataset is rebuilt.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.caption(
        "Research project by Heesun Joo · Python · cvxpy · pandas · scikit-learn · Plotly · "
        "Price data: ENTSO-E Transparency Platform (CC BY 4.0)"
    )

with tab_provenance:
    st.subheader("Trace every headline to its source")
    st.write(
        "The public app is built from three versioned artifacts. The optimiser and model code "
        "remain separate from presentation logic so the displayed KPIs can be reproduced in tests."
    )
    provenance = pd.DataFrame(
        [
            {"Layer": "Market prices", "Artifact": "data/public/da_prices.csv.gz", "Purpose": "Complete DE-LU hourly delivery days", "Source": "ENTSO-E Transparency"},
            {"Layer": "Headline KPIs", "Artifact": "data/processed/kpi_summary.json", "Purpose": "Asset assumptions, revenue and capture", "Source": "Backtest pipeline"},
            {"Layer": "Model ladder", "Artifact": "data/processed/forecast_gap_recovery.csv", "Purpose": "Common-sample tier comparison", "Source": "Walk-forward forecast evaluation"},
        ]
    )
    st.dataframe(provenance, hide_index=True, width="stretch")
    left_doc, right_doc = st.columns(2)
    with left_doc:
        st.markdown("#### Reproduction contract")
        st.markdown(
            """
            1. Validate local data health.
            2. Rebuild forecast tiers walk-forward.
            3. Settle every fixed schedule against realised prices.
            4. Regenerate the structured KPI artifacts.
            5. Run the full test suite before publishing.
            """
        )
    with right_doc:
        st.markdown("#### Interpretation contract")
        st.markdown(
            """
            - Perfect foresight is a ceiling, never a strategy.
            - Every percentage is a simulated backtest result.
            - Fundamental tiers remain an upper bound until point-in-time forecasts are rebuilt.
            - No intraday, balancing, tax, grid-fee or realised P&L claim is included.
            """
        )
    st.info(
        "Current public milestone: historical monitor complete. Next milestone: publish genuine "
        "point-in-time forward results after the paper-trading ledger has enough settled observations."
    )
