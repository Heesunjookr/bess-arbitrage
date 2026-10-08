"""Public Streamlit app for the German BESS dispatch research project."""

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import ROOT
from src.optimize.lp_dispatch import BatteryParams
from src.webapp.dashboard import available_dates, compare_dispatch_day, forecast_ladder


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
    .hero { padding: 1.5rem 0 .6rem 0; }
    .eyebrow { color: #197a54; font-weight: 700; letter-spacing: .08em;
        text-transform: uppercase; font-size: .78rem; }
    .hero h1 { font-size: 3rem; line-height: 1.05; margin: .35rem 0 .6rem; }
    .hero p { color: #4b5b52; max-width: 850px; font-size: 1.08rem; }
    .note { border-left: 4px solid #e2a93b; padding: .6rem 1rem;
        background: #fffaf0; border-radius: 4px; }
    </style>
    <div class="hero">
      <div class="eyebrow">OpenBESS Lab · Germany DE-LU</div>
      <h1>What is a day-ahead forecast worth to a battery?</h1>
      <p>An interactive, reproducible study of battery dispatch, forecast quality,
      and data integrity in the German day-ahead power market.</p>
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
    "The 92.3% result uses a common 1,542-day sample; the 77.2% headline KPI uses the full sample."
)

tab_ladder, tab_explorer, tab_validation = st.tabs(
    ["Forecast ladder", "Dispatch explorer", "Validation & limitations"]
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
    st.plotly_chart(chart, use_container_width=True)
    st.success(
        "The residual-load blend captured 92.3% of the ceiling and recovered "
        "65.7% of the D-1 persistence gap. A compact rolling OLS beat the GBM."
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
            st.plotly_chart(fig, use_container_width=True)

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
