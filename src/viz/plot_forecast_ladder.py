"""Render the forecast capture ladder as interactive HTML and static PNG.

Run:  python -m src.viz.plot_forecast_ladder
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from src.config import ROOT, get_logger, load_config

logger = get_logger("plot_forecast_ladder")

TEMPLATE = "plotly_white"
LADDER_ORDER = ["persist_d1", "persist_d7", "blend", "gbm", "rl_quad", "blend_rl"]
LADDER_LABELS = {
    "persist_d1": "Persistence D-1",
    "persist_d7": "Persistence D-7",
    "blend": "Price blend",
    "gbm": "Gradient boosting",
    "rl_quad": "Residual load only",
    "blend_rl": "Blend + residual load",
}
LADDER_DETAILS = {
    "persist_d1": "D-1 price curve",
    "persist_d7": "D-7 price curve",
    "blend": "D-1 / D-7 · rolling OLS",
    "gbm": "prices + fundamentals",
    "rl_quad": "RL + RL² · rolling OLS",
    "blend_rl": "rolling OLS",
}
BAR_COLOR = "#8FAADC"
HIGHLIGHT_COLOR = "#2563EB"
TRACK_COLOR = "#EEF2F7"
TEXT_COLOR = "#172033"
MUTED_COLOR = "#697386"


def plot_forecast_ladder(rec: pd.DataFrame, outdir: Path) -> None:
    rec = rec.set_index("tier").loc[LADDER_ORDER]
    tiers = list(reversed(LADDER_ORDER))
    labels = [LADDER_LABELS[t] for t in tiers]
    tick_labels = [
        f"<b>{LADDER_LABELS[t]}</b><br>"
        f"<span style='font-size:11px;color:{MUTED_COLOR}'>{LADDER_DETAILS[t]}</span>"
        for t in tiers
    ]
    values = [100 * rec.loc[t, "capture"] for t in tiers]
    colors = [HIGHLIGHT_COLOR if t == "blend_rl" else BAR_COLOR for t in tiers]

    fig = go.Figure()

    # Pale tracks make the distance to perfect foresight visible without a
    # heavy reference line dominating the chart.
    fig.add_trace(go.Bar(
        x=[100] * len(tiers),
        y=labels,
        orientation="h",
        marker=dict(color=TRACK_COLOR, line_width=0),
        hoverinfo="skip",
        showlegend=False,
        width=0.48,
    ))
    fig.add_trace(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=colors, line_width=0),
        customdata=[[LADDER_DETAILS[t]] for t in tiers],
        hovertemplate=(
            "<b>%{y}</b><br>%{customdata[0]}<br>Capture %{x:.1f}%<extra></extra>"
        ),
        showlegend=False,
        width=0.48,
    ))

    for tier, label, value in zip(tiers, labels, values):
        fig.add_annotation(
            x=value + 0.8,
            y=label,
            text=f"<b>{value:.1f}%</b>",
            showarrow=False,
            xanchor="left",
            font=dict(
                family="Arial, sans-serif",
                size=17,
                color=HIGHLIGHT_COLOR if tier == "blend_rl" else TEXT_COLOR,
            ),
        )
    fig.add_vline(x=100, line_width=1.3, line_dash="dot", line_color="#AAB2C0")
    fig.add_annotation(
        x=100,
        y=1.06,
        yref="paper",
        text="PERFECT FORESIGHT",
        showarrow=False,
        xanchor="right",
        font=dict(family="Arial, sans-serif", size=10, color=MUTED_COLOR),
    )
    fig.update_layout(
        title=dict(
            text=(
                "<b>Forecast capture improves with richer signals</b>"
                f"<br><span style='font-size:14px;color:{MUTED_COLOR}'>"
                f"Same dispatch LP · {int(rec['n_days'].iloc[0]):,} days · zero lookahead"
                "</span>"
            ),
            x=0.04,
            xanchor="left",
            y=0.96,
            yanchor="top",
            font=dict(family="Arial, sans-serif", size=25, color=TEXT_COLOR),
        ),
        template=TEMPLATE,
        barmode="overlay",
        bargap=0.5,
        width=1200,
        height=720,
        margin=dict(l=275, r=90, t=125, b=70),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", color=TEXT_COLOR),
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="Arial, sans-serif"),
    )
    fig.update_xaxes(
        range=[0, 104],
        tickvals=[0, 20, 40, 60, 80, 100],
        ticksuffix="%",
        title=None,
        showgrid=False,
        zeroline=False,
        showline=False,
        tickfont=dict(size=12, color=MUTED_COLOR),
    )
    fig.update_yaxes(
        title=None,
        showgrid=False,
        ticks="",
        tickmode="array",
        tickvals=labels,
        ticktext=tick_labels,
        tickfont=dict(size=15, color=TEXT_COLOR),
        ticklabelstandoff=18,
        automargin=True,
    )
    fig.write_html(outdir / "forecast_capture_ladder.html", include_plotlyjs="cdn")
    fig.write_image(outdir / "forecast_capture_ladder.png", width=1200, height=720, scale=2)
    logger.info("saved forecast_capture_ladder.{html,png}")


def main() -> None:
    cfg = load_config()
    outdir = ROOT / cfg["paths"]["processed"]["dir"]
    rec = pd.read_csv(outdir / "forecast_gap_recovery.csv")
    plot_forecast_ladder(rec, outdir)


if __name__ == "__main__":
    main()
