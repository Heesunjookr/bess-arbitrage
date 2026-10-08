"""
plots.py
--------
Backtest result visualization (Plotly) and structured KPI exports.

Every chart is written twice to data/processed/:
  - <name>.html : interactive (Plotly, CDN js)
  - <name>.png  : static render via kaleido (embedded in the README)

Charts:
  - dispatch_profile : example-day price curve + charge/discharge schedule
  - soc_curve        : example-day battery state of charge
  - revenue_by_year  : per-year per-tier revenue (EUR/MW/yr)
  - capture_ratio    : executable / perfect-foresight per year
  - sensitivity      : PF revenue vs efficiency / duration / throughput cost

KPI exports:
  - kpi_yearly.csv   : tidy long format (year, tier, metric, value)
  - kpi_summary.json : full-sample headline numbers for programmatic use

Run:  python -m src.viz.plots   (regenerates everything, ~2 min)
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import ROOT, get_logger, load_config
from src.data.load_prices import load_prices
from src.optimize.lp_dispatch import BatteryParams, solve_day
from src.backtest.revenue import run_backtest, summarize_yearly, build_findings_table

logger = get_logger("plots")

TEMPLATE = "plotly_white"
TIER_LABELS = {
    "pf_da": "Perfect foresight (ceiling)",
    "persist_d1": "Persistence D-1 (executable)",
    "persist_d7": "Persistence D-7 (executable)",
    "blend": "D-1/D-7 rolling-OLS blend (executable)",
}
TIER_COLORS = {
    "pf_da": "#1f77b4",
    "persist_d1": "#d62728",
    "persist_d7": "#ff7f0e",
    "blend": "#2ca02c",
}


def _outdir(cfg: dict) -> Path:
    d = ROOT / cfg["paths"]["processed"]["dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig: go.Figure, outdir: Path, name: str,
          width: int = 1000, height: int = 500) -> None:
    fig.write_html(outdir / ("%s.html" % name), include_plotlyjs="cdn")
    fig.write_image(outdir / ("%s.png" % name), width=width, height=height, scale=2)
    logger.info("saved %s.{html,png}", name)


def plot_dispatch_and_soc(df: pd.DataFrame, p: BatteryParams, outdir: Path,
                          example_date: Optional[str] = None) -> None:
    """Dispatch profile and SoC curve for the highest-spread day."""
    if example_date is None:
        rng = df.groupby("date")["da_price"].agg(lambda x: x.max() - x.min())
        example_date = str(rng.idxmax())
    sub = df[df["date"].astype(str) == example_date].sort_index()
    prices = sub["da_price"].to_numpy(dtype=float)
    r = solve_day(prices, p)
    hours = np.arange(len(prices))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=hours, y=r.charge, name="Charge (MW)",
                         marker_color="#1f77b4"))
    fig.add_trace(go.Bar(x=hours, y=-r.discharge, name="Discharge (MW)",
                         marker_color="#d62728"))
    fig.add_trace(go.Scatter(x=hours, y=prices, name="DA price",
                             mode="lines+markers", line=dict(color="black")),
                  secondary_y=True)
    fig.update_layout(
        title="Perfect-foresight dispatch — %s (revenue %.0f EUR)"
              % (example_date, r.revenue),
        barmode="relative", template=TEMPLATE,
        xaxis_title="Hour of day",
    )
    fig.update_yaxes(title_text="Power (MW) [+charge / −discharge]",
                     secondary_y=False)
    fig.update_yaxes(title_text="DA price (EUR/MWh)", secondary_y=True)
    _save(fig, outdir, "dispatch_profile")

    fig = go.Figure(go.Scatter(x=hours, y=r.soc, mode="lines+markers",
                               line=dict(color="#2ca02c"), name="SoC"))
    fig.add_hline(y=p.e_max_mwh, line_dash="dash", line_color="gray",
                  annotation_text="E_max")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title="Battery state of charge — %s" % example_date,
                      xaxis_title="Hour of day",
                      yaxis_title="State of charge (MWh)",
                      template=TEMPLATE)
    _save(fig, outdir, "soc_curve", height=400)


def plot_revenue_by_year(findings: pd.DataFrame, outdir: Path) -> None:
    """Grouped bars: per-year revenue by tier (EUR/MW/yr)."""
    fig = go.Figure()
    for tier, label in TIER_LABELS.items():
        col = "%s_per_mw" % tier
        if col in findings.columns:
            fig.add_trace(go.Bar(x=findings.index.astype(str), y=findings[col],
                                 name=label, marker_color=TIER_COLORS[tier]))
    fig.update_layout(
        title="BESS day-ahead arbitrage revenue by year and tier "
              "(DE-LU, 2 MWh / 1 MW)",
        yaxis_title="Revenue (EUR / MW / yr)", barmode="group",
        template=TEMPLATE,
    )
    _save(fig, outdir, "revenue_by_year")


def plot_capture_ratio(findings: pd.DataFrame, outdir: Path) -> None:
    """Executable revenue as a share of the perfect-foresight ceiling."""
    fig = go.Figure()
    for tier in ("persist_d1", "persist_d7", "blend"):
        col = "capture_%s" % tier
        if col in findings.columns:
            fig.add_trace(go.Bar(
                x=findings.index.astype(str), y=findings[col],
                name=TIER_LABELS[tier], marker_color=TIER_COLORS[tier],
                text=(findings[col] * 100).round(0).astype(int).astype(str) + "%",
                textposition="outside",
            ))
    fig.update_layout(
        title="Capture ratio: executable bidding tiers / "
              "perfect-foresight ceiling",
        yaxis_title="Capture ratio", yaxis_range=[0, 1], barmode="group",
        template=TEMPLATE,
    )
    _save(fig, outdir, "capture_ratio")


def compute_sensitivity(df: pd.DataFrame, base: BatteryParams,
                        sample_every: int = 5) -> dict:
    """
    One-at-a-time parameter sensitivity of PF-DA revenue, on every
    `sample_every`-th day (annualized). Returns {param: DataFrame}.
    """
    dates = sorted(df["date"].unique())[::sample_every]
    sample = df[df["date"].isin(dates)]
    n_days = len(dates)
    ann = 365.0 / n_days

    def run(p: BatteryParams) -> float:
        total = 0.0
        for _, sub in sample.groupby("date"):
            sub = sub.sort_index()
            if len(sub) != 24:
                continue
            total += solve_day(sub["da_price"].to_numpy(float), p).revenue
        return total * ann / p.p_max_mw

    out = {}
    effs = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    out["efficiency"] = pd.DataFrame(
        {"per_mw": [run(BatteryParams(base.e_max_mwh, base.p_max_mw, e,
                                      base.throughput_cost_eur_mwh)) for e in effs]},
        index=effs)

    durations = [1.0, 2.0, 4.0, 6.0, 8.0]  # h = E_max / P_max
    out["duration"] = pd.DataFrame(
        {"per_mw": [run(BatteryParams(d * base.p_max_mw, base.p_max_mw,
                                      base.eta_round_trip,
                                      base.throughput_cost_eur_mwh))
                    for d in durations]},
        index=durations)

    tputs = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
    out["throughput"] = pd.DataFrame(
        {"per_mw": [run(BatteryParams(base.e_max_mwh, base.p_max_mw,
                                      base.eta_round_trip, tc)) for tc in tputs]},
        index=tputs)
    return out


def plot_sensitivity(sens: dict, outdir: Path) -> None:
    """Three-panel sensitivity figure."""
    specs = [
        ("efficiency", "Round-trip efficiency"),
        ("duration", "Duration (h = E_max/P_max)"),
        ("throughput", "Throughput cost (EUR/MWh)"),
    ]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[t for _, t in specs])
    for i, (key, _) in enumerate(specs, start=1):
        d = sens[key]
        fig.add_trace(go.Scatter(x=d.index, y=d["per_mw"],
                                 mode="lines+markers",
                                 marker_color="#1f77b4", showlegend=False),
                      row=1, col=i)
    fig.update_yaxes(title_text="PF-DA revenue (EUR/MW/yr)", row=1, col=1)
    fig.update_layout(
        title="Perfect-foresight DA revenue sensitivity "
              "(sampled days, annualized)",
        template=TEMPLATE,
    )
    _save(fig, outdir, "sensitivity", width=1400, height=450)


def export_kpis(yearly: pd.DataFrame, findings: pd.DataFrame,
                p: BatteryParams, outdir: Path) -> None:
    """
    Structured KPI exports for downstream/reporting use.

    kpi_yearly.csv : tidy long format — one row per (year, tier, metric).
    kpi_summary.json : battery config + full-sample averages + per-year table.
    """
    tiers = [c[len("rev_"):] for c in yearly.columns if c.startswith("rev_")]
    metric_cols = {"revenue_eur_per_mw_yr": "per_mw_",
                   "revenue_eur_per_mwh_yr": "per_mwh_",
                   "equivalent_full_cycles_yr": "cycles_"}
    rows = []
    for year in yearly.index:
        for tier in tiers:
            for metric, prefix in metric_cols.items():
                col = prefix + tier
                if col in yearly.columns:
                    rows.append({"year": int(year), "tier": tier,
                                 "metric": metric,
                                 "value": round(float(yearly.loc[year, col]), 1)})
            if tier != "pf_da":
                for metric, prefix in [("capture_ratio", "capture_"),
                                       ("forecast_value_gap_eur_per_mw_yr", "gap_")]:
                    col = prefix + tier
                    if col in findings.columns:
                        rows.append({"year": int(year), "tier": tier,
                                     "metric": metric,
                                     "value": round(float(findings.loc[year, col]), 3)})
    tidy = pd.DataFrame(rows)
    tidy.to_csv(outdir / "kpi_yearly.csv", index=False)

    summary = {
        "asset": {
            "market": "DE-LU day-ahead",
            "e_max_mwh": p.e_max_mwh,
            "p_max_mw": p.p_max_mw,
            "eta_round_trip": p.eta_round_trip,
            "throughput_cost_eur_mwh": p.throughput_cost_eur_mwh,
        },
        "sample": {
            "years": [int(y) for y in yearly.index],
            "n_days_total": int(yearly["n_days"].sum()),
        },
        "full_sample_mean": {
            "pf_da_eur_per_mw_yr": round(float(findings["pf_da_per_mw"].mean()), 0),
            "persist_d1_eur_per_mw_yr": round(float(findings["persist_d1_per_mw"].mean()), 0),
            "blend_eur_per_mw_yr": round(float(findings["blend_per_mw"].mean()), 0),
            "capture_persist_d1": round(float(findings["capture_persist_d1"].mean()), 3),
            "capture_blend": round(float(findings["capture_blend"].mean()), 3),
            "forecast_value_gap_d1_eur_per_mw_yr": round(float(findings["gap_persist_d1"].mean()), 0),
            "forecast_value_gap_blend_eur_per_mw_yr": round(float(findings["gap_blend"].mean()), 0),
        },
        "by_year": json.loads(findings.round(3).to_json(orient="index")),
    }
    with open(outdir / "kpi_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("saved kpi_yearly.csv (%d rows) and kpi_summary.json", len(tidy))


def main() -> None:
    cfg = load_config()
    p = BatteryParams.from_config(cfg)
    outdir = _outdir(cfg)
    df = load_prices(cfg)

    daily = run_backtest(df=df, p=p, cfg=cfg)
    yearly = summarize_yearly(daily, p)
    findings = build_findings_table(yearly)

    daily.to_parquet(outdir / "daily_revenue.parquet")
    yearly.to_csv(outdir / "yearly_summary.csv")
    findings.to_csv(outdir / "findings_table.csv")
    export_kpis(yearly, findings, p, outdir)

    plot_dispatch_and_soc(df, p, outdir)
    plot_revenue_by_year(findings, outdir)
    plot_capture_ratio(findings, outdir)

    sens = compute_sensitivity(df, p)
    plot_sensitivity(sens, outdir)
    for k, v in sens.items():
        v.to_csv(outdir / ("sensitivity_%s.csv" % k))

    logger.info("all artifacts written -> %s", outdir)


if __name__ == "__main__":
    main()
