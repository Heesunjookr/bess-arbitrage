"""
validate.py
-----------
Data-quality checks for the raw price series.

Motivation (post-mortem, 2026-07): the series originally used as the German
intraday price (`id_prices.parquet`, inherited from the sibling da-id study)
turned out NOT to be intraday prices at all. The upstream fetch labeled SMARD
filters 4996/4997 as "Intraday Index", but per the official SMARD API spec:

    filter 4996 = "Marktpreis: Belgien"    (Belgium day-ahead price)
    filter 4997 = "Marktpreis: Norwegen 2" (Norway NO2 day-ahead price)

SMARD publishes no German intraday index at all. This module encodes the
checks that exposed the problem, so the same class of error is caught
automatically instead of by accident:

  1. Structural completeness: duplicate timestamps, missing hours, DST days.
  2. Hour-of-day shape: a DE-LU delivery-period price must show the solar
     "duck curve" (midday trough, morning/evening peaks). NO2 hydro prices
     are nearly flat across the day -> flagged.
  3. Cross-market coupling fingerprint: share of hours exactly equal to the
     DE-LU DA price. Belgium is market-coupled to DE-LU, so its DA price
     matches DE-LU to the cent in ~20% of hours -- far too often for an
     intraday index, which is a volume-weighted average of trades.
  4. MTU-switch signature: a quarter-hourly series that is constant within
     the hour before 2025-10 (hourly products) and varies within the hour
     afterwards is a day-ahead series that followed the 15-minute MTU
     switch -- another DA fingerprint.

Run:  python -m src.data.validate
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.config import ROOT, get_logger, load_config

logger = get_logger("validate")

# A DE-LU price series should have a pronounced daily shape. Ratio of the
# candidate's mean hour-of-day profile range to the DA profile range below
# this threshold means "suspiciously flat for this bidding zone".
SHAPE_RATIO_MIN = 0.5
# Share of hours exactly equal (to the cent) to DE-LU DA above this
# threshold means "this is a coupled day-ahead price, not a traded index".
EXACT_MATCH_MAX = 0.05


def structural_checks(s: pd.Series, name: str) -> Dict[str, float]:
    """Duplicate timestamps, gaps, and basic sanity for one hourly series."""
    dups = int(s.index.duplicated().sum())
    s = s[~s.index.duplicated()]
    full = pd.date_range(s.index.min(), s.index.max(), freq="1h", tz=s.index.tz)
    missing = int(len(full) - len(s.dropna().reindex(full).dropna()))
    out = {
        "n_obs": int(s.notna().sum()),
        "duplicate_timestamps": dups,
        "missing_hours": missing,
    }
    logger.info("%s: %d obs, %d duplicate ts, %d missing hours",
                name, out["n_obs"], dups, missing)
    return out


def hour_profile_ratio(candidate: pd.Series, reference_da: pd.Series) -> float:
    """
    Range of the candidate's mean hour-of-day profile relative to DA's.

    ~1.0 -> same daily shape as the DE-LU duck curve; << 1 -> flat profile,
    i.e. the series does not describe DE-LU delivery periods.
    """
    j = pd.concat([reference_da, candidate], axis=1, join="inner").dropna()
    j.columns = ["da", "x"]
    prof = j.groupby(j.index.hour).mean()
    da_range = prof["da"].max() - prof["da"].min()
    x_range = prof["x"].max() - prof["x"].min()
    return float(x_range / da_range) if da_range > 0 else np.nan


def exact_match_share(candidate: pd.Series, reference_da: pd.Series,
                      tol: float = 0.01) -> float:
    """Share of hours where the candidate equals DE-LU DA to within `tol` EUR."""
    j = pd.concat([reference_da, candidate], axis=1, join="inner").dropna()
    j.columns = ["da", "x"]
    return float((np.abs(j["da"] - j["x"]) < tol).mean())


def intra_hour_constancy_by_year(quarter_series: pd.Series) -> pd.Series:
    """
    For a quarter-hourly series: share of hours whose four quarters are all
    equal, per year. A jump from ~1.0 to ~0.0 at the 2025-10 15-minute MTU
    switch is a day-ahead fingerprint.
    """
    s = quarter_series.dropna()
    s = s[~s.index.duplicated()]
    q = s.tz_convert("UTC").to_frame("q")
    q["hour"] = q.index.floor("h")
    nuniq = q.groupby("hour")["q"].nunique()
    nuniq.index = nuniq.index.tz_convert(s.index.tz)
    return nuniq.groupby(nuniq.index.year).apply(lambda g: float((g == 1).mean()))


def validate_series(candidate: pd.Series, reference_da: pd.Series,
                    name: str) -> Dict[str, object]:
    """Run all delivery-period plausibility checks for one hourly series."""
    shape = hour_profile_ratio(candidate, reference_da)
    exact = exact_match_share(candidate, reference_da)
    corr = pd.concat([reference_da, candidate], axis=1,
                     join="inner").dropna().corr().iloc[0, 1]
    flags = []
    if shape < SHAPE_RATIO_MIN:
        flags.append("FLAT_PROFILE (no DE-LU duck curve -> wrong market/period)")
    if exact > EXACT_MATCH_MAX:
        flags.append("DA_COUPLING (matches DE-LU DA to the cent too often)")
    verdict = "FAIL" if flags else "ok"
    logger.info("%s: shape_ratio=%.2f exact_match=%.1f%% corr_da=%.3f -> %s %s",
                name, shape, exact * 100, corr, verdict, "; ".join(flags))
    return {"name": name, "shape_ratio": shape, "exact_match_share": exact,
            "corr_with_da": float(corr), "verdict": verdict, "flags": flags}


def make_diagnostic_figure(da: pd.Series, candidates: Dict[str, pd.Series],
                           out_path) -> None:
    """Mean hour-of-day profiles: the one-glance view that exposed the issue."""
    import plotly.graph_objects as go

    fig = go.Figure()
    j = pd.concat([da] + list(candidates.values()), axis=1, join="inner").dropna()
    j.columns = ["DE-LU DA"] + list(candidates.keys())
    prof = j.groupby(j.index.hour).mean()
    for col in prof.columns:
        fig.add_trace(go.Scatter(x=prof.index, y=prof[col], mode="lines+markers",
                                 name=col))
    fig.update_layout(
        title=("Mean hour-of-day price profile — the check that exposed the "
               "mislabeled 'intraday' series"),
        xaxis_title="Hour of day (Europe/Berlin)",
        yaxis_title="Mean price (EUR/MWh)",
        template="plotly_white", height=450,
    )
    fig.write_html(str(out_path) + ".html", include_plotlyjs="cdn")
    fig.write_image(str(out_path) + ".png", width=1000, height=450, scale=2)
    logger.info("diagnostic figure saved -> %s.{html,png}", out_path)


def main() -> None:
    cfg = load_config()
    da = pd.read_parquet(ROOT / cfg["paths"]["raw"]["da_prices"])["da_price"]
    da = da[~da.index.duplicated()]

    neigh = pd.read_parquet(ROOT / cfg["paths"]["raw"]["neighbor_prices"])
    # Columns were originally (mis)labeled id_price_hour / id_price_quarter.
    hour = neigh["id_price_hour"].dropna()
    hour = hour[~hour.index.duplicated()]
    quarter_raw = neigh["id_price_quarter"].dropna()
    quarter_raw = quarter_raw[~quarter_raw.index.duplicated()]
    quarter_hourly = quarter_raw.resample("1h").mean()

    print("\n=== structural checks ===")
    structural_checks(da, "da_price (DE-LU DA)")
    structural_checks(hour, "'id_price_hour' (actually SMARD 4997 = NO2 DA)")

    print("\n=== delivery-period plausibility vs DE-LU DA ===")
    print("(reference: DE-LU DA has shape_ratio=1.0 by definition)")
    results = [
        validate_series(hour, da, "'id_price_hour' -> SMARD 4997 (Norway NO2 DA)"),
        validate_series(quarter_hourly, da,
                        "'id_price_quarter' -> SMARD 4996 (Belgium DA)"),
    ]

    print("\n=== 15-min MTU-switch signature ('id_price_quarter') ===")
    const = intra_hour_constancy_by_year(quarter_raw)
    print(const.round(2).to_string())
    print("(constant within the hour until 2025, then 15-min values -> "
          "day-ahead series that followed the 2025-10 MTU switch)")

    outdir = ROOT / cfg["paths"]["processed"]["dir"]
    outdir.mkdir(parents=True, exist_ok=True)
    make_diagnostic_figure(
        da,
        {"'id_price_hour' (= NO2 DA)": hour,
         "'id_price_quarter' (= Belgium DA)": quarter_hourly},
        outdir / "data_quality_profiles",
    )

    failed = [r["name"] for r in results if r["verdict"] == "FAIL"]
    print("\nVERDICT: %d series failed delivery-period checks: %s"
          % (len(failed), failed))
    print("=> neither series is a German intraday price; "
          "the backtest uses DE-LU DA only.")


if __name__ == "__main__":
    main()
