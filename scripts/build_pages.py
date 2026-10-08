"""Build the zero-backend GitHub Pages edition of the public monitor."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_site"
LABELS = {
    "persist_d1": "D-1 persistence",
    "persist_d7": "D-7 persistence",
    "blend": "D-1/D-7 rolling OLS",
    "gbm": "Gradient boosting",
    "rl_quad": "Residual-load OLS",
    "blend_rl": "Price + residual-load OLS",
}
INFO = {
    "persist_d1": "Yesterday's cleared price curve",
    "persist_d7": "Same weekday's cleared price curve",
    "blend": "D-1 and D-7 price curves",
    "gbm": "Prices, residual load and calendar",
    "rl_quad": "Day-ahead residual-load forecast",
    "blend_rl": "D-1, D-7 and residual-load forecast",
}


def load_inputs() -> tuple[dict, list[dict]]:
    summary = json.loads((ROOT / "data/processed/kpi_summary.json").read_text())
    with (ROOT / "data/processed/forecast_gap_recovery.csv").open(newline="") as handle:
        ladder = list(csv.DictReader(handle))
    return summary, ladder


def model_rows(ladder: list[dict]) -> str:
    rows = []
    for item in ladder:
        tier = item["tier"]
        capture = float(item["capture"]) * 100
        recovery = float(item["gap_recovered_vs_persist_d1"]) * 100
        rows.append(
            f"""<tr><td><strong>{html.escape(LABELS.get(tier, tier))}</strong><small>{html.escape(INFO.get(tier, ''))}</small></td>
            <td>{int(item['n_days']):,}</td><td>{capture:.1f}%</td><td>{recovery:.1f}%</td></tr>"""
        )
    return "\n".join(rows)


def bars(ladder: list[dict]) -> str:
    return "\n".join(
        f"""<div class="bar-row"><span>{html.escape(LABELS.get(item['tier'], item['tier']))}</span>
        <div class="track"><i style="width:{float(item['capture']) * 100:.1f}%"></i></div>
        <b>{float(item['capture']) * 100:.1f}%</b></div>"""
        for item in ladder
    )


def build() -> Path:
    summary, ladder = load_inputs()
    headline = summary["full_sample_mean"]
    best = max(float(row["capture"]) for row in ladder)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Open, reproducible German BESS forecast-value research by Heesun Joo.">
<title>German BESS Forecast Value Monitor</title>
<style>
:root{{--ink:#17211b;--muted:#526158;--green:#197a54;--pale:#e8f3ed;--paper:#f7f7f2;--line:#d8e0da}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui,-apple-system,sans-serif}}
main{{max-width:1120px;margin:auto;padding:36px 24px 70px}} a{{color:var(--green)}}
nav{{display:flex;justify-content:space-between;align-items:center;margin-bottom:58px}} nav strong{{letter-spacing:.04em}} nav div{{display:flex;gap:18px}}
.eyebrow{{color:var(--green);font-size:13px;font-weight:800;letter-spacing:.1em;text-transform:uppercase}}
h1{{font-size:clamp(42px,7vw,76px);line-height:1.01;letter-spacing:-.045em;max-width:900px;margin:10px 0 20px}}
.lead{{font-size:20px;color:var(--muted);max-width:780px}} .chips{{display:flex;gap:8px;flex-wrap:wrap;margin:25px 0 44px}}
.chip{{background:var(--pale);border:1px solid #c8dfd2;border-radius:999px;padding:6px 12px;font-size:13px;font-weight:700}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}} .card{{background:white;border:1px solid var(--line);border-radius:14px;padding:20px}}
.card small{{display:block;color:var(--muted)}} .card b{{font-size:30px;letter-spacing:-.03em}}
.brief{{background:var(--ink);color:white;border-radius:18px;padding:28px;margin:24px 0 55px;font-size:18px}} .brief strong{{color:#87ddb8}}
section{{margin:58px 0}} h2{{font-size:34px;letter-spacing:-.03em;margin-bottom:8px}} .sub{{color:var(--muted);max-width:760px}}
.bar-row{{display:grid;grid-template-columns:220px 1fr 65px;gap:14px;align-items:center;margin:14px 0}} .track{{height:16px;background:#dfe6e1;border-radius:9px;overflow:hidden}}
.track i{{height:100%;display:block;background:linear-gradient(90deg,#95cdb0,var(--green));border-radius:9px}}
table{{width:100%;border-collapse:collapse;background:white;border:1px solid var(--line)}} th,td{{padding:13px;text-align:left;border-bottom:1px solid var(--line)}} th{{font-size:12px;text-transform:uppercase;color:var(--muted)}} td small{{display:block;color:var(--muted)}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}} .panel{{border:1px solid var(--line);background:white;border-radius:14px;padding:22px}}
footer{{border-top:1px solid var(--line);padding-top:25px;color:var(--muted);display:flex;justify-content:space-between;gap:18px}}
@media(max-width:760px){{.metrics,.grid{{grid-template-columns:1fr 1fr}}.bar-row{{grid-template-columns:130px 1fr 52px;font-size:13px}}table{{font-size:13px}}nav{{margin-bottom:35px}}}}
@media(max-width:520px){{.metrics,.grid{{grid-template-columns:1fr}}nav div{{display:none}}}}
</style></head><body><main>
<nav><strong>HEESUN JOO · OPEN RESEARCH</strong><div><a href="#results">Results</a><a href="#method">Method</a><a href="https://github.com/Heesunjookr/bess-arbitrage">GitHub</a></div></nav>
<div class="eyebrow">German BESS Forecast Value Monitor · DE-LU</div>
<h1>Forecast quality, measured in battery value.</h1>
<p class="lead">A reproducible study that converts day-ahead forecast quality into dispatch value, compares executable model tiers and exposes every material assumption.</p>
<div class="chips"><span class="chip">1,632 delivery days</span><span class="chip">No-lookahead schedules</span><span class="chip">Open methodology</span><span class="chip">44 tests passing</span></div>
<div class="metrics"><div class="card"><small>Backtest decisions</small><b>{summary['sample']['n_days_total']:,}</b></div>
<div class="card"><small>D-1 capture</small><b>{headline['capture_persist_d1']:.1%}</b></div>
<div class="card"><small>Best model capture</small><b>{best:.1%}</b></div>
<div class="card"><small>Forecast-value gap</small><b>€{headline['forecast_value_gap_d1_eur_per_mw_yr']/1000:.1f}k</b><small>per MW / year</small></div></div>
<div class="brief"><strong>Decision brief.</strong> On the common sample, D-1 persistence captures 77.6% of the perfect-foresight ceiling. Adding a day-ahead residual-load forecast lifts capture to 92.3% and recovers 65.7% of the gap. Information set and objective mattered more than model complexity.</div>
<section id="results"><div class="eyebrow">Executive results</div><h2>One optimisation harness. Six information sets.</h2><p class="sub">Every tier submits a fixed schedule and settles it against realised prices. Perfect foresight is a research ceiling, never an executable strategy.</p>
{bars(ladder)}
<table><thead><tr><th>Model tier</th><th>Days</th><th>Capture</th><th>Gap recovered</th></tr></thead><tbody>{model_rows(ladder)}</tbody></table></section>
<section id="method"><div class="eyebrow">Credibility layer</div><h2>Designed to be challenged.</h2><div class="grid">
<div class="panel"><h3>Reproduction contract</h3><ol><li>Validate data health.</li><li>Rebuild every tier walk-forward.</li><li>Freeze schedules before settlement.</li><li>Regenerate structured KPI artifacts.</li><li>Run the test suite before publishing.</li></ol></div>
<div class="panel"><h3>Interpretation contract</h3><ul><li>All results are simulated backtests.</li><li>Negative findings remain visible.</li><li>SMARD fundamental tiers are an upper bound pending point-in-time data.</li><li>No realised P&amp;L, intraday or balancing claim.</li></ul></div></div></section>
<section><div class="eyebrow">Data provenance</div><h2>Three artifacts behind the public monitor.</h2>
<table><thead><tr><th>Layer</th><th>Versioned artifact</th><th>Source</th></tr></thead><tbody>
<tr><td>Market prices</td><td>data/public/da_prices.csv.gz</td><td>ENTSO-E Transparency</td></tr>
<tr><td>Headline KPIs</td><td>data/processed/kpi_summary.json</td><td>Backtest pipeline</td></tr>
<tr><td>Model ladder</td><td>data/processed/forecast_gap_recovery.csv</td><td>Walk-forward evaluation</td></tr></tbody></table></section>
<footer><span>Research by Heesun Joo · Python · cvxpy · pandas · scikit-learn</span><span><a href="https://github.com/Heesunjookr/bess-arbitrage">Source &amp; tests</a> · <a href="https://github.com/Heesunjookr/bess-arbitrage/blob/main/BUILDING_IN_PUBLIC.md">Roadmap</a></span></footer>
</main></body></html>"""
    OUT.mkdir(exist_ok=True)
    target = OUT / "index.html"
    target.write_text(page, encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build())
