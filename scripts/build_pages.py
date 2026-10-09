"""Build the zero-backend, interactive GitHub Pages dispatch lab."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "web"
OUT = ROOT / "_site"


def build_price_bundle() -> dict:
    """Publish complete local-price days with a complete D-1 comparison curve."""
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with gzip.open(ROOT / "data/public/da_prices.csv.gz", "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            stamp = datetime.fromisoformat(row["timestamp_utc"]).astimezone(ZoneInfo("Europe/Berlin"))
            grouped[stamp.date().isoformat()].append((stamp.hour, float(row["da_price"])))
    complete = {day: [price for _, price in sorted(values)] for day, values in grouped.items()
                if len(values) == 24 and len({hour for hour, _ in values}) == 24}
    days = sorted(complete)
    eligible = [(previous, day) for previous, day in zip(days, days[1:])
                if (datetime.fromisoformat(day).date() - datetime.fromisoformat(previous).date()).days == 1]
    return {"market": "DE-LU day-ahead", "currency": "EUR/MWh", "resolution": "hourly",
            "days": {day: {"realised": complete[day], "d1": complete[previous]}
                     for previous, day in eligible}}


def build_research_bundle() -> dict:
    """Expose only reviewed study outputs used by the commercial screening tab."""
    summary = json.loads(
        (ROOT / "data/processed/kpi_summary.json").read_text(encoding="utf-8")
    )
    mean = summary["full_sample_mean"]
    return {
        "sample": summary["sample"],
        "revenue_eur_per_mw_yr": {
            "blend": mean["blend_eur_per_mw_yr"],
            "d1": mean["persist_d1_eur_per_mw_yr"],
            "ceiling": mean["pf_da_eur_per_mw_yr"],
        },
        "labels": {
            "blend": "Forecast-based blend",
            "d1": "D-1 persistence",
            "ceiling": "Perfect-foresight ceiling",
        },
        "intraday": {
            "status": "blocked",
            "reason": "No validated German intraday target or OOS signal is available.",
            "recommendation": None,
        },
    }


def build() -> Path:
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(SOURCE, OUT)
    app_hash = hashlib.sha256((OUT / "app.js").read_bytes()).hexdigest()[:12]
    index = OUT / "index.html"
    index.write_text(
        index.read_text(encoding="utf-8").replace("__BUILD_ID__", app_hash),
        encoding="utf-8",
    )
    (OUT / "prices.json").write_text(
        json.dumps(build_price_bundle(), separators=(",", ":")), encoding="utf-8"
    )
    (OUT / "research.json").write_text(
        json.dumps(build_research_bundle(), separators=(",", ":")), encoding="utf-8"
    )
    return index


if __name__ == "__main__":
    print(build())
