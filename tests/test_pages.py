import json

from scripts.build_pages import build, build_price_bundle, build_research_bundle


def test_price_bundle_only_contains_complete_day_pairs():
    bundle = build_price_bundle()
    assert bundle["market"] == "DE-LU day-ahead"
    assert len(bundle["days"]) > 1_000
    assert all(len(item["realised"]) == len(item["d1"]) == 24
               for item in bundle["days"].values())


def test_pages_build_contains_interactive_assets():
    target = build()
    assert target.is_file()
    assert "Run dispatch" in target.read_text()
    assert "__BUILD_ID__" not in target.read_text()
    assert "app.js?build=" in target.read_text()
    assert (target.parent / "app.js").is_file()
    assert (target.parent / "palette.css").is_file()
    payload = json.loads((target.parent / "prices.json").read_text())
    assert payload["days"]
    page = target.read_text()
    assert '>Day-Ahead</button>' in page
    assert '>Intraday</button>' in page
    assert '>Investment</button>' in page
    assert "ML exceeds the time-of-day baseline" in page
    assert "NO LIVE TRADE RECOMMENDATION" not in page
    research = json.loads((target.parent / "research.json").read_text())
    assert research["intraday"]["data_status"] == "validated"
    assert research["intraday"]["model_status"] == "research_only"
    assert research["intraday"]["incremental_value_eur"] < 0
    assert research["intraday"]["v2_holdout"]["ml_r2"] > 0
    assert research["intraday"]["production_ready"] is False
    assert research["intraday"]["recommendation"] is None


def test_research_bundle_uses_reviewed_da_outputs_only():
    bundle = build_research_bundle()
    assert bundle["sample"]["n_days_total"] > 1_000
    assert set(bundle["revenue_eur_per_mw_yr"]) == {"blend", "d1", "ceiling"}
    assert all(value > 0 for value in bundle["revenue_eur_per_mw_yr"].values())
