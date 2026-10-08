import json

from scripts.build_pages import build, build_price_bundle


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
    payload = json.loads((target.parent / "prices.json").read_text())
    assert payload["days"]
