from __future__ import annotations

import pandas as pd
import pytest

from portfolio.construction import PortfolioConstructor


def test_portfolio_construction_decile_long_only_equal_weight() -> None:
    scores = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 10,
            "ticker": list("ABCDEFGHIJ"),
            "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )

    out = PortfolioConstructor(use_score_weighting=False, use_long_short=False).build_weights(scores)
    positive = out[out["weight"] > 0].sort_values("rank")
    negative = out[out["weight"] < 0]

    assert len(positive) == 2
    assert len(negative) == 0
    assert set(positive["ticker"].tolist()) == {"H", "I"}
    assert positive["weight"].tolist() == [0.5, 0.5]
    assert out["weight"].sum() == 1.0
    assert out["weight"].abs().sum() == 1.0


def test_portfolio_construction_decile_long_only_score_weighted() -> None:
    scores = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 10,
            "ticker": list("ABCDEFGHIJ"),
            "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )

    out = PortfolioConstructor(use_score_weighting=True, use_long_short=False).build_weights(scores)
    positive = out[out["weight"] > 0].set_index("ticker")

    assert positive.loc["H", "raw_score"] == pytest.approx(8.0)
    assert positive.loc["I", "raw_score"] == pytest.approx(9.0)
    assert positive.loc["H", "score_weight"] == pytest.approx(8.0)
    assert positive.loc["I", "score_weight"] == pytest.approx(9.0)
    assert positive.loc["H", "normalized_weight"] == pytest.approx(8.0 / 17.0)
    assert positive.loc["I", "normalized_weight"] == pytest.approx(9.0 / 17.0)
    assert positive.loc["H", "weight"] == pytest.approx(8.0 / 17.0)
    assert positive.loc["I", "weight"] == pytest.approx(9.0 / 17.0)
    assert out["weight"].sum() == pytest.approx(1.0)


def test_portfolio_construction_handles_missing_side_with_no_positions() -> None:
    scores = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 2,
            "ticker": ["A", "B"],
            "score": [0.9, 0.7],
        }
    )

    out = PortfolioConstructor(use_long_short=False).build_weights(scores)
    assert (out["weight"] == 0.0).all()


def test_portfolio_construction_sector_cap_limits_exposure() -> None:
    scores = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 10,
            "ticker": list("ABCDEFGHIJ"),
            "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )
    sector_map = {
        "H": "Tech",
        "I": "Tech",
    }

    out = PortfolioConstructor(max_sector_weight=0.2, use_long_short=False).build_weights(scores, sector_by_ticker=sector_map)
    positive = out[out["weight"] > 0].copy()
    positive["sector"] = positive["ticker"].map(sector_map).fillna("UNKNOWN")
    sector_weights = positive.groupby("sector")["weight"].sum()

    assert (sector_weights <= 0.2 + 1e-9).all()
    assert out["weight"].sum() <= 1.0


def test_portfolio_construction_can_disable_score_weighting_for_equal_weight() -> None:
    scores = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 10,
            "ticker": list("ABCDEFGHIJ"),
            "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )

    out = PortfolioConstructor(use_score_weighting=False, use_long_short=False).build_weights(scores)
    positive = out[out["weight"] > 0].sort_values("rank")

    assert positive["score_weight"].tolist() == [1.0, 1.0]
    assert positive["normalized_weight"].tolist() == [0.5, 0.5]
    assert positive["weight"].tolist() == [0.5, 0.5]
    assert out["weight"].sum() == pytest.approx(1.0)


def test_portfolio_construction_inverse_volatility_weights() -> None:
    scores = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 10,
            "ticker": list("ABCDEFGHIJ"),
            "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )
    vol_snapshot = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 2,
            "ticker": ["H", "I"],
            "realized_vol_60d": [0.2, 0.4],
        }
    )

    out = PortfolioConstructor(use_volatility_targeting=True, use_long_short=False).build_weights(
        scores=scores,
        volatility_snapshot=vol_snapshot,
    )
    positive = out[out["weight"] > 0].set_index("ticker")
    expected_h = (1 / 0.2) / ((1 / 0.2) + (1 / 0.4))
    expected_i = (1 / 0.4) / ((1 / 0.2) + (1 / 0.4))

    assert positive.loc["H", "volatility_60d"] == 0.2
    assert positive.loc["I", "volatility_60d"] == 0.4
    assert positive.loc["H", "score_weight"] == pytest.approx(8.0)
    assert positive.loc["I", "score_weight"] == pytest.approx(9.0)
    assert positive.loc["H", "weight"] == pytest.approx(expected_h)
    assert positive.loc["I", "weight"] == pytest.approx(expected_i)
    assert out["weight"].sum() == pytest.approx(1.0)


def test_portfolio_construction_long_short_score_weighted_is_market_neutral() -> None:
    scores = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 10,
            "ticker": list("ABCDEFGHIJ"),
            "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )

    out = PortfolioConstructor(use_long_short=True, use_score_weighting=True).build_weights(scores)
    longs = out[out["weight"] > 0].set_index("ticker")
    shorts = out[out["weight"] < 0].set_index("ticker")

    assert set(longs.index.tolist()) == {"H", "I"}
    assert set(shorts.index.tolist()) == {"A", "B"}
    assert longs["side"].tolist() == ["long", "long"]
    assert shorts["side"].tolist() == ["short", "short"]
    assert longs["weight"].sum() == pytest.approx(1.0)
    assert shorts["weight"].sum() == pytest.approx(-1.0)
    assert out["weight"].sum() == pytest.approx(0.0)
    assert out["weight"].abs().sum() == pytest.approx(2.0)


def test_portfolio_construction_long_short_sector_cap_applies_to_both_sides() -> None:
    scores = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 10,
            "ticker": list("ABCDEFGHIJ"),
            "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )
    sector_map = {
        "A": "Tech",
        "B": "Tech",
        "H": "Health",
        "I": "Health",
    }

    out = PortfolioConstructor(max_sector_weight=0.2, use_long_short=True).build_weights(
        scores,
        sector_by_ticker=sector_map,
    )
    longs = out[out["weight"] > 0].copy()
    shorts = out[out["weight"] < 0].copy()
    longs["sector"] = longs["ticker"].map(sector_map).fillna("UNKNOWN")
    shorts["sector"] = shorts["ticker"].map(sector_map).fillna("UNKNOWN")

    long_sector_weights = longs.groupby("sector")["weight"].sum()
    short_sector_weights = shorts.groupby("sector")["weight"].sum().abs()

    assert (long_sector_weights <= 0.2 + 1e-9).all()
    assert (short_sector_weights <= 0.2 + 1e-9).all()
