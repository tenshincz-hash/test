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

    out = PortfolioConstructor().build_weights(scores)
    positive = out[out["weight"] > 0].sort_values("rank")
    negative = out[out["weight"] < 0]

    assert len(positive) == 2
    assert len(negative) == 0
    assert set(positive["ticker"].tolist()) == {"H", "I"}
    assert positive["weight"].tolist() == [0.5, 0.5]
    assert out["weight"].sum() == 1.0
    assert out["weight"].abs().sum() == 1.0


def test_portfolio_construction_handles_missing_side_with_no_positions() -> None:
    scores = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 2,
            "ticker": ["A", "B"],
            "score": [0.9, 0.7],
        }
    )

    out = PortfolioConstructor().build_weights(scores)
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

    out = PortfolioConstructor(max_sector_weight=0.2).build_weights(scores, sector_by_ticker=sector_map)
    positive = out[out["weight"] > 0].copy()
    positive["sector"] = positive["ticker"].map(sector_map).fillna("UNKNOWN")
    sector_weights = positive.groupby("sector")["weight"].sum()

    assert (sector_weights <= 0.2 + 1e-9).all()
    assert out["weight"].sum() <= 1.0


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

    out = PortfolioConstructor(use_volatility_targeting=True).build_weights(
        scores=scores,
        volatility_snapshot=vol_snapshot,
    )
    positive = out[out["weight"] > 0].set_index("ticker")
    expected_h = (1 / 0.2) / ((1 / 0.2) + (1 / 0.4))
    expected_i = (1 / 0.4) / ((1 / 0.2) + (1 / 0.4))

    assert positive.loc["H", "volatility_60d"] == 0.2
    assert positive.loc["I", "volatility_60d"] == 0.4
    assert positive.loc["H", "weight"] == pytest.approx(expected_h)
    assert positive.loc["I", "weight"] == pytest.approx(expected_i)
    assert out["weight"].sum() == pytest.approx(1.0)
