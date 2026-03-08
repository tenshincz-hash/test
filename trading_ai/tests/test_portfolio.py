from __future__ import annotations

import pandas as pd

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
