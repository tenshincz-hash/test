from __future__ import annotations

import numpy as np
import pandas as pd

from feature_engineering.features import FeatureEngineer


def test_factor_features_are_created_with_past_only_transforms() -> None:
    dates = pd.bdate_range("2024-01-01", periods=300)
    base = np.arange(300, dtype=float)

    prices = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAA"] * 300,
            "open": 100.0 + base,
            "high": 101.0 + base,
            "low": 99.0 + base,
            "close": 100.0 + base,
            "volume": 1_000_000.0 + base,
        }
    )
    sentiments = pd.DataFrame({"date": dates, "ticker": ["AAA"] * 300, "sentiment": [0.0] * 300})
    spy = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["SPY"] * 300,
            "open": 200.0 + base * 1.2,
            "high": 201.0 + base * 1.2,
            "low": 199.0 + base * 1.2,
            "close": 200.0 + base * 1.2,
            "volume": 1_000_000.0,
        }
    )

    out = FeatureEngineer().transform(prices=prices, sentiments=sentiments, benchmark_prices=spy)

    for col in [
        "mom_1m",
        "mom_3m",
        "mom_6m",
        "mom_12m",
        "realized_vol_20d",
        "realized_vol_60d",
        "rel_strength_3m",
        "rel_strength_6m",
        "rel_strength_12m",
    ]:
        assert col in out.columns

    row = out.iloc[260]
    expected_mom_12m = prices["close"].iloc[260] / prices["close"].iloc[20] - 1.0
    expected_spy_12m = spy["close"].iloc[260] / spy["close"].iloc[20] - 1.0

    assert np.isclose(row["mom_12m"], expected_mom_12m)
    assert np.isclose(row["rel_strength_12m"], expected_mom_12m - expected_spy_12m)
