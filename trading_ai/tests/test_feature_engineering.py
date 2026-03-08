from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering.features import FeatureEngineer


def test_feature_engineering_adds_technical_features_for_mock_style_prices() -> None:
    periods = 80
    dates = pd.date_range("2022-01-03", periods=periods, freq="B")
    close = pd.Series(np.linspace(100.0, 180.0, periods))

    prices = pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAPL",
            "close": close,
        }
    )
    sentiments = pd.DataFrame({"date": dates, "ticker": "AAPL", "sentiment": 0.1})

    out = FeatureEngineer().transform(prices, sentiments)

    expected_columns = {
        "ret_10d",
        "ret_20d",
        "vol_10d",
        "vol_30d",
        "SMA_10",
        "SMA_20",
        "SMA_50",
        "distance_from_SMA_10",
        "distance_from_SMA_20",
        "distance_from_SMA_50",
        "RSI_14",
        "MACD",
        "MACD_signal",
        "volume_zscore",
        "volume_change_5d",
        "high_low_range_pct",
        "close_open_gap_pct",
    }

    for col in expected_columns:
        assert col in out.columns

    idx = 30
    expected_ret_10d = close.iloc[idx] / close.iloc[idx - 10] - 1.0
    assert out.loc[idx, "ret_10d"] == pytest.approx(expected_ret_10d)

    expected_sma_10 = close.iloc[idx - 9 : idx + 1].mean()
    assert out.loc[idx, "SMA_10"] == pytest.approx(expected_sma_10)

    # Mock prices do not include OHLCV, so fallback behavior should produce zero structure features.
    assert out["high_low_range_pct"].dropna().eq(0.0).all()
    assert out["close_open_gap_pct"].dropna().eq(0.0).all()


def test_feature_engineering_uses_ohlcv_when_available() -> None:
    periods = 80
    dates = pd.date_range("2022-01-03", periods=periods, freq="B")
    close = pd.Series(np.linspace(100.0, 180.0, periods))
    open_ = close * 0.99
    high = close * 1.01
    low = close * 0.98
    volume = pd.Series(np.linspace(1_000_000, 2_000_000, periods))

    prices = pd.DataFrame(
        {
            "date": dates,
            "ticker": "MSFT",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    sentiments = pd.DataFrame({"date": dates, "ticker": "MSFT", "sentiment": -0.1})

    out = FeatureEngineer().transform(prices, sentiments)
    idx = 40
    expected_range = (high.iloc[idx] - low.iloc[idx]) / close.iloc[idx]
    expected_gap = (close.iloc[idx] - open_.iloc[idx]) / open_.iloc[idx]

    assert out.loc[idx, "high_low_range_pct"] == pytest.approx(expected_range)
    assert out.loc[idx, "close_open_gap_pct"] == pytest.approx(expected_gap)
