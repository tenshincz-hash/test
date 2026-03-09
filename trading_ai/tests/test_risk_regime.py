from __future__ import annotations

import pandas as pd

from risk.manager import RiskManager


def test_build_market_regime_uses_spy_close_above_sma_200_rule() -> None:
    dates = pd.bdate_range("2025-01-01", periods=205)
    closes = [100.0] * 200 + [99.0, 101.0, 102.0, 103.0, 104.0]
    spy = pd.DataFrame({"date": dates, "close": closes})

    regime = RiskManager.build_market_regime(spy, ma_window=200)

    assert not regime.empty
    assert regime["regime_favorable"].iloc[199] == False
    assert regime["regime_favorable"].iloc[200] == False
    assert regime["regime_favorable"].iloc[201] == True


def test_apply_market_regime_forces_cash_when_unfavorable() -> None:
    weights = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")],
            "ticker": ["AAPL", "AAPL"],
            "weight": [0.2, 0.2],
        }
    )
    regime = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")],
            "spy_close": [600.0, 590.0],
            "spy_sma_200": [595.0, 595.0],
            "regime_favorable": [True, False],
        }
    )

    out = RiskManager.apply_market_regime(weights, regime)

    assert out["weight"].tolist() == [0.2, 0.0]
    assert out["regime_favorable"].tolist() == [True, False]
