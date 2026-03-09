from __future__ import annotations

import pandas as pd


class RiskManager:
    def __init__(self, max_position: float = 0.2, target_vol: float = 0.12) -> None:
        self.max_position = max_position
        self.target_vol = target_vol

    def apply(self, weights: pd.DataFrame, vol_snapshot: pd.DataFrame) -> pd.DataFrame:
        merged = weights.merge(vol_snapshot, on=["date", "ticker"], how="left")
        merged["vol_20d"] = merged["vol_20d"].replace(0, pd.NA).fillna(0.02)

        # Vol targeting via scalar applied per date.
        date_vol = merged.groupby("date")["vol_20d"].mean().replace(0, 0.02)
        scaler = (self.target_vol / date_vol).clip(upper=2.0)
        merged = merged.join(scaler.rename("vol_scalar"), on="date")

        merged["weight"] = (merged["weight"] * merged["vol_scalar"]).clip(
            lower=-self.max_position,
            upper=self.max_position,
        )
        return merged[["date", "ticker", "weight"]]

    @staticmethod
    def build_market_regime(spy_prices: pd.DataFrame, ma_window: int = 200) -> pd.DataFrame:
        if spy_prices.empty:
            return pd.DataFrame(columns=["date", "spy_close", "spy_sma_200", "regime_favorable"])

        regime = spy_prices.copy()
        regime["date"] = pd.to_datetime(regime["date"])
        regime = regime.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        regime["spy_close"] = pd.to_numeric(regime["close"], errors="coerce")
        regime["spy_sma_200"] = regime["spy_close"].rolling(ma_window, min_periods=ma_window).mean()
        regime["regime_favorable"] = (regime["spy_close"] > regime["spy_sma_200"]).fillna(False)
        return regime[["date", "spy_close", "spy_sma_200", "regime_favorable"]]

    @staticmethod
    def apply_market_regime(weights: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
        if weights.empty:
            return weights

        if regime.empty:
            out = weights.copy()
            out["spy_close"] = pd.NA
            out["spy_sma_200"] = pd.NA
            out["regime_favorable"] = True
            return out

        out = weights.copy()
        out["date"] = pd.to_datetime(out["date"])
        reg = regime.copy()
        reg["date"] = pd.to_datetime(reg["date"])
        out = out.merge(reg, on="date", how="left")
        out["regime_favorable"] = out["regime_favorable"].fillna(False)
        out["weight"] = out["weight"].where(out["regime_favorable"], 0.0)
        return out
