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
