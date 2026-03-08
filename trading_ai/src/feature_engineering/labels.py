from __future__ import annotations

import pandas as pd


class LabelGenerator:
    def __init__(self, horizon: int = 5, threshold: float = 0.0) -> None:
        self.horizon = horizon
        self.threshold = threshold

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy().sort_values(["ticker", "date"])
        out["future_return"] = (
            out.groupby("ticker")["close"].shift(-self.horizon) / out["close"] - 1.0
        )
        out["label"] = (out["future_return"] > self.threshold).astype(int)
        return out
