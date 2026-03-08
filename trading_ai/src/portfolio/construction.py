from __future__ import annotations

import numpy as np
import pandas as pd


class PortfolioConstructor:
    def __init__(self, top_n: int = 10, gross_leverage: float = 2.0) -> None:
        self.top_n = top_n
        self.gross_leverage = gross_leverage

    def build_weights(self, scores: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for dt, grp in scores.groupby("date", sort=True):
            frame = grp.copy()
            frame["decile"] = np.ceil(frame["score"].rank(method="first", pct=True) * 10).astype(int).clip(1, 10)
            ranked = frame.sort_values("score", ascending=False).reset_index(drop=True)
            ranked["rank"] = ranked.index + 1
            if ranked.empty:
                continue

            frame = ranked[["date", "ticker", "score", "rank"]].copy()
            frame["weight"] = 0.0
            frame["decile"] = ranked["decile"]

            long_mask = frame["decile"].isin([8, 9])
            long_count = int(long_mask.sum())
            if long_count == 0:
                rows.append(frame)
                continue
            long_weight = 1.0 / long_count

            frame.loc[long_mask, "weight"] = long_weight
            rows.append(frame)

        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
