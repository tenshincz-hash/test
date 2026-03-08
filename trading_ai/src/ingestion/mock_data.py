from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MockMarketDataIngestor:
    seed: int = 42

    def generate_prices(
        self,
        tickers: list[str],
        start: str = "2020-01-01",
        periods: int = 500,
        freq: str = "B",
    ) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        dates = pd.date_range(start=start, periods=periods, freq=freq)
        records = []

        for ticker in tickers:
            drift = rng.uniform(0.00005, 0.0006)
            vol = rng.uniform(0.008, 0.02)
            shocks = rng.normal(loc=drift, scale=vol, size=periods)
            close = 100 * np.exp(np.cumsum(shocks))
            for dt, px in zip(dates, close):
                records.append({"date": dt, "ticker": ticker, "close": float(px)})

        return pd.DataFrame(records).sort_values(["date", "ticker"]).reset_index(drop=True)

    def generate_news(self, prices: pd.DataFrame) -> pd.DataFrame:
        positive = [
            "beats earnings expectations",
            "raises guidance",
            "strong demand outlook",
            "announces strategic partnership",
        ]
        negative = [
            "misses earnings forecast",
            "cuts guidance",
            "regulatory pressure increases",
            "supply chain disruptions continue",
        ]
        neutral = [
            "holds investor day",
            "executes planned product update",
            "maintains current strategy",
            "reports quarterly operating metrics",
        ]

        rng = np.random.default_rng(self.seed + 7)
        rows = []
        for row in prices.itertuples(index=False):
            bucket = rng.choice([positive, negative, neutral], p=[0.35, 0.30, 0.35])
            headline = f"{row.ticker} {rng.choice(bucket)}"
            rows.append({"date": row.date, "ticker": row.ticker, "headline": headline})

        return pd.DataFrame(rows)
