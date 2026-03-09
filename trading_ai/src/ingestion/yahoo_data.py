from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class YahooFinanceDataProvider:
    seed: int = 42

    @staticmethod
    def _configure_yfinance_cache() -> None:
        import yfinance as yf

        cache_dir = Path("results/.yfinance_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        yf.cache.set_cache_location(str(cache_dir.resolve()))
        yf.set_tz_cache_location(str(cache_dir.resolve()))

    def generate_prices(
        self,
        tickers: list[str],
        start: str = "2020-01-01",
        periods: int = 500,
        freq: str = "B",
    ) -> pd.DataFrame:
        del freq  # Yahoo daily bars do not need pandas frequency input.

        if not tickers:
            return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])

        import yfinance as yf
        self._configure_yfinance_cache()

        raw = yf.download(
            tickers=tickers,
            start=start,
            interval="1d",
            auto_adjust=False,
            group_by="ticker",
            progress=False,
            threads=False,
        )

        records: list[pd.DataFrame] = []
        for ticker in tickers:
            ticker_frame = self._extract_ticker_frame(raw, ticker, len(tickers) == 1)
            if ticker_frame.empty:
                continue

            bars = (
                ticker_frame.rename(
                    columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    }
                )[["open", "high", "low", "close", "volume"]]
                .dropna(subset=["close"])
                .tail(periods)
                .copy()
            )
            bars["ticker"] = ticker
            bars["date"] = pd.to_datetime(bars.index).tz_localize(None)
            records.append(bars.reset_index(drop=True))

        if not records:
            raise ValueError("No Yahoo Finance data returned for requested tickers and date range.")

        out = pd.concat(records, ignore_index=True)
        out["volume"] = out["volume"].fillna(0).astype(float)
        return out[["date", "ticker", "open", "high", "low", "close", "volume"]].sort_values(
            ["date", "ticker"]
        ).reset_index(drop=True)

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

    @staticmethod
    def _extract_ticker_frame(raw: pd.DataFrame, ticker: str, single_ticker: bool) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame()
        if single_ticker:
            if isinstance(raw.columns, pd.MultiIndex):
                level0 = raw.columns.get_level_values(0)
                if ticker in level0:
                    return raw[ticker].copy()
                if "Price" in level0:
                    return raw["Price"].copy()
                return raw.copy()
            return raw.copy()
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker not in raw.columns.get_level_values(0):
                return pd.DataFrame()
            return raw[ticker].copy()
        return pd.DataFrame()
