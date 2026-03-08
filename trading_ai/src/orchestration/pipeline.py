from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtesting.walk_forward import WalkForwardBacktester
from execution.simulator import ExecutionSimulator
from feature_engineering.features import FeatureEngineer
from feature_engineering.labels import LabelGenerator
from ingestion.mock_data import MockMarketDataIngestor
from ingestion.yahoo_data import YahooFinanceDataProvider
from nlp_engine.sentiment import SentimentPipeline
from portfolio.construction import PortfolioConstructor
from risk.manager import RiskManager
from storage.data_store import InMemoryDataStore


@dataclass
class ResearchConfig:
    tickers: list[str]
    periods: int = 520
    seed: int = 42
    data_provider: str = "mock"
    top_n: int = 10


class TradingResearchPipeline:
    FEATURE_COLUMNS = [
        "ret_1d",
        "ret_5d",
        "ret_10d",
        "ret_20d",
        "vol_10d",
        "vol_20d",
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
        "sentiment",
        "sentiment_5d",
    ]

    def __init__(self, config: ResearchConfig) -> None:
        self.config = config
        self.store = InMemoryDataStore()

    def run(self) -> dict[str, pd.DataFrame | float]:
        if self.config.data_provider == "mock":
            ingestor = MockMarketDataIngestor(seed=self.config.seed)
        elif self.config.data_provider == "yahoo":
            ingestor = YahooFinanceDataProvider(seed=self.config.seed)
        else:
            raise ValueError(f"Unsupported data_provider='{self.config.data_provider}'")

        prices = ingestor.generate_prices(self.config.tickers, periods=self.config.periods)
        news = ingestor.generate_news(prices)

        sentiment = SentimentPipeline().transform(news)
        features = FeatureEngineer().transform(prices, sentiment)
        labeled = LabelGenerator(horizon=5).transform(features)

        self.store.write("prices", prices)
        self.store.write("news", news)
        self.store.write("dataset", labeled)

        backtester = WalkForwardBacktester(features=self.FEATURE_COLUMNS)
        scores = backtester.run(labeled)

        portfolio = PortfolioConstructor(top_n=max(1, self.config.top_n)).build_weights(scores)
        vol_snapshot = labeled[["date", "ticker", "vol_20d"]].drop_duplicates()
        risk_adjusted = RiskManager().apply(portfolio[["date", "ticker", "weight"]], vol_snapshot)

        execution = ExecutionSimulator()
        performance = execution.simulate(risk_adjusted, prices)

        sharpe = self._annualized_sharpe(performance["strategy_return"]) if not performance.empty else float("nan")

        return {
            "prices": prices,
            "news": news,
            "dataset": labeled,
            "scores": scores,
            "portfolio": risk_adjusted,
            "constructed_portfolio": portfolio,
            "performance": performance,
            "sharpe": sharpe,
        }

    @staticmethod
    def _annualized_sharpe(returns: pd.Series) -> float:
        if returns.std(ddof=0) == 0:
            return 0.0
        return float(np.sqrt(252) * returns.mean() / returns.std(ddof=0))
