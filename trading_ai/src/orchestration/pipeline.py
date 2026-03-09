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
    rebalance_frequency: str = "daily"
    sector_by_ticker: dict[str, str] | None = None
    max_sector_weight: float = 1.0
    use_volatility_targeting: bool = True


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
        "mom_1m",
        "mom_3m",
        "mom_6m",
        "mom_12m",
        "realized_vol_20d",
        "realized_vol_60d",
        "rel_strength_3m",
        "rel_strength_6m",
        "rel_strength_12m",
        "sentiment",
        "sentiment_5d",
    ]

    def __init__(self, config: ResearchConfig) -> None:
        if config.rebalance_frequency not in {"daily", "weekly", "biweekly"}:
            raise ValueError(
                f"Unsupported rebalance_frequency='{config.rebalance_frequency}'. "
                "Use one of: daily, weekly, biweekly."
            )
        self.config = config
        self.store = InMemoryDataStore()

    def run(self) -> dict[str, pd.DataFrame | float]:
        regime = pd.DataFrame(columns=["date", "spy_close", "spy_sma_200", "regime_favorable"])
        spy_prices = pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])
        if self.config.data_provider == "mock":
            ingestor = MockMarketDataIngestor(seed=self.config.seed)
        elif self.config.data_provider == "yahoo":
            ingestor = YahooFinanceDataProvider(seed=self.config.seed)
        else:
            raise ValueError(f"Unsupported data_provider='{self.config.data_provider}'")

        prices = ingestor.generate_prices(self.config.tickers, periods=self.config.periods)
        if self.config.data_provider == "yahoo":
            spy_prices = ingestor.generate_prices(["SPY"], periods=self.config.periods + 250)
        news = ingestor.generate_news(prices)

        sentiment = SentimentPipeline().transform(news)
        features = FeatureEngineer().transform(
            prices,
            sentiment,
            benchmark_prices=spy_prices if self.config.data_provider == "yahoo" else None,
        )
        labeled = LabelGenerator(horizon=5).transform(features)

        self.store.write("prices", prices)
        self.store.write("news", news)
        self.store.write("dataset", labeled)

        backtester = WalkForwardBacktester(features=self.FEATURE_COLUMNS)
        scores = backtester.run(labeled)

        portfolio = PortfolioConstructor(
            top_n=max(1, self.config.top_n),
            max_sector_weight=self.config.max_sector_weight,
            use_volatility_targeting=self.config.use_volatility_targeting,
        ).build_weights(
            scores,
            sector_by_ticker=self.config.sector_by_ticker,
            volatility_snapshot=labeled[["date", "ticker", "realized_vol_60d"]].drop_duplicates(),
        )
        vol_snapshot = labeled[["date", "ticker", "vol_20d"]].drop_duplicates()
        risk_adjusted = RiskManager().apply(portfolio[["date", "ticker", "weight"]], vol_snapshot)
        scheduled = self._apply_rebalance_schedule(
            risk_adjusted,
            frequency=self.config.rebalance_frequency,
        )
        if self.config.data_provider == "yahoo":
            regime = RiskManager.build_market_regime(spy_prices)
        scheduled = RiskManager.apply_market_regime(scheduled, regime)

        execution = ExecutionSimulator()
        performance = execution.simulate(scheduled, prices)

        sharpe = self._annualized_sharpe(performance["strategy_return"]) if not performance.empty else float("nan")

        return {
            "prices": prices,
            "news": news,
            "dataset": labeled,
            "scores": scores,
            "portfolio": scheduled,
            "constructed_portfolio": portfolio,
            "performance": performance,
            "sharpe": sharpe,
            "regime": regime,
        }

    @staticmethod
    def _annualized_sharpe(returns: pd.Series) -> float:
        if returns.std(ddof=0) == 0:
            return 0.0
        return float(np.sqrt(252) * returns.mean() / returns.std(ddof=0))

    @staticmethod
    def _rebalance_step(frequency: str) -> int:
        if frequency == "daily":
            return 1
        if frequency == "weekly":
            return 5
        if frequency == "biweekly":
            return 10
        raise ValueError(f"Unsupported rebalance_frequency='{frequency}'")

    @classmethod
    def _apply_rebalance_schedule(cls, portfolio: pd.DataFrame, frequency: str) -> pd.DataFrame:
        if portfolio.empty or frequency == "daily":
            return portfolio

        step = cls._rebalance_step(frequency)
        out = portfolio.copy()
        unique_dates = sorted(pd.to_datetime(out["date"]).unique())
        rebalance_dates = set(unique_dates[::step])
        out = out.sort_values(["ticker", "date"])
        out["is_rebalance_date"] = out["date"].isin(rebalance_dates)
        out["weight"] = (
            out["weight"]
            .where(out["is_rebalance_date"])
            .groupby(out["ticker"])
            .ffill()
            .fillna(0.0)
        )
        return out.drop(columns=["is_rebalance_date"])
