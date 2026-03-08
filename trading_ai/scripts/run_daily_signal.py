from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.features import FeatureEngineer
from feature_engineering.labels import LabelGenerator
from models.baseline import BaselineClassifier
from nlp_engine.sentiment import SentimentPipeline
from orchestration.pipeline import TradingResearchPipeline
from portfolio.construction import PortfolioConstructor
from ingestion.yahoo_data import YahooFinanceDataProvider

SP500_50_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "TSLA",
    "JPM", "WMT", "XOM", "UNH", "V", "MA", "ORCL", "COST", "PG", "JNJ",
    "HD", "BAC", "ABBV", "KO", "CRM", "NFLX", "CVX", "MRK", "CSCO", "WFC",
    "ACN", "IBM", "LIN", "MCD", "ABT", "PM", "GE", "AMD", "INTU", "DIS",
    "TXN", "T", "VZ", "CAT", "PFE", "AMGN", "QCOM", "NOW", "SPGI", "INTC",
]
TRANSACTION_COST_RATE = 0.0005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate daily long-only decile 8-9 signal targets from Yahoo data")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=SP500_50_TICKERS,
        help="Ticker universe to rank (default: S&P 500 sample used by run_research)",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=520,
        help="Lookback bars to download from Yahoo before generating features and training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used by existing synthetic-news pipeline",
    )
    parser.add_argument(
        "--current-portfolio",
        type=Path,
        default=Path("results/current_portfolio.csv"),
        help="Optional current holdings CSV. If missing, actions default to BUY/HOLD placeholders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/daily_signals.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--rebalance-frequency",
        choices=["daily", "weekly", "biweekly"],
        default="daily",
        help="Rebalance schedule for transitioning from current to target weights",
    )
    return parser.parse_args()


def _load_current_portfolio(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "current_weight"])

    current = pd.read_csv(path)
    expected_cols = {"ticker", "current_weight"}
    missing = expected_cols.difference(current.columns)
    if missing:
        raise ValueError(
            f"Invalid current portfolio format in {path}. Missing columns: {sorted(missing)}"
        )

    out = current[["ticker", "current_weight"]].copy()
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["current_weight"] = pd.to_numeric(out["current_weight"], errors="coerce").fillna(0.0)
    out = out.groupby("ticker", as_index=False)["current_weight"].sum()
    return out


def _derive_action(target_weight: float, current_weight: float, has_current_file: bool, tol: float = 1e-6) -> str:
    if not has_current_file:
        return "BUY" if target_weight > tol else "HOLD"

    if abs(target_weight - current_weight) <= tol:
        return "HOLD"
    if target_weight <= tol and current_weight > tol:
        return "EXIT"
    if target_weight > current_weight + tol:
        return "BUY"
    if target_weight > tol and current_weight > target_weight + tol:
        return "REDUCE"
    if target_weight <= tol and current_weight <= tol:
        return "HOLD"
    return "SELL"


def _rebalance_step(frequency: str) -> int:
    if frequency == "daily":
        return 1
    if frequency == "weekly":
        return 5
    if frequency == "biweekly":
        return 10
    raise ValueError(f"Unsupported rebalance_frequency='{frequency}'")


def _is_rebalance_day(dates: pd.Series, frequency: str) -> bool:
    step = _rebalance_step(frequency)
    if step == 1:
        return True
    unique_dates = pd.Index(pd.to_datetime(dates).dropna().sort_values().unique())
    if unique_dates.empty:
        return True
    return (len(unique_dates) - 1) % step == 0


def main() -> None:
    args = parse_args()

    # Ensure yfinance cache writes to a project-local path that is writable.
    try:
        import yfinance as yf

        cache_dir = Path("results/.yfinance_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        yf.cache.set_cache_location(str(cache_dir.resolve()))
        yf.set_tz_cache_location(str(cache_dir.resolve()))
    except Exception:
        # Cache config is best-effort; download can still work with defaults.
        pass

    prices = YahooFinanceDataProvider(seed=args.seed).generate_prices(
        tickers=args.tickers,
        periods=args.periods,
    )
    news = YahooFinanceDataProvider(seed=args.seed).generate_news(prices)
    sentiment = SentimentPipeline().transform(news)
    features = FeatureEngineer().transform(prices, sentiment)
    labeled = LabelGenerator(horizon=5).transform(features)

    feature_cols = TradingResearchPipeline.FEATURE_COLUMNS
    modeled = labeled.dropna(subset=feature_cols + ["future_return"]).copy()
    if modeled.empty:
        raise ValueError("No valid rows after feature/label preparation. Increase --periods or review Yahoo data.")

    latest_date = modeled["date"].max()
    train = modeled[modeled["date"] < latest_date]
    today = modeled[modeled["date"] == latest_date]
    if train.empty or today.empty:
        raise ValueError("Not enough data to train and score latest day. Increase --periods.")

    model = BaselineClassifier(random_state=args.seed)
    model.fit(train[feature_cols].to_numpy(), train["label"].to_numpy())

    today_scores = today[["date", "ticker"]].copy()
    today_scores["score"] = model.predict_proba(today[feature_cols].to_numpy())

    ranked = today_scores.sort_values("score", ascending=False).reset_index(drop=True)
    ranked["decile"] = np.ceil(ranked["score"].rank(method="first", pct=True) * 10).astype(int).clip(1, 10)

    daily_targets = PortfolioConstructor(top_n=10).build_weights(ranked)[
        ["date", "ticker", "score", "decile", "weight"]
    ].rename(columns={"weight": "target_weight"})
    daily_targets = daily_targets[daily_targets["date"] == latest_date].copy()

    current = _load_current_portfolio(args.current_portfolio)
    has_current_file = args.current_portfolio.exists()
    rebalance_due = _is_rebalance_day(modeled["date"], args.rebalance_frequency)

    if not rebalance_due:
        if has_current_file:
            # Hold current weights between scheduled rebalances.
            daily_targets = current.copy().assign(
                date=latest_date,
                score=np.nan,
                decile=pd.NA,
            )[["date", "ticker", "score", "decile", "current_weight"]].rename(
                columns={"current_weight": "target_weight"}
            )
        else:
            daily_targets["target_weight"] = 0.0

    merged = daily_targets.merge(current, on="ticker", how="outer")
    merged["date"] = merged["date"].fillna(latest_date)
    merged["score"] = merged["score"].astype(float)
    merged["decile"] = merged["decile"].astype("Int64")
    merged["target_weight"] = merged["target_weight"].fillna(0.0)
    merged["current_weight"] = merged["current_weight"].fillna(0.0)
    merged["turnover"] = (merged["target_weight"] - merged["current_weight"]).abs()
    merged["estimated_cost"] = merged["turnover"] * TRANSACTION_COST_RATE
    merged["action"] = merged.apply(
        lambda row: _derive_action(
            target_weight=float(row["target_weight"]),
            current_weight=float(row["current_weight"]),
            has_current_file=has_current_file,
        ),
        axis=1,
    )

    output = merged[
        ["date", "ticker", "score", "decile", "target_weight", "turnover", "estimated_cost", "action"]
    ].copy()
    output["date"] = pd.to_datetime(output["date"]).dt.normalize()
    output = output.sort_values(["target_weight", "score"], ascending=[False, False], na_position="last")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    print("=== Daily Signal Bot ===")
    print(f"Date: {latest_date.date()}")
    print(f"Rebalance frequency: {args.rebalance_frequency}")
    print(f"Rebalance due today: {'yes' if rebalance_due else 'no'}")
    print(f"Universe size scored: {len(today_scores)}")
    print(f"Target holdings (decile 8-9): {(output['target_weight'] > 0).sum()}")
    print(f"Estimated one-way transition cost: {output['estimated_cost'].sum():.4%}")
    print(f"Output: {args.output}")
    if has_current_file:
        print(f"Current portfolio compared: {args.current_portfolio}")
    else:
        print(f"No current portfolio file found at {args.current_portfolio}; emitted BUY/HOLD placeholders.")


if __name__ == "__main__":
    main()
