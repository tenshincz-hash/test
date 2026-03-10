from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from execution.signal_schedule import is_rebalance_day, next_rebalance_date
from feature_engineering.features import FeatureEngineer
from feature_engineering.labels import LabelGenerator
from models.baseline import BaselineClassifier
from nlp_engine.sentiment import SentimentPipeline
from orchestration.pipeline import TradingResearchPipeline
from portfolio.construction import PortfolioConstructor
from ingestion.yahoo_data import YahooFinanceDataProvider
from risk.manager import RiskManager
from universe_loader import load_tickers_from_csv

DEFAULT_UNIVERSE_FILE = Path("configs/universe_500.csv")
TRANSACTION_COST_RATE = 0.0005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate daily long-only decile 8-9 signal targets from Yahoo data")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Ticker universe to rank (overrides --universe-file)",
    )
    parser.add_argument(
        "--universe-file",
        type=Path,
        default=DEFAULT_UNIVERSE_FILE,
        help="CSV file containing ticker universe (columns: ticker or symbol)",
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
        "--portfolio-value",
        type=float,
        default=None,
        help="Optional total portfolio value (e.g., 100000) to emit current/target/delta values.",
    )
    parser.add_argument(
        "--rebalance-frequency",
        choices=["daily", "weekly", "biweekly"],
        default="weekly",
        help="Rebalance schedule for transitioning from current to target weights",
    )
    parser.add_argument(
        "--disable-score-weighting",
        action="store_true",
        help="Disable score-based sizing and use equal weights for selected stocks",
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


def main() -> None:
    args = parse_args()
    tickers = args.tickers if args.tickers is not None else load_tickers_from_csv(args.universe_file)

    data_provider = YahooFinanceDataProvider(seed=args.seed)

    prices = data_provider.generate_prices(
        tickers=tickers,
        periods=args.periods,
    )
    news = data_provider.generate_news(prices)
    sentiment = SentimentPipeline().transform(news)
    features = FeatureEngineer().transform(prices, sentiment)
    labeled = LabelGenerator(horizon=5).transform(features)

    feature_cols = TradingResearchPipeline.FEATURE_COLUMNS
    modeled = labeled.dropna(subset=feature_cols + ["future_return"]).copy()
    if modeled.empty:
        raise ValueError("No valid rows after feature/label preparation. Increase --periods or review Yahoo data.")

    latest_date = modeled["date"].max()
    spy_prices = data_provider.generate_prices(["SPY"], periods=args.periods + 250)
    regime = RiskManager.build_market_regime(spy_prices)
    regime_today = regime[regime["date"] == latest_date]
    if regime_today.empty:
        raise ValueError(
            f"No SPY regime state available for latest model date {latest_date.date()}. Increase --periods."
        )
    regime_favorable = bool(regime_today["regime_favorable"].iloc[0])
    spy_close = float(regime_today["spy_close"].iloc[0])
    spy_sma_200 = float(regime_today["spy_sma_200"].iloc[0]) if pd.notna(regime_today["spy_sma_200"].iloc[0]) else np.nan

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

    model_targets = PortfolioConstructor(
        top_n=10,
        use_score_weighting=not args.disable_score_weighting,
    ).build_weights(ranked)[
        ["date", "ticker", "score", "raw_score", "score_weight", "normalized_weight", "decile", "weight"]
    ].rename(columns={"weight": "model_target_weight"})
    model_targets = model_targets[model_targets["date"] == latest_date].copy()
    if not regime_favorable:
        model_targets["model_target_weight"] = 0.0
    else:
        total_model_weight = float(model_targets["model_target_weight"].sum())
        if not np.isclose(total_model_weight, 1.0, atol=1e-9):
            raise ValueError(
                f"Model target weights must sum to 1.0 on rebalance dates. Observed {total_model_weight:.12f}."
            )

    current = _load_current_portfolio(args.current_portfolio)
    has_current_file = args.current_portfolio.exists()
    rebalance_due = is_rebalance_day(modeled["date"], args.rebalance_frequency)
    next_rebalance_date_value = next_rebalance_date(latest_date, args.rebalance_frequency)

    if rebalance_due:
        execution_targets = (
            model_targets[["date", "ticker", "model_target_weight"]]
            .rename(columns={"model_target_weight": "target_weight"})
            .copy()
        )
    elif has_current_file:
        # Hold current weights between scheduled rebalances.
        execution_targets = current.copy().assign(
            date=latest_date,
            target_weight=lambda df: df["current_weight"],
        )[["date", "ticker", "target_weight"]]
    else:
        # If current holdings are unknown, emit model targets as executable placeholders.
        execution_targets = (
            model_targets[["date", "ticker", "model_target_weight"]]
            .rename(columns={"model_target_weight": "target_weight"})
            .copy()
        )

    if not regime_favorable:
        execution_targets["target_weight"] = 0.0

    merged = execution_targets.merge(current, on="ticker", how="outer")
    merged = merged.merge(
        model_targets[
            [
                "ticker",
                "score",
                "raw_score",
                "score_weight",
                "normalized_weight",
                "decile",
                "model_target_weight",
            ]
        ],
        on="ticker",
        how="outer",
    )
    merged["date"] = merged["date"].fillna(latest_date)
    merged["score"] = merged["score"].astype(float)
    merged["raw_score"] = pd.to_numeric(merged["raw_score"], errors="coerce")
    merged["score_weight"] = pd.to_numeric(merged["score_weight"], errors="coerce").fillna(0.0)
    merged["normalized_weight"] = pd.to_numeric(merged["normalized_weight"], errors="coerce").fillna(0.0)
    merged["decile"] = merged["decile"].astype("Int64")
    merged["target_weight"] = merged["target_weight"].fillna(0.0)
    merged["model_target_weight"] = merged["model_target_weight"].fillna(0.0)
    merged["current_weight"] = pd.to_numeric(merged["current_weight"], errors="coerce")
    current_weight_for_calc = merged["current_weight"].fillna(0.0)
    merged["delta_weight"] = merged["target_weight"] - current_weight_for_calc
    merged["turnover"] = merged["delta_weight"].abs()
    merged["estimated_cost"] = merged["turnover"] * TRANSACTION_COST_RATE
    merged["action"] = merged.apply(
        lambda row: _derive_action(
            target_weight=float(row["target_weight"]),
            current_weight=float(0.0 if pd.isna(row["current_weight"]) else row["current_weight"]),
            has_current_file=has_current_file,
        ),
        axis=1,
    )
    merged["rebalance_due"] = rebalance_due
    merged["signal_type"] = "risk_off" if not regime_favorable else ("rebalance" if rebalance_due else "hold")
    merged["next_rebalance_date"] = next_rebalance_date_value
    merged["valid_until"] = next_rebalance_date_value
    merged["regime_favorable"] = regime_favorable
    merged["spy_close"] = spy_close
    merged["spy_sma_200"] = spy_sma_200

    if args.portfolio_value is not None:
        if args.portfolio_value < 0:
            raise ValueError("--portfolio-value must be non-negative.")
        merged["current_value"] = merged["current_weight"] * args.portfolio_value
        merged["target_value"] = merged["target_weight"] * args.portfolio_value
        merged["delta_value"] = merged["target_value"] - merged["current_value"].fillna(0.0)
    else:
        merged["current_value"] = np.nan
        merged["target_value"] = np.nan
        merged["delta_value"] = np.nan

    if not has_current_file:
        merged["current_weight"] = np.nan
        merged["delta_weight"] = np.nan
        merged["current_value"] = np.nan
        merged["delta_value"] = np.nan

    output = merged[
        [
            "date",
            "ticker",
            "score",
            "raw_score",
            "score_weight",
            "normalized_weight",
            "decile",
            "current_weight",
            "model_target_weight",
            "target_weight",
            "delta_weight",
            "current_value",
            "target_value",
            "delta_value",
            "turnover",
            "estimated_cost",
            "action",
            "signal_type",
            "rebalance_due",
            "next_rebalance_date",
            "valid_until",
            "regime_favorable",
            "spy_close",
            "spy_sma_200",
        ]
    ].copy()
    output["date"] = pd.to_datetime(output["date"]).dt.normalize()
    output["next_rebalance_date"] = pd.to_datetime(output["next_rebalance_date"]).dt.normalize()
    output["valid_until"] = pd.to_datetime(output["valid_until"]).dt.normalize()
    # Round for practical manual execution readability.
    output["score"] = output["score"].round(4)
    output["raw_score"] = output["raw_score"].round(4)
    output["score_weight"] = output["score_weight"].round(6)
    output["normalized_weight"] = output["normalized_weight"].round(6)
    output["current_weight"] = output["current_weight"].round(4)
    output["model_target_weight"] = output["model_target_weight"].round(4)
    output["target_weight"] = output["target_weight"].round(4)
    output["delta_weight"] = output["delta_weight"].round(4)
    output["turnover"] = output["turnover"].round(4)
    output["estimated_cost"] = output["estimated_cost"].round(6)
    output["current_value"] = output["current_value"].round(2)
    output["target_value"] = output["target_value"].round(2)
    output["delta_value"] = output["delta_value"].round(2)
    output["spy_close"] = output["spy_close"].round(4)
    output["spy_sma_200"] = output["spy_sma_200"].round(4)
    output = output.sort_values(["target_weight", "score"], ascending=[False, False], na_position="last")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    print("=== Daily Signal Bot ===")
    print(f"Date: {latest_date.date()}")
    print(f"Rebalance frequency: {args.rebalance_frequency}")
    print(f"Rebalance due today: {'yes' if rebalance_due else 'no'}")
    print(
        "Market regime (SPY close > 200D SMA): "
        f"{'favorable' if regime_favorable else 'unfavorable'} "
        f"(close={spy_close:.2f}, sma200={spy_sma_200:.2f})"
    )
    print(f"Next rebalance date: {next_rebalance_date_value.date()}")
    print(f"Universe size scored: {len(today_scores)}")
    print(f"Model target holdings (decile 8-9): {(output['model_target_weight'] > 0).sum()}")
    print(f"Executable target holdings today: {(output['target_weight'] > 0).sum()}")
    print(f"Estimated one-way transition cost: {output['estimated_cost'].sum():.4%}")
    print(f"Output: {args.output}")
    if has_current_file:
        print(f"Current portfolio compared: {args.current_portfolio}")
    else:
        print(f"No current portfolio file found at {args.current_portfolio}; emitted BUY/HOLD placeholders.")


if __name__ == "__main__":
    main()
