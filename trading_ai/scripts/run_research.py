from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.baseline import BaselineClassifier
from orchestration.pipeline import ResearchConfig, TradingResearchPipeline

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"]
SP500_50_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "TSLA",
    "JPM", "WMT", "XOM", "UNH", "V", "MA", "ORCL", "COST", "PG", "JNJ",
    "HD", "BAC", "ABBV", "KO", "CRM", "NFLX", "CVX", "MRK", "CSCO", "WFC",
    "ACN", "IBM", "LIN", "MCD", "ABT", "PM", "GE", "AMD", "INTU", "DIS",
    "TXN", "T", "VZ", "CAT", "PFE", "AMGN", "QCOM", "NOW", "SPGI", "INTC",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end trading AI research pipeline")
    parser.add_argument(
        "--data-provider",
        choices=["mock", "yahoo"],
        default="mock",
        help="Market data provider: mock synthetic prices or Yahoo Finance daily bars",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="List of ticker symbols",
    )
    parser.add_argument("--periods", type=int, default=520, help="Number of business-day observations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for mock data")
    parser.add_argument("--top-n", type=int, default=10, help="Number of tickers per side (top N long, bottom N short)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tickers is None:
        tickers = SP500_50_TICKERS if args.data_provider == "yahoo" else DEFAULT_TICKERS
    else:
        tickers = args.tickers

    config = ResearchConfig(
        tickers=tickers,
        periods=args.periods,
        seed=args.seed,
        data_provider=args.data_provider,
        top_n=args.top_n,
    )
    pipeline = TradingResearchPipeline(config)
    results = pipeline.run()

    perf = results["performance"]
    total_return = perf["equity_curve"].iloc[-1] - 1 if not perf.empty else 0.0
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    results["scores"].to_csv(results_dir / "predictions.csv", index=False)
    trade_source = results.get("constructed_portfolio", results["portfolio"])
    trades = trade_source[trade_source["weight"] != 0].copy()
    trades["side"] = trades["weight"].map(lambda w: "long" if w > 0 else "short")
    trades.to_csv(results_dir / "trades.csv", index=False)
    perf.to_csv(results_dir / "equity_curve.csv", index=False)

    # Decile analysis: rank scores per rebalance date and evaluate next-day returns by score decile.
    scores = results["scores"].copy()
    prices = results["prices"].copy().sort_values(["ticker", "date"])
    prices["subsequent_return"] = prices.groupby("ticker")["close"].pct_change().shift(-1)
    scored = scores.merge(prices[["date", "ticker", "subsequent_return"]], on=["date", "ticker"], how="left")
    scored = scored.dropna(subset=["score", "subsequent_return"]).copy()
    if not scored.empty:
        scored["score_rank_pct"] = scored.groupby("date")["score"].rank(method="first", pct=True)
        scored["decile"] = np.ceil(scored["score_rank_pct"] * 10).astype(int).clip(lower=1, upper=10)

        decile_daily = (
            scored.groupby(["date", "decile"], as_index=False)
            .agg(mean_subsequent_return=("subsequent_return", "mean"), n_tickers=("ticker", "count"))
            .sort_values(["date", "decile"])
        )
    else:
        decile_daily = pd.DataFrame(columns=["date", "decile", "mean_subsequent_return", "n_tickers"])

    decile_daily.to_csv(results_dir / "decile_analysis.csv", index=False)

    decile_mean = (
        decile_daily.groupby("decile", as_index=False)["mean_subsequent_return"].mean().sort_values("decile")
        if not decile_daily.empty
        else pd.DataFrame({"decile": list(range(1, 11)), "mean_subsequent_return": [0.0] * 10})
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(decile_mean["decile"], decile_mean["mean_subsequent_return"])
    ax.set_title("Average Subsequent Return by Score Decile")
    ax.set_xlabel("Score Decile (1=Lowest, 10=Highest)")
    ax.set_ylabel("Mean Subsequent Return")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(range(1, 11))
    fig.tight_layout()
    fig.savefig(results_dir / "decile_analysis.png", dpi=150)
    plt.close(fig)

    feature_columns = TradingResearchPipeline.FEATURE_COLUMNS
    model_frame = results["dataset"].dropna(subset=feature_columns + ["label"])
    feature_importance = [{"feature": feature, "importance": 0.0} for feature in feature_columns]
    if not model_frame.empty:
        model = BaselineClassifier(random_state=args.seed)
        model.fit(model_frame[feature_columns].to_numpy(), model_frame["label"].to_numpy())
        feature_importance = [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in zip(feature_columns, model.model.feature_importances_, strict=True)
        ]

    feature_importance_df = (
        pd.DataFrame(feature_importance).sort_values("importance", ascending=False).reset_index(drop=True)
    )
    feature_importance_df.to_csv(results_dir / "feature_importance.csv", index=False)

    metrics = {
        "data_provider": args.data_provider,
        "tickers": args.tickers,
        "periods": args.periods,
        "seed": args.seed,
        "top_n": args.top_n,
        "rows_in_dataset": int(len(results["dataset"])),
        "backtest_days": int(len(perf)),
        "total_return": float(total_return),
        "annualized_sharpe": float(results["sharpe"]),
    }
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10, 5))
    if not perf.empty:
        ax.plot(perf["date"], perf["equity_curve"], label="Equity Curve", linewidth=2)
    ax.set_title("Strategy Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "equity_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(feature_importance_df["feature"], feature_importance_df["importance"])
    ax.set_title("Feature Importance")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(results_dir / "feature_importance.png", dpi=150)
    plt.close(fig)

    print("=== Trading AI Research Run ===")
    print(f"Data provider: {args.data_provider}")
    print(f"Tickers ({len(tickers)}): {', '.join(tickers)}")
    print(f"Rows in dataset: {len(results['dataset'])}")
    print(f"Backtest days: {len(perf)}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Sharpe: {results['sharpe']:.2f}")


if __name__ == "__main__":
    main()
