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
ROBUSTNESS_PERIODS = [500, 800, 1200, 1600, 2000]


def _download_spy_prices(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    import yfinance as yf

    # yfinance end date is exclusive; add one day to include the final session.
    raw = yf.download(
        tickers="SPY",
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw.empty:
        return pd.DataFrame(columns=["date", "close"])

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    close_col = "Adj Close" if "Adj Close" in raw.columns else "Close"
    spy = raw[[close_col]].rename(columns={close_col: "close"}).dropna()
    spy["date"] = pd.to_datetime(spy.index).tz_localize(None)
    return spy.reset_index(drop=True)[["date", "close"]]


def _compute_benchmark_curve(perf: pd.DataFrame, spy_prices: pd.DataFrame) -> pd.DataFrame:
    if perf.empty or spy_prices.empty:
        return pd.DataFrame(columns=["date", "strategy_equity", "benchmark_equity"])

    strategy = perf[["date", "equity_curve"]].copy()
    strategy["date"] = pd.to_datetime(strategy["date"])
    strategy = strategy.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    backtest_dates = pd.DatetimeIndex(strategy["date"].tolist())
    spy_close = (
        spy_prices.assign(date=pd.to_datetime(spy_prices["date"]))
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .set_index("date")["close"]
    )
    spy_aligned = spy_close.reindex(backtest_dates).ffill().bfill()
    spy_returns = spy_aligned.pct_change().fillna(0.0)
    benchmark_equity = (1.0 + spy_returns).cumprod()

    out = strategy.rename(columns={"equity_curve": "strategy_equity"}).set_index("date")
    out["benchmark_equity"] = benchmark_equity
    return out.reset_index()

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
    parser.add_argument(
        "--rebalance-frequency",
        choices=["daily", "weekly", "biweekly"],
        default="daily",
        help="Rebalance schedule for portfolio weights",
    )
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
        rebalance_frequency=args.rebalance_frequency,
    )
    pipeline = TradingResearchPipeline(config)
    results = pipeline.run()

    perf = results["performance"]
    total_return = perf["equity_curve"].iloc[-1] - 1 if not perf.empty else 0.0
    cost_adjusted_return = float(total_return)
    cost_adjusted_sharpe = float(results["sharpe"])
    benchmark_return = 0.0
    excess_return = float(total_return)
    benchmark_comparison = pd.DataFrame(columns=["date", "strategy_equity", "benchmark_equity"])
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

    if args.data_provider == "yahoo":
        market_prices = results["prices"]
        market_start = pd.to_datetime(market_prices["date"]).min()
        market_end = pd.to_datetime(market_prices["date"]).max()
        spy_prices = _download_spy_prices(market_start, market_end)
        benchmark_comparison = _compute_benchmark_curve(perf, spy_prices)
        if not benchmark_comparison.empty:
            benchmark_return = float(benchmark_comparison["benchmark_equity"].iloc[-1] - 1.0)
            excess_return = float(total_return - benchmark_return)

    metrics = {
        "data_provider": args.data_provider,
        "tickers": args.tickers,
        "periods": args.periods,
        "seed": args.seed,
        "top_n": args.top_n,
        "rebalance_frequency": args.rebalance_frequency,
        "rows_in_dataset": int(len(results["dataset"])),
        "backtest_days": int(len(perf)),
        "total_return": float(total_return),
        "annualized_sharpe": float(results["sharpe"]),
        "cost_adjusted_return": float(cost_adjusted_return),
        "cost_adjusted_sharpe": float(cost_adjusted_sharpe),
        "benchmark_return": float(benchmark_return),
        "excess_return": float(excess_return),
    }
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.data_provider == "yahoo":
        robustness_rows: list[dict[str, int | float]] = []
        for periods in ROBUSTNESS_PERIODS:
            robustness_config = ResearchConfig(
                tickers=tickers,
                periods=periods,
                seed=args.seed,
                data_provider=args.data_provider,
                top_n=args.top_n,
                rebalance_frequency="weekly",
            )
            robustness_results = TradingResearchPipeline(robustness_config).run()
            robustness_perf = robustness_results["performance"]
            robustness_total_return = (
                robustness_perf["equity_curve"].iloc[-1] - 1 if not robustness_perf.empty else 0.0
            )
            robustness_benchmark_return = 0.0
            robustness_excess_return = float(robustness_total_return)
            if not robustness_perf.empty:
                robustness_prices = robustness_results["prices"]
                robust_start = pd.to_datetime(robustness_prices["date"]).min()
                robust_end = pd.to_datetime(robustness_prices["date"]).max()
                robust_spy = _download_spy_prices(robust_start, robust_end)
                robust_benchmark_curve = _compute_benchmark_curve(robustness_perf, robust_spy)
                if not robust_benchmark_curve.empty:
                    robustness_benchmark_return = float(robust_benchmark_curve["benchmark_equity"].iloc[-1] - 1.0)
                    robustness_excess_return = float(robustness_total_return - robustness_benchmark_return)
            robustness_rows.append(
                {
                    "periods": periods,
                    "rows_in_dataset": int(len(robustness_results["dataset"])),
                    "backtest_days": int(len(robustness_perf)),
                    "total_return": float(robustness_total_return),
                    "annualized_sharpe": float(robustness_results["sharpe"]),
                    "benchmark_return": float(robustness_benchmark_return),
                    "excess_return": float(robustness_excess_return),
                }
            )

        robustness_summary = pd.DataFrame(
            robustness_rows,
            columns=[
                "periods",
                "rows_in_dataset",
                "backtest_days",
                "total_return",
                "annualized_sharpe",
                "benchmark_return",
                "excess_return",
            ],
        ).sort_values("periods")
        robustness_summary.to_csv(results_dir / "robustness_summary.csv", index=False)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(robustness_summary["periods"], robustness_summary["total_return"], marker="o", linewidth=2)
        ax1.set_title("Long-Only Decile 8-9 Robustness Across Lookback Windows")
        ax1.set_ylabel("Total Return")
        ax1.grid(alpha=0.3)

        ax2.plot(robustness_summary["periods"], robustness_summary["annualized_sharpe"], marker="o", linewidth=2)
        ax2.set_xlabel("Lookback Periods")
        ax2.set_ylabel("Annualized Sharpe")
        ax2.grid(alpha=0.3)
        ax2.set_xticks(ROBUSTNESS_PERIODS)

        fig.tight_layout()
        fig.savefig(results_dir / "robustness_summary.png", dpi=150)
        plt.close(fig)

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

    if args.data_provider == "yahoo":
        fig, ax = plt.subplots(figsize=(10, 5))
        if not benchmark_comparison.empty:
            ax.plot(
                benchmark_comparison["date"],
                benchmark_comparison["strategy_equity"],
                label="Strategy",
                linewidth=2,
            )
            ax.plot(
                benchmark_comparison["date"],
                benchmark_comparison["benchmark_equity"],
                label="SPY Buy-and-Hold",
                linewidth=2,
            )
        ax.set_title("Strategy vs SPY Buy-and-Hold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(results_dir / "benchmark_comparison.png", dpi=150)
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
    print(f"Rebalance frequency: {args.rebalance_frequency}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Sharpe: {results['sharpe']:.2f}")
    if args.data_provider == "yahoo":
        print(f"SPY Benchmark Return: {benchmark_return:.2%}")
        print(f"Excess Return vs SPY: {excess_return:.2%}")
        print(f"Robustness summary saved: {results_dir / 'robustness_summary.csv'}")
        print(f"Robustness chart saved: {results_dir / 'robustness_summary.png'}")
        print(f"Benchmark comparison chart saved: {results_dir / 'benchmark_comparison.png'}")


if __name__ == "__main__":
    main()
