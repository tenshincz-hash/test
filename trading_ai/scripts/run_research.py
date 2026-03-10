from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.baseline import BaselineClassifier
from orchestration.pipeline import ResearchConfig, TradingResearchPipeline
from universe_loader import load_universe_metadata

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"]
DEFAULT_UNIVERSE_FILE = Path("configs/universe_500.csv")
ROBUSTNESS_PERIODS = [500, 1200]


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


def _max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0
    return float(drawdown.min())


def _build_sector_exposure(portfolio: pd.DataFrame, sector_map: pd.DataFrame) -> pd.DataFrame:
    if portfolio.empty or sector_map.empty:
        return pd.DataFrame(columns=["date", "sector", "sector_weight", "sector_share"])

    merged = portfolio.merge(sector_map, on="ticker", how="left")
    merged["sector"] = merged["sector"].fillna("UNKNOWN")
    merged["long_weight"] = merged["weight"].clip(lower=0.0)
    grouped = (
        merged.groupby(["date", "sector"], as_index=False)["long_weight"]
        .sum()
        .rename(columns={"long_weight": "sector_weight"})
    )
    total = grouped.groupby("date")["sector_weight"].transform("sum")
    grouped["sector_share"] = np.where(total > 0, grouped["sector_weight"] / total, 0.0)
    return grouped.sort_values(["date", "sector"]).reset_index(drop=True)


def _sector_concentration_metrics(exposure: pd.DataFrame) -> dict[str, float]:
    if exposure.empty:
        return {
            "avg_max_sector_share": 0.0,
            "max_sector_share_observed": 0.0,
            "avg_sector_hhi": 0.0,
        }

    max_share = exposure.groupby("date")["sector_share"].max()
    hhi = exposure.groupby("date")["sector_share"].apply(lambda s: float((s**2).sum()))
    return {
        "avg_max_sector_share": float(max_share.mean()),
        "max_sector_share_observed": float(max_share.max()),
        "avg_sector_hhi": float(hhi.mean()),
    }


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
    parser.add_argument(
        "--universe-file",
        type=Path,
        default=DEFAULT_UNIVERSE_FILE,
        help="CSV file containing ticker universe for Yahoo mode (columns: ticker or symbol)",
    )
    parser.add_argument(
        "--max-sector-weight",
        type=float,
        default=0.20,
        help="Maximum sector exposure on long book (0-1, default 0.20)",
    )
    parser.add_argument(
        "--disable-volatility-targeting",
        action="store_true",
        help="Disable inverse-volatility weighting and fall back to score weighting or equal weights",
    )
    parser.add_argument(
        "--disable-score-weighting",
        action="store_true",
        help="Disable score-based weighting and use equal weights when volatility targeting is off",
    )
    parser.add_argument(
        "--disable-long-short",
        action="store_true",
        help="Disable long-short construction and run the legacy long-only portfolio",
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
    if args.max_sector_weight <= 0:
        raise ValueError("--max-sector-weight must be positive.")

    universe_metadata = pd.DataFrame(columns=["ticker", "sector"])
    sector_by_ticker: dict[str, str] | None = None
    if args.tickers is None:
        if args.data_provider == "yahoo":
            universe_metadata = load_universe_metadata(args.universe_file)
            tickers = universe_metadata["ticker"].tolist()
        else:
            tickers = DEFAULT_TICKERS
    else:
        tickers = args.tickers
        if args.data_provider == "yahoo" and args.universe_file.exists():
            universe_metadata = load_universe_metadata(args.universe_file)

    if args.data_provider == "yahoo" and not universe_metadata.empty:
        sector_by_ticker = {
            row.ticker: row.sector
            for row in universe_metadata.itertuples(index=False)
            if pd.notna(row.sector)
        }

    config = ResearchConfig(
        tickers=tickers,
        periods=args.periods,
        seed=args.seed,
        data_provider=args.data_provider,
        top_n=args.top_n,
        rebalance_frequency=args.rebalance_frequency,
        sector_by_ticker=sector_by_ticker,
        max_sector_weight=args.max_sector_weight,
        use_volatility_targeting=not args.disable_volatility_targeting,
        use_score_weighting=not args.disable_score_weighting,
        use_long_short=not args.disable_long_short,
    )
    pipeline = TradingResearchPipeline(config)
    results = pipeline.run()

    perf = results["performance"]
    requested_tickers_count = len(tickers)
    downloaded_tickers_count = int(results["prices"]["ticker"].nunique()) if not results["prices"].empty else 0
    scored_tickers_count = int(results["scores"]["ticker"].nunique()) if not results["scores"].empty else 0
    total_return = perf["equity_curve"].iloc[-1] - 1 if not perf.empty else 0.0
    max_drawdown = _max_drawdown(perf["equity_curve"]) if not perf.empty else 0.0
    cost_adjusted_return = float(total_return)
    cost_adjusted_sharpe = float(results["sharpe"])
    benchmark_return = 0.0
    excess_return = float(total_return)
    sector_metrics = {
        "avg_max_sector_share": 0.0,
        "max_sector_share_observed": 0.0,
        "avg_sector_hhi": 0.0,
    }
    benchmark_comparison = pd.DataFrame(columns=["date", "strategy_equity", "benchmark_equity"])
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    sector_map_df = (
        universe_metadata[["ticker", "sector"]].drop_duplicates(subset=["ticker"], keep="first")
        if not universe_metadata.empty
        else pd.DataFrame(columns=["ticker", "sector"])
    )
    scores_out = results["scores"].merge(sector_map_df, on="ticker", how="left")
    scores_out.to_csv(results_dir / "predictions.csv", index=False)
    score_columns = ["date", "ticker", "lgbm_score", "rf_score", "linear_score", "final_score", "score"]
    available_score_columns = [col for col in score_columns if col in scores_out.columns]
    if available_score_columns:
        scores_out[available_score_columns].to_csv(results_dir / "model_scores.csv", index=False)
    trade_source = results.get("constructed_portfolio", results["portfolio"])
    if "sector" not in trade_source.columns:
        trade_source = trade_source.merge(sector_map_df, on="ticker", how="left")
    trades = trade_source[trade_source["weight"] != 0].copy()
    trades["side"] = trades["weight"].map(lambda w: "long" if w > 0 else "short")
    trades.to_csv(results_dir / "trades.csv", index=False)
    perf.to_csv(results_dir / "equity_curve.csv", index=False)
    sector_exposure = _build_sector_exposure(results["portfolio"], sector_map_df)
    sector_exposure.to_csv(results_dir / "sector_exposure.csv", index=False)
    sector_metrics = _sector_concentration_metrics(sector_exposure)

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
        "max_sector_weight": args.max_sector_weight,
        "use_volatility_targeting": bool(not args.disable_volatility_targeting),
        "use_score_weighting": bool(not args.disable_score_weighting),
        "use_long_short": bool(not args.disable_long_short),
        "universe_file": str(args.universe_file),
        "requested_tickers_count": int(requested_tickers_count),
        "downloaded_tickers_count": int(downloaded_tickers_count),
        "scored_tickers_count": int(scored_tickers_count),
        "avg_max_sector_share": float(sector_metrics["avg_max_sector_share"]),
        "max_sector_share_observed": float(sector_metrics["max_sector_share_observed"]),
        "avg_sector_hhi": float(sector_metrics["avg_sector_hhi"]),
        "rows_in_dataset": int(len(results["dataset"])),
        "backtest_days": int(len(perf)),
        "total_return": float(total_return),
        "annualized_sharpe": float(results["sharpe"]),
        "cost_adjusted_return": float(cost_adjusted_return),
        "cost_adjusted_sharpe": float(cost_adjusted_sharpe),
        "max_drawdown": float(max_drawdown),
        "benchmark_return": float(benchmark_return),
        "excess_return": float(excess_return),
    }
    regime = results.get("regime")
    if isinstance(regime, pd.DataFrame) and not regime.empty:
        favorable = regime["regime_favorable"].fillna(False)
        metrics.update(
            {
                "regime_proxy": "SPY",
                "regime_rule": "SPY close > SPY 200-day moving average",
                "regime_favorable_days": int(favorable.sum()),
                "regime_unfavorable_days": int((~favorable).sum()),
                "regime_favorable_ratio": float(favorable.mean()),
                "regime_latest_favorable": bool(favorable.iloc[-1]),
                "regime_latest_spy_close": float(regime["spy_close"].iloc[-1]),
                "regime_latest_spy_sma_200": float(regime["spy_sma_200"].iloc[-1]),
            }
        )
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
                sector_by_ticker=sector_by_ticker,
                max_sector_weight=args.max_sector_weight,
                use_volatility_targeting=not args.disable_volatility_targeting,
                use_score_weighting=not args.disable_score_weighting,
                use_long_short=not args.disable_long_short,
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
            robustness_sector_exposure = _build_sector_exposure(robustness_results["portfolio"], sector_map_df)
            robustness_sector_metrics = _sector_concentration_metrics(robustness_sector_exposure)
            robustness_rows.append(
                {
                    "periods": periods,
                    "requested_tickers_count": int(len(tickers)),
                    "downloaded_tickers_count": int(
                        robustness_results["prices"]["ticker"].nunique()
                    ) if not robustness_results["prices"].empty else 0,
                    "scored_tickers_count": int(
                        robustness_results["scores"]["ticker"].nunique()
                    ) if not robustness_results["scores"].empty else 0,
                    "rows_in_dataset": int(len(robustness_results["dataset"])),
                    "backtest_days": int(len(robustness_perf)),
                    "total_return": float(robustness_total_return),
                    "annualized_sharpe": float(robustness_results["sharpe"]),
                    "max_drawdown": float(_max_drawdown(robustness_perf["equity_curve"]) if not robustness_perf.empty else 0.0),
                    "benchmark_return": float(robustness_benchmark_return),
                    "excess_return": float(robustness_excess_return),
                    "avg_max_sector_share": float(robustness_sector_metrics["avg_max_sector_share"]),
                    "max_sector_share_observed": float(robustness_sector_metrics["max_sector_share_observed"]),
                    "avg_sector_hhi": float(robustness_sector_metrics["avg_sector_hhi"]),
                }
            )

        robustness_summary = pd.DataFrame(
            robustness_rows,
            columns=[
                "periods",
                "requested_tickers_count",
                "downloaded_tickers_count",
                "scored_tickers_count",
                "rows_in_dataset",
                "backtest_days",
                "total_return",
                "annualized_sharpe",
                "max_drawdown",
                "benchmark_return",
                "excess_return",
                "avg_max_sector_share",
                "max_sector_share_observed",
                "avg_sector_hhi",
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
    preview = ", ".join(tickers[:10])
    suffix = " ..." if len(tickers) > 10 else ""
    print(f"Tickers ({len(tickers)}): {preview}{suffix}")
    print(
        "Universe validation: "
        f"requested={requested_tickers_count}, downloaded={downloaded_tickers_count}, scored={scored_tickers_count}"
    )
    print(f"Rows in dataset: {len(results['dataset'])}")
    print(f"Backtest days: {len(perf)}")
    print(f"Rebalance frequency: {args.rebalance_frequency}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Sharpe: {results['sharpe']:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(
        "Sector concentration: "
        f"avg max sector share={sector_metrics['avg_max_sector_share']:.2%}, "
        f"worst max sector share={sector_metrics['max_sector_share_observed']:.2%}, "
        f"avg HHI={sector_metrics['avg_sector_hhi']:.4f}"
    )
    if args.data_provider == "yahoo":
        print(f"SPY Benchmark Return: {benchmark_return:.2%}")
        print(f"Excess Return vs SPY: {excess_return:.2%}")
        print(f"Robustness summary saved: {results_dir / 'robustness_summary.csv'}")
        print(f"Robustness chart saved: {results_dir / 'robustness_summary.png'}")
        print(f"Benchmark comparison chart saved: {results_dir / 'benchmark_comparison.png'}")


if __name__ == "__main__":
    main()
