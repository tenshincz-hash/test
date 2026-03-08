from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from execution.simulator import ExecutionSimulator
from orchestration.pipeline import ResearchConfig, TradingResearchPipeline
from portfolio.construction import PortfolioConstructor
from risk.manager import RiskManager

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"]
SP500_50_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "TSLA",
    "JPM", "WMT", "XOM", "UNH", "V", "MA", "ORCL", "COST", "PG", "JNJ",
    "HD", "BAC", "ABBV", "KO", "CRM", "NFLX", "CVX", "MRK", "CSCO", "WFC",
    "ACN", "IBM", "LIN", "MCD", "ABT", "PM", "GE", "AMD", "INTU", "DIS",
    "TXN", "T", "VZ", "CAT", "PFE", "AMGN", "QCOM", "NOW", "SPGI", "INTC",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose whether model signal direction is inverted")
    parser.add_argument(
        "--data-provider",
        choices=["mock", "yahoo"],
        default="yahoo",
        help="Market data provider",
    )
    parser.add_argument("--periods", type=int, default=1200, help="Number of business-day observations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--top-n", type=int, default=10, help="Number of tickers per side")
    parser.add_argument("--tickers", nargs="+", default=None, help="Optional custom tickers")
    return parser.parse_args()


def annualized_sharpe(returns: pd.Series) -> float:
    if returns.empty or returns.std(ddof=0) == 0:
        return 0.0
    return float((252**0.5) * returns.mean() / returns.std(ddof=0))


def evaluate_from_scores(
    scores: pd.DataFrame,
    labeled: pd.DataFrame,
    prices: pd.DataFrame,
    top_n: int,
) -> dict[str, float]:
    portfolio = PortfolioConstructor(top_n=max(1, top_n)).build_weights(scores)
    vol_snapshot = labeled[["date", "ticker", "vol_20d"]].drop_duplicates()
    risk_adjusted = RiskManager().apply(portfolio[["date", "ticker", "weight"]], vol_snapshot)
    performance = ExecutionSimulator().simulate(risk_adjusted, prices)
    total_return = float(performance["equity_curve"].iloc[-1] - 1) if not performance.empty else 0.0
    sharpe = annualized_sharpe(performance["strategy_return"]) if not performance.empty else 0.0
    return {
        "total_return": total_return,
        "annualized_sharpe": float(sharpe),
        "backtest_days": int(len(performance)),
    }


def main() -> None:
    args = parse_args()
    tickers = args.tickers
    if tickers is None:
        tickers = SP500_50_TICKERS if args.data_provider == "yahoo" else DEFAULT_TICKERS

    config = ResearchConfig(
        tickers=tickers,
        periods=args.periods,
        seed=args.seed,
        data_provider=args.data_provider,
        top_n=args.top_n,
    )
    results = TradingResearchPipeline(config).run()

    scores = results["scores"].copy()
    labeled = results["dataset"]
    prices = results["prices"]

    original = evaluate_from_scores(scores=scores, labeled=labeled, prices=prices, top_n=args.top_n)

    inverted_scores = scores.copy()
    inverted_scores["score"] = -inverted_scores["score"]
    inverted = evaluate_from_scores(scores=inverted_scores, labeled=labeled, prices=prices, top_n=args.top_n)

    comparison = {
        "data_provider": args.data_provider,
        "tickers_count": len(tickers),
        "periods": args.periods,
        "seed": args.seed,
        "top_n": args.top_n,
        "original": original,
        "inverted": inverted,
        "likely_inverted_signal": inverted["annualized_sharpe"] > original["annualized_sharpe"],
    }

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "signal_direction_comparison.json"
    output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print(f"original return / Sharpe: {original['total_return']:.2%} / {original['annualized_sharpe']:.4f}")
    print(f"inverted return / Sharpe: {inverted['total_return']:.2%} / {inverted['annualized_sharpe']:.4f}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
