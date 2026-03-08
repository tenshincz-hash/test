from __future__ import annotations

import pandas as pd


class ExecutionSimulator:
    def __init__(self, slippage_bps: float = 0.0, trading_cost_bps: float = 5.0) -> None:
        self.slippage = slippage_bps / 10000.0
        self.cost = trading_cost_bps / 10000.0

    def simulate(self, portfolio: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
        market = market.copy().sort_values(["ticker", "date"])
        market["next_ret"] = market.groupby("ticker")["close"].pct_change().shift(-1)

        px = portfolio.merge(market[["date", "ticker", "next_ret"]], on=["date", "ticker"], how="left")
        px = px.sort_values(["ticker", "date"])
        px["turnover"] = px.groupby("ticker")["weight"].diff().abs().fillna(px["weight"].abs())
        px["gross_pnl"] = px["weight"] * px["next_ret"].fillna(0.0)
        px["transaction_cost"] = px["turnover"] * (self.slippage + self.cost)
        px["pnl"] = px["gross_pnl"] - px["transaction_cost"]

        daily = (
            px.groupby("date", as_index=False)
            .agg(
                gross_return=("gross_pnl", "sum"),
                transaction_cost=("transaction_cost", "sum"),
                strategy_return=("pnl", "sum"),
            )
        )
        daily["gross_equity_curve"] = (1 + daily["gross_return"]).cumprod()
        daily["equity_curve"] = (1 + daily["strategy_return"]).cumprod()
        return daily
