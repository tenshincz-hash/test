import pandas as pd

signals = pd.read_csv("results/daily_signals.csv")

trades = signals[signals["action"].isin(["BUY", "SELL", "REDUCE", "EXIT"])].copy()

trades["amount"] = trades["delta_value"].abs()

trades = trades[["ticker", "action", "amount"]]

trades.to_csv("results/weekly_trades.csv", index=False)

print("eToro trade list saved to results/weekly_trades.csv")