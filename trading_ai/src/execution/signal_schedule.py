from __future__ import annotations

import pandas as pd
from pandas.tseries.offsets import Week


def scheduled_rebalance_dates(dates: pd.Series, frequency: str) -> pd.DatetimeIndex:
    unique_dates = pd.DatetimeIndex(pd.to_datetime(dates).dropna().sort_values().unique()).normalize()
    if unique_dates.empty:
        return unique_dates
    if frequency == "daily":
        return unique_dates
    if frequency not in {"weekly", "biweekly"}:
        raise ValueError(f"Unsupported rebalance_frequency='{frequency}'")

    weekly = (
        pd.DataFrame({"date": unique_dates, "period": unique_dates.to_period("W-SUN")})
        .groupby("period", as_index=False)["date"]
        .min()["date"]
        .sort_values()
    )
    if frequency == "weekly":
        return pd.DatetimeIndex(weekly)
    return pd.DatetimeIndex(weekly.iloc[::2])


def is_rebalance_day(dates: pd.Series, frequency: str) -> bool:
    scheduled = scheduled_rebalance_dates(dates, frequency)
    if scheduled.empty:
        return True
    latest_date = pd.Timestamp(pd.to_datetime(dates).max()).normalize()
    return latest_date in set(scheduled)


def next_rebalance_date(signal_date: pd.Timestamp, frequency: str) -> pd.Timestamp:
    signal_date = pd.Timestamp(signal_date).normalize()
    if frequency == "daily":
        return signal_date
    if frequency == "weekly":
        return signal_date + Week(weekday=0)
    if frequency == "biweekly":
        return signal_date + Week(weekday=0) + Week(1)
    raise ValueError(f"Unsupported rebalance_frequency='{frequency}'")
