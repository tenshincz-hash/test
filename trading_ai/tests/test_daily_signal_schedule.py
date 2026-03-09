from __future__ import annotations

import pandas as pd

from execution.signal_schedule import is_rebalance_day, next_rebalance_date, scheduled_rebalance_dates


def test_scheduled_rebalance_dates_weekly_and_biweekly() -> None:
    dates = pd.Series(pd.bdate_range("2026-01-05", periods=15))

    weekly = scheduled_rebalance_dates(dates, "weekly")
    biweekly = scheduled_rebalance_dates(dates, "biweekly")

    assert [d.strftime("%Y-%m-%d") for d in weekly] == ["2026-01-05", "2026-01-12", "2026-01-19"]
    assert [d.strftime("%Y-%m-%d") for d in biweekly] == ["2026-01-05", "2026-01-19"]


def test_is_rebalance_day_uses_latest_business_date() -> None:
    due_dates = pd.Series(pd.bdate_range("2026-01-05", periods=11))
    not_due_dates = pd.Series(pd.bdate_range("2026-01-05", periods=12))

    assert is_rebalance_day(due_dates, "weekly")
    assert not is_rebalance_day(not_due_dates, "weekly")


def test_next_rebalance_date_calendar_behavior() -> None:
    assert next_rebalance_date(pd.Timestamp("2026-01-22"), "weekly") == pd.Timestamp("2026-01-26")
    assert next_rebalance_date(pd.Timestamp("2026-01-23"), "weekly") == pd.Timestamp("2026-01-26")
    assert next_rebalance_date(pd.Timestamp("2026-01-23"), "biweekly") == pd.Timestamp("2026-02-02")
