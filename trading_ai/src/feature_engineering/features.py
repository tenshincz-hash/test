from __future__ import annotations

import pandas as pd


class FeatureEngineer:
    def transform(self, prices: pd.DataFrame, sentiments: pd.DataFrame) -> pd.DataFrame:
        px = prices.copy().sort_values(["ticker", "date"])

        if "open" not in px.columns:
            px["open"] = px["close"]
        if "high" not in px.columns:
            px["high"] = px["close"]
        if "low" not in px.columns:
            px["low"] = px["close"]
        if "volume" not in px.columns:
            px["volume"] = 1.0

        close_by_ticker = px.groupby("ticker")["close"]
        volume_by_ticker = px.groupby("ticker")["volume"]

        px["ret_1d"] = px.groupby("ticker")["close"].pct_change()
        px["ret_5d"] = px.groupby("ticker")["close"].pct_change(5)
        px["ret_10d"] = close_by_ticker.pct_change(10)
        px["ret_20d"] = close_by_ticker.pct_change(20)

        px["vol_10d"] = (
            px.groupby("ticker")["ret_1d"]
            .rolling(window=10, min_periods=10)
            .std()
            .reset_index(level=0, drop=True)
        )
        px["vol_20d"] = (
            px.groupby("ticker")["ret_1d"]
            .rolling(window=20, min_periods=20)
            .std()
            .reset_index(level=0, drop=True)
        )
        px["vol_30d"] = (
            px.groupby("ticker")["ret_1d"]
            .rolling(window=30, min_periods=30)
            .std()
            .reset_index(level=0, drop=True)
        )

        px["SMA_10"] = close_by_ticker.transform(lambda s: s.rolling(window=10, min_periods=10).mean())
        px["SMA_20"] = close_by_ticker.transform(lambda s: s.rolling(window=20, min_periods=20).mean())
        px["SMA_50"] = close_by_ticker.transform(lambda s: s.rolling(window=50, min_periods=50).mean())

        close_nonzero = px["close"].replace(0, pd.NA)
        px["distance_from_SMA_10"] = (px["close"] - px["SMA_10"]) / close_nonzero
        px["distance_from_SMA_20"] = (px["close"] - px["SMA_20"]) / close_nonzero
        px["distance_from_SMA_50"] = (px["close"] - px["SMA_50"]) / close_nonzero

        delta = close_by_ticker.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        avg_gain = gains.groupby(px["ticker"]).transform(lambda s: s.rolling(window=14, min_periods=14).mean())
        avg_loss = losses.groupby(px["ticker"]).transform(lambda s: s.rolling(window=14, min_periods=14).mean())
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.where(avg_loss != 0, 100.0)
        rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
        px["RSI_14"] = rsi

        ema_12 = close_by_ticker.transform(lambda s: s.ewm(span=12, adjust=False).mean())
        ema_26 = close_by_ticker.transform(lambda s: s.ewm(span=26, adjust=False).mean())
        px["MACD"] = ema_12 - ema_26
        px["MACD_signal"] = px.groupby("ticker")["MACD"].transform(lambda s: s.ewm(span=9, adjust=False).mean())

        volume_mean_20 = volume_by_ticker.transform(lambda s: s.rolling(window=20, min_periods=20).mean())
        volume_std_20 = volume_by_ticker.transform(lambda s: s.rolling(window=20, min_periods=20).std())
        volume_std_20 = volume_std_20.replace(0, 1.0).fillna(1.0)
        px["volume_zscore"] = (px["volume"] - volume_mean_20) / volume_std_20
        px["volume_change_5d"] = volume_by_ticker.pct_change(5)

        px["high_low_range_pct"] = (px["high"] - px["low"]) / close_nonzero
        px["close_open_gap_pct"] = (px["close"] - px["open"]) / px["open"].replace(0, pd.NA)

        merged = px.merge(sentiments, on=["date", "ticker"], how="left")
        merged["sentiment"] = merged["sentiment"].fillna(0.0)
        merged["sentiment_5d"] = (
            merged.groupby("ticker")["sentiment"]
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        return merged
