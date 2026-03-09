from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_universe_metadata(path: str | Path) -> pd.DataFrame:
    universe_path = Path(path)
    if not universe_path.exists():
        raise FileNotFoundError(f"Universe file not found: {universe_path}")

    frame = pd.read_csv(universe_path)
    candidate_ticker_columns = ["ticker", "symbol", "Ticker", "Symbol"]
    ticker_column = next((col for col in candidate_ticker_columns if col in frame.columns), None)
    if ticker_column is None:
        raise ValueError(
            f"Universe file {universe_path} must include one of columns: {candidate_ticker_columns}"
        )

    out = frame.copy()
    out["ticker"] = (
        out[ticker_column]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".", "-", regex=False)
    )
    out = out[out["ticker"].ne("NAN") & out["ticker"].ne("")].copy()
    if "sector" in out.columns:
        out["sector"] = out["sector"].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA})
    elif "GICS Sector" in out.columns:
        out["sector"] = out["GICS Sector"].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA})
    else:
        out["sector"] = pd.NA
    out = out[["ticker", "sector"]].drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)
    return out


def load_tickers_from_csv(path: str | Path) -> list[str]:
    return load_universe_metadata(path)["ticker"].tolist()
