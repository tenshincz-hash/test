from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


POSITIVE_WORDS = {
    "beats",
    "raises",
    "strong",
    "partnership",
    "growth",
    "outlook",
}
NEGATIVE_WORDS = {
    "misses",
    "cuts",
    "pressure",
    "disruptions",
    "decline",
    "risk",
}


@dataclass
class SentimentPipeline:
    def score_text(self, text: str) -> float:
        tokens = text.lower().split()
        pos = sum(1 for tok in tokens if tok in POSITIVE_WORDS)
        neg = sum(1 for tok in tokens if tok in NEGATIVE_WORDS)
        if pos == 0 and neg == 0:
            return 0.0
        return float((pos - neg) / (pos + neg))

    def transform(self, news_df: pd.DataFrame) -> pd.DataFrame:
        required = {"date", "ticker", "headline"}
        missing = required - set(news_df.columns)
        if missing:
            raise ValueError(f"News dataframe missing columns: {missing}")

        out = news_df.copy()
        out["sentiment"] = out["headline"].astype(str).map(self.score_text)
        return out[["date", "ticker", "sentiment"]]
