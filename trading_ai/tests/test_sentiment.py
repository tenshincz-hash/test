from __future__ import annotations

from nlp_engine.sentiment import SentimentPipeline


def test_sentiment_scoring() -> None:
    pipe = SentimentPipeline()
    assert pipe.score_text("AAPL beats outlook") > 0
    assert pipe.score_text("AAPL misses guidance") < 0
    assert pipe.score_text("AAPL investor day") == 0
