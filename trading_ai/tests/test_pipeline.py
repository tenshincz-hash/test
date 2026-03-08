from __future__ import annotations

from orchestration.pipeline import ResearchConfig, TradingResearchPipeline


def test_pipeline_runs_end_to_end() -> None:
    config = ResearchConfig(tickers=["AAPL", "MSFT", "NVDA"], periods=520, seed=7)
    results = TradingResearchPipeline(config).run()

    assert not results["prices"].empty
    assert not results["news"].empty
    assert not results["dataset"].empty
    assert not results["scores"].empty
    assert not results["portfolio"].empty
    assert not results["performance"].empty

    perf = results["performance"]
    assert "strategy_return" in perf.columns
    assert "equity_curve" in perf.columns
    assert perf["equity_curve"].iloc[-1] > 0
