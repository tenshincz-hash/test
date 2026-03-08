# Trading AI Research Platform

A modular, end-to-end research stack for systematic trading experiments.

## Architecture

- `src/ingestion`: mock market and news ingestion
- `src/storage`: in-memory dataset storage
- `src/nlp_engine`: sentiment scoring pipeline
- `src/feature_engineering`: feature creation and label generation
- `src/models`: baseline ML classifier
- `src/backtesting`: walk-forward training and scoring
- `src/portfolio`: portfolio construction from model scores
- `src/risk`: exposure and volatility-based risk management
- `src/execution`: execution simulation and return attribution
- `src/orchestration`: end-to-end research pipeline

## Quickstart

```bash
cd trading_ai
python -m pip install -e .[dev]
python scripts/run_research.py
```

## Run Tests

```bash
cd trading_ai
pytest -q
```

## End-to-End Flow

1. Generate mock prices and headlines.
2. Score headline sentiment.
3. Build technical + sentiment features.
4. Create forward-return labels.
5. Run walk-forward model training and out-of-sample scoring.
6. Convert scores into long/short portfolio weights.
7. Apply risk constraints and vol targeting.
8. Simulate execution and compute performance metrics.

## Notes

- This is intentionally lightweight and deterministic for rapid research iteration.
- Replace `ingestion/mock_data.py` and `execution/simulator.py` first when moving toward live data and production assumptions.
