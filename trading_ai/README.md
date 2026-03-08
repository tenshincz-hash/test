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

## Daily Signal Bot (Yahoo, Long-Only Decile 8-9)

Generate today's target portfolio weights from the existing Yahoo strategy:

```bash
python scripts/run_daily_signal.py
```

Output is written to `results/daily_signals.csv` with:
- `date`
- `ticker`
- `score`
- `decile`
- `target_weight`
- `action`

Optional current portfolio input:
- Path: `results/current_portfolio.csv`
- Required columns: `ticker,current_weight`
- Example:

```csv
ticker,current_weight
AAPL,0.25
MSFT,0.25
NVDA,0.20
AMZN,0.00
```

If `results/current_portfolio.csv` is missing, the script still runs and emits BUY/HOLD placeholder actions.

Validation checklist:
1. Run `python scripts/run_daily_signal.py`.
2. Confirm `results/daily_signals.csv` exists and is non-empty.
3. Confirm `target_weight` is positive only for deciles 8-9 and sums to ~1.0 across positive rows.
4. If `results/current_portfolio.csv` exists, verify `action` reflects differences vs `current_weight` (`BUY`, `HOLD`, `REDUCE`, `EXIT`; `SELL` reserved for edge cases).

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
