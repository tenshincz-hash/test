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
- `current_weight`
- `decile`
- `model_target_weight` (raw model portfolio for this date)
- `target_weight`
- `delta_weight`
- `current_value` (if `--portfolio-value` is provided)
- `target_value` (if `--portfolio-value` is provided)
- `delta_value` (if `--portfolio-value` is provided)
- `signal_type` (`rebalance` or `hold`)
- `rebalance_due`
- `next_rebalance_date`
- `valid_until`
- `action`

Default behavior is now `--rebalance-frequency weekly` for practical weekly execution while still generating output daily.

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
In that case, `current_weight`, `delta_weight`, `current_value`, and `delta_value` are left blank while `target_weight` remains populated.

To include dollar values in output, pass `--portfolio-value`:

```bash
python scripts/run_daily_signal.py --portfolio-value 100000
```

Rebalance scheduling:
- `daily`: executable `target_weight` is refreshed every run.
- `weekly`: executable `target_weight` updates on the first available trading day of each calendar week.
- `biweekly`: same weekly anchor, but every second rebalance week.
- Between scheduled rebalances, the script keeps current holdings as executable targets when `current_portfolio.csv` is present, while still showing `model_target_weight` for preview.

How to populate `current_portfolio.csv`:
1. List each currently held ticker once.
2. Set `current_weight` as a decimal portfolio fraction (for example `0.1250` = 12.50%).
3. Ensure weights reflect your live book and approximately sum to `1.0` (or less if holding cash).
4. Use uppercase ticker symbols.

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
