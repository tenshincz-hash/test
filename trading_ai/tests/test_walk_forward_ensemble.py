from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.walk_forward import WalkForwardBacktester


def test_walk_forward_outputs_individual_and_ensemble_scores() -> None:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2023-01-02", periods=260)
    tickers = ["AAA", "BBB", "CCC"]

    rows: list[dict[str, float | pd.Timestamp | str | int]] = []
    for dt in dates:
        for tk in tickers:
            f1 = float(rng.normal())
            f2 = float(rng.normal())
            future_return = 0.02 * f1 - 0.01 * f2 + float(rng.normal(scale=0.01))
            rows.append(
                {
                    "date": dt,
                    "ticker": tk,
                    "f1": f1,
                    "f2": f2,
                    "future_return": future_return,
                    "label": int(future_return > 0.0),
                }
            )

    df = pd.DataFrame(rows)
    out = WalkForwardBacktester(features=["f1", "f2"]).run(df)

    for col in ["lgbm_score", "rf_score", "linear_score", "final_score", "score"]:
        assert col in out.columns

    recomputed = 0.5 * out["lgbm_score"] + 0.3 * out["rf_score"] + 0.2 * out["linear_score"]
    assert np.allclose(out["final_score"].to_numpy(), recomputed.to_numpy())
    assert np.allclose(out["score"].to_numpy(), out["final_score"].to_numpy())
