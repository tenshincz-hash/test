from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from models.baseline import BaselineClassifier


@dataclass
class WalkForwardBacktester:
    features: list[str]
    train_window: int = 200
    test_window: int = 20
    min_obs: int = 250

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.dropna(subset=self.features + ["label"]).sort_values("date").reset_index(drop=True)
        unique_dates = sorted(data["date"].unique())

        if len(unique_dates) < self.min_obs:
            raise ValueError(
                f"Not enough observations for walk-forward run: {len(unique_dates)} dates < {self.min_obs}"
            )

        score_slices = []
        start = self.train_window
        while start < len(unique_dates):
            train_dates = unique_dates[start - self.train_window : start]
            test_dates = unique_dates[start : start + self.test_window]
            if not test_dates:
                break

            train = data[data["date"].isin(train_dates)]
            test = data[data["date"].isin(test_dates)]

            X_train = train[self.features].to_numpy()
            y_train = train["label"].to_numpy()
            X_test = test[self.features].to_numpy()

            model = BaselineClassifier()
            model.fit(X_train, y_train)
            test_scores = model.predict_proba(X_test)

            out = test[["date", "ticker"]].copy()
            out["score"] = test_scores
            score_slices.append(out)
            start += self.test_window

        return pd.concat(score_slices, ignore_index=True) if score_slices else pd.DataFrame()
