from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

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

            X_train_frame = train[self.features].replace([np.inf, -np.inf], np.nan)
            fill_values = X_train_frame.median(numeric_only=True).fillna(0.0)
            X_train = X_train_frame.fillna(fill_values).to_numpy(dtype=float)
            y_train_cls = train["label"].to_numpy()
            y_train_reg = train["future_return"].fillna(0.0).to_numpy()
            X_test = (
                test[self.features]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(fill_values)
                .to_numpy(dtype=float)
            )

            lgbm_model = BaselineClassifier()
            lgbm_model.fit(X_train, y_train_cls)
            lgbm_scores = lgbm_model.predict_proba(X_test)

            rf_model = RandomForestRegressor(
                n_estimators=20,
                max_depth=6,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=1,
            )
            rf_model.fit(X_train, y_train_reg)
            rf_scores = rf_model.predict(X_test)

            linear_model = LinearRegression()
            linear_model.fit(X_train, y_train_reg)
            linear_scores = linear_model.predict(X_test)

            final_scores = 0.5 * lgbm_scores + 0.3 * rf_scores + 0.2 * linear_scores
            final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)

            out = test[["date", "ticker"]].copy()
            out["lgbm_score"] = lgbm_scores
            out["rf_score"] = rf_scores
            out["linear_score"] = linear_scores
            out["final_score"] = final_scores
            out["score"] = final_scores
            score_slices.append(out)
            start += self.test_window

        return pd.concat(score_slices, ignore_index=True) if score_slices else pd.DataFrame()
