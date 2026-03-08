from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lightgbm import LGBMClassifier


@dataclass
class BaselineClassifier:
    random_state: int = 42

    def __post_init__(self) -> None:
        self.model = LGBMClassifier(
            objective="binary",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=self.random_state,
            verbosity=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = self.model.predict_proba(X)
        return probs[:, 1]
