"""LGBM-Q base learner: one LightGBM gradient-boosted model per quantile.

Strong on the tabular / clear-sky features (NWP, NESO, calendar, solar, lags).
Native NaN handling, so no imputation needed. Trains an independent model for
each quantile with the pinball objective.
"""
from __future__ import annotations

import numpy as np


class LGBMQuantile:
    def __init__(self, quantiles=(0.1, 0.5, 0.9), params: dict | None = None, seed: int = 42):
        self.quantiles = tuple(quantiles)
        self.params = dict(params or {})
        self.seed = seed
        self.models_: dict[float, object] = {}

    def fit(self, X: np.ndarray, y: np.ndarray,
            eval_set: tuple | None = None) -> "LGBMQuantile":
        from lightgbm import LGBMRegressor, early_stopping, log_evaluation
        for q in self.quantiles:
            m = LGBMRegressor(objective="quantile", alpha=q,
                              random_state=self.seed, **self.params)
            cb, es = [], eval_set
            if es is not None:
                cb = [early_stopping(50, verbose=False), log_evaluation(0)]
            m.fit(X, y, eval_set=[es] if es is not None else None, callbacks=cb or None)
            self.models_[q] = m
        return self

    def predict(self, X: np.ndarray) -> dict[float, np.ndarray]:
        return {q: m.predict(X).astype("float32") for q, m in self.models_.items()}

    def feature_importance(self):
        return {q: m.feature_importances_ for q, m in self.models_.items()}
