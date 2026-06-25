"""Linear-Q meta-learner that stacks the base quantile predictions.

Meta input per row = the 6 base quantiles (TCN q10/q50/q90 + LGBM q10/q50/q90)
plus Clear-Sky GHI. A separate Linear Quantile Regression is fit for each output
quantile; predictions are then sorted per row so the final q10<=q50<=q90 never
cross. Trained on OUT-OF-FOLD base predictions so the meta-learner never sees a
base prediction made by a model that trained on the same row.
"""
from __future__ import annotations

import numpy as np


def assemble_meta_X(tcn: dict[float, np.ndarray], lgbm: dict[float, np.ndarray],
                    clearsky: np.ndarray, quantiles) -> np.ndarray:
    cols = [tcn[q] for q in quantiles] + [lgbm[q] for q in quantiles] + [clearsky]
    return np.column_stack(cols).astype("float32")


class LinearQuantileStacker:
    def __init__(self, quantiles=(0.1, 0.5, 0.9)):
        self.quantiles = tuple(quantiles)
        self.models_: dict[float, object] = {}

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "LinearQuantileStacker":
        from sklearn.linear_model import QuantileRegressor
        for q in self.quantiles:
            # alpha=0 -> plain pinball linear regression; 'highs' is the LP solver
            self.models_[q] = QuantileRegressor(quantile=q, alpha=0.0,
                                                solver="highs").fit(Z, y)
        return self

    def predict(self, Z: np.ndarray) -> dict[float, np.ndarray]:
        P = np.column_stack([self.models_[q].predict(Z) for q in self.quantiles])
        P = np.sort(P, axis=1)                       # re-calibrate: enforce monotonicity
        return {q: P[:, i].astype("float32") for i, q in enumerate(self.quantiles)}
