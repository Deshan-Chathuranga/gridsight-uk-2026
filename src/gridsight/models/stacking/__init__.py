"""Probabilistic solar-forecasting stack: TCN-Q + LGBM-Q -> Linear-Q.

Run:  python -m modeling --fast
"""
from .config import ModelConfig
from .train import run
from .predict import predict_gold
# NOTE: import evaluate lazily (`from modeling.evaluate import make_charts`) so that
# `python -m modeling.evaluate` doesn't double-import and warn.

__all__ = ["ModelConfig", "run", "predict_gold"]
