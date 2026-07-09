"""Probabilistic solar-forecasting standalone LSTM-Q model.

Run:  python -m lstm_q.train --fast
"""
from .config import ModelConfig
from .train import run_lstm_pipeline
from .predict import predict_lstm_gold

__all__ = ["ModelConfig", "run_lstm_pipeline", "predict_lstm_gold"]
