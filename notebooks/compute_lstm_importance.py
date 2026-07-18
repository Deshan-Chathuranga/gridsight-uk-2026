#!/usr/bin/env python3
import sys
import os
import json
import numpy as np
import pandas as pd
import joblib
import torch

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
sys.path.insert(0, os.getcwd())

from gridsight.models.lstm.config import ModelConfig
from gridsight.models.lstm.data import make_sequences, prepare
from gridsight.models.lstm.predict import load_lstm_stack
from gridsight.models.lstm.base.lstm_q import pinball_loss_torch

def main():
    lstm_dir = "artifacts/lstm"
    if not (os.path.exists(os.path.join(lstm_dir, "lstm.joblib")) and os.path.exists(os.path.join(lstm_dir, "lstm.pt"))):
        print(json.dumps({"error": "LSTM model artifacts not found."}))
        return

    art, lstm = load_lstm_stack(lstm_dir)
    cfg = art["cfg"]
    feats = art["features"]
    std = art["standardizer"]

    ds = prepare(cfg)
    df = ds.df.sort_values("timestamp_utc").reset_index(drop=True)
    tr_mask, va_mask, te_mask = ds.split_masks()
    score = ds.score_mask()

    test_indices = np.where(te_mask)[0]
    if len(test_indices) == 0:
        test_indices = np.where(va_mask)[0]
    if len(test_indices) == 0:
        test_indices = np.arange(len(df))

    test_indices = test_indices[-min(8000, len(test_indices)):]
    start_idx = max(0, test_indices[0] - cfg.seq_len)
    end_idx = test_indices[-1]
    df_eval = df.iloc[start_idx : end_idx + 1].copy().reset_index(drop=True)

    def get_loss(df_input):
        V = df_input[feats].to_numpy("float32")
        V = np.nan_to_num(V, nan=0.0)
        Vs = std.transform(V)
        seqs, end_idx_seq = make_sequences(Vs, cfg.seq_len)

        original_indices = np.arange(start_idx, end_idx + 1)
        pred_orig_indices = original_indices[end_idx_seq]
        score_mask_sub = score[pred_orig_indices]

        if not np.any(score_mask_sub):
            return 0.0

        y_sub = df.iloc[pred_orig_indices[score_mask_sub]][cfg.target].to_numpy("float32")
        seqs_sub = seqs[score_mask_sub]

        with torch.no_grad():
            xb = torch.from_numpy(seqs_sub)
            preds = lstm.model_(xb)
            yb = torch.from_numpy(y_sub)
            loss = pinball_loss_torch(preds, yb, cfg.quantiles)
            return float(loss.item())

    baseline_loss = get_loss(df_eval)

    imp = {}
    for col in feats:
        df_perm = df_eval.copy()
        df_perm[col] = np.random.permutation(df_perm[col].values)
        perm_loss = get_loss(df_perm)
        imp[col] = max(0.0, perm_loss - baseline_loss)

    print(json.dumps(imp))

if __name__ == '__main__':
    main()
