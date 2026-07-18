from __future__ import annotations

import numpy as np


def device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ChronosQuantile:
    def __init__(self, cfg):
        self.cfg = cfg
        self.quantiles = tuple(cfg.quantiles)
        self.pipe_ = None

    def load(self) -> "ChronosQuantile":
        import torch
        from chronos import BaseChronosPipeline
        torch.set_num_threads(self.cfg.torch_threads)
        dtype = getattr(torch, self.cfg.torch_dtype, torch.float32)
        self.pipe_ = BaseChronosPipeline.from_pretrained(
            self.cfg.model_name,
            device_map=device(),
            torch_dtype=dtype,
        )
        return self

    def forecast(self, contexts: list[np.ndarray], prediction_length: int) -> np.ndarray:
        import torch
        levels = list(self.quantiles)
        out = np.empty((len(contexts), prediction_length, len(levels)), dtype="float32")
        B = self.cfg.chronos_batch
        for i in range(0, len(contexts), B):
            chunk = contexts[i:i + B]
            tensors = [torch.as_tensor(c, dtype=torch.float32) for c in chunk]
            import torch
            q, _ = self.pipe_.predict_quantiles(
                tensors,
                prediction_length=prediction_length,
                quantile_levels=levels,
            )
            if isinstance(q, list):
                q = torch.cat(q, dim=0)
            out[i:i + len(chunk)] = q.to(torch.float32).cpu().numpy()
        return out

    def forecast_h_ahead(self, series: np.ndarray, origins: np.ndarray,
                         horizon: int) -> dict[float, np.ndarray]:
        C = self.cfg.context_len
        contexts = []
        for o in origins:
            lo = max(0, o - C + 1)
            contexts.append(series[lo:o + 1])
        preds = self.forecast(contexts, horizon)
        last = preds[:, horizon - 1, :]
        last = np.sort(last, axis=1)
        return {q: last[:, i].astype("float32") for i, q in enumerate(self.quantiles)}
