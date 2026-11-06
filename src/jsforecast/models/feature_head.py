"""
feature_head.py
---------------
A lightweight MLP that reads the Jane Street feature columns from the context
window and produces a per-horizon *correction delta* added on top of TimesFM's
base forecast:

    context_features (B, context_len, F) -> mean over time -> MLP -> delta (B, horizon_len)
    final_pred = base_pred + delta

`NullHead` is the "no features" variant: it always returns a zero delta, so the
forecaster never has to branch on whether a head exists.

Both heads store themselves next to the LoRA adapter as `feature_head.pt`;
`load_head` returns a `NullHead` when that file does not exist.
"""

import os

import torch
import torch.nn as nn

FEATURE_HEAD_FILENAME = "feature_head.pt"


class NullHead(nn.Module):
    """Zero correction: `base_pred + NullHead(features) == base_pred`."""

    def __init__(self, horizon_len: int):
        super().__init__()
        self.horizon_len = horizon_len

    def forward(self, context_features: torch.Tensor) -> torch.Tensor:
        return context_features.new_zeros(context_features.shape[0], self.horizon_len)

    def save(self, output_dir: str) -> None:
        """Nothing to store."""


class FeatureCorrectionHead(nn.Module):
    """
    2-layer MLP correction head. The time axis is reduced by average pooling, so
    `context_len` is only kept as metadata. The output is unbounded (no tanh).
    """

    def __init__(
        self,
        num_features: int,
        context_len: int,
        horizon_len: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon_len),
        )

    def forward(self, context_features: torch.Tensor) -> torch.Tensor:
        """(batch, context_len, num_features) -> delta (batch, horizon_len)"""
        return self.net(context_features.mean(dim=1))

    def save(self, output_dir: str) -> None:
        path = os.path.join(output_dir, FEATURE_HEAD_FILENAME)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "num_features": self.num_features,
                    "context_len": self.context_len,
                    "horizon_len": self.horizon_len,
                    "hidden_dim": self.hidden_dim,
                },
            },
            path,
        )
        print(f"  -> feature head saved to {path}")


def load_head(adapter_dir: str, device: str, horizon_len: int) -> nn.Module:
    """
    Loads `feature_head.pt` from `adapter_dir`, or a `NullHead` if the file is absent.

    Loaded heads are for inference, so they come back in eval mode (dropout off).
    """
    path = os.path.join(adapter_dir, FEATURE_HEAD_FILENAME)
    if not os.path.isfile(path):
        return NullHead(horizon_len).to(device).eval()
    ckpt = torch.load(path, map_location=device)
    head = FeatureCorrectionHead(**ckpt["config"])
    head.load_state_dict(ckpt["state_dict"])
    return head.to(device).eval()
