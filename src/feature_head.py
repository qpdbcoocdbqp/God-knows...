"""
feature_head.py
---------------
A lightweight MLP that reads the 79 Jane Street feature columns from the
context window and produces a per-horizon *correction delta* to be added
on top of TimesFM's base forecast.

Architecture
------------
context_features: (batch, context_len, num_features)
    └─ mean(dim=1) ─► (batch, num_features)   [temporal avg-pool]
        └─ Linear(num_features → hidden_dim) → LayerNorm → GELU
            └─ Linear(hidden_dim → hidden_dim) → LayerNorm → GELU
                └─ Linear(hidden_dim → horizon_len)
                    └─ delta: (batch, horizon_len)

final_pred = timesfm_base_pred + delta

The correction head is intentionally small so it regularises well with
limited data and doesn't dominate the base model's signal.  The output is
*unbounded* (no tanh) so the optimiser can freely scale corrections.
"""

import torch
import torch.nn as nn


class FeatureCorrectionHead(nn.Module):
    """
    2-layer MLP correction head that consumes context-window features and
    produces an additive delta on top of the TimesFM base forecast.

    Parameters
    ----------
    num_features : int
        Number of input feature columns (79 for Jane Street default).
    context_len : int
        Length of the context window (used only for documentation; the
        module itself reduces the time dimension via average pooling).
    horizon_len : int
        Number of future steps to predict (must match the base model).
    hidden_dim : int
        Width of the two hidden layers.  Default 128.
    dropout : float
        Dropout probability applied after each hidden activation.
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
            # Temporal average pooling happens *before* this net (in forward).
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

        # Initialise the last linear layer near zero so the head starts as a
        # near-identity correction (i.e. doesn't disturb the base model at
        # the beginning of training).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        context_features : Tensor, shape (batch, context_len, num_features)

        Returns
        -------
        delta : Tensor, shape (batch, horizon_len)
        """
        # Average-pool over the time axis → (batch, num_features)
        pooled = context_features.mean(dim=1)
        # MLP → (batch, horizon_len)
        return self.net(pooled)


# --------------------------------------------------------------------------
# Save / load helpers (stored alongside the LoRA adapter in the same dir).
# --------------------------------------------------------------------------
FEATURE_HEAD_FILENAME = "feature_head.pt"


def save_feature_head(head: FeatureCorrectionHead, output_dir: str) -> None:
    import os
    path = os.path.join(output_dir, FEATURE_HEAD_FILENAME)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "config": {
                "num_features": head.num_features,
                "context_len": head.context_len,
                "horizon_len": head.horizon_len,
                "hidden_dim": head.hidden_dim,
            },
        },
        path,
    )
    print(f"  -> feature head saved to {path}")


def load_feature_head(output_dir: str, device: str = "cpu") -> FeatureCorrectionHead:
    import os
    path = os.path.join(output_dir, FEATURE_HEAD_FILENAME)
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    head = FeatureCorrectionHead(**cfg)
    head.load_state_dict(ckpt["state_dict"])
    return head.to(device)
