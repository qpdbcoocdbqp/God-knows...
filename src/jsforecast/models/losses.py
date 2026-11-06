"""
losses.py
---------
Weighted loss for Jane Street style targets.
"""

import torch
import torch.nn as nn


class JaneStreetWeightedMSELoss(nn.Module):
    """Weighted MSE: mean_i( weight_i * (pred_i - true_i)^2 ). Shapes: (batch, horizon)."""

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return (((y_pred - y_true) ** 2) * weight).mean()
