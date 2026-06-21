"""
losses.py
---------
Weighted loss(es) for Jane Street style targets.

`JaneStreetWeightedMSELoss` is the practical loss used by this pipeline: it
applies the official per-row trading `weight` to a single target
(responder_6 by default), matching what `data.JaneStreetSeriesDataset`
produces.

`JaneStreetMultitaskLoss` is kept (and fixed) for users who extend the
pipeline to predict multiple responders at once with a multi-head model;
the original snippet had two bugs fixed here:
  1. `self.target_weights` was reassigned every forward() instead of being
     registered as a buffer (works, but doesn't move with .to(device)/.half()).
  2. No protection against div-by-zero when a target's batch loss is ~0.
"""

import torch
import torch.nn as nn


class JaneStreetWeightedMSELoss(nn.Module):
    """Weighted MSE: mean_i( weight_i * (pred_i - true_i)^2 )."""

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # shapes: (batch, horizon)
        se = (y_pred - y_true) ** 2
        weighted = se * weight
        return weighted.mean()


class JaneStreetMultitaskLoss(nn.Module):
    def __init__(self, target_weights=None, num_targets: int = 9, eps: float = 1e-8):
        super().__init__()
        if target_weights is None:
            # default: responder_6 (index 6) weighted highest
            target_weights = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1])
        else:
            target_weights = torch.as_tensor(target_weights, dtype=torch.float32)
        assert target_weights.numel() == num_targets, "target_weights must match num_targets"
        self.register_buffer("target_weights", target_weights)
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        y_pred: (batch, num_targets)
        y_true: (batch, num_targets)
        weights: (batch, 1) trading weight, broadcast across targets
        """
        individual_losses = (y_pred - y_true) ** 2 * weights.view(-1, 1)
        mean_losses = individual_losses.mean(dim=0)  # (num_targets,)

        # scale each target's loss to ~1 (detached) so target_weights controls
        # the relative emphasis rather than raw target scale
        scaled_losses = mean_losses / (mean_losses.detach() + self.eps)

        total_loss = torch.sum(scaled_losses * self.target_weights)
        return total_loss
