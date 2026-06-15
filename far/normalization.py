"""A3: Instance normalization utilities (RevIN-style).

Under marginal shift P(X), raw levels drift (e.g. FX trends up). Without
per-instance normalization, retrieval compares *levels* instead of
*shape / outcome*. We normalize every window (query, KB-past and KB-future)
consistently so the retrieval metric is level-invariant.
"""

import torch
import torch.nn as nn


def instance_normalize(x, eps=1e-5, return_stats=False):
    """Per-instance, per-channel z-normalization over the time dimension.

    Args:
        x: tensor of shape (B, L, C).
        eps: numerical stabilizer.
        return_stats: if True, also return (mean, std) for de-normalization.

    Returns:
        x_norm (and optionally mean, std), broadcastable over the time dim.
    """
    mean = x.mean(dim=1, keepdim=True)
    std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + eps)
    x_norm = (x - mean) / std
    if return_stats:
        return x_norm, mean, std
    return x_norm


class RevIN(nn.Module):
    """Reversible Instance Normalization.

    Reference: Kim et al., "Reversible Instance Normalization for Accurate
    Time-Series Forecasting against Distribution Shift", ICLR 2022.

    Used inside the FAR encoder so the learned embedding is invariant to the
    absolute level / scale of each input window.
    """

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError(f"Unknown RevIN mode: {mode}")

    def _get_statistics(self, x):
        # x: (B, L, C)
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(
            x.var(dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev + self.mean
        return x
