"""B5: Hard-negative mining.

The richest training signal for FAR is the set of "past-similar but
future-divergent" pairs: these are exactly the cases where a naive
past-similarity retriever is fooled into pulling in toxic futures. Explicitly
up-weighting them as negatives sharpens the encoder's discriminative power.

We return a (N, N) weight matrix that scales each negative's contribution in
the InfoNCE denominator. Weight grows with (small past distance) AND
(large future distance).
"""

import torch

from far.future_similarity import pairwise_future_distance


def _minmax(x, eps=1e-8):
    lo = x.min()
    hi = x.max()
    return (x - lo) / (hi - lo + eps)


def hard_negative_weights(past_windows, future_windows, neg_mask,
                          metric="shape", base_weight=1.0, hard_scale=3.0,
                          gamma=0.1):
    """Compute per-negative weights emphasizing hard negatives.

    Args:
        past_windows: (N, S, C) past windows of the batch.
        future_windows: (N, P, C) future windows of the batch.
        neg_mask: (N, N) bool, valid negatives.
        metric: future-distance metric (must match the one used for labeling).
        base_weight: baseline weight applied to every negative.
        hard_scale: max extra weight added to the hardest negatives.
        gamma: soft-DTW smoothing (if metric == "softdtw").

    Returns:
        (N, N) float weight matrix (>= 0); zero where not a negative.
    """
    N = past_windows.shape[0]

    past_flat = past_windows.reshape(N, -1)
    past_dist = torch.cdist(past_flat, past_flat, p=2)          # small = similar past
    fut_dist = pairwise_future_distance(
        future_windows, metric=metric, gamma=gamma
    )                                                           # large = divergent future

    past_sim = 1.0 - _minmax(past_dist)   # in [0,1], 1 = identical past
    fut_div = _minmax(fut_dist)           # in [0,1], 1 = most divergent future

    hardness = past_sim * fut_div         # high only for past-similar & future-divergent
    weights = base_weight + hard_scale * hardness
    weights = weights * neg_mask.float()
    return weights


def mine_hard_negative_pairs(past_windows, future_windows, top_pairs=100,
                             metric="shape", gamma=0.1):
    """Return indices of the hardest (past-similar, future-divergent) pairs.

    Useful for logging / diagnostics and for the synthetic validation plot.

    Returns:
        LongTensor of shape (top_pairs, 2) with (i, j) index pairs.
    """
    N = past_windows.shape[0]
    past_flat = past_windows.reshape(N, -1)
    past_dist = torch.cdist(past_flat, past_flat, p=2)
    fut_dist = pairwise_future_distance(future_windows, metric=metric, gamma=gamma)

    eye = torch.eye(N, dtype=torch.bool, device=past_windows.device)
    hardness = (1.0 - _minmax(past_dist)) * _minmax(fut_dist)
    hardness = hardness.masked_fill(eye, -1.0)

    flat = hardness.flatten()
    k = min(top_pairs, flat.numel())
    idx = torch.topk(flat, k).indices
    rows = idx // N
    cols = idx % N
    return torch.stack([rows, cols], dim=1)
