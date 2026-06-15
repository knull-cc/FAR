"""A1: Future-Similarity label definition + positive/negative pair construction.

This module defines the *ground truth* notion of "similar future" that drives
all contrastive learning. It is the linchpin of FAR: positives are pairs whose
FUTURES are alike (not whose pasts are alike).

All distances are computed on the *instance-normalized* future window so that
similarity reflects shape / outcome rather than absolute level (this is what
makes the metric robust under marginal shift P(X)).

Supported metrics (configurable via `metric`):
    - "shape"   : Euclidean distance on instance-normalized Y (cheap soft-DTW proxy).
    - "euclid"  : plain Euclidean distance on raw Y.
    - "corr"    : 1 - Pearson correlation of Y (direction / co-movement).
    - "softdtw" : soft-DTW on instance-normalized Y (alignment-aware, default).
    - "slope"   : Euclidean distance on the first-difference (direction/turning points).
"""

import torch

from far.normalization import instance_normalize


def _flatten_channels(y):
    """(B, P, C) -> (B, P*C). Treats the multivariate future jointly."""
    return y.reshape(y.shape[0], -1)


def _soft_dtw_distance(a, b, gamma=0.1):
    """Batched soft-DTW between two sets of univariate sequences.

    Args:
        a: (B, P) anchor sequences.
        b: (M, P) candidate sequences.
    Returns:
        (B, M) soft-DTW distances.

    Implemented with a vectorized DP over the (B, M) cross-product. P is small
    for typical forecasting horizons so the O(P^2) DP is affordable.
    """
    B, P = a.shape
    M, _ = b.shape
    # pairwise squared cost matrix: (B, M, P, P)
    a_e = a.unsqueeze(1).unsqueeze(-1)   # (B, 1, P, 1)
    b_e = b.unsqueeze(0).unsqueeze(2)    # (1, M, 1, P)
    cost = (a_e - b_e) ** 2              # (B, M, P, P)

    inf = 1e10
    R = torch.full((B, M, P + 1, P + 1), inf, device=a.device, dtype=a.dtype)
    R[:, :, 0, 0] = 0.0
    for i in range(1, P + 1):
        for j in range(1, P + 1):
            r0 = -R[:, :, i - 1, j - 1] / gamma
            r1 = -R[:, :, i - 1, j] / gamma
            r2 = -R[:, :, i, j - 1] / gamma
            rmax = torch.maximum(torch.maximum(r0, r1), r2)
            softmin = -gamma * (
                rmax + torch.log(
                    torch.exp(r0 - rmax) + torch.exp(r1 - rmax) + torch.exp(r2 - rmax)
                )
            )
            R[:, :, i, j] = cost[:, :, i - 1, j - 1] + softmin
    return R[:, :, P, P]


def pairwise_future_distance(Y, metric="shape", gamma=0.1, normalize=True):
    """Pairwise distance matrix between future windows.

    Args:
        Y: (N, P, C) future windows (e.g. one minibatch of KB entries).
        metric: one of {"shape", "euclid", "corr", "softdtw", "slope"}.
        gamma: soft-DTW smoothing.
        normalize: apply instance normalization on Y before measuring distance.

    Returns:
        D: (N, N) distance matrix (0 on the diagonal, larger = more divergent).
    """
    if normalize and metric not in ("corr",):
        Y = instance_normalize(Y)

    if metric == "softdtw":
        # average soft-DTW over channels (channels treated independently)
        N, P, C = Y.shape
        D = torch.zeros(N, N, device=Y.device, dtype=Y.dtype)
        for c in range(C):
            D = D + _soft_dtw_distance(Y[:, :, c], Y[:, :, c], gamma=gamma)
        return D / C

    if metric == "slope":
        Y = Y[:, 1:, :] - Y[:, :-1, :]

    if metric == "corr":
        f = _flatten_channels(Y)
        f = f - f.mean(dim=1, keepdim=True)
        f = torch.nn.functional.normalize(f, dim=1)
        sim = f @ f.t()
        return 1.0 - sim

    f = _flatten_channels(Y)
    return torch.cdist(f, f, p=2)


def future_similarity(y_i, y_j, metric="shape", gamma=0.1, normalize=True):
    """Scalar future-similarity (negative distance) between two windows.

    Convenience wrapper matching the deliverable signature in the spec:
        future_similarity(y_i, y_j) -> scalar
    Larger = more similar future.

    Args:
        y_i, y_j: (P, C) future windows.
    """
    Y = torch.stack([y_i, y_j], dim=0)  # (2, P, C)
    D = pairwise_future_distance(Y, metric=metric, gamma=gamma, normalize=normalize)
    return -D[0, 1]


def build_pos_neg(Y, metric="shape", pos_k=5, neg_ratio=1.0, gamma=0.1,
                  normalize=True, return_distance=False):
    """Construct positive / negative sets from future similarity.

    Within a batch of future windows, the `pos_k` nearest neighbors (in future
    space) of each anchor are labeled positive; everything else is a candidate
    negative. The diagonal (self) is excluded.

    Args:
        Y: (N, P, C) future windows of the batch.
        metric: future-distance metric (see pairwise_future_distance).
        pos_k: number of positives per anchor.
        neg_ratio: kept for API symmetry; all non-positives serve as negatives
            in the in-batch InfoNCE, so this is informational only.
        gamma: soft-DTW smoothing.
        normalize: instance-normalize futures before measuring distance.
        return_distance: also return the (N, N) future-distance matrix.

    Returns:
        pos_mask: (N, N) bool, True where j is a future-positive of i.
        neg_mask: (N, N) bool, True where j is a valid negative of i.
        (optional) D: (N, N) future-distance matrix.
    """
    N = Y.shape[0]
    D = pairwise_future_distance(Y, metric=metric, gamma=gamma, normalize=normalize)

    eye = torch.eye(N, dtype=torch.bool, device=Y.device)
    D_masked = D.masked_fill(eye, float("inf"))

    k = min(pos_k, N - 1)
    pos_idx = torch.topk(D_masked, k, dim=1, largest=False).indices  # nearest futures
    pos_mask = torch.zeros(N, N, dtype=torch.bool, device=Y.device)
    pos_mask.scatter_(1, pos_idx, True)

    neg_mask = ~pos_mask & ~eye

    if return_distance:
        return pos_mask, neg_mask, D
    return pos_mask, neg_mask
