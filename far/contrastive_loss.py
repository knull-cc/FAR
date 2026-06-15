"""InfoNCE / supervised-contrastive loss over future-defined pairs.

The encoder embeds the PAST window. Positives/negatives are defined by FUTURE
similarity (see future_similarity.build_pos_neg). We therefore train the past
encoder to pull together windows whose futures are alike and push apart windows
whose futures diverge -- i.e. we make "past embedding similarity" predict
"future similarity".

This is a supervised-contrastive generalization of InfoNCE that supports:
    - multiple positives per anchor (averaged log-likelihood),
    - per-negative weighting (used by hard-negative mining, module B5).
"""

import torch
import torch.nn.functional as F


def info_nce_loss(embeddings, pos_mask, neg_mask=None, temperature=0.1,
                  neg_weights=None):
    """Supervised InfoNCE with future-defined positives.

    Args:
        embeddings: (N, d) L2-normalizable embeddings of the PAST windows.
        pos_mask: (N, N) bool, True where j is a future-positive of anchor i.
        neg_mask: (N, N) bool, True where j is a valid negative of anchor i.
            If None, all non-positive, non-self entries are negatives.
        temperature: InfoNCE temperature.
        neg_weights: optional (N, N) >= 0 multiplicative weights applied to the
            negative logits' contribution (e.g. up-weight hard negatives).

    Returns:
        scalar loss.
    """
    N = embeddings.shape[0]
    device = embeddings.device

    z = F.normalize(embeddings, dim=1)
    logits = z @ z.t() / temperature  # (N, N)

    eye = torch.eye(N, dtype=torch.bool, device=device)
    if neg_mask is None:
        neg_mask = ~pos_mask & ~eye

    # numerical stability
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    exp_logits = torch.exp(logits)
    exp_logits = exp_logits.masked_fill(eye, 0.0)

    # denominator = positives + (optionally weighted) negatives
    denom_weights = torch.zeros_like(logits)
    denom_weights[pos_mask] = 1.0
    if neg_weights is not None:
        denom_weights = denom_weights + neg_weights * neg_mask.float()
    else:
        denom_weights = denom_weights + neg_mask.float()

    denom = (exp_logits * denom_weights).sum(dim=1, keepdim=True) + 1e-12
    log_prob = logits - torch.log(denom)

    pos_counts = pos_mask.sum(dim=1).clamp(min=1)
    mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1) / pos_counts

    valid = pos_mask.sum(dim=1) > 0
    if valid.sum() == 0:
        return torch.zeros((), device=device, requires_grad=True)
    return -mean_log_prob_pos[valid].mean()
