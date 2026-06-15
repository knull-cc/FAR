"""B4: Confidence-aware retrieval gating.

FAR knows whether a *truly future-aligned* neighbor exists. If the query enters
a novel regime (no good match in embedding space), retrieval should be
down-weighted and the host backbone trusted; if a highly aligned neighbor
exists, retrieval should be up-weighted.

gate(query_emb, retrieved_sims) -> scalar weight in [0, 1] that modulates the
retrieval contribution during fusion. The gate is a function of the retrieval
confidence (top-k similarities), so it can be computed online from the past
only -- no future leakage.
"""

import torch
import torch.nn as nn


class RetrievalGate(nn.Module):
    """Maps retrieval-confidence features to a fusion weight in [0, 1].

    Confidence features are derived from the top-k retrieval similarities:
        [max_sim, mean_topk_sim, std_topk_sim, gap(top1 - topk)].
    These summarize "is there a confidently aligned neighbor?".
    """

    def __init__(self, hidden=16, learnable=True, bias_init=2.0):
        super().__init__()
        self.learnable = learnable
        if learnable:
            self.net = nn.Sequential(
                nn.Linear(4, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            # start near "open gate" (retrieval trusted) so training is stable
            nn.init.zeros_(self.net[-1].weight)
            nn.init.constant_(self.net[-1].bias, bias_init)

    @staticmethod
    def confidence_features(topk_sims):
        """Build confidence features from top-k similarities.

        Args:
            topk_sims: (B, k) similarities of the retrieved neighbors,
                sorted descending (cosine in [-1, 1] or correlation).
        Returns:
            (B, 4) feature tensor.
        """
        max_sim = topk_sims[:, :1]
        mean_sim = topk_sims.mean(dim=1, keepdim=True)
        std_sim = topk_sims.std(dim=1, keepdim=True)
        gap = topk_sims[:, :1] - topk_sims[:, -1:]
        return torch.cat([max_sim, mean_sim, std_sim, gap], dim=1)

    def forward(self, topk_sims):
        """Args:
            topk_sims: (B, k) descending retrieval similarities.
        Returns:
            (B, 1) gate weights in [0, 1].
        """
        feats = self.confidence_features(topk_sims)
        if not self.learnable:
            # heuristic fallback: gate = normalized max similarity
            return ((feats[:, :1] + 1.0) / 2.0).clamp(0.0, 1.0)
        return torch.sigmoid(self.net(feats))
