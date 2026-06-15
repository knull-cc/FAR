"""FAR retriever: train the encoder(s), index the KB, and do Top-K lookup.

This is the object that a host pipeline plugs in to replace its
cosine/euclidean past-similarity key. It exposes:

    fit(...)          -> contrastively train the future-aligned encoder(s)
    encode_kb(...)    -> build the embedding index over the knowledge base
    query_similarity  -> (G, B, T) similarity of queries to every KB entry

Per-grain retrieval
-------------------
The host (RAFT) decomposes every window into ``n_grains`` multi-grain views
(progressively smoothed copies of the series). The baseline correlation key is
computed *per grain*, so each grain retrieves its own neighbors and the
downstream multi-grain aggregation is genuinely diverse. FAR mirrors this: it
holds one future-aligned encoder per grain and produces a separate
future-aligned ranking for every grain, where "future" is that grain's
(smoothed) future window. Everything downstream (multi-grain future
aggregation + linear fusion head) is left unchanged.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from far.encoder import FAREncoder
from far.gating import RetrievalGate
from far.contrastive_loss import info_nce_loss
from far.future_similarity import build_pos_neg
from far.hard_negative import hard_negative_weights
from far.normalization import instance_normalize


class FARRetriever:
    def __init__(self, seq_len, pred_len, channels, cov_channels=0,
                 emb_dim=128, d_model=128, n_blocks=3, dropout=0.1,
                 use_revin=True, temperature=0.1, pos_k=5,
                 future_metric="shape", soft_dtw_gamma=0.1,
                 use_hard_neg=False, hard_scale=3.0, use_gating=False,
                 n_grains=1, aux_weight=1.0):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.cov_channels = cov_channels

        self.temperature = temperature
        self.pos_k = pos_k
        self.future_metric = future_metric
        self.soft_dtw_gamma = soft_dtw_gamma
        self.use_hard_neg = use_hard_neg
        self.hard_scale = hard_scale
        self.use_gating = use_gating
        self.n_grains = n_grains
        # Weight of the auxiliary future-trend regression loss. > 0 distills
        # dense future-trend supervision into the past embedding during
        # training (see fit); the regression heads are dropped at inference.
        self.aux_weight = aux_weight

        # One future-aligned encoder per grain (mirrors RAFT's per-grain key).
        self.encoders = [
            FAREncoder(
                seq_len=seq_len,
                in_channels=channels,
                cov_channels=cov_channels,
                d_model=d_model,
                emb_dim=emb_dim,
                n_blocks=n_blocks,
                dropout=dropout,
                use_revin=use_revin,
            )
            for _ in range(n_grains)
        ]
        # Per-grain auxiliary head: past embedding -> (normalized) future trend
        # at that grain. Training-only; never used for retrieval/inference.
        self.future_heads = None
        if aux_weight > 0:
            self.future_heads = [
                nn.Linear(emb_dim, pred_len * channels)
                for _ in range(n_grains)
            ]
        self.gate = RetrievalGate(learnable=True) if use_gating else None

        self.kb_emb = [None] * n_grains  # list of (T, emb_dim) per grain

    def to(self, device):
        self.encoders = [enc.to(device) for enc in self.encoders]
        if self.future_heads is not None:
            self.future_heads = [h.to(device) for h in self.future_heads]
        if self.gate is not None:
            self.gate = self.gate.to(device)
        return self

    def parameters(self):
        params = []
        for enc in self.encoders:
            params += list(enc.parameters())
        if self.future_heads is not None:
            for h in self.future_heads:
                params += list(h.parameters())
        if self.gate is not None:
            params += list(self.gate.parameters())
        return params

    def _train_mode(self):
        for enc in self.encoders:
            enc.train()
        if self.future_heads is not None:
            for h in self.future_heads:
                h.train()

    def _eval_mode(self):
        for enc in self.encoders:
            enc.eval()
        if self.future_heads is not None:
            for h in self.future_heads:
                h.eval()

    # ------------------------------------------------------------------ train
    def fit(self, past_all_mg, future_all_mg, cov_all=None,
            device=torch.device("cpu"), epochs=10, batch_size=256, lr=1e-3,
            verbose=True):
        """Contrastively train one future-aligned encoder per grain.

        Positives/negatives are defined by FUTURE similarity (at each grain);
        the encoder only ever sees the PAST (+ covariates). No future leakage
        at inference.

        Args:
            past_all_mg: (G, T, seq_len, C) per-grain KB past windows.
            future_all_mg: (G, T, pred_len, C) per-grain KB future windows.
            cov_all: (T, seq_len, cov_channels) or None (shared across grains).
        """
        self.to(device)
        self._train_mode()

        n = past_all_mg.shape[1]
        idx_ds = TensorDataset(torch.arange(n))
        loader = DataLoader(idx_ds, batch_size=batch_size, shuffle=True,
                            drop_last=True)
        optim = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            losses = []
            it = tqdm(loader, disable=not verbose,
                      desc=f"FAR encoder epoch {epoch + 1}/{epochs}")
            for (batch_idx,) in it:
                cov = None
                if cov_all is not None and self.cov_channels > 0:
                    cov = cov_all[batch_idx].to(device)

                total_loss = 0.0
                for g in range(self.n_grains):
                    past = past_all_mg[g, batch_idx].to(device)
                    future = future_all_mg[g, batch_idx].to(device)

                    emb = self.encoders[g](past, cov)

                    pos_mask, neg_mask = build_pos_neg(
                        future, metric=self.future_metric, pos_k=self.pos_k,
                        gamma=self.soft_dtw_gamma,
                    )

                    neg_weights = None
                    if self.use_hard_neg:
                        neg_weights = hard_negative_weights(
                            past, future, neg_mask,
                            metric=self.future_metric,
                            hard_scale=self.hard_scale,
                            gamma=self.soft_dtw_gamma,
                        )

                    grain_loss = info_nce_loss(
                        emb, pos_mask, neg_mask,
                        temperature=self.temperature,
                        neg_weights=neg_weights,
                    )

                    # Auxiliary future-trend regression: force the PAST
                    # embedding to predict the (instance-normalized) FUTURE
                    # trend at this grain. This is dense per-sample supervision
                    # that aligns past trend with future trend; the head is
                    # training-only and never touches inference (no leakage).
                    if self.future_heads is not None:
                        target = instance_normalize(future)  # B, P, C (shape/trend)
                        target = target.reshape(target.shape[0], -1)
                        pred_trend = self.future_heads[g](emb)
                        aux_loss = F.mse_loss(pred_trend, target)
                        grain_loss = grain_loss + self.aux_weight * aux_loss

                    total_loss = total_loss + grain_loss

                loss = total_loss / self.n_grains

                optim.zero_grad()
                loss.backward()
                optim.step()
                losses.append(loss.item())
                if verbose:
                    it.set_postfix(loss=sum(losses) / len(losses))

        self._eval_mode()
        return self

    # ------------------------------------------------------------------ index
    @torch.no_grad()
    def encode_kb(self, past_all_mg, cov_all=None, device=torch.device("cpu"),
                  batch_size=1024):
        """Build a per-grain embedding index over the knowledge base.

        Args:
            past_all_mg: (G, T, seq_len, C) per-grain KB past windows.
            cov_all: (T, seq_len, cov_channels) or None.
        """
        self.to(device)
        self._eval_mode()
        n = past_all_mg.shape[1]
        for g in range(self.n_grains):
            embs = []
            for i in range(math.ceil(n / batch_size)):
                sl = slice(i * batch_size, (i + 1) * batch_size)
                past = past_all_mg[g, sl].to(device)
                cov = None
                if cov_all is not None and self.cov_channels > 0:
                    cov = cov_all[sl].to(device)
                embs.append(self.encoders[g](past, cov).cpu())
            self.kb_emb[g] = torch.cat(embs, dim=0)  # (T, d)
        return self.kb_emb

    # ------------------------------------------------------------------ query
    @torch.no_grad()
    def query_similarity(self, past_mg, cov=None, device=None):
        """Per-grain cosine similarity between query and every KB entry.

        Args:
            past_mg: (G, B, seq_len, C) per-grain query past windows.
            cov: (B, seq_len, cov_channels) or None (shared across grains).
        Returns:
            (G, B, T) similarity in [-1, 1].
        """
        if device is None:
            device = past_mg.device
        self._eval_mode()
        cov_d = None if cov is None else cov.to(device)

        sims = []
        for g in range(self.n_grains):
            q = self.encoders[g](past_mg[g].to(device), cov_d)
            kb = self.kb_emb[g].to(device)
            sims.append(F.normalize(q, dim=1) @ F.normalize(kb, dim=1).t())
        return torch.stack(sims, dim=0)  # (G, B, T)
