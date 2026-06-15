"""FAR retriever: train the encoder, index the KB, and do Top-K lookup.

This is the object that a host pipeline plugs in to replace its
cosine/euclidean past-similarity key. It exposes:

    fit(...)          -> contrastively train the future-aligned encoder
    encode_kb(...)    -> build the embedding index over the knowledge base
    query_similarity  -> (B, T) similarity of queries to every KB entry

The actual fusion of retrieved futures is left to the host (FAR is a plug-in
retriever, not a backbone). In RAFT, `layers/Retrieval.py` consumes the
similarity matrix produced here and reuses RAFT's multi-grain future
aggregation + linear fusion head unchanged.
"""

import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from far.encoder import FAREncoder
from far.gating import RetrievalGate
from far.contrastive_loss import info_nce_loss
from far.future_similarity import build_pos_neg
from far.hard_negative import hard_negative_weights


class FARRetriever:
    def __init__(self, seq_len, pred_len, channels, cov_channels=0,
                 emb_dim=128, d_model=128, n_blocks=3, dropout=0.1,
                 use_revin=True, temperature=0.1, pos_k=5,
                 future_metric="shape", soft_dtw_gamma=0.1,
                 use_hard_neg=False, hard_scale=3.0, use_gating=False):
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

        self.encoder = FAREncoder(
            seq_len=seq_len,
            in_channels=channels,
            cov_channels=cov_channels,
            d_model=d_model,
            emb_dim=emb_dim,
            n_blocks=n_blocks,
            dropout=dropout,
            use_revin=use_revin,
        )
        self.gate = RetrievalGate(learnable=True) if use_gating else None

        self.kb_emb = None  # (T, emb_dim)

    def to(self, device):
        self.encoder = self.encoder.to(device)
        if self.gate is not None:
            self.gate = self.gate.to(device)
        return self

    def parameters(self):
        params = list(self.encoder.parameters())
        if self.gate is not None:
            params += list(self.gate.parameters())
        return params

    # ------------------------------------------------------------------ train
    def fit(self, past_all, future_all, cov_all=None, device=torch.device("cpu"),
            epochs=10, batch_size=256, lr=1e-3, verbose=True):
        """Contrastively train the future-aligned encoder.

        Positives/negatives are defined by FUTURE similarity; the encoder only
        ever sees the PAST (+ covariates). No future leakage at inference.

        Args:
            past_all: (T, seq_len, C) KB past windows.
            future_all: (T, pred_len, C) KB future windows.
            cov_all: (T, seq_len, cov_channels) or None.
        """
        self.to(device)
        self.encoder.train()

        n = past_all.shape[0]
        idx_ds = TensorDataset(torch.arange(n))
        loader = DataLoader(idx_ds, batch_size=batch_size, shuffle=True,
                            drop_last=True)
        optim = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            losses = []
            it = tqdm(loader, disable=not verbose,
                      desc=f"FAR encoder epoch {epoch + 1}/{epochs}")
            for (batch_idx,) in it:
                past = past_all[batch_idx].to(device)
                future = future_all[batch_idx].to(device)
                cov = None
                if cov_all is not None and self.cov_channels > 0:
                    cov = cov_all[batch_idx].to(device)

                emb = self.encoder(past, cov)

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

                loss = info_nce_loss(
                    emb, pos_mask, neg_mask,
                    temperature=self.temperature,
                    neg_weights=neg_weights,
                )

                optim.zero_grad()
                loss.backward()
                optim.step()
                losses.append(loss.item())
                if verbose:
                    it.set_postfix(loss=sum(losses) / len(losses))

        self.encoder.eval()
        return self

    # ------------------------------------------------------------------ index
    @torch.no_grad()
    def encode_kb(self, past_all, cov_all=None, device=torch.device("cpu"),
                  batch_size=1024):
        """Build the embedding index over the knowledge base."""
        self.to(device)
        self.encoder.eval()
        embs = []
        n = past_all.shape[0]
        for i in range(math.ceil(n / batch_size)):
            sl = slice(i * batch_size, (i + 1) * batch_size)
            past = past_all[sl].to(device)
            cov = None
            if cov_all is not None and self.cov_channels > 0:
                cov = cov_all[sl].to(device)
            embs.append(self.encoder(past, cov).cpu())
        self.kb_emb = torch.cat(embs, dim=0)  # (T, d)
        return self.kb_emb

    # ------------------------------------------------------------------ query
    @torch.no_grad()
    def query_embeddings(self, past, cov=None):
        self.encoder.eval()
        return self.encoder(past, cov)

    def query_similarity(self, past, cov=None, device=None):
        """Cosine similarity between query past windows and every KB entry.

        Args:
            past: (B, seq_len, C) query past windows.
            cov: (B, seq_len, cov_channels) or None.
        Returns:
            (B, T) similarity matrix in [-1, 1].
        """
        if device is None:
            device = past.device
        q = self.query_embeddings(past.to(device), None if cov is None else cov.to(device))
        kb = self.kb_emb.to(device)
        return F.normalize(q, dim=1) @ F.normalize(kb, dim=1).t()
