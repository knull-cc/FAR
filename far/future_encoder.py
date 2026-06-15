"""Future-trend encoder for FAR (Future-Aligned Retrieval).

Minimal addition on top of RAFT. A small channel-independent encoder maps a
*past* window to an embedding. During (offline) training a regression head
forces this embedding to predict the *instance-normalized future window*
(i.e. the future trend / shape). The future supervision is therefore
distilled into the past representation without any test-time leakage: the
encoder only ever sees past windows at retrieval time.

After offline training the regression head is discarded and only the encoder
embedding is kept, used to produce a cosine `far_sim` that is blended (with a
small weight alpha) into RAFT's correlation similarity.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader


class FutureTrendEncoder(nn.Module):
    def __init__(self, seq_len, pred_len, emb_dim=64, hidden=128):
        super(FutureTrendEncoder, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.emb_dim = emb_dim

        # Channel-independent temporal encoder: (..., seq_len) -> (..., emb_dim)
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, hidden),
            nn.GELU(),
            nn.Linear(hidden, emb_dim),
        )
        # Regression head: embedding -> instance-normalized future shape.
        # Used only during offline training, discarded for retrieval.
        self.head = nn.Linear(emb_dim, pred_len)

    @staticmethod
    def _instance_norm(x, eps=1e-5):
        # x: (..., L); normalize per-instance, per-channel over the time axis.
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + eps
        return (x - mean) / std

    def _encode_channels(self, x):
        # x: (B, S, C) -> per-channel embedding (B, C, emb_dim)
        x = x.permute(0, 2, 1)            # B, C, S
        x = self._instance_norm(x)        # normalize past shape
        return self.encoder(x)            # B, C, D

    def embed(self, x):
        """Past window (B, S, C) -> single instance embedding (B, emb_dim)."""
        e = self._encode_channels(x)      # B, C, D
        return e.mean(dim=1)              # mean-pool channels -> B, D

    def forward(self, x):
        """Returns (predicted_normalized_future (B, C, P), embedding (B, D))."""
        e = self._encode_channels(x)      # B, C, D
        pred = self.head(e)               # B, C, P
        emb = e.mean(dim=1)               # B, D
        return pred, emb


def train_future_encoder(model, train_data, device, epochs=10, lr=1e-3,
                         batch_size=256, num_workers=4):
    """Offline-train the encoder to predict the instance-normalized future.

    `train_data` is a dataset yielding (index, seq_x, seq_y, x_mark, y_mark).
    The future window is `seq_y[-pred_len:]`, instance-normalized per channel.
    """
    loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    pred_len = model.pred_len

    model.train()
    for epoch in range(epochs):
        losses = []
        for index, batch_x, batch_y, _, _ in loader:
            batch_x = batch_x.float().to(device)                 # B, S, C
            future = batch_y[:, -pred_len:, :].float().to(device)  # B, P, C
            future = future.permute(0, 2, 1)                     # B, C, P
            future_n = FutureTrendEncoder._instance_norm(future)  # trend/shape

            pred, _ = model(batch_x)                             # B, C, P
            loss = F.mse_loss(pred, future_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print('[FAR] encoder epoch {}/{} | mse {:.6f}'.format(
            epoch + 1, epochs, float(np.mean(losses))))

    model.eval()
    return model
