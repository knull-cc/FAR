"""FAR future-aligned retrieval encoder (A2 + A3).

forward(past, covariates) -> retrieval_embedding

Design notes:
    * A3 (instance normalization): a RevIN layer normalizes the multivariate
      past window so the embedding is invariant to absolute level / scale.
    * A2 (covariate-augmented input): optional time/calendar covariates (and
      exogenous channels) are concatenated to the target past so the future is
      *identifiable* from the inputs. If same-past->different-future is driven
      by unobserved exogenous variables, no encoder can separate them; A2 is
      how FAR lifts that ceiling when covariates are available.

The backbone is a lightweight dilated temporal CNN followed by global pooling
and a projection head. The output embedding is L2-normalized so cosine
similarity in embedding space is the retrieval metric (drop-in replacement for
the host's cosine/euclidean key, per the RATF/RAFT integration point).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from far.normalization import RevIN


class _DilatedConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        x = self.dropout(self.act(self.conv1(x)))
        x = self.conv2(x)
        x = self.norm(x + res)
        return self.act(x)


class FAREncoder(nn.Module):
    """Maps a past window (+ covariates) to an L2-normalized embedding.

    Args:
        seq_len: length of the past window.
        in_channels: number of target channels.
        cov_channels: number of covariate channels concatenated to the input
            (0 disables A2 covariate augmentation).
        d_model: width of the temporal backbone.
        emb_dim: dimensionality of the output retrieval embedding.
        n_blocks: number of dilated conv blocks.
        dropout: dropout rate.
        use_revin: enable A3 instance normalization on the target past.
    """

    def __init__(self, seq_len, in_channels, cov_channels=0, d_model=128,
                 emb_dim=128, n_blocks=3, dropout=0.1, use_revin=True):
        super().__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.cov_channels = cov_channels
        self.use_revin = use_revin

        if use_revin:
            self.revin = RevIN(in_channels)

        total_in = in_channels + cov_channels
        self.input_proj = nn.Conv1d(total_in, d_model, kernel_size=1)
        self.blocks = nn.ModuleList([
            _DilatedConvBlock(d_model, dilation=2 ** i, dropout=dropout)
            for i in range(n_blocks)
        ])
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, emb_dim),
        )

    def forward(self, past, covariates=None, normalize=True):
        """Args:
            past: (B, seq_len, in_channels) target past window.
            covariates: (B, seq_len, cov_channels) or None.
            normalize: L2-normalize the output embedding.
        Returns:
            (B, emb_dim) embedding.
        """
        x = past
        if self.use_revin:
            x = self.revin(x, "norm")

        if self.cov_channels > 0 and covariates is not None:
            x = torch.cat([x, covariates], dim=-1)

        x = x.permute(0, 2, 1)          # (B, C, L)
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=-1)              # global average pool -> (B, d_model)
        emb = self.proj_head(x)
        if normalize:
            emb = F.normalize(emb, dim=-1)
        return emb
