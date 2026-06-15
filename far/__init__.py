"""FAR: Future-Aligned Retrieval for robust time series forecasting.

A model-agnostic, plug-in retriever that replaces past-similarity with
future-alignment in retrieval-augmented time series forecasting (RAG-TSF).

The only thing FAR changes relative to a host RAG-TSF pipeline is the
*retrieval similarity metric*: instead of "the past looks alike", FAR
retrieves neighbors whose *futures* will be alike. The future is used as a
supervision signal during training only; at inference the encoder sees only
the past window (plus available covariates).
"""

from far.normalization import RevIN, instance_normalize
from far.future_similarity import (
    future_similarity,
    pairwise_future_distance,
    build_pos_neg,
)
from far.contrastive_loss import info_nce_loss
from far.encoder import FAREncoder
from far.gating import RetrievalGate
from far.hard_negative import hard_negative_weights

__all__ = [
    "RevIN",
    "instance_normalize",
    "future_similarity",
    "pairwise_future_distance",
    "build_pos_neg",
    "info_nce_loss",
    "FAREncoder",
    "RetrievalGate",
    "hard_negative_weights",
]
