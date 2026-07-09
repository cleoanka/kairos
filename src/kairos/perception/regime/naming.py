"""Name unsupervised clusters by their micro-structure *signature*.

Real markets carry no ground-truth regime labels — which is precisely why a
label-free method matters. After clustering, we still want interpretable names,
so we read the signature straight from the features:

  * the cluster with the most **cancel-flow** is TOXIC (spoofing / phantom depth),
  * of the rest, the one with the most **trade volume** is TREND (directional flow),
  * the quiet remainder is RANGE.

This works identically on synthetic and live data, with no labels.
"""
from __future__ import annotations

import numpy as np

from ..schema import Regime, block_slices


def cluster_signatures(X: np.ndarray, pred: np.ndarray, k: int = 3) -> dict:
    sl = block_slices()
    cancel = X[:, sl["bid_cxl"]].sum(1) + X[:, sl["ask_cxl"]].sum(1)
    trade = X[:, sl["trade"].start + 1]   # log total trade flow
    sig = {}
    for c in range(k):
        m = pred == c
        sig[c] = {
            "cancel": float(cancel[m].mean()) if m.any() else 0.0,
            "trade": float(trade[m].mean()) if m.any() else 0.0,
            "n": int(m.sum()),
        }
    return sig


def name_clusters_by_signature(X: np.ndarray, pred: np.ndarray, k: int = 3) -> dict:
    """Map cluster id -> Regime id by micro-structure signature."""
    sig = cluster_signatures(X, pred, k)
    clusters = list(range(k))
    toxic = max(clusters, key=lambda c: sig[c]["cancel"])
    rest = [c for c in clusters if c != toxic]
    trend = max(rest, key=lambda c: sig[c]["trade"]) if rest else toxic
    out = {toxic: int(Regime.TOXIC)}
    for c in rest:
        out[c] = int(Regime.TREND) if c == trend else int(Regime.RANGE)
    return out
