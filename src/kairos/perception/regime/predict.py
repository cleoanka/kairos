"""Live regime inference (Component D/F).

Turns the *detected* regimes (clustered on training data) into a reusable
*predictor* for new/live snapshots: embed with the frozen encoder, standardise
the latent with the training stats, assign to the nearest KMeans centroid, and
map the cluster to its regime via the saved majority map. This is the inference
API the live path (live-dry, the TUI, the strategy) needs — the system can now
*label* the regime of a fresh book, not just cluster the training set.

Label-free training (Rule 4) is preserved: the centroids and cluster→regime map
are produced by unsupervised clustering; ground-truth regimes only *named* the
clusters after the fact.
"""
from __future__ import annotations

import numpy as np

from ..models.embedder import embed, load_trained
from ..schema import REGIME_NAMES, Regime, featurize, featurize_from_raw


class RegimePredictor:
    def __init__(self, model, stats, z_mean, z_std, centroids, cluster_to_regime):
        self.model = model
        self.stats = stats
        self.z_mean = np.asarray(z_mean)
        self.z_std = np.asarray(z_std)
        self.centroids = np.asarray(centroids)            # (k, latent) standardised
        self.cluster_to_regime = np.asarray(cluster_to_regime)

    @classmethod
    def load(cls, weights: str = "artifacts/lob_encoder.safetensors",
             latents: str = "artifacts/latents.npz",
             regime_model: str = "artifacts/regime_model.npz") -> RegimePredictor:
        model, stats = load_trained(weights, latents)
        rm = np.load(regime_model)
        return cls(model, stats, rm["z_mean"], rm["z_std"],
                   rm["centroids"], rm["cluster_to_regime"])

    def predict_features(self, X: np.ndarray) -> np.ndarray:
        """Map a feature matrix ``(n, FEATURE_DIM)`` to regime ids ``(n,)``."""
        X = np.asarray(X)
        z = embed(self.model, X, self.stats)
        zs = (z - self.z_mean) / self.z_std
        d = ((zs[:, None, :] - self.centroids[None, :, :]) ** 2).sum(-1)  # (n, k)
        clusters = np.nan_to_num(d, nan=np.inf).argmin(axis=1)
        regimes = self.cluster_to_regime[clusters].astype(np.int64).copy()
        # Fail-safe: a corrupt observation (non-finite features or latent) is
        # labelled TOXIC, so the strategy STANDS ASIDE — never a confident wrong
        # regime that would drive a bad trade.
        bad = ~np.isfinite(X).all(axis=1) | ~np.isfinite(np.asarray(z)).all(axis=1)
        if bad.any():
            regimes[bad] = int(Regime.TOXIC)
        return regimes

    def predict_from_raw(self, raw: np.ndarray) -> np.ndarray:
        X, _ = featurize_from_raw(raw)
        return self.predict_features(X)

    def predict(self, df) -> np.ndarray:
        X, _ = featurize(df)
        return self.predict_features(X)

    def names(self, ids: np.ndarray) -> list[str]:
        return [REGIME_NAMES.get(int(i), str(int(i))) for i in ids]
