"""Offline regime detection (Component, Milestone 5).

Takes the self-supervised embeddings and clusters them with no labels, then
*scores* the clustering against the held-out ground-truth regimes purely to
prove the embedding is regime-aware. Produces a 2D projection figure and a
machine-readable report. The Phase-1 acceptance criterion lives here: TOXIC
(spoofing) and TREND (real directional flow) must land in distinct clusters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.model_selection import cross_val_score

from ..schema import REGIME_NAMES, Regime, featurize


def _purity(true: np.ndarray, pred: np.ndarray) -> tuple[float, dict]:
    mapping = {}
    correct = 0
    for c in np.unique(pred):
        members = true[pred == c]
        if len(members) == 0:
            continue
        maj = int(np.bincount(members).argmax())
        mapping[int(c)] = REGIME_NAMES.get(maj, str(maj))
        correct += int((members == maj).sum())
    return correct / len(true), mapping


def _confusion(true: np.ndarray, pred: np.ndarray, k: int) -> np.ndarray:
    m = np.zeros((len(Regime), k), dtype=int)
    for t, p in zip(true, pred):
        m[t, p] += 1
    return m


def analyze(latents: Path, data: Path, fig: Path, report: Path, k: int = 3) -> dict:
    df = pd.read_parquet(data)
    regime = df["regime"].to_numpy()
    z = np.load(latents)["z"]

    # Standardise the latent so no single dimension dominates the distance metric.
    zs = (z - z.mean(0)) / (z.std(0) + 1e-6)
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    pred = km.fit_predict(zs)

    ari = float(adjusted_rand_score(regime, pred))
    nmi = float(normalized_mutual_info_score(regime, pred))
    purity, mapping = _purity(regime, pred)
    conf = _confusion(regime, pred, k)

    # Baseline: clustering the RAW features directly. The whole point of the
    # self-supervised embedding is that it makes regimes geometrically
    # clusterable where the raw 63-d book is not.
    Xr, _ = featurize(df)
    Xrs = (Xr - Xr.mean(0)) / (Xr.std(0) + 1e-6)
    ari_raw = float(adjusted_rand_score(
        regime, KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xrs)))

    # Strong headline: is the embedding *linearly* separable by regime? (A
    # supervised probe used only as a measurement — it trains nothing in the
    # pipeline.) ~1.0 means the regimes occupy distinct half-spaces.
    lin_sep = float(cross_val_score(
        LogisticRegression(max_iter=2000), zs, regime, cv=5).mean())

    # Dominant cluster per true regime — the heart of the spec's acceptance test.
    dom = {int(r): int(conf[r].argmax()) for r in range(len(Regime))}
    toxic_vs_trend_separated = dom[int(Regime.TOXIC)] != dom[int(Regime.TREND)]

    _plot(z, regime, pred, fig)

    # Persist the fitted regime model so new/live snapshots can be *labelled*
    # (not just clustered): latent standardisation + KMeans centroids (in the
    # standardised latent space) + the cluster→regime majority map. See
    # lob_core.regime.predict.RegimePredictor.
    np.savez(
        report.parent / "regime_model.npz",
        z_mean=z.mean(0).astype(np.float32),
        z_std=(z.std(0) + 1e-6).astype(np.float32),
        centroids=km.cluster_centers_.astype(np.float32),
        cluster_to_regime=conf.argmax(axis=0).astype(np.int64),
    )

    # Acceptance follows the spec's actual Phase-1 criterion (spoofing vs real
    # trend land in distinct mathematical clusters), backed by two objective
    # checks: the embedding is near-perfectly regime-separable, and it beats raw
    # clustering by a wide margin.
    accept = bool(
        toxic_vs_trend_separated
        and lin_sep >= 0.9
        and ari >= 2.0 * ari_raw
    )
    result = {
        "n": int(len(z)),
        "latent_dim": int(z.shape[1]),
        "linear_separability_5cv": round(lin_sep, 4),
        "adjusted_rand_index": round(ari, 4),
        "adjusted_rand_index_raw_baseline": round(ari_raw, 4),
        "ari_gain_over_raw": round(ari / ari_raw, 2) if ari_raw > 1e-6 else None,
        "normalized_mutual_info": round(nmi, 4),
        "cluster_purity": round(purity, 4),
        "cluster_to_regime": mapping,
        "dominant_cluster_per_regime": {REGIME_NAMES[r]: c for r, c in dom.items()},
        "confusion_true_by_cluster": conf.tolist(),
        "toxic_vs_trend_separated": bool(toxic_vs_trend_separated),
        "acceptance_pass": accept,
        "figure": str(fig),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2))
    return result


def _plot(z: np.ndarray, regime: np.ndarray, pred: np.ndarray, fig: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    proj = PCA(n_components=2, random_state=0).fit_transform(z)
    fig.parent.mkdir(parents=True, exist_ok=True)
    f, ax = plt.subplots(1, 2, figsize=(13, 5.5))

    colors = {0: "#22b8cf", 1: "#37b24d", 2: "#f03e3e"}  # RANGE/TREND/TOXIC
    for r in np.unique(regime):
        m = regime == r
        ax[0].scatter(proj[m, 0], proj[m, 1], s=5, alpha=0.5,
                      c=colors.get(int(r), "#888"), label=REGIME_NAMES[int(r)])
    ax[0].set_title("Latent space — TRUE regime (ground truth, eval only)")
    ax[0].legend(markerscale=2, loc="best")

    for c in np.unique(pred):
        m = pred == c
        ax[1].scatter(proj[m, 0], proj[m, 1], s=5, alpha=0.5, label=f"cluster {c}")
    ax[1].set_title("Latent space — KMeans clusters (label-free)")
    ax[1].legend(markerscale=2, loc="best")

    for a in ax:
        a.set_xlabel("PC1")
        a.set_ylabel("PC2")
    f.suptitle("LOB-Core — self-supervised regime separation (PCA of MLX embedding)")
    f.tight_layout()
    f.savefig(fig, dpi=130)
    plt.close(f)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cluster LOB embeddings into regimes")
    ap.add_argument("--latents", type=Path, default=Path("artifacts/latents.npz"))
    ap.add_argument("--data", type=Path, default=Path("data/synthetic.parquet"))
    ap.add_argument("--fig", type=Path, default=Path("artifacts/regime_clusters.png"))
    ap.add_argument("--report", type=Path, default=Path("artifacts/report.json"))
    args = ap.parse_args(argv)

    r = analyze(args.latents, args.data, args.fig, args.report)
    print(json.dumps(r, indent=2))
    verdict = "PASS ✓" if r["acceptance_pass"] else "FAIL ✗"
    print(f"\nPhase-1 acceptance (spoofing vs trend separated, "
          f"linear-sep>=0.9, ARI>=2x raw): {verdict}")
    print(f"  linear_separability={r['linear_separability_5cv']}  "
          f"ARI={r['adjusted_rand_index']} (raw baseline {r['adjusted_rand_index_raw_baseline']}, "
          f"{r['ari_gain_over_raw']}x gain)")
    print(f"  NMI={r['normalized_mutual_info']}  purity={r['cluster_purity']}  -> {args.fig}")
    return 0 if r["acceptance_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
