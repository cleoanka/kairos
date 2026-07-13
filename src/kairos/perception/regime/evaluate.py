"""Out-of-sample generalisation check for the regime embedder.

Trains nothing. It freezes the trained encoder and the *training* normalisation
stats, generates a FRESH synthetic stream with a different seed, embeds it, and
clusters — scoring against that stream's held-out ground-truth regimes. This
verifies the embedding *generalises* rather than memorising the training
snapshots: the in-sample ``report.json`` is transductive (same data trained and
evaluated), whereas this is genuinely out-of-sample.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import cross_val_score

from ..models.embedder import embed, load_trained
from ..schema import Regime, featurize
from ..synthetic.generate import generate


def evaluate_oos(seed: int = 99, n_steps: int = 6000,
                 weights: str = "artifacts/lob_encoder.safetensors",
                 latents: str = "artifacts/latents.npz") -> dict:
    model, stats = load_trained(weights, latents)
    df = generate(n_steps=n_steps, seed=seed)          # fresh, unseen stream
    X, _ = featurize(df)
    regime = df["regime"].to_numpy()
    z = embed(model, X, stats)                          # frozen encoder + train stats

    zs = (z - z.mean(0)) / (z.std(0) + 1e-6)
    pred = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(zs)
    ari = float(adjusted_rand_score(regime, pred))
    nmi = float(normalized_mutual_info_score(regime, pred))
    lin = float(cross_val_score(LogisticRegression(max_iter=2000), zs, regime, cv=5).mean())

    conf = np.zeros((len(Regime), 3), dtype=int)
    for t, p in zip(regime, pred, strict=True):
        conf[t, p] += 1
    dom = {int(r): int(conf[r].argmax()) for r in range(len(Regime))}
    purity = float(conf.max(axis=0).sum() / len(regime))
    toxic_vs_trend = dom[int(Regime.TOXIC)] != dom[int(Regime.TREND)]

    return {
        "oos_seed": seed,
        "n": int(len(z)),
        "oos_linear_separability_5cv": round(lin, 4),
        "oos_adjusted_rand_index": round(ari, 4),
        "oos_normalized_mutual_info": round(nmi, 4),
        "oos_cluster_purity": round(purity, 4),
        "oos_toxic_vs_trend_separated": bool(toxic_vs_trend),
        "generalises": bool(toxic_vs_trend and ari >= 0.7),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Out-of-sample regime-embedding evaluation")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--weights", default="artifacts/lob_encoder.safetensors")
    ap.add_argument("--latents", default="artifacts/latents.npz")
    ap.add_argument("--out", type=Path, default=Path("artifacts/report_oos.json"))
    args = ap.parse_args(argv)

    if not Path(args.weights).exists():
        print(f"no trained encoder at {args.weights} — run `lob_core --mode synthetic` first.")
        return 1
    r = evaluate_oos(args.seed, args.steps, args.weights, args.latents)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))
    print(f"\nOut-of-sample generalisation (seed {args.seed}): "
          f"{'PASS ✓' if r['generalises'] else 'FAIL ✗'}  "
          f"ARI={r['oos_adjusted_rand_index']}  linear-sep={r['oos_linear_separability_5cv']}")
    return 0 if r["generalises"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
