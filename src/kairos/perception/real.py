"""Real-market pipeline.

Record live Bybit L2 / trade data, train the label-free VICReg embedder on it,
cluster the discovered regimes, and name them by their micro-structure signature.
Real markets carry no ground-truth regime labels — which is exactly why a
label-free method is the right tool. Everything is written under ``artifacts/real/``
so the synthetic reproducibility model (``artifacts/``) is left untouched.

    lob_core record   # capture a live session to artifacts/real/real.parquet
    lob_core real     # record (or reuse) -> train -> cluster -> name -> save model
    lob_core --mode live-dry --real      # live inference with the real-trained model
    lob_core web --real                  # dashboard on the real-trained model
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .ingest.capturer import run_live_capture
from .regime.naming import cluster_signatures, name_clusters_by_signature
from .schema import REGIME_NAMES, column_names, featurize, snapshot_to_row

REAL_DIR = Path("artifacts/real")
DEFAULT_URL = "wss://stream.bybit.com/v5/public/spot"


def real_model_paths():
    return (str(REAL_DIR / "lob_encoder.safetensors"),
            str(REAL_DIR / "latents.npz"),
            str(REAL_DIR / "regime_model.npz"))


def has_real_model() -> bool:
    return all(Path(p).exists() for p in real_model_paths())


def record(url: str, symbol: str, n: int, tick_size: float, out: Path) -> pd.DataFrame:
    snaps: list = []
    asyncio.run(run_live_capture(url, symbol=symbol, tick_size=tick_size,
                                 on_snapshot=snaps.append, max_snapshots=n))
    df = pd.DataFrame([snapshot_to_row(s) for s in snaps], columns=column_names())
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, engine="pyarrow", index=False)
    return df


def build_real_model(df: pd.DataFrame, epochs: int = 120, k: int = 3) -> dict:
    from sklearn.cluster import KMeans

    from .models.embedder import embed, train

    X, _ = featurize(df)
    model, stats, _ = train(X, epochs=epochs)
    z = embed(model, X, stats)

    REAL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(REAL_DIR / "latents.npz", z=z.astype(np.float32),
             mu=stats["mu"], sd=stats["sd"])
    model.save_weights(str(REAL_DIR / "lob_encoder.safetensors"))

    zs = (z - z.mean(0)) / (z.std(0) + 1e-6)
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    pred = km.fit_predict(zs)
    c2r = name_clusters_by_signature(X, pred, k)
    np.savez(REAL_DIR / "regime_model.npz",
             z_mean=z.mean(0).astype(np.float32),
             z_std=(z.std(0) + 1e-6).astype(np.float32),
             centroids=km.cluster_centers_.astype(np.float32),
             cluster_to_regime=np.array([c2r[c] for c in range(k)], dtype=np.int64))

    named = np.array([c2r[c] for c in pred])
    sig = cluster_signatures(X, pred, k)
    dist = {REGIME_NAMES[r]: int((named == r).sum()) for r in sorted(set(named.tolist()))}
    return {"n": int(len(df)), "regime_dist": dist,
            "signatures": {REGIME_NAMES[c2r[c]]: sig[c] for c in range(k)}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Real-market pipeline (record + train on live Bybit)")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--n", type=int, default=2500, help="snapshots to record")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--tick", type=float, default=0.1)
    ap.add_argument("--data", default=None,
                    help="use an existing recorded parquet instead of recording")
    args = ap.parse_args(argv)

    out = REAL_DIR / "real.parquet"
    if args.data and Path(args.data).exists():
        df = pd.read_parquet(args.data)
        print(f"using recorded data: {args.data} ({len(df)} snapshots)")
    elif (args.data is None) and out.exists():
        df = pd.read_parquet(out)
        print(f"reusing {out} ({len(df)} snapshots) — delete it to re-record")
    else:
        print(f"recording {args.n} snapshots from {args.url} ({args.symbol})…")
        try:
            df = record(args.url, args.symbol, args.n, args.tick, out)
        except Exception as e:
            print(f"recording failed (network?): {e!r}")
            return 1
        print(f"recorded {len(df)} snapshots -> {out}")

    print(f"training label-free VICReg on {len(df)} real snapshots…")
    r = build_real_model(df, epochs=args.epochs)
    print(f"\ndiscovered regimes (named by micro-structure signature): {r['regime_dist']}")
    print(json.dumps(r["signatures"], indent=2))
    print(f"\nreal model -> {REAL_DIR}/  ·  use: lob_core --mode live-dry --real  |  lob_core web --real")
    return 0


def record_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Record a live Bybit session to parquet")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--tick", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=REAL_DIR / "real.parquet")
    args = ap.parse_args(argv)
    print(f"recording {args.n} snapshots from {args.url} ({args.symbol})…")
    try:
        df = record(args.url, args.symbol, args.n, args.tick, args.out)
    except Exception as e:
        print(f"recording failed (network?): {e!r}")
        return 1
    print(f"recorded {len(df)} snapshots -> {args.out}")
    return 0
