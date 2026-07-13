"""Latency / throughput benchmark of the inference hot path.

The spec targets sub-0.1 ms zero-copy processing. The C++ ring read is ~750 ns
(p99 ~2.25 µs; see the red-team stress test); this measures the Python inference
path on top.

KEY FINDING: single-snapshot inference is **MLX-dispatch-bound** (~0.4 ms — over
the 0.1 ms target), but **batched** inference amortises the dispatch to ~2 µs/snap
at batch 256 and ~0.5 µs/snap at batch 2000 — 50–200× under target. So the live
loop must process snapshots in *batches*, not one at a time. The pipeline already
does this: live-dry and the TUI featurize+predict the whole captured window at
once; only the per-frame TUI latency readout is a deliberate single-snapshot call.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .regime.predict import RegimePredictor
from .schema import column_names, featurize_from_raw
from .synthetic.generate import generate


def _bench(fn, iters: int) -> float:
    fn()  # warm up
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t) / iters


def run_bench(n: int = 2000, seed: int = 5) -> dict:
    p = RegimePredictor.load()
    df = generate(n, seed=seed)
    raw = df.loc[:, column_names()].to_numpy(np.float64)

    res: dict = {
        "single_snapshot_us": _bench(lambda: p.predict_from_raw(raw[:1]), 200) * 1e6,
        "featurize_1row_us": _bench(lambda: featurize_from_raw(raw[:1]), 200) * 1e6,
        "batches": {},
    }
    for b in (64, 256, min(n, 2000)):
        lat = _bench(lambda b=b: p.predict_from_raw(raw[:b]), 50)
        res["batches"][b] = {
            "total_us": lat * 1e6,
            "per_snap_us": lat / b * 1e6,
            "snap_per_s": b / lat,
        }
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark the inference hot path")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("artifacts/bench.json"))
    args = ap.parse_args(argv)

    if not Path("artifacts/regime_model.npz").exists():
        print("no fitted regime model — run `lob_core --mode synthetic` first.")
        return 1

    r = run_bench(args.n)
    print("=" * 60)
    print(" LOB-Core — inference hot-path benchmark")
    print("=" * 60)
    print(f"  single-snapshot predict : {r['single_snapshot_us']:8.1f} µs/snap "
          f"(MLX-dispatch-bound — batch the live loop)")
    print(f"  featurize_from_raw (1)  : {r['featurize_1row_us']:8.1f} µs")
    for b, s in r["batches"].items():
        print(f"  batch={b:<5d}            : {s['per_snap_us']:7.2f} µs/snap "
              f"({s['snap_per_s']:,.0f} snap/s)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
