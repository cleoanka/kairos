"""Build + serve the LOB-Core web dashboard.

Runs the real engine — synthetic *or* live-recorded Bybit data → label-free regime
inference → paper backtest — dumps a compact JSON the static dashboard animates,
and serves it over a local HTTP server. No new heavy dependencies (stdlib
``http.server``). The dashboard is a *monitor* (no orders) — it never touches the
trading path (Constitution Rule 3).
"""
from __future__ import annotations

import contextlib
import functools
import http.server
import json
import shutil
import socketserver
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

from ..execution.risk import RiskGate
from ..execution.simulator import run_backtest
from ..schema import N_LEVELS, Regime, featurize
from ..strategy.avellaneda import AvellanedaStoikovMaker
from ..synthetic.generate import generate

WEB_DIR = Path(__file__).resolve().parent


def _load_metrics(real: bool) -> dict:
    m = {"bridge_read_ns": 750, "bridge_throughput_mps": 45.8}
    art = Path("artifacts/real") if real else Path("artifacts")
    rep = art / "report.json"
    if rep.exists():
        d = json.loads(rep.read_text())
        m["ari"] = d.get("adjusted_rand_index")
        m["linear_sep"] = d.get("linear_separability_5cv")
        m["purity"] = d.get("cluster_purity")
    oos = Path("artifacts") / "report_oos.json"
    if not real and oos.exists():
        m["oos_ari"] = json.loads(oos.read_text()).get("oos_adjusted_rand_index")
    bench = Path("artifacts") / "bench.json"
    if bench.exists():
        b = json.loads(bench.read_text()).get("batches", {})
        key = "256" if "256" in b else (next(iter(b)) if b else None)
        if key:
            m["batched_latency_us"] = round(b[key]["per_snap_us"], 2)
    return m


def build_dashboard_data(n: int = 500, seed: int = 4, real: bool = False) -> dict:
    from ..real import REAL_DIR, has_real_model, real_model_paths

    use_real = real and has_real_model()
    if use_real:
        df = pd.read_parquet(REAL_DIR / "real.parquet")
        pred_paths = real_model_paths()
        lat_path = REAL_DIR / "latents.npz"
        source = "Bybit BTCUSDT · live"
    else:
        df = generate(n_steps=n, seed=seed)
        pred_paths = None
        lat_path = Path("artifacts/latents.npz")
        source = "synthetic"

    X, _ = featurize(df)
    regime_gt = df["regime"].to_numpy()

    predicted = None
    try:
        from ..regime.predict import RegimePredictor
        pr = RegimePredictor.load(*pred_paths) if pred_paths else RegimePredictor.load()
        predicted = pr.predict_features(X)
    except Exception:
        predicted = None

    bt = run_backtest(df, AvellanedaStoikovMaker(), RiskGate())
    pnl, inv = bt["pnl_curve"], bt["inv_curve"]

    # Direction (BULL/BEAR/FLAT) is *orthogonal* to the regime: the embedding is
    # deliberately direction-invariant (up- and down-trend share the TREND regime),
    # so the bias is read separately from the signed mid drift over a short window.
    midarr = df["mid"].to_numpy()
    W = 20
    _drifts = np.array([float(midarr[i] - midarr[max(0, i - W)]) for i in range(len(midarr))])
    _thr = max(1e-6, 0.4 * float(_drifts.std()))   # adaptive: scales with the market's own volatility

    def _dir(i: int) -> int:
        d = _drifts[i]
        return 1 if d > _thr else (-1 if d < -_thr else 0)

    # Pull every column the loop reads to contiguous numpy once, so each snapshot
    # is a cheap row slice instead of a fresh `df.iloc[i]` Series + label lookups.
    def _cols(prefix: str) -> np.ndarray:
        return df[[f"{prefix}_{k}" for k in range(N_LEVELS)]].to_numpy()
    mid = df["mid"].to_numpy()
    bpx, bsz = _cols("bid_px"), _cols("bid_sz")
    apx, asz = _cols("ask_px"), _cols("ask_sz")
    cxl = (df[[f"bid_cxl_{k}" for k in range(N_LEVELS)]].to_numpy()
           + df[[f"ask_cxl_{k}" for k in range(N_LEVELS)]].to_numpy()).sum(axis=1)
    tf = df["trade_buy"].to_numpy() - df["trade_sell"].to_numpy()

    snaps = []
    for i in range(len(df)):
        rp = int(predicted[i]) if predicted is not None else int(regime_gt[i])
        rt = rp if use_real else int(regime_gt[i])   # no ground truth on real markets
        snaps.append({
            "dir": _dir(i),
            "mid": round(float(mid[i]), 2),
            "bpx": [round(float(v), 1) for v in bpx[i]],
            "bsz": [round(float(v), 1) for v in bsz[i]],
            "apx": [round(float(v), 1) for v in apx[i]],
            "asz": [round(float(v), 1) for v in asz[i]],
            "cxl": round(float(cxl[i]), 1),
            "tf": round(float(tf[i]), 1),
            "rt": rt, "rp": rp,
            "pnl": round(float(pnl[i]), 1) if i < len(pnl) else 0.0,
            "inv": round(float(inv[i]), 2) if i < len(inv) else 0.0,
        })

    # Scatter = the full *training* latent space (not the snapshot stream): for
    # synthetic colour by ground truth; for real colour by the model's own labels.
    scatter = None
    try:
        from sklearn.decomposition import PCA
        z = np.load(lat_path)["z"] if lat_path.exists() else None
        if use_real:
            colour = predicted if (z is not None and len(z) == len(df)) else None
        else:
            parq = Path("data/synthetic.parquet")
            colour = (pd.read_parquet(parq, columns=["regime"])["regime"].to_numpy()
                      if parq.exists() else None)
        if z is not None and colour is not None and len(z) == len(colour):
            proj = PCA(n_components=2, random_state=0).fit_transform(z)
            idx = np.linspace(0, len(proj) - 1, min(1500, len(proj))).astype(int)
            scatter = {
                "x": [round(float(v), 3) for v in proj[idx, 0]],
                "y": [round(float(v), 3) for v in proj[idx, 1]],
                "r": [int(colour[i]) for i in idx],
            }
    except Exception:
        scatter = None

    return {
        "regime_names": {str(r.value): r.name for r in Regime},
        "predicted": predicted is not None,
        "meta": {"real": use_real, "source": source},
        "metrics": _load_metrics(use_real),
        "scatter": scatter,
        "snapshots": snaps,
    }


def write_bundle(out_dir: Path, n: int = 500, seed: int = 4, real: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = build_dashboard_data(n=n, seed=seed, real=real)
    (out_dir / "dashboard_data.json").write_text(json.dumps(data))
    shutil.copy(WEB_DIR / "index.html", out_dir / "index.html")
    return out_dir


def serve(port: int = 8000, n: int = 500, seed: int = 4, open_browser: bool = True,
          real: bool = False) -> int:
    out = Path("artifacts/web")
    print(f"building dashboard data ({'real Bybit' if real else 'synthetic'}, {n} snapshots)…")
    write_bundle(out, n=n, seed=seed, real=real)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out))
    handler.log_message = lambda *a, **k: None  # quiet
    try:
        httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    except OSError as e:
        print(f"could not start server on port {port}: {e}\n"
              f"the port may be in use — try `lob_core web --port {port + 1}`.")
        return 1
    with httpd:
        url = f"http://127.0.0.1:{port}/"
        print(f"\n  LOB-Core dashboard > {url}\n  (Ctrl+C to stop)\n")
        if open_browser:
            with contextlib.suppress(Exception):
                webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Serve the LOB-Core web dashboard")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--n", type=int, default=500, help="snapshots to stream")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--real", action="store_true", help="use the real-Bybit-trained model")
    ap.add_argument("--live", action="store_true",
                    help="real-time streaming dashboard (live Bybit, coin-selectable)")
    ap.add_argument("--symbol", default="BTCUSDT", help="initial coin for --live")
    ap.add_argument("--no-open", action="store_true", help="don't open a browser")
    args = ap.parse_args(argv)
    if args.live:
        from .. import stream
        return stream.serve_live(port=args.port, symbol=args.symbol,
                                 real=args.real, open_browser=not args.no_open)
    return serve(port=args.port, n=args.n, seed=args.seed,
                 open_browser=not args.no_open, real=args.real)


if __name__ == "__main__":
    raise SystemExit(main())
