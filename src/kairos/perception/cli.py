"""lob_core — command-line entry point.

Usage::

    lob_core --version
    lob_core --mode synthetic          # full offline PoC: generate -> train -> cluster
    lob_core --mode live-dry [--url …] # paper: live Bybit WS or offline replay + regime inference
    lob_core --mode hft-live           # institutional execution (guarded — refuses real money)

    lob_core web        [--port N]     # browser dashboard (live regime monitor)
    lob_core gen        [--scenario …] # synthetic chaos generator
    lob_core train      [--epochs N]   # self-supervised embedder (MLX)
    lob_core cluster                   # offline regime detection
    lob_core eval-oos   [--seed N]     # out-of-sample generalisation check
    lob_core backtest   [--strategy …] # paper maker backtest
    lob_core optimize-maker            # Avellaneda-Stoikov (gamma,k) grid
    lob_core bench                     # inference hot-path benchmark
    lob_core monitor    [--live]       # rich TUI (replay or --live)
    lob_core soul-check                # run the Constitution enforcer
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from . import __version__

_ROOT = Path(__file__).resolve().parents[3]


def _run_synthetic() -> int:
    from .models import embedder
    from .regime import cluster
    from .synthetic import generate

    print("=" * 64)
    print(" LOB-Core — MODE: synthetic  (offline self-supervised PoC)")
    print("=" * 64)
    print("\n[1/3] Synthetic chaos generator (MarketMaker / Retail / Spoofer)…")
    rc = generate.main([])
    if rc:
        return rc
    print("\n[2/3] Self-supervised LOB embedder (MLX, masked + contrastive)…")
    rc = embedder.main([])
    if rc:
        return rc
    print("\n[3/3] Offline regime detection (label-free clustering)…")
    return cluster.main([])


def _run_live_dry(url: str | None = None, symbol: str = "BTCUSDT",
                  n: int = 200, tick_size: float = 0.1, via_ring: bool = False,
                  real: bool = False) -> int:
    import asyncio
    import os
    import sys
    from pathlib import Path

    import numpy as np

    from .ingest.bybit import synth_bybit_stream
    from .ingest.capturer import consume, replay_messages, run_live_capture
    from .schema import block_slices, column_names, featurize_from_raw, snapshot_to_row

    print("=" * 64)
    print(" LOB-Core — MODE: live-dry  (paper / shadow — NO orders sent)")
    print("=" * 64)

    # Optional: route every snapshot through the C++ zero-copy ring (native live
    # path: WS -> LiveOrderBook -> ring.push -> zero-copy view -> featurize -> MLX).
    ring = None
    if via_ring:
        bdir = str(_ROOT / "build")
        if bdir not in sys.path:
            sys.path.insert(0, bdir)
        try:
            import lob_bridge
            cap = max(256, 1 << (int(n).bit_length() + 1))
            ring = lob_bridge.create(f"aura_live_{os.getpid()}", cap)
            print("routing through C++ zero-copy ring (Bybit → ring → view → MLX).")
        except Exception as e:
            print(f"(C++ ring unavailable — using Python path: {e!r})")
            ring = None

    snaps: list = []
    if ring is not None:
        from .ingest.ring_bridge import make_ring_sink
        sink = make_ring_sink(ring, also=snaps.append)
    else:
        sink = snaps.append

    if url:
        print(f"connecting WebSocket {url}  symbol={symbol}  (Rule 3: WS, no REST)…")
        try:
            asyncio.run(run_live_capture(url, symbol=symbol, tick_size=tick_size,
                                         on_snapshot=sink, max_snapshots=n))
        except Exception as e:  # network/endpoint problems -> fall back, don't crash
            print(f"live WS error: {e!r}\nfalling back to offline replay.")
    if not snaps:
        if not url:
            print("no --url → offline Bybit-format replay (no network, no account, no orders).")
        msgs = synth_bybit_stream(n_updates=n * 3, seed=1, symbol=symbol, tick=tick_size)
        asyncio.run(consume(replay_messages(msgs), sink,
                            tick_size=tick_size, max_snapshots=n))

    if not snaps:
        print("no snapshots produced.")
        return 1

    if ring is not None:
        view = np.asarray(ring.latest_view())
        raw = view[:len(snaps)]
        print(f"  ring zero-copy: base_addr aliased = "
              f"{view.ctypes.data == ring.base_addr()}, view {view.shape}")
    else:
        raw = np.array([[snapshot_to_row(s)[c] for c in column_names()] for s in snaps],
                       dtype=np.float64)
    X, _ = featurize_from_raw(raw)
    print(f"captured {len(snaps)} snapshots → featurized {X.shape} (label-free).")

    # Live regime inference if the fitted regime model is present (frozen encoder
    # + KMeans centroids + cluster->regime map; see regime/predict.py).
    from .schema import REGIME_NAMES
    regimes = None
    if real:
        from .real import has_real_model, real_model_paths
        paths, model_ok, src = real_model_paths(), has_real_model(), "real-Bybit-trained"
    else:
        paths = ("artifacts/lob_encoder.safetensors", "artifacts/latents.npz",
                 "artifacts/regime_model.npz")
        model_ok, src = Path(paths[0]).exists() and Path(paths[2]).exists(), "synthetic-trained"
    if model_ok:
        try:
            from collections import Counter

            from .regime.predict import RegimePredictor
            regimes = RegimePredictor.load(*paths).predict_features(X)
            dist = Counter(int(r) for r in regimes)
            dist_str = "  ".join(f"{REGIME_NAMES[r]}={dist.get(r, 0)}" for r in sorted(REGIME_NAMES))
            print(f"live regime inference ({src}): {dist_str}")
        except Exception as e:
            print(f"(regime inference skipped: {e!r})")
    else:
        nxt = "lob_core real" if real else "lob_core --mode synthetic"
        print(f"(run `{nxt}` first to enable regime inference.)")

    sl = block_slices()
    cxl = X[:, sl["bid_cxl"]].sum(1) + X[:, sl["ask_cxl"]].sum(1)
    tsig = X[:, sl["trade"].start + 0]
    print("last snapshots  |   mid       regime   cancel-flow  trade-signed")
    for i in range(max(0, len(snaps) - 5), len(snaps)):
        rg = REGIME_NAMES.get(int(regimes[i]), "?") if regimes is not None else "—"
        print(f"  #{i:4d}  | {snaps[i].mid:9.2f}  {rg:6s}  {cxl[i]:9.3f}   {tsig[i]:+.3f}")
    return 0


def _run_hft_live() -> int:
    print("MODE hft-live — institutional execution against a real venue.")
    print("\n\033[31mREFUSED:\033[0m live order routing with real funds is intentionally")
    print("not wired in this PoC. It requires, by design and for safety:")
    print("  • explicit operator confirmation flag (--i-understand-real-money),")
    print("  • exchange API credentials supplied via env / secrets (never committed),")
    print("  • the WebSocket execution engine + risk limits (future milestone).")
    print("See NEXT_STEPS.md and docs/ for the execution roadmap.")
    return 1


def _run_soul_check() -> int:
    return subprocess.call([sys.executable, str(_ROOT / "scripts" / "soul_check.py")])


def _run_backtest(data: str = "data/synthetic.parquet", base_size: float = 1.0,
                  max_inventory: float = 20.0, max_drawdown: float = 1e9,
                  strategy: str = "baseline") -> int:
    import json
    from pathlib import Path

    import pandas as pd

    from .execution.risk import RiskGate
    from .execution.simulator import run_backtest

    p = Path(data)
    if not p.exists():
        print(f"no dataset at {data} — run `lob_core --mode synthetic` first.")
        return 1
    if strategy == "avellaneda":
        from .strategy.avellaneda import AvellanedaStoikovMaker
        strat = AvellanedaStoikovMaker(base_size=base_size, max_inventory=max_inventory)
    else:
        from .strategy.maker import MakerStrategy
        strat = MakerStrategy(base_size=base_size, max_inventory=max_inventory)

    df = pd.read_parquet(p)
    print("=" * 64)
    print(f" LOB-Core — paper backtest  strategy={strategy}  (NO real orders)")
    print("=" * 64)
    rep = run_backtest(df, strat,
                       RiskGate(max_inventory=max_inventory, max_drawdown=max_drawdown))
    curves = {"pnl": rep.pop("pnl_curve"), "inv": rep.pop("inv_curve")}
    print(json.dumps(rep, indent=2))

    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/backtest.json").write_text(json.dumps(rep, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        ax[0].plot(curves["pnl"], color="#1c7ed6")
        ax[0].set_ylabel("PnL (mark-to-market)")
        ax[0].set_title("LOB-Core — paper maker backtest")
        ax[1].plot(curves["inv"], color="#e8590c")
        ax[1].axhline(0, color="k", lw=0.5)
        ax[1].set_ylabel("inventory")
        ax[1].set_xlabel("snapshot")
        f.tight_layout()
        f.savefig("artifacts/backtest_pnl.png", dpi=120)
        plt.close(f)
        print("saved artifacts/backtest_pnl.png + artifacts/backtest.json")
    except Exception as e:
        print(f"(figure skipped: {e!r})")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(
        prog="lob_core",
        description="LOB-Core — self-supervised LOB micro-structure engine "
                    "(Project AURA subsystem). Don't predict, understand.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"lob_core {__version__}")
    p.add_argument("--mode", choices=["synthetic", "live-dry", "hft-live"],
                   help="run a full pipeline mode")
    p.add_argument("--url", default=None,
                   help="exchange WebSocket URL for live-dry (omit = offline replay)")
    p.add_argument("--symbol", default="BTCUSDT", help="instrument symbol (live-dry)")
    p.add_argument("--n", type=int, default=200, help="snapshots to capture (live-dry)")
    p.add_argument("--via-ring", action="store_true",
                   help="route live snapshots through the C++ zero-copy ring (live-dry)")
    p.add_argument("--real", action="store_true",
                   help="use the real-Bybit-trained model (live-dry); see `lob_core real`")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("gen", help="run the synthetic chaos generator")
    sub.add_parser("train", help="train the self-supervised embedder")
    sub.add_parser("cluster", help="run offline regime detection")
    sub.add_parser("eval-oos", help="out-of-sample regime-embedding generalisation check")
    sub.add_parser("soul-check", help="run the Constitution enforcer")
    mon = sub.add_parser("monitor", help="rich TUI dashboard (replay, or --live from the ingest path)")
    mon.add_argument("--frames", type=int, default=60)
    mon.add_argument("--delay", type=float, default=0.06)
    mon.add_argument("--live", action="store_true",
                     help="drive from the live ingestion path + regime inference")
    mon.add_argument("--n", type=int, default=200, dest="mon_n",
                     help="live snapshots to capture (--live)")
    bt = sub.add_parser("backtest", help="paper maker backtest over the synthetic stream")
    bt.add_argument("--data", default="data/synthetic.parquet")
    bt.add_argument("--strategy", choices=["baseline", "avellaneda"], default="baseline")
    bt.add_argument("--base-size", type=float, default=1.0)
    bt.add_argument("--max-inventory", type=float, default=20.0)
    bt.add_argument("--max-drawdown", type=float, default=1e9)
    sub.add_parser("optimize-maker", help="grid-search Avellaneda-Stoikov maker params")
    sub.add_parser("bench", help="benchmark the inference hot path (single vs batched)")
    sub.add_parser("web", help="serve the browser dashboard (live regime monitor)")
    sub.add_parser("record", help="record a live Bybit session to parquet")
    sub.add_parser("real", help="record + train the label-free model on live Bybit data")

    args, extra = p.parse_known_args(argv)

    if args.mode == "synthetic":
        return _run_synthetic()
    if args.mode == "live-dry":
        return _run_live_dry(url=args.url, symbol=args.symbol, n=args.n,
                             via_ring=args.via_ring, real=args.real)
    if args.mode == "hft-live":
        return _run_hft_live()

    if args.cmd == "gen":
        from .synthetic import generate
        return generate.main(extra)
    if args.cmd == "train":
        from .models import embedder
        return embedder.main(extra)
    if args.cmd == "cluster":
        from .regime import cluster
        return cluster.main(extra)
    if args.cmd == "eval-oos":
        from .regime import evaluate
        return evaluate.main(extra)
    if args.cmd == "soul-check":
        return _run_soul_check()
    if args.cmd == "monitor":
        from .tui import dashboard
        if args.live:
            return dashboard.run_live(frames=args.frames, n=args.mon_n, delay=args.delay)
        return dashboard.run(frames=args.frames, delay=args.delay)
    if args.cmd == "backtest":
        return _run_backtest(args.data, args.base_size, args.max_inventory,
                             args.max_drawdown, args.strategy)
    if args.cmd == "optimize-maker":
        from .execution import optimize
        return optimize.main(extra)
    if args.cmd == "bench":
        from . import bench
        return bench.main(extra)
    if args.cmd == "web":
        from .web import build
        return build.main(extra)
    if args.cmd == "record":
        from . import real
        return real.record_main(extra)
    if args.cmd == "real":
        from . import real
        return real.main(extra)

    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
