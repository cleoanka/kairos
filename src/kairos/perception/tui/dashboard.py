"""Rich TUI replay dashboard (Component: TUI).

Replays the generated LOB stream and shows, live: the L2 depth ladder, the
ground-truth vs label-free predicted regime, the model's reconstruction loss,
the running cancel-flow / trade-flow, and a *measured* per-snapshot MLX
inference latency. In Phase 1 this drives the offline synthetic stream; the
same renderer will later attach to the C++ zero-copy ring for the live feed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import N_LEVELS, REGIME_NAMES, REGIME_STYLE, Regime, featurize


def _load(latents: Path, data: Path, report: Path):
    df = pd.read_parquet(data)
    z = np.load(latents)["z"]
    rep = json.loads(report.read_text()) if report.exists() else {}
    return df, z, rep


def _predict_clusters(z, rep):
    from sklearn.cluster import KMeans
    zs = (z - z.mean(0)) / (z.std(0) + 1e-6)
    pred = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(zs)
    mapping = {int(k): v for k, v in rep.get("cluster_to_regime", {}).items()}
    names = np.array([mapping.get(int(c), f"c{c}") for c in pred])
    return names


def _bar(value: float, vmax: float, width: int = 18) -> str:
    n = 0 if vmax <= 0 else int(round(width * min(1.0, value / vmax)))
    return "█" * n + "·" * (width - n)


def run(frames: int = 60, delay: float = 0.06,
        latents: str | Path = "artifacts/latents.npz",
        data: str | Path = "data/synthetic.parquet",
        report: str | Path = "artifacts/report.json") -> int:
    latents, data, report = Path(latents), Path(data), Path(report)
    if not data.exists() or not latents.exists():
        print("Missing artifacts. Run `lob_core --mode synthetic` first.")
        return 1

    from rich.console import Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table

    df, z, rep = _load(latents, data, report)
    pred_names = _predict_clusters(z, rep)
    final_loss = rep.get("linear_separability_5cv", None)

    # Load the trained encoder to measure real inference latency per snapshot.
    enc = None
    try:
        import mlx.core as mx

        from ..models.embedder import LobAutoEncoder
        w = Path("artifacts/lob_encoder.safetensors")
        if w.exists():
            enc = LobAutoEncoder()
            enc.load_weights(str(w))
            X_all, _ = featurize(df)
            mu, sd = np.load(latents)["mu"], np.load(latents)["sd"]
            Xn_all = ((X_all - mu) / sd).astype(np.float32)
    except Exception:
        enc = None

    n = len(df)
    step = max(1, n // frames)
    cols = df.columns

    def render(i: int):
        row = df.iloc[i]
        mid = float(row["mid"])
        true_r = int(row["regime"])
        pred_r = pred_names[i]

        # Measure a real single-snapshot MLX encode latency.
        lat_us = float("nan")
        if enc is not None:
            t0 = time.perf_counter()
            zz = enc.encode(mx.array(Xn_all[i:i + 1]))
            mx.eval(zz)
            lat_us = (time.perf_counter() - t0) * 1e6

        bid_sz = np.array([row[f"bid_sz_{k}"] for k in range(N_LEVELS)])
        ask_sz = np.array([row[f"ask_sz_{k}"] for k in range(N_LEVELS)])
        bid_px = np.array([row[f"bid_px_{k}"] for k in range(N_LEVELS)])
        ask_px = np.array([row[f"ask_px_{k}"] for k in range(N_LEVELS)])
        vmax = max(bid_sz.max(), ask_sz.max(), 1.0)
        cxl = sum(row[f"bid_cxl_{k}"] + row[f"ask_cxl_{k}"] for k in range(N_LEVELS))
        tflow = row["trade_buy"] - row["trade_sell"]

        book = Table(show_header=True, header_style="bold", expand=True)
        book.add_column("bid size", justify="left")
        book.add_column("Δtick", justify="center", style="dim")
        book.add_column("ask size", justify="right")
        show = min(8, N_LEVELS)
        for k in range(show):
            book.add_row(
                f"[cyan]{_bar(bid_sz[k], vmax)}[/] {bid_sz[k]:6.0f}",
                f"-{bid_px[k]:.0f} | +{ask_px[k]:.0f}",
                f"{ask_sz[k]:6.0f} [magenta]{_bar(ask_sz[k], vmax)}[/]",
            )

        ts = f"[{REGIME_STYLE[Regime(true_r)]}]{REGIME_NAMES[true_r]}[/]"
        pstyle = REGIME_STYLE.get(Regime[pred_r], "white") if pred_r in REGIME_NAMES.values() else "white"
        ps = f"[{pstyle}]{pred_r}[/]"
        hit = "✓" if pred_r == REGIME_NAMES[true_r] else "·"
        stats = Table.grid(padding=(0, 2))
        stats.add_row("frame", f"{i:5d}/{n}", "mid", f"{mid:9.2f}")
        stats.add_row("regime (truth)", ts, "regime (model)", f"{ps} {hit}")
        stats.add_row("cancel-flow", f"{cxl:8.0f}", "trade-flow", f"{tflow:+8.1f}")
        lat_txt = "n/a" if lat_us != lat_us else f"{lat_us:7.1f} µs"
        sep = "n/a" if final_loss is None else f"{final_loss:.3f}"
        stats.add_row("MLX encode latency", lat_txt, "linear-sep", sep)

        return Group(
            Panel(stats, title="[bold]LOB-Core — live regime monitor[/]",
                  subtitle="don't predict, understand", border_style="green"),
            Panel(book, title=f"L2 depth (top {show})", border_style="blue"),
        )

    with Live(render(0), refresh_per_second=max(1, int(1 / max(delay, 1e-3)))) as live:
        for f in range(frames):
            i = min(n - 1, f * step)
            live.update(render(i))
            time.sleep(delay)
    print(f"replayed {frames} frames over {n:,} snapshots.")
    return 0


def run_live(frames: int = 60, n: int = 200, delay: float = 0.06,
             tick_size: float = 0.1) -> int:
    """Live monitor: drive the dashboard from the live ingestion path (Bybit-format
    replay → LiveOrderBook) with real regime inference (RegimePredictor) and a
    *measured* per-snapshot featurize+predict latency. Same renderer family as
    run(), but sourced from the live stream rather than the parquet replay."""
    import asyncio
    from pathlib import Path

    from ..ingest.bybit import synth_bybit_stream
    from ..ingest.capturer import consume, replay_messages
    from ..schema import snapshot_to_vector

    snaps: list = []
    asyncio.run(consume(replay_messages(synth_bybit_stream(n_updates=n * 3, seed=1, tick=tick_size)),
                        snaps.append, tick_size=tick_size, max_snapshots=n))
    if not snaps:
        print("no live snapshots produced.")
        return 1

    predictor, regimes = None, None
    if Path("artifacts/regime_model.npz").exists():
        try:
            from ..regime.predict import RegimePredictor
            predictor = RegimePredictor.load()
            raw = np.array([snapshot_to_vector(s) for s in snaps], dtype=np.float64)
            regimes = predictor.predict_from_raw(raw)
        except Exception:
            predictor = None

    from rich.console import Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table

    step = max(1, len(snaps) // frames)

    def render(i: int):
        s = snaps[i]
        lat_us = float("nan")
        if predictor is not None:
            t0 = time.perf_counter()
            predictor.predict_from_raw(snapshot_to_vector(s)[None, :])
            lat_us = (time.perf_counter() - t0) * 1e6

        vmax = max(float(s.bid_sz.max()), float(s.ask_sz.max()), 1.0)
        cxl = float(s.bid_cxl.sum() + s.ask_cxl.sum())
        tflow = s.trade_buy - s.trade_sell
        rg = "—"
        rstyle = "white"
        if regimes is not None:
            rg = REGIME_NAMES.get(int(regimes[i]), "?")
            rstyle = REGIME_STYLE.get(Regime(int(regimes[i])), "white") if rg in REGIME_NAMES.values() else "white"

        book = Table(show_header=True, header_style="bold", expand=True)
        book.add_column("bid size", justify="left")
        book.add_column("Δtick", justify="center", style="dim")
        book.add_column("ask size", justify="right")
        show = min(8, N_LEVELS)
        for k in range(show):
            book.add_row(
                f"[cyan]{_bar(float(s.bid_sz[k]), vmax)}[/] {s.bid_sz[k]:6.0f}",
                f"-{s.bid_px[k]:.0f} | +{s.ask_px[k]:.0f}",
                f"{s.ask_sz[k]:6.0f} [magenta]{_bar(float(s.ask_sz[k]), vmax)}[/]",
            )

        stats = Table.grid(padding=(0, 2))
        stats.add_row("frame", f"{i:5d}/{len(snaps)}", "mid", f"{s.mid:9.2f}")
        stats.add_row("regime (model)", f"[{rstyle}]{rg}[/]", "source", "LIVE (Bybit replay)")
        stats.add_row("cancel-flow", f"{cxl:8.1f}", "trade-flow", f"{tflow:+8.1f}")
        lat_txt = "n/a" if lat_us != lat_us else f"{lat_us:7.1f} µs"
        stats.add_row("featurize+predict", lat_txt, "orders", "NONE (paper)")

        return Group(
            Panel(stats, title="[bold]LOB-Core — LIVE regime monitor[/]",
                  subtitle="don't predict, understand", border_style="green"),
            Panel(book, title=f"L2 depth (top {show})", border_style="blue"),
        )

    with Live(render(0), refresh_per_second=max(1, int(1 / max(delay, 1e-3)))) as live:
        for f in range(frames):
            live.update(render(min(len(snaps) - 1, f * step)))
            time.sleep(delay)
    print(f"replayed {frames} live frames over {len(snaps)} snapshots "
          f"({'regime inference on' if predictor else 'no model — run --mode synthetic'}).")
    return 0
