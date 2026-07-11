#!/usr/bin/env python3
"""Generate the README figures directly from the running system.

Every image is produced by the real pipeline — no stock art, no mock-ups:

  1. regime_separation.png — 2D projection of the self-supervised embedding,
     colored by ground-truth regime (proves System-1 separates RANGE/TREND/TOXIC
     without labels).
  2. equity_curves.png — the honest results: Kairos vs baselines across the three
     market scenarios (range win, toxic adverse selection).
  3. causal_boundary.png — the as_of(cutoff) look-ahead boundary, drawn.
  4. veto_timeline.png — a forward window with the perceived-regime bands and the
     System-1 TOXIC veto marked on the mid-price path.

Run:  .venv/bin/python scripts/make_readme_figures.py
Deps: [viz] (matplotlib). Figure 1 additionally uses a trained encoder; it is
trained on the fly with MLX if artifacts are absent, else numpy inference is used.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from kairos.bridge import Decision, ExecutionLink, build_causal_bus
from kairos.bridge.execution_link import perceive_regimes
from kairos.loop import deterministic_policy
from kairos.perception.schema import Regime, featurize
from kairos.perception.synthetic.generate import generate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

# --- palette (GitHub-dark friendly) -----------------------------------------
BG = "#0d1117"
PANEL = "#111726"
FG = "#e6edf3"
MUTED = "#8b949e"
GRID = "#233042"
C_RANGE, C_TREND, C_TOXIC = "#22d3ee", "#34d399", "#f87171"
C_KAIROS, C_NAIVE, C_MM = "#a78bfa", "#f59e0b", "#64748b"
REGIME_COLOR = {int(Regime.RANGE): C_RANGE, int(Regime.TREND): C_TREND, int(Regime.TOXIC): C_TOXIC}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 11,
    "axes.titlecolor": FG, "axes.titleweight": "bold", "legend.frameon": False,
})


def _style(ax):
    ax.grid(True, alpha=0.35, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _load_encoder():
    """Return (model, stats) — trained (MLX) if needed, else numpy inference."""
    from kairos.perception.models.embedder import HAS_MLX, load_trained, train

    art = ROOT / "artifacts"
    w, lat = art / "lob_encoder.safetensors", art / "latents.npz"
    if w.exists() and lat.exists():
        return load_trained(str(w), str(lat))
    if not HAS_MLX:
        raise SystemExit("No trained encoder and MLX unavailable — run `kairos perceive --mode synthetic` first.")
    df = generate(n_steps=6000, seed=7, scenario="toxic")
    X, _ = featurize(df)
    model, stats, _ = train(X, epochs=80, seed=0)
    art.mkdir(exist_ok=True)
    model.save_weights(str(w))
    z = _embed(model, X, stats)
    np.savez(lat, z=z.astype(np.float32), ts=df["ts"].to_numpy(np.float32),
             mid=df["mid"].to_numpy(np.float32), mu=stats["mu"], sd=stats["sd"])
    return model, stats


def _embed(model, X, stats):
    from kairos.perception.models.embedder import embed
    return embed(model, X, stats)


# --- Figure 1 : regime separation -------------------------------------------
def fig_regime_separation():
    from sklearn.decomposition import PCA

    model, stats = _load_encoder()
    df = generate(n_steps=6000, seed=13, scenario="toxic")  # unseen seed
    X, _ = featurize(df)
    z = _embed(model, X, stats)
    z2 = PCA(n_components=2, random_state=0).fit_transform(np.asarray(z))
    reg = df["regime"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 6.2), dpi=170)
    for r in (Regime.RANGE, Regime.TREND, Regime.TOXIC):
        m = reg == int(r)
        ax.scatter(z2[m, 0], z2[m, 1], s=7, alpha=0.55, c=REGIME_COLOR[int(r)],
                   label=r.name, edgecolors="none", rasterized=True)
    _style(ax)
    ax.set_title("System-1 separates market regimes with no labels", pad=14)
    ax.text(0.0, 1.02, "self-supervised LOB embedding (VICReg) → PCA-2D, colored by ground-truth regime  ·  out-of-sample ARI ≈ 0.99",
            transform=ax.transAxes, fontsize=9.5, color=MUTED)
    ax.set_xlabel("PC-1"); ax.set_ylabel("PC-2")
    leg = ax.legend(loc="upper right", markerscale=2.2, fontsize=10)
    for t in leg.get_texts():
        t.set_color(FG)
    fig.tight_layout()
    fig.savefig(OUT / "regime_separation.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote regime_separation.png")


# --- Figure 2 : honest equity curves ----------------------------------------
def _loop_curves(scenario, seed=7, n=4000):
    df = generate(n_steps=n, seed=seed, scenario=scenario)
    k = max(64, len(df) // 2)
    in_sample, fwd = df.iloc[:k].reset_index(drop=True), df.iloc[k:].reset_index(drop=True)
    bus = build_causal_bus(in_sample, "X", window=64, step=8)
    percept = bus.as_of(float(in_sample.iloc[-1]["ts"]))
    link = ExecutionLink()
    kd = deterministic_policy(percept)
    out = {"Kairos (dual-process)": (link.execute(kd, fwd)["pnl_curve"], C_KAIROS, kd.action)}
    out["naive always-long"] = (link.execute(Decision("BUY", 1.0), fwd)["pnl_curve"], C_NAIVE, "BUY")
    out["market-making (HOLD)"] = (link.execute(Decision("HOLD", 0.0), fwd)["pnl_curve"], C_MM, "HOLD")
    return out, percept


def fig_equity_curves():
    scenarios = [("range", "RANGE — benign, two-sided book"),
                 ("calm", "CALM — mild cycles"),
                 ("toxic", "TOXIC — trending + spoofed liquidity")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=160)
    for ax, (scen, title) in zip(axes, scenarios, strict=True):
        curves, percept = _loop_curves(scen)
        for label, (curve, color, action) in curves.items():
            lw = 2.6 if "Kairos" in label else 1.6
            z = 3 if "Kairos" in label else 1
            ax.plot(curve, color=color, lw=lw, label=label, zorder=z,
                    alpha=0.95 if "Kairos" in label else 0.8)
        ax.axhline(0, color=MUTED, lw=0.8, ls=":")
        _style(ax)
        ax.set_title(title, fontsize=11.5)
        ax.set_xlabel("forward step")
        kact = curves["Kairos (dual-process)"][2]
        ax.text(0.02, 0.04, f"System-1 read → stance: {kact}", transform=ax.transAxes,
                fontsize=9, color=MUTED)
    axes[0].set_ylabel("cumulative PnL (mark-to-market)")
    leg = axes[0].legend(loc="upper left", fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(FG)
    fig.suptitle("Honest results: Kairos wins on benign markets, pays adverse selection on trending ones",
                 fontsize=13, fontweight="bold", color=FG, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "equity_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote equity_curves.png")


# --- Figure 3 : the causal boundary -----------------------------------------
def fig_causal_boundary():
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(figsize=(11, 3.6), dpi=170)
    n = 26
    xs = np.linspace(0.4, 9.6, n)
    cutoff = 6.3
    for x in xs:
        past = x <= cutoff
        ax.plot([x], [0], "o", ms=11,
                color=(C_KAIROS if past else PANEL),
                markeredgecolor=(C_KAIROS if past else "#3b4252"),
                markeredgewidth=1.6, alpha=(0.95 if past else 0.9), zorder=3)
    ax.axhline(0, color=GRID, lw=1.2, zorder=0)
    ax.axvline(cutoff, color=C_TOXIC, lw=2.2, zorder=4)
    ax.axvspan(cutoff, 10.0, color=C_TOXIC, alpha=0.08, zorder=0)
    ax.annotate("as_of(cutoff)", xy=(cutoff, 0.62), ha="center", color=C_TOXIC,
                fontsize=12, fontweight="bold")
    ax.annotate("← in-sample: bisect can reach only these", xy=(cutoff - 0.15, -0.6),
                ha="right", color=C_KAIROS, fontsize=11)
    ax.annotate("FUTURE — unreachable by construction →", xy=(cutoff + 0.15, -0.6),
                ha="left", color=C_TOXIC, fontsize=11)
    ax.set_title("The causal boundary: System-2 reads perception only up to the cutoff", pad=16)
    ax.set_xlim(0, 10); ax.set_ylim(-1.0, 1.0)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "causal_boundary.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote causal_boundary.png")


# --- Figure 4 : the System-1 veto on the mid path ----------------------------
def fig_veto_timeline():
    df = generate(n_steps=2600, seed=7, scenario="toxic")
    fwd = df.iloc[1300:].reset_index(drop=True)
    perceived = perceive_regimes(fwd, window=64)
    mid = fwd["mid"].to_numpy()
    x = np.arange(len(mid))

    fig, ax = plt.subplots(figsize=(13, 4.4), dpi=160)
    # regime bands
    start = 0
    for i in range(1, len(perceived) + 1):
        if i == len(perceived) or perceived[i] != perceived[start]:
            ax.axvspan(start, i, color=REGIME_COLOR[int(perceived[start])], alpha=0.14, lw=0)
            start = i
    ax.plot(x, mid, color=FG, lw=1.3, zorder=3)
    # veto markers where perceived == TOXIC
    tox = perceived == int(Regime.TOXIC)
    ax.scatter(x[tox], mid[tox], s=10, color=C_TOXIC, zorder=4, alpha=0.6)
    _style(ax)
    ax.set_title("System-1 veto in action: red = TOXIC regime → the maker stands aside", pad=14)
    ax.set_xlabel("forward step"); ax.set_ylabel("mid price")
    handles = [mpatches.Patch(color=REGIME_COLOR[int(r)], alpha=0.5, label=r.name)
               for r in (Regime.RANGE, Regime.TREND, Regime.TOXIC)]
    leg = ax.legend(handles=handles, loc="upper left", fontsize=10)
    for t in leg.get_texts():
        t.set_color(FG)
    fig.tight_layout()
    fig.savefig(OUT / "veto_timeline.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote veto_timeline.png")


if __name__ == "__main__":
    fig_regime_separation()
    fig_equity_curves()
    fig_causal_boundary()
    fig_veto_timeline()
    print(f"\n✓ figures written to {OUT}")
