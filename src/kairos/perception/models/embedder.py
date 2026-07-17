"""Self-supervised LOB embedder (Component D) — MLX, Apple Silicon.

Two label-free objectives train the encoder jointly:
  * Masked-LOB modelling — random whole price-*level* columns (and sometimes the
    trade block) are blanked and reconstructed from the surrounding liquidity,
    tying the latent to real book structure.
  * VICReg (negative-free) — a snapshot and a temporal-neighbour / mid-reflected
    view are pulled together (invariance), while variance + covariance terms
    spread and decorrelate the latent. Because there are no negatives, distant
    same-regime snapshots are never pushed apart — the failure mode that capped
    an earlier InfoNCE objective — so regimes form compact, KMeans-separable
    clusters (RANGE / TREND / TOXIC) with no labels.

No price-direction target is ever used (Constitution Rule 4); the only signals
are masked structure, the time index, and side symmetry.

This module is intentionally ``regime``-blind: it consumes only the feature
matrix X. Evaluation against ground-truth regimes happens in
``lob_core.regime.cluster``.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import struct
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

# MLX is the *training* backend and is Apple-Silicon only. Inference has a
# dependency-free numpy fallback (``numpy_backend.NumpyEncoder``), so the whole
# perception package imports and *labels* live books anywhere — only training
# requires MLX. Guard the import so a non-MLX host (CI/Linux/reasoning
# container) can still ``import kairos.perception.*``.
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    HAS_MLX = True
except Exception:  # pragma: no cover - platform dependent
    mx = nn = optim = None  # type: ignore
    HAS_MLX = False

from ..schema import (
    FEATURE_DIM,
    block_slices,
    feature_weights,
    featurize,
    level_column_groups,
    mirror_spec,
)
from .numpy_backend import NumpyEncoder


def _require_mlx(action: str) -> None:
    if not HAS_MLX:
        raise RuntimeError(
            f"{action} requires MLX (Apple-Silicon training backend), which is not "
            "installed. Install with `pip install 'kairos[mlx]'` on Apple Silicon, "
            "or use pre-trained weights for numpy inference (load_trained)."
        )


if HAS_MLX:
    # --- Model ---------------------------------------------------------------
    class LobAutoEncoder(nn.Module):
        def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 128, latent: int = 16):
            super().__init__()
            self.enc1 = nn.Linear(in_dim, hidden)
            self.enc2 = nn.Linear(hidden, hidden)
            self.enc3 = nn.Linear(hidden, latent)
            self.dec1 = nn.Linear(latent, hidden)
            self.dec2 = nn.Linear(hidden, hidden)
            self.dec3 = nn.Linear(hidden, in_dim)

        def encode(self, x):
            h = nn.gelu(self.enc1(x))
            h = nn.gelu(self.enc2(h))
            return self.enc3(h)

        def decode(self, z):
            h = nn.gelu(self.dec1(z))
            h = nn.gelu(self.dec2(h))
            return self.dec3(h)

        def __call__(self, x):
            return self.decode(self.encode(x))
else:  # pragma: no cover - platform dependent
    LobAutoEncoder = None  # type: ignore


# --- Masking -----------------------------------------------------------------
def _build_mask(batch: int, rng: np.random.Generator, p_level: float = 0.35) -> np.ndarray:
    """Mask whole price-level column groups (px+sz+cxl per level), per sample."""
    groups = level_column_groups()
    sl = block_slices()
    mask = np.zeros((batch, FEATURE_DIM), dtype=np.float32)
    for b in range(batch):
        for g in groups:
            if rng.random() < p_level:
                mask[b, g] = 1.0
        if rng.random() < 0.25:  # occasionally hide the trade block too
            mask[b, sl["trade"]] = 1.0
    return mask


def _vic_terms(z):
    """VICReg variance + covariance terms for one embedding batch."""
    zc = z - z.mean(axis=0, keepdims=True)
    std = mx.sqrt(zc.var(axis=0) + 1e-4)
    var = nn.relu(1.0 - std).mean()                 # hinge: keep each dim active
    b, d = z.shape
    cov = (zc.T @ zc) / (b - 1)                      # (D, D)
    cov_off = (cov * cov).sum() - (mx.diagonal(cov) ** 2).sum()
    return var, cov_off / d                          # decorrelate the dims


def _loss_fn(model, x, mask, x_pos, w):
    # (1) Masked-LOB reconstruction — auxiliary: keeps the learned space tied to
    #     real book structure and gives a meaningful reconstruction metric.
    z_in = model.encode(x * (1.0 - mask))
    recon = model.decode(z_in)
    se = w * (recon - x) ** 2
    recon_loss = (se * mask).sum() / (mask.sum() + 1e-6) + 0.1 * se.mean()

    # (2) VICReg (negative-free) — invariance to a temporal-neighbour / mid-reflected
    #     view (regimes are contiguous in time; up/down-trend collapse), plus variance
    #     and covariance terms that spread and decorrelate the latent. No negatives,
    #     so distant same-regime snapshots are never repelled (the InfoNCE failure
    #     mode). Label-free (Rule 4): the only signal is time index + side symmetry.
    za = model.encode(x)
    zb = model.encode(x_pos)
    inv = ((za - zb) ** 2).mean()
    va, ca = _vic_terms(za)
    vb, cb = _vic_terms(zb)
    vicreg = 25.0 * inv + 25.0 * (va + vb) + 1.0 * (ca + cb)

    return vicreg + 0.5 * recon_loss


# --- Train / embed -----------------------------------------------------------
def train(
    X: np.ndarray,
    epochs: int = 120,
    batch: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
):
    _require_mlx("Training the LOB embedder")
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)   # seed MLX too: weight init was non-deterministic, so a
                           # fresh train could land below the ARI target run-to-run
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-6
    Xn = (X - mu) / sd

    w = mx.array(feature_weights())
    midx, msign = mirror_spec()
    window = 5  # temporal-neighbour radius for the VICReg positive view

    model = LobAutoEncoder()
    mx.eval(model.parameters())
    opt = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, _loss_fn)

    n = Xn.shape[0]
    history = []
    for ep in range(epochs):
        order = rng.permutation(n)
        ep_loss = 0.0
        nb = 0
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb_np = Xn[idx]
            xb = mx.array(xb_np)
            # Positive view: a temporal neighbour, half of them side-reflected.
            delta = rng.integers(1, window + 1) * rng.choice([-1, 1], size=len(idx))
            pos_idx = np.clip(idx + delta, 0, n - 1)
            xp_np = Xn[pos_idx].copy()
            flip = rng.random(len(idx)) < 0.5
            xp_np[flip] = xp_np[flip][:, midx] * msign
            xp = mx.array(xp_np)
            mb = mx.array(_build_mask(len(idx), rng))
            loss, grads = loss_and_grad(model, xb, mb, xp, w)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
            ep_loss += float(loss)
            nb += 1
        ep_loss /= max(nb, 1)
        history.append(ep_loss)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:3d}  train_loss(VICReg+recon)={ep_loss:.5f}")

    stats = {"mu": mu.astype(np.float32), "sd": sd.astype(np.float32)}
    return model, stats, history


def embed(model, X: np.ndarray, stats: dict) -> np.ndarray:
    """Encode features to the regime embedding. VICReg's invariance term already
    makes the encoder side-symmetric, so no inference-time symmetrisation is needed.
    """
    Xn = ((X - stats["mu"]) / stats["sd"]).astype(np.float32)
    if isinstance(model, NumpyEncoder):
        return model.encode(Xn)
    z = model.encode(mx.array(Xn))
    mx.eval(z)
    return np.array(z)


def load_trained(weights: str = "artifacts/lob_encoder.safetensors",
                 latents: str = "artifacts/latents.npz"):
    """Load a frozen trained encoder + its training normalisation stats.

    Uses the MLX module when MLX is installed; otherwise loads the same
    ``.safetensors`` weights into the pure-numpy :class:`NumpyEncoder`, so
    inference is identical across platforms.
    """
    npz = np.load(latents)
    stats = {"mu": npz["mu"], "sd": npz["sd"]}
    if HAS_MLX:
        model = LobAutoEncoder()
        model.load_weights(str(weights))
    else:
        model = NumpyEncoder.load(str(weights))
    return model, stats


def save_weights_with_run_id(model, path: str, run_id: str) -> None:
    """Write the encoder ``.safetensors`` with a ``run_id`` stamped in its metadata.

    ``nn.Module.save_weights`` cannot carry metadata, so we flatten the params and
    call ``mx.save_safetensors`` directly. The stamp is the encoder's half of the
    three-file coherence check (:func:`regime.predict._reject_mixed_provenance`):
    latents/regime already carry it, and stamping the weights lets a load reject a
    mixed set even when the crash committed the *new* encoder first (real.py swaps
    weights before latents). Only ever called on the MLX training path.
    """
    from mlx.utils import tree_flatten

    flat = tree_flatten(model.parameters(), destination={})
    mx.save_safetensors(path, flat, metadata={"run_id": run_id})


def read_weights_run_id(weights: str) -> str | None:
    """Return the ``run_id`` stamped in a ``.safetensors`` header, or ``None``.

    Dependency-free (the provenance guard runs on non-MLX hosts too): the format
    is an 8-byte little-endian header length, then a JSON header whose optional
    ``__metadata__`` map holds the stamp. Older weights without it read as
    ``None`` (backward-compatible — nothing to compare).
    """
    with open(weights, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len).decode("utf-8"))
    meta = header.get("__metadata__") or {}
    rid = meta.get("run_id")
    return str(rid) if rid is not None else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train the self-supervised LOB embedder (MLX)")
    ap.add_argument("--data", type=Path, default=Path("data/synthetic.parquet"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--latents", type=Path, default=Path("artifacts/latents.npz"))
    ap.add_argument("--weights", type=Path, default=Path("artifacts/lob_encoder.safetensors"))
    args = ap.parse_args(argv)

    _require_mlx("Training the LOB embedder")
    df = pd.read_parquet(args.data)
    X, _ = featurize(df)
    print(f"training on {X.shape[0]:,} snapshots, dim={X.shape[1]}, device={mx.default_device()}")
    model, stats, history = train(X, epochs=args.epochs)
    z = embed(model, X, stats)

    args.latents.parent.mkdir(parents=True, exist_ok=True)
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    # Latents/stats and encoder weights are ONE coherent model; a crash between
    # the two in-place writes would leave mismatched files that load silently.
    # Stage each to a unique pid+uuid temp and os.replace them into place only
    # after BOTH are written, so a crash never overwrites a good file with a
    # partial one; a shared run_id lets a later load reject a mixed set. Temps
    # are cleaned on any BaseException. See stockstats_utils.py for the pattern.
    run_id = uuid.uuid4().hex
    stem = f"{os.getpid()}.{uuid.uuid4().hex}.tmp"
    # save_weights / np.savez validate the extension, so put the unique stem
    # *before* the real suffix so each temp still ends in .npz / .safetensors.
    l_tmp = f"{args.latents}.{stem}.npz"
    w_tmp = f"{args.weights}.{stem}.safetensors"
    try:
        # Carry ts/mid (NOT regime) so the embedder stays strictly label-free.
        # ts is float64: float32 has a 128s ULP near a modern UNIX epoch, which
        # would quantise sub-2-minute snapshots to an identical timestamp.
        np.savez(
            l_tmp, z=z.astype(np.float32),
            ts=df["ts"].to_numpy(np.float64), mid=df["mid"].to_numpy(np.float64),
            mu=stats["mu"], sd=stats["sd"], final_loss=np.float32(history[-1]),
            run_id=run_id,
        )
        save_weights_with_run_id(model, w_tmp, run_id)
        # Only now that both temps are fully written do we swap them into place.
        os.replace(l_tmp, args.latents)
        os.replace(w_tmp, args.weights)
    except BaseException:
        for tmp in (l_tmp, w_tmp):
            with contextlib.suppress(OSError):
                os.remove(tmp)
        raise
    print(f"saved latents {z.shape} -> {args.latents}; weights -> {args.weights}")
    print(f"final masked-recon loss: {history[-1]:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
