"""System-1 perception: raw book window -> :class:`Percept`.

Two regime backends share one causal feature computation:

* **Heuristic** (always available, no model, no MLX) — a transparent mapping
  from raw microstructure signals (toxicity, order-flow) to RANGE/TREND/TOXIC.
  This is the portable default so the whole loop runs in CI and on any host.
* **Learned** (:class:`LearnedRegime`) — wraps a trained
  :class:`~kairos.perception.regime.predict.RegimePredictor` (self-supervised
  VICReg encoder + label-free KMeans). Used on Apple Silicon (MLX) or anywhere
  a ``.safetensors`` encoder is present (numpy inference). Adds a calibrated
  per-regime confidence via a softmax over centroid distances.

Every signal is a function of the window's own rows only — no future row is ever
read — so a Percept is causal by construction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from kairos.perception.schema import N_LEVELS, Regime, column_names, featurize

from .percept import BEAR, BULL, NEUTRAL, Percept


@dataclass(frozen=True)
class MicrostructureConfig:
    toxic_threshold: float = 0.35      # cancel/phantom ratio above which the book is TOXIC
    trend_threshold: float = 0.35      # |order-flow imbalance| above which flow is a TREND
    direction_threshold: float = 0.08  # |blended signal| dead-zone for NEUTRAL
    intensity_scale: float = 8.0       # trades/step that saturate trade_intensity
    ofi_weight: float = 0.6            # blend weight of flow vs resting depth for direction


def raw_signals(window, cfg: MicrostructureConfig) -> dict:
    """Compute the causal microstructure signals from a window of raw rows.

    ``window`` is a DataFrame whose columns follow ``schema.column_names()``.
    The last row is the current book; trade/cancel flow is aggregated across the
    window. Returns finite, bounded signals (a corrupt book yields a
    stand-aside-friendly, TOXIC-leaning reading rather than raising).
    """
    n = len(window)
    last = window.iloc[-1]
    mid = float(last["mid"])

    best_bid_off = float(last["bid_px_0"])
    best_ask_off = float(last["ask_px_0"])
    spread_ticks = best_bid_off + best_ask_off

    bid_depth = float(sum(last[f"bid_sz_{i}"] for i in range(N_LEVELS)))
    ask_depth = float(sum(last[f"ask_sz_{i}"] for i in range(N_LEVELS)))
    rest = bid_depth + ask_depth
    depth_imbalance = (bid_depth - ask_depth) / rest if rest > 1e-9 else 0.0

    buy = float(window["trade_buy"].sum())
    sell = float(window["trade_sell"].sum())
    flow = buy + sell
    ofi = (buy - sell) / flow if flow > 1e-9 else 0.0

    cxl = float(sum(window[f"bid_cxl_{i}"].sum() + window[f"ask_cxl_{i}"].sum()
                    for i in range(N_LEVELS)))
    avg_cxl = cxl / max(n, 1)
    toxicity = avg_cxl / (avg_cxl + rest + 1e-9)

    trades = float(window["trade_n"].sum())
    trade_intensity = 1.0 - math.exp(-trades / (max(n, 1) * cfg.intensity_scale))

    blend = cfg.ofi_weight * ofi + (1.0 - cfg.ofi_weight) * depth_imbalance
    if not all(map(math.isfinite, (mid, spread_ticks, toxicity, blend))):
        # Corrupt/one-sided book (e.g. NaN mid or NaN depth/size): force a
        # stand-aside reading. A NaN toxicity must fail SAFE to TOXIC — a
        # silent RANGE would drop the safety veto on a broken book.
        return {"mid": mid, "spread_ticks": spread_ticks, "order_flow_imbalance": 0.0,
                "depth_imbalance": 0.0, "toxicity": 1.0, "trade_intensity": 0.0,
                "blend": 0.0, "corrupt": True}

    return {
        "mid": mid, "spread_ticks": max(spread_ticks, 0.0),
        "order_flow_imbalance": float(np.clip(ofi, -1.0, 1.0)),
        "depth_imbalance": float(np.clip(depth_imbalance, -1.0, 1.0)),
        "toxicity": float(np.clip(toxicity, 0.0, 1.0)),
        "trade_intensity": float(np.clip(trade_intensity, 0.0, 1.0)),
        "blend": float(np.clip(blend, -1.0, 1.0)), "corrupt": False,
    }


def _direction(blend: float, cfg: MicrostructureConfig) -> tuple[str, float]:
    if blend > cfg.direction_threshold:
        return BULL, min(1.0, abs(blend))
    if blend < -cfg.direction_threshold:
        return BEAR, min(1.0, abs(blend))
    return NEUTRAL, min(1.0, abs(blend))


def heuristic_regime(sig: dict, cfg: MicrostructureConfig) -> tuple[int, float, str]:
    """Transparent regime read: TOXIC if phantom liquidity dominates, TREND on
    strong one-sided flow, else RANGE. Confidence scales with signal margin."""
    if sig.get("corrupt"):
        return int(Regime.TOXIC), 1.0, "heuristic"
    tox, ofi = sig["toxicity"], abs(sig["order_flow_imbalance"])
    if tox > cfg.toxic_threshold:
        conf = 0.5 + 0.5 * min(1.0, (tox - cfg.toxic_threshold) / (1.0 - cfg.toxic_threshold))
        return int(Regime.TOXIC), conf, "heuristic"
    if ofi > cfg.trend_threshold:
        conf = 0.5 + 0.5 * min(1.0, (ofi - cfg.trend_threshold) / (1.0 - cfg.trend_threshold))
        return int(Regime.TREND), conf, "heuristic"
    tox_ref = max(cfg.toxic_threshold, 1e-9)
    trend_ref = max(cfg.trend_threshold, 1e-9)
    conf = 0.5 + 0.5 * (1.0 - max(tox / tox_ref, ofi / trend_ref))
    return int(Regime.RANGE), max(0.5, min(conf, 1.0)), "heuristic"


class LearnedRegime:
    """Callable regime backend backed by a trained RegimePredictor.

    Returns ``(regime_id, confidence, "learned")``. Confidence is the softmax
    mass (over KMeans centroid distances) assigned to the winning regime — a
    genuine, calibrated read rather than a placeholder. Falls back to TOXIC on a
    non-finite embedding (mirrors the predictor's own fail-safe: never a
    confident wrong regime that could drive a bad trade).
    """

    def __init__(self, predictor, temperature: float = 1.0):
        self.p = predictor
        self.temperature = max(temperature, 1e-6)

    def __call__(self, window, sig: dict) -> tuple[int, float, str]:
        from kairos.perception.models.embedder import embed

        X, _ = featurize(window.iloc[[-1]].loc[:, column_names()])
        if not np.isfinite(X).all():
            return int(Regime.TOXIC), 1.0, "learned"
        z = embed(self.p.model, X, self.p.stats)
        if not np.isfinite(np.asarray(z)).all():
            return int(Regime.TOXIC), 1.0, "learned"
        zs = (z - self.p.z_mean) / self.p.z_std
        d = ((zs[:, None, :] - self.p.centroids[None, :, :]) ** 2).sum(-1)[0]  # (k,)
        logits = -d / self.temperature
        logits -= logits.max()
        w = np.exp(logits)
        w /= w.sum()
        cluster = int(np.argmin(d))
        regime = int(self.p.cluster_to_regime[cluster])
        # Aggregate probability mass over all clusters mapping to this regime.
        mask = np.asarray(self.p.cluster_to_regime).astype(int) == regime
        conf = float(w[mask].sum())
        return regime, conf, "learned"


def percept_from_window(window, symbol: str, ts: float, *,
                        regime_backend=None, cfg: MicrostructureConfig | None = None,
                        as_of: str | None = None) -> Percept:
    """Build a causal :class:`Percept` from a raw book window.

    ``regime_backend`` is either ``None`` (heuristic), or a
    :class:`LearnedRegime` (or any ``callable(window, sig) -> (regime, conf,
    source)``). All directional / flow signals are heuristic and always present;
    only the *regime label* is swapped out by the backend.
    """
    cfg = cfg or MicrostructureConfig()
    sig = raw_signals(window, cfg)
    if regime_backend is None:
        regime, conf, source = heuristic_regime(sig, cfg)
    else:
        regime, conf, source = regime_backend(window, sig)
    direction, strength = _direction(sig["blend"], cfg)
    # A TOXIC book has no trustworthy direction — System-1 stands aside.
    if int(regime) == int(Regime.TOXIC):
        direction, strength = NEUTRAL, 0.0
    return Percept(
        ts=float(ts), symbol=symbol, mid=sig["mid"], spread_ticks=sig["spread_ticks"],
        order_flow_imbalance=sig["order_flow_imbalance"], depth_imbalance=sig["depth_imbalance"],
        toxicity=sig["toxicity"], trade_intensity=sig["trade_intensity"],
        regime=int(regime), regime_confidence=float(conf), regime_source=source,
        direction=direction, direction_strength=float(strength),
        n_observations=int(len(window)), as_of=as_of,
    )
