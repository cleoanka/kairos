"""Avellaneda–Stoikov inventory-optimal market maker (Component E, research).

Quotes around an *inventory-adjusted reservation price* with a spread that prices
in inventory risk and order-flow intensity (Avellaneda & Stoikov, 2008), instead
of the baseline maker's crude size-skew:

    reservation r = mid − inventory · γ · σ² · (T−t)
    half-spread δ = ½ γ σ² (T−t) + (1/γ) ln(1 + γ/k)
    quotes       = (r − δ,  r + δ)

σ is estimated online from recent mid moves. Regime overlay (same philosophy as
the baseline): stand aside in TOXIC, and in TREND quote only the inventory-
flattening side. Label-free (Rule 4): keyed on the regime id and inventory.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from ..schema import Regime
from .maker import Quote


class AvellanedaStoikovMaker:
    def __init__(self, base_size: float = 1.0, gamma: float = 0.02,
                 k: float = 0.5, horizon: float = 1.0, vol_window: int = 50,
                 max_inventory: float = 20.0, default_sigma: float = 2.0):
        self.base_size = base_size
        self.gamma = gamma
        self.k = k
        self.horizon = horizon
        self.max_inv = max_inventory
        self.default_sigma = default_sigma
        self._mids: deque[float] = deque(maxlen=vol_window + 1)

    def _sigma(self, mid: float) -> float:
        self._mids.append(mid)
        if len(self._mids) < 5:
            return self.default_sigma
        d = np.diff(np.asarray(self._mids, dtype=float))
        s = float(d.std())
        return s if s > 1e-9 else self.default_sigma

    def decide(self, regime: int, best_bid: float, best_ask: float,
               inventory: float) -> Quote:
        if not (math.isfinite(best_bid) and math.isfinite(best_ask)):
            return Quote(None, 0.0, None, 0.0)   # corrupt book — do not quote
        if not (self.k > 0.0 and self.gamma > 0.0):
            return Quote(None, 0.0, None, 0.0)   # degenerate spread params — stand aside
        mid = 0.5 * (best_bid + best_ask)
        if not math.isfinite(mid):
            return Quote(None, 0.0, None, 0.0)   # finite touch, overflowing mid — stand aside
        sigma = self._sigma(mid)
        if regime == Regime.TOXIC:
            return Quote(None, 0.0, None, 0.0)

        var = sigma * sigma * self.horizon
        if not math.isfinite(var):
            return Quote(None, 0.0, None, 0.0)   # volatility overflowed to inf — stand aside
        r = mid - inventory * self.gamma * var                       # reservation price
        half = 0.5 * self.gamma * var + (1.0 / self.gamma) * math.log(1.0 + self.gamma / self.k)
        # Stay competitive: clamp to at-or-improving the touch (a wider A-S spread
        # would rest behind the book and never fill), and never cross mid — the
        # reservation skew leans the quotes within [mid, touch] to manage inventory.
        ask_px = max(min(r + half, best_ask), mid)
        bid_px = min(max(r - half, best_bid), mid)

        bid_sz = ask_sz = self.base_size
        if regime == Regime.TREND:
            # Reduce-only: never add exposure into a directional book.
            if inventory > 1e-9:
                bid_sz = 0.0          # long  -> sell-only
            elif inventory < -1e-9:
                ask_sz = 0.0          # short -> buy-only
            else:
                bid_sz = ask_sz = 0.0
        return Quote(bid_px if bid_sz > 0 else None, bid_sz,
                     ask_px if ask_sz > 0 else None, ask_sz)
