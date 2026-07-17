"""Passive market-maker decision layer (Component E).

Turns the *detected regime* into resting limit quotes. The thesis: provide
liquidity and earn the spread when the book is balanced (RANGE), defend against
adverse selection when flow turns one-sided (TREND), and stand aside entirely
when displayed liquidity is phantom (TOXIC / spoofing). Inventory is mean-reverted
by skewing quote sizes.

Label-free (Constitution Rule 4): the strategy consumes the regime *id* produced
by the self-supervised clustering — never a price-direction target.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..schema import Regime


@dataclass(slots=True)
class Quote:
    bid_px: float | None
    bid_sz: float
    ask_px: float | None
    ask_sz: float

    @property
    def two_sided(self) -> bool:
        return self.bid_px is not None and self.ask_px is not None


@runtime_checkable
class Maker(Protocol):
    """Structural contract every maker satisfies: regime + touch + inventory ->
    a resting Quote. Lets run_backtest accept any maker (the baseline skew-maker
    below or the Avellaneda-Stoikov sibling) soundly, without nominal inheritance."""

    def decide(self, regime: int, best_bid: float, best_ask: float,
               inventory: float) -> Quote:
        ...


class MakerStrategy:
    def __init__(self, base_size: float = 1.0, tick: float = 1.0,
                 max_inventory: float = 20.0):
        self.base_size = base_size
        self.tick = tick
        self.max_inv = max_inventory

    def decide(self, regime: int, best_bid: float, best_ask: float,
               inventory: float) -> Quote:
        if not (math.isfinite(best_bid) and math.isfinite(best_ask)):
            return Quote(None, 0.0, None, 0.0)  # corrupt book — do not quote
        if regime == Regime.TOXIC:
            return Quote(None, 0.0, None, 0.0)  # phantom depth — do not quote into it

        if regime == Regime.TREND:
            # Reduce-only: never add exposure into a directional book. Quote ONLY
            # the side that flattens current inventory (label-free — keyed on the
            # sign of inventory, not on any price-direction prediction).
            size = self.base_size * 0.5
            bid_sz = size if inventory < -1e-9 else 0.0   # short -> buy back
            ask_sz = size if inventory > 1e-9 else 0.0    # long  -> sell down
            return Quote(best_bid if bid_sz > 0 else None, bid_sz,
                         best_ask if ask_sz > 0 else None, ask_sz)

        # RANGE — tight two-sided liquidity, inventory-skewed to mean-revert.
        mid = 0.5 * (best_bid + best_ask)
        spread = best_ask - best_bid
        improve = self.tick if spread > 2 * self.tick else 0.0
        skew = max(-1.0, min(1.0, inventory / self.max_inv))  # long -> less bid/more ask
        bid_sz = max(0.0, self.base_size * (1.0 - skew))
        ask_sz = max(0.0, self.base_size * (1.0 + skew))
        # Clamp to mid so quotes never cross (a crossed/negative-offset input book
        # would otherwise yield bid_px > ask_px) — mirrors the A-S maker's defense.
        return Quote(min(best_bid + improve, mid), bid_sz,
                     max(best_ask - improve, mid), ask_sz)
