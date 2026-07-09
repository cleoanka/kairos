"""Risk gate for the execution layer (Component E).

A hard safety layer in front of any order routing: caps inventory per side and
trips a kill-switch on drawdown. The same gate guards the paper simulator and
(in future) the live execution engine, so risk limits are enforced identically
whether or not real orders are sent.

Constitution Rule 3: this module performs no I/O — no REST, no network. It is a
pure decision gate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RiskGate:
    max_inventory: float = 20.0   # absolute position cap per side
    max_drawdown: float = 1e9     # halt if mark-to-market PnL falls this far below peak
    halted: bool = False
    _peak: float = 0.0
    _peak_set: bool = False

    def update(self, pnl: float) -> bool:
        """Record PnL; trip the kill-switch on excess drawdown. Returns whether
        trading may continue."""
        if not math.isfinite(pnl):
            self.halted = True   # a broken mark-to-market is a halt condition
            return False
        if not self._peak_set or pnl > self._peak:
            self._peak = pnl
            self._peak_set = True
        if self._peak - pnl > self.max_drawdown:
            self.halted = True
        return not self.halted

    def clamp(self, quote, inventory: float):
        """Zero out the side that would push inventory past its cap."""
        if inventory >= self.max_inventory:
            quote.bid_sz = 0.0   # already max long — stop buying
        if inventory <= -self.max_inventory:
            quote.ask_sz = 0.0   # already max short — stop selling
        return quote
