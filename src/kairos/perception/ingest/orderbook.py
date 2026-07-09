"""Live L2 order book for streaming ingestion (Component B).

Applies exchange depth snapshots/deltas and trade prints to a running book and
photographs it into a :class:`~lob_core.schema.Snapshot` — the SAME contract the
synthetic engine emits, so the self-supervised model and the C++ zero-copy bridge
consume live and synthetic data through one code path.

Cancel-flow: a net *decrease* in resting size at a level between deltas is
attributed to cancellation (the anti-spoofing signal — spoofers add huge size
then pull it). This is a documented simplification: L2 alone cannot always
separate a cancel from a trade-driven decrease; the dedicated trade stream is
tracked independently for the trade-flow features.
"""
from __future__ import annotations

import math

import numpy as np

from ..schema import N_LEVELS, Snapshot

LIVE_REGIME_UNKNOWN = -1  # no ground-truth regime on a live feed (eval label absent)


class LiveOrderBook:
    def __init__(self, tick_size: float = 0.1):
        self.tick = float(tick_size)
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self._cxl_bid = np.zeros(N_LEVELS, dtype=np.float64)
        self._cxl_ask = np.zeros(N_LEVELS, dtype=np.float64)
        self._tbuy = 0.0
        self._tsell = 0.0
        self._tn = 0
        self._seq = 0
        self._got_snapshot = False

    # --- queries -------------------------------------------------------------
    def best_bid(self):
        return max(self.bids) if self.bids else None

    def best_ask(self):
        return min(self.asks) if self.asks else None

    def ready(self) -> bool:
        return self._got_snapshot and bool(self.bids) and bool(self.asks)

    def mid(self) -> float:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return math.nan
        return (bb + ba) / 2.0

    # --- mutations -----------------------------------------------------------
    def apply_depth(self, bids, asks, is_snapshot: bool) -> None:
        if is_snapshot:
            self.bids.clear()
            self.asks.clear()
            self._got_snapshot = True
        self._apply_side(self.bids, self._cxl_bid, bids)
        self._apply_side(self.asks, self._cxl_ask, asks)

    def _apply_side(self, book, cxl, updates) -> None:
        mid = self.mid()
        for price, size in updates:
            if not (math.isfinite(price) and math.isfinite(size)):
                continue  # never let non-finite values enter the book
            prev = book.get(price, 0.0)
            if size <= 0.0:
                if prev > 0.0:
                    self._add_cxl(cxl, price, mid, prev)
                book.pop(price, None)
            else:
                if size < prev:
                    self._add_cxl(cxl, price, mid, prev - size)
                book[price] = size

    def _add_cxl(self, cxl, price, mid, vol) -> None:
        off = 0 if (mid != mid) else int(
            min(N_LEVELS - 1, max(0, round(abs(price - mid) / self.tick))))
        cxl[off] += vol

    def apply_trades(self, trades) -> None:
        for price, qty, side in trades:
            if not (math.isfinite(price) and math.isfinite(qty)) or qty <= 0:
                continue
            if str(side).lower().startswith("b"):
                self._tbuy += qty
            else:
                self._tsell += qty
            self._tn += 1

    # --- photography ---------------------------------------------------------
    def _side_levels(self, book, is_bid: bool, mid: float):
        prices = (sorted(book, reverse=True) if is_bid else sorted(book))[:N_LEVELS]
        px = np.zeros(N_LEVELS, dtype=np.float64)
        sz = np.zeros(N_LEVELS, dtype=np.float64)
        for i, p in enumerate(prices):
            px[i] = (mid - p) / self.tick if is_bid else (p - mid) / self.tick
            sz[i] = book[p]
        for i in range(len(prices), N_LEVELS):
            px[i] = (px[i - 1] + 1.0) if i > 0 else float(i + 1)
        return px, sz

    def snapshot(self, ts: float | None = None,
                 regime: int = LIVE_REGIME_UNKNOWN) -> Snapshot:
        mid = self.mid()
        bpx, bsz = self._side_levels(self.bids, True, mid)
        apx, asz = self._side_levels(self.asks, False, mid)
        snap = Snapshot(
            ts=float(self._seq if ts is None else ts), mid=mid,
            bid_px=bpx, bid_sz=bsz, ask_px=apx, ask_sz=asz,
            bid_cxl=self._cxl_bid.copy(), ask_cxl=self._cxl_ask.copy(),
            trade_buy=self._tbuy, trade_sell=self._tsell,
            trade_n=self._tn, regime=regime,
        )
        self._cxl_bid[:] = 0.0
        self._cxl_ask[:] = 0.0
        self._tbuy = self._tsell = 0.0
        self._tn = 0
        self._seq += 1
        return snap
