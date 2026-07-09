"""Synthetic L2 limit-order-book matching engine (Component A core).

A minimal but faithful price-time order book over integer tick prices. It is
deliberately mechanism-only: it knows how to rest, cancel and match orders and
how to photograph itself into a :class:`~lob_core.schema.Snapshot`. The market
*behaviour* (who trades and why) lives in the agents (``generate.py``); this
keeps the matching logic honest and reusable.

Cancellations are bucketed by tick-offset-from-mid at cancel time and emitted as
the dedicated cancel-flow vector — the signal that lets the self-supervised
embedder see through spoofed (phantom) depth.
"""
from __future__ import annotations

import numpy as np

from ..schema import N_LEVELS, Snapshot


class OrderBook:
    """Aggregated price-level book. ``bids``/``asks`` map tick price -> size."""

    def __init__(self, mid_tick: int = 10_000):
        self.bids: dict[int, float] = {}
        self.asks: dict[int, float] = {}
        self._init_mid = mid_tick
        # Per-snapshot accumulators (reset on each snapshot()).
        self._cxl_bid = np.zeros(N_LEVELS, dtype=np.float64)
        self._cxl_ask = np.zeros(N_LEVELS, dtype=np.float64)
        self._trade_buy = 0.0   # aggressive buy volume
        self._trade_sell = 0.0  # aggressive sell volume
        self._trade_n = 0

    # --- queries -------------------------------------------------------------
    def best_bid(self) -> int | None:
        return max(self.bids) if self.bids else None

    def best_ask(self) -> int | None:
        return min(self.asks) if self.asks else None

    def mid_tick(self) -> float:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None and ba is None:
            return float(self._init_mid)
        if bb is None:
            return float(ba)
        if ba is None:
            return float(bb)
        return (bb + ba) / 2.0

    # --- mutations -----------------------------------------------------------
    def add_limit(self, side: str, price: int, size: float) -> None:
        book = self.bids if side == "bid" else self.asks
        book[price] = book.get(price, 0.0) + size

    def cancel(self, side: str, price: int, size: float) -> float:
        """Remove up to ``size`` from a resting level; record the cancel-flow."""
        book = self.bids if side == "bid" else self.asks
        resting = book.get(price, 0.0)
        if resting <= 0.0:
            return 0.0
        removed = min(size, resting)
        left = resting - removed
        if left <= 1e-9:
            book.pop(price, None)
        else:
            book[price] = left
        mid = self.mid_tick()
        off = int(min(N_LEVELS - 1, max(0, round(abs(price - mid)))))
        (self._cxl_bid if side == "bid" else self._cxl_ask)[off] += removed
        return removed

    def market_order(self, side: str, size: float) -> float:
        """Aggress ``size`` against the opposite side; return filled volume.

        ``side`` is the aggressor direction: 'buy' lifts asks, 'sell' hits bids.
        """
        remaining = size
        filled = 0.0
        if side == "buy":
            book, prices = self.asks, sorted(self.asks)
        else:
            book, prices = self.bids, sorted(self.bids, reverse=True)
        for px in prices:
            if remaining <= 1e-9:
                break
            avail = book[px]
            take = min(avail, remaining)
            book[px] = avail - take
            if book[px] <= 1e-9:
                book.pop(px, None)
            remaining -= take
            filled += take
        if filled > 0:
            if side == "buy":
                self._trade_buy += filled
            else:
                self._trade_sell += filled
            self._trade_n += 1
        return filled

    # --- photography ---------------------------------------------------------
    def _side_levels(self, side: str, mid: float) -> tuple[np.ndarray, np.ndarray]:
        if side == "bid":
            prices = sorted(self.bids, reverse=True)[:N_LEVELS]
            book = self.bids
            offs = [mid - p for p in prices]
        else:
            prices = sorted(self.asks)[:N_LEVELS]
            book = self.asks
            offs = [p - mid for p in prices]
        px = np.zeros(N_LEVELS, dtype=np.float64)
        sz = np.zeros(N_LEVELS, dtype=np.float64)
        for i, (p, o) in enumerate(zip(prices, offs)):
            px[i] = o
            sz[i] = book[p]
        # Pad empty deeper levels with a monotone synthetic offset, zero size.
        for i in range(len(prices), N_LEVELS):
            px[i] = (px[i - 1] + 1.0) if i > 0 else float(i + 1)
        return px, sz

    def snapshot(self, ts: float, regime: int) -> Snapshot:
        mid = self.mid_tick()
        bid_px, bid_sz = self._side_levels("bid", mid)
        ask_px, ask_sz = self._side_levels("ask", mid)
        snap = Snapshot(
            ts=ts, mid=mid,
            bid_px=bid_px, bid_sz=bid_sz, ask_px=ask_px, ask_sz=ask_sz,
            bid_cxl=self._cxl_bid.copy(), ask_cxl=self._cxl_ask.copy(),
            trade_buy=self._trade_buy, trade_sell=self._trade_sell,
            trade_n=self._trade_n, regime=regime,
        )
        # Reset per-snapshot accumulators.
        self._cxl_bid[:] = 0.0
        self._cxl_ask[:] = 0.0
        self._trade_buy = self._trade_sell = 0.0
        self._trade_n = 0
        return snap
