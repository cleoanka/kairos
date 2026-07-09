"""Bybit v5 public-WebSocket message parsing + an offline replay generator.

Constitution Rule 3: market-data ingestion is WebSocket-only — nothing in this
module (or the capturer) uses REST. The parser is a pure function so the whole
ingest path is unit-testable with no network.

Reference message shapes (Bybit v5 public):
  orderbook : {"topic":"orderbook.50.BTCUSDT","type":"snapshot"|"delta",
               "data":{"s":"BTCUSDT","b":[["price","size"],...],"a":[...]}}
  trade     : {"topic":"publicTrade.BTCUSDT",
               "data":[{"T":ts,"s":"BTCUSDT","S":"Buy"|"Sell","v":"qty","p":"price"}]}
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..schema import N_LEVELS


@dataclass(slots=True)
class DepthMsg:
    is_snapshot: bool
    bids: list  # [(price, size), ...]
    asks: list


@dataclass(slots=True)
class TradeMsg:
    trades: list  # [(price, qty, side), ...]


def build_subscribe(symbol: str = "BTCUSDT", depth: int = 50) -> dict:
    return {"op": "subscribe",
            "args": [f"orderbook.{depth}.{symbol}", f"publicTrade.{symbol}"]}


def _num(x):
    """Parse a finite float, or None — exchange feeds are untrusted input."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _levels(rows):
    """Parse a list of [price, size] pairs, skipping any malformed / non-finite /
    negative-size entry (a single bad row must not crash the whole feed)."""
    out = []
    for row in (rows or []):
        try:
            p, s = _num(row[0]), _num(row[1])
        except (TypeError, IndexError, KeyError):
            continue
        if p is not None and s is not None and s >= 0.0:
            out.append((p, s))
    return out


def parse_message(msg: dict):
    """Parse one decoded JSON message into a DepthMsg/TradeMsg, or None. Malformed
    entries are skipped rather than raising — a hostile or buggy feed cannot crash
    the ingestion loop or inject non-finite values downstream."""
    if not isinstance(msg, dict):
        return None
    topic = msg.get("topic", "")
    if topic.startswith("orderbook"):
        data = msg.get("data") or {}
        return DepthMsg(is_snapshot=(msg.get("type") == "snapshot"),
                        bids=_levels(data.get("b", [])), asks=_levels(data.get("a", [])))
    if topic.startswith("publicTrade"):
        trades = []
        for t in (msg.get("data") or []):
            if not isinstance(t, dict):
                continue
            p, v, s = _num(t.get("p")), _num(t.get("v")), t.get("S")
            if p is not None and v is not None and v > 0 and s is not None:
                trades.append((p, v, s))
        return TradeMsg(trades)
    return None


def _depth_delta(symbol, depth, side, price, size):
    book = {"s": symbol, "b": [], "a": []}
    book["b" if side == "b" else "a"].append([f"{price:.2f}", f"{size:.3f}"])
    return {"topic": f"orderbook.{depth}.{symbol}", "type": "delta", "data": book}


def synth_bybit_stream(n_updates: int = 600, seed: int = 0, symbol: str = "BTCUSDT",
                       depth: int = 50, mid0: float = 30_000.0,
                       tick: float = 0.1) -> list[dict]:
    """A plausible Bybit v5 message list: an initial snapshot, then deltas and
    trades — including spoof-like add/cancel bursts — so the live ingest path can
    be exercised end-to-end with no network."""
    rng = np.random.default_rng(seed)

    def px(side, i):
        return round(mid0 + (i + 1) * tick * (1 if side == "a" else -1), 2)

    snap_b = [[f"{px('b', i):.2f}", f"{rng.uniform(1, 5):.3f}"] for i in range(N_LEVELS)]
    snap_a = [[f"{px('a', i):.2f}", f"{rng.uniform(1, 5):.3f}"] for i in range(N_LEVELS)]
    msgs = [{"topic": f"orderbook.{depth}.{symbol}", "type": "snapshot",
             "data": {"s": symbol, "b": snap_b, "a": snap_a}}]

    pending_spoof = None
    for k in range(n_updates):
        if pending_spoof is not None:  # cancel last round's phantom wall
            side, price = pending_spoof
            pending_spoof = None
            msgs.append(_depth_delta(symbol, depth, side, price, 0.0))
        r = rng.random()
        if r < 0.30:  # spoof: large add a few levels deep, cancelled next round
            side = "b" if rng.random() < 0.5 else "a"
            price = px(side, int(rng.integers(2, 6)))
            msgs.append(_depth_delta(symbol, depth, side, price, rng.uniform(50, 150)))
            pending_spoof = (side, price)
        elif r < 0.70:  # ordinary resize at a near level
            side = "b" if rng.random() < 0.5 else "a"
            price = px(side, int(rng.integers(0, N_LEVELS)))
            msgs.append(_depth_delta(symbol, depth, side, price, rng.uniform(0.5, 5)))
        else:  # a trade print
            s = "Buy" if rng.random() < 0.5 else "Sell"
            msgs.append({"topic": f"publicTrade.{symbol}",
                         "data": [{"T": k, "s": symbol, "S": s,
                                   "v": f"{rng.uniform(0.1, 2):.3f}", "p": f"{mid0:.2f}"}]})
    return msgs
