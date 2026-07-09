"""Async WebSocket LOB capturer (Component B).

Constitution Rule 3: trade-data ingestion is WebSocket-only — no REST. The
dispatch core (:func:`consume`) is decoupled from the socket via an async message
iterator, so the parser → book → snapshot logic is fully testable offline; the
live path (:func:`run_live_capture`) is a thin ``websockets.connect`` wrapper.

Each depth update emits a :class:`~lob_core.schema.Snapshot` to ``on_snapshot``,
which can push into the C++ zero-copy ring or featurize for live regime inference.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterable

from .bybit import DepthMsg, TradeMsg, build_subscribe, parse_message
from .orderbook import LiveOrderBook


async def consume(messages: AsyncIterator[dict],
                  on_snapshot: Callable[[object], None],
                  tick_size: float = 0.1,
                  max_snapshots: int | None = None) -> int:
    """Drive a LiveOrderBook from an async stream of decoded messages.

    Returns the number of snapshots emitted. Stops after ``max_snapshots`` if set.
    """
    book = LiveOrderBook(tick_size=tick_size)
    n = 0
    async for msg in messages:
        ev = parse_message(msg)
        if isinstance(ev, DepthMsg):
            book.apply_depth(ev.bids, ev.asks, ev.is_snapshot)
            if book.ready():
                on_snapshot(book.snapshot())
                n += 1
                if max_snapshots is not None and n >= max_snapshots:
                    break
        elif isinstance(ev, TradeMsg):
            book.apply_trades(ev.trades)
    return n


async def replay_messages(msgs: Iterable[dict]) -> AsyncIterator[dict]:
    """Wrap a finite list of messages as an async iterator (offline replay)."""
    for m in msgs:
        yield m


async def _ws_messages(ws) -> AsyncIterator[dict]:
    async for raw in ws:
        try:
            yield json.loads(raw)
        except (ValueError, TypeError):
            continue


async def run_live_capture(url: str, symbol: str = "BTCUSDT", depth: int = 50,
                           tick_size: float = 0.1,
                           on_snapshot: Callable[[object], None] | None = None,
                           max_snapshots: int | None = None) -> int:
    """Connect to a Bybit v5 public WebSocket, subscribe, and consume. No REST,
    no orders — pure market-data ingestion."""
    import websockets  # local import: only needed on the live path

    async with websockets.connect(url, ping_interval=20) as ws:
        await ws.send(json.dumps(build_subscribe(symbol, depth)))
        return await consume(_ws_messages(ws), on_snapshot or (lambda s: None),
                             tick_size, max_snapshots)
