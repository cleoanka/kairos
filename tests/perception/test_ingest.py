"""Tests for live ingestion (Component B): Bybit parsing, the live order book,
cancel-flow, and the async dispatch core driven by an offline Bybit-format
replay (no network, no exchange account)."""
from __future__ import annotations

import asyncio

import numpy as np

from kairos.perception.ingest.bybit import (
    DepthMsg,
    TradeMsg,
    build_subscribe,
    parse_message,
    synth_bybit_stream,
)
from kairos.perception.ingest.capturer import consume, replay_messages
from kairos.perception.ingest.orderbook import LIVE_REGIME_UNKNOWN, LiveOrderBook
from kairos.perception.schema import (
    FEATURE_DIM,
    RAW_SNAPSHOT_DOUBLES,
    block_slices,
    featurize_from_raw,
    snapshot_to_vector,
)


# --- parser ------------------------------------------------------------------
def test_parse_depth_snapshot_and_trade():
    d = parse_message({"topic": "orderbook.50.BTCUSDT", "type": "snapshot",
                       "data": {"s": "BTCUSDT", "b": [["100.0", "2"]], "a": [["101.0", "3"]]}})
    assert isinstance(d, DepthMsg) and d.is_snapshot
    assert d.bids == [(100.0, 2.0)] and d.asks == [(101.0, 3.0)]

    t = parse_message({"topic": "publicTrade.BTCUSDT",
                       "data": [{"T": 1, "s": "BTCUSDT", "S": "Buy", "v": "1.5", "p": "100.5"}]})
    assert isinstance(t, TradeMsg) and t.trades == [(100.5, 1.5, "Buy")]

    assert parse_message({"topic": "tickers.BTCUSDT", "data": {}}) is None


def test_build_subscribe_is_websocket_op():
    sub = build_subscribe("ETHUSDT", depth=50)
    assert sub["op"] == "subscribe"
    assert "orderbook.50.ETHUSDT" in sub["args"]
    assert "publicTrade.ETHUSDT" in sub["args"]


# --- live order book ---------------------------------------------------------
def test_orderbook_snapshot_emits_valid_schema_snapshot():
    b = LiveOrderBook(tick_size=0.1)
    bids = [(100.0 - i * 0.1, 2.0 + i) for i in range(10)]
    asks = [(100.1 + i * 0.1, 2.0 + i) for i in range(10)]
    b.apply_depth(bids, asks, is_snapshot=True)
    assert b.ready()
    snap = b.snapshot()
    assert snap.regime == LIVE_REGIME_UNKNOWN
    row = snapshot_to_vector(snap)
    assert row.shape == (RAW_SNAPSHOT_DOUBLES,)
    X, _ = featurize_from_raw(row[None, :])
    assert X.shape == (1, FEATURE_DIM) and np.isfinite(X).all()


def test_cancel_flow_records_size_decrease():
    b = LiveOrderBook(tick_size=0.1)
    b.apply_depth([(100.0, 50.0)], [(100.1, 1.0)], is_snapshot=True)
    b.snapshot()  # clear accumulators
    # shrink the bid level from 50 -> 5  => 45 cancelled at the touch
    b.apply_depth([(100.0, 5.0)], [], is_snapshot=False)
    snap = b.snapshot()
    assert snap.bid_cxl.sum() == 45.0


def test_cancel_flow_full_removal_counts():
    b = LiveOrderBook(tick_size=0.1)
    b.apply_depth([(99.9, 30.0), (100.0, 4.0)], [(100.1, 4.0)], is_snapshot=True)
    b.snapshot()
    b.apply_depth([(99.9, 0.0)], [], is_snapshot=False)  # pull the whole level
    snap = b.snapshot()
    assert snap.bid_cxl.sum() == 30.0


# --- async dispatch via offline replay --------------------------------------
def test_consume_replay_produces_featurizable_snapshots():
    msgs = synth_bybit_stream(n_updates=300, seed=3)
    assert any(m.get("type") == "snapshot" for m in msgs)
    assert any(m["topic"].startswith("publicTrade") for m in msgs)

    snaps: list = []
    n = asyncio.run(consume(replay_messages(msgs), snaps.append,
                            tick_size=0.1, max_snapshots=120))
    assert n == len(snaps) == 120
    raw = np.array([snapshot_to_vector(s) for s in snaps], dtype=np.float64)
    X, _ = featurize_from_raw(raw)
    assert X.shape == (120, FEATURE_DIM) and np.isfinite(X).all()
    # spoof bursts must register cancel-flow somewhere in the stream
    sl = block_slices()
    assert (X[:, sl["bid_cxl"]].sum() + X[:, sl["ask_cxl"]].sum()) > 0.0


def test_live_monitor_runs():
    """The live TUI drives from the ingest replay path without error (works with or
    without a trained regime model)."""
    from kairos.perception.tui.dashboard import run_live
    assert run_live(frames=3, n=20, delay=0.0) == 0
