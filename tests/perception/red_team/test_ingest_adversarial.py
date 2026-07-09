"""Red-team: a live exchange feed is UNTRUSTED input. Malformed or adversarial
WebSocket messages must not crash the ingestion loop or inject non-finite values
into the feature matrix (which would poison the model / regime inference)."""
from __future__ import annotations

import asyncio

import numpy as np

from kairos.perception.ingest.bybit import parse_message, synth_bybit_stream
from kairos.perception.ingest.capturer import consume, replay_messages
from kairos.perception.ingest.orderbook import LiveOrderBook
from kairos.perception.schema import RAW_SNAPSHOT_DOUBLES, featurize_from_raw, snapshot_to_vector

MALFORMED = [
    {"topic": "orderbook.50.X", "type": "delta", "data": {"b": [["abc", "1"]], "a": []}},   # non-numeric
    {"topic": "orderbook.50.X", "type": "delta", "data": {"b": [["1.0"]], "a": []}},          # not a pair
    {"topic": "orderbook.50.X", "type": "snapshot", "data": None},                            # None data
    {"topic": "orderbook.50.X", "type": "delta"},                                             # missing data
    {"topic": "orderbook.50.X", "type": "delta",
     "data": {"b": [["nan", "5"]], "a": [["inf", "5"]]}},                                     # NaN/inf price
    {"topic": "orderbook.50.X", "type": "delta", "data": {"b": [["100", "-5"]], "a": []}},    # negative size
    {"topic": "orderbook.50.X", "type": "delta", "data": {"b": [["100", "inf"]], "a": []}},   # inf size
    {"topic": "publicTrade.X", "data": [{"p": "1.0"}]},                                       # trade missing v/S
    {"topic": "publicTrade.X", "data": [{"p": "nan", "v": "1", "S": "Buy"}]},                 # NaN trade price
    {"topic": "tickers.X", "data": {}},                                                       # unknown topic
    {}, [], None, "garbage",                                                                  # not even a dict
]


def test_parser_never_crashes_on_malformed():
    for m in MALFORMED:
        parse_message(m)  # must not raise


def test_book_rejects_bad_entries_and_emits_finite_snapshots():
    b = LiveOrderBook()
    for m in MALFORMED:
        ev = parse_message(m)
        if ev is None:
            continue
        if hasattr(ev, "bids"):
            b.apply_depth(ev.bids, ev.asks, ev.is_snapshot)
        else:
            b.apply_trades(ev.trades)
    # the adversarial entries must not have polluted the book
    assert all(np.isfinite(p) for p in list(b.bids) + list(b.asks))
    # a subsequent valid snapshot must be fully finite
    b.apply_depth([(100.0 - i * 0.1, 2.0) for i in range(10)],
                  [(100.1 + i * 0.1, 2.0) for i in range(10)], is_snapshot=True)
    assert b.ready()
    v = snapshot_to_vector(b.snapshot())
    assert v.shape == (RAW_SNAPSHOT_DOUBLES,) and np.isfinite(v).all()
    X, _ = featurize_from_raw(v[None, :])
    assert np.isfinite(X).all()


def test_consume_survives_malformed_stream_and_features_stay_finite():
    good = synth_bybit_stream(n_updates=240, seed=1)
    mixed = []
    for i, g in enumerate(good):
        mixed.append(g)
        if i % 8 == 0:
            mixed.append(MALFORMED[i % len(MALFORMED)])
    snaps: list = []
    n = asyncio.run(consume(replay_messages(mixed), snaps.append, max_snapshots=80))
    assert n == len(snaps) > 0
    raw = np.array([snapshot_to_vector(s) for s in snaps], dtype=np.float64)
    assert np.isfinite(raw).all()
    X, _ = featurize_from_raw(raw)
    assert np.isfinite(X).all()
