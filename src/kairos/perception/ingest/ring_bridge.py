"""Route live ingestion snapshots into the C++ zero-copy ring (Component B↔C).

Closes the loop between live ingestion and the native zero-copy contract so the
live path is byte-identical to the synthetic one:

    Bybit WS → LiveOrderBook → ring.push → zero-copy view → featurize_from_raw → MLX

The ``ring`` is a ``lob_bridge`` handle created by the caller (the bridge is an
optional native build). Constitution Rule 3: this module performs no network/REST.
"""
from __future__ import annotations

from ..schema import snapshot_to_vector


def push_snapshot(ring, snap) -> bool:
    """Push one LiveOrderBook snapshot into the shared-memory ring. Returns False
    if the ring is full (back-pressure)."""
    return ring.push_snapshot(snapshot_to_vector(snap))


def make_ring_sink(ring, also=None):
    """Build an ``on_snapshot`` callback that pushes into ``ring`` (and optionally
    forwards to another sink, e.g. a list for reporting)."""
    def sink(snap):
        push_snapshot(ring, snap)
        if also is not None:
            also(snap)
    return sink
