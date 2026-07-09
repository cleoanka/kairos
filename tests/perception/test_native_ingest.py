"""Live-ingestion ↔ C++ zero-copy ring integration (Component B↔C).

Replays a Bybit-format stream into a LiveOrderBook, pushes every snapshot into
the shared-memory ring, then reads the snapshots back through the zero-copy view
and asserts they byte-match the direct snapshot vectors and featurize identically
— proving the live path is the same native zero-copy contract as the synthetic
one. Skips cleanly if the C++ bridge is not built.
"""
from __future__ import annotations

import asyncio
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "build"), os.path.join(_ROOT, "src", "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import lob_bridge as L  # type: ignore
    _IMPORT_ERR = None
except Exception as exc:  # pragma: no cover
    L = None
    _IMPORT_ERR = exc

from kairos.perception.ingest.bybit import synth_bybit_stream  # noqa: E402
from kairos.perception.ingest.capturer import consume, replay_messages  # noqa: E402
from kairos.perception.ingest.ring_bridge import make_ring_sink  # noqa: E402
from kairos.perception.schema import featurize_from_raw, snapshot_to_vector  # noqa: E402

pytestmark = pytest.mark.skipif(
    L is None, reason=f"lob_bridge not built (run scripts/build_cpp.sh): {_IMPORT_ERR!r}")


def test_live_snapshots_round_trip_through_zero_copy_ring():
    n = 100
    ring = L.create(f"aura_native_ingest_{os.getpid()}", 256)
    try:
        msgs = synth_bybit_stream(n_updates=n * 4, seed=9)
        snaps: list = []
        sink = make_ring_sink(ring, also=snaps.append)
        got = asyncio.run(consume(replay_messages(msgs), sink, tick_size=0.1, max_snapshots=n))
        assert got == len(snaps) == n

        view = np.asarray(ring.latest_view())
        assert view.ctypes.data == ring.base_addr()           # zero-copy alias
        assert view.shape[1] == L.N_DOUBLES == 66

        direct = np.array([snapshot_to_vector(s) for s in snaps], dtype=np.float64)
        np.testing.assert_array_equal(view[:n], direct)        # ring == source, no corruption

        X_ring, _ = featurize_from_raw(view[:n])
        X_direct, _ = featurize_from_raw(direct)
        np.testing.assert_array_equal(X_ring, X_direct)        # featurize parity
    finally:
        ring.detach()
