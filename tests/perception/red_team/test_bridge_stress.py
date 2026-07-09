"""Red-team stress test for the LOB-Core zero-copy SPSC ring bridge.

ATTACKS SIMULATED
-----------------
1. High-rate hammering: a producer pushes >= 10,000 snapshots/sec while a
   consumer drains via the zero-copy (capacity, 64) memoryview. We assert every
   snapshot the consumer reads is byte-for-byte intact (sequence id + checksum)
   and that ordering is strictly monotonic (SPSC FIFO contract).
2. Buffer overflow: we deliberately push past capacity with NO consumer and
   assert the DOCUMENTED behaviour. Empirically (and per push() in
   lockfree_ring.hpp) this ring chose BACK-PRESSURE: push() returns False when
   full and NEVER overwrites an unread slot. We assert exactly that — no oldest
   silently dropped, no corruption of the items already in the ring.
3. Latency: we measure per-snapshot read latency through the zero-copy view.

If lob_bridge is not importable (build/ has no .so), the whole module SKIPS.
"""
from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np
import pytest

# --- Locate and import the compiled bridge, or skip the whole module ---------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BUILD_DIR = os.path.join(_REPO_ROOT, "build")
if _BUILD_DIR not in sys.path:
    sys.path.insert(0, _BUILD_DIR)

try:
    import lob_bridge as L  # type: ignore
    _IMPORT_ERR = None
except Exception as exc:  # pragma: no cover - exercised only on broken builds
    L = None
    _IMPORT_ERR = exc

_skip_reason = (
    f"lob_bridge not importable (build the .so via scripts/build_cpp.sh first): "
    f"{_IMPORT_ERR!r}"
)
pytestmark = pytest.mark.skipif(L is None, reason=_skip_reason)

N_DOUBLES = L.N_DOUBLES if L is not None else 66  # track the bridge's contract width
SEQ_COL = 0       # we stamp a monotonic sequence id into column 0
CHK_COL = 1       # and an order-independent checksum into column 1


def _make_snapshot(seq: int) -> np.ndarray:
    """Build a 64-double snapshot whose payload is fully determined by `seq`,
    so the consumer can recompute the expected checksum and detect any torn or
    stale bytes."""
    a = np.empty(N_DOUBLES, dtype=np.float64)
    # Deterministic, seq-dependent payload across ALL columns so a partial/torn
    # write (e.g. only some doubles updated) is detectable.
    a[:] = (np.arange(N_DOUBLES, dtype=np.float64) + 1.0) * (seq + 1.0)
    a[SEQ_COL] = float(seq)
    a[CHK_COL] = 0.0
    a[CHK_COL] = float(np.sum(a))  # checksum over the rest (CHK_COL was 0)
    return a


def _verify_snapshot(row: np.ndarray, expected_seq: int) -> None:
    seq = int(row[SEQ_COL])
    assert seq == expected_seq, f"out-of-order: expected seq {expected_seq}, got {seq}"
    stored_chk = row[CHK_COL]
    recomputed = row.copy()
    recomputed[CHK_COL] = 0.0
    assert np.isclose(stored_chk, float(np.sum(recomputed))), (
        f"checksum mismatch on seq {seq}: torn / corrupt snapshot"
    )


@pytest.fixture
def bridge():
    """A fresh creator-side bridge on a unique segment name; detached after."""
    name = f"aura_rt_stress_{os.getpid()}_{int(time.time()*1e6) % 1_000_000}"
    b = L.create(name, 4096)
    try:
        yield b
    finally:
        b.detach()


def test_zero_copy_view_aliases_shared_pages(bridge):
    """The numpy view must literally alias the shared slot base (no copy)."""
    view = np.asarray(bridge.latest_view())
    assert view.dtype == np.float64
    assert view.shape == (bridge.capacity(), N_DOUBLES)
    assert view.ctypes.data == bridge.base_addr(), (
        "latest_view() copied instead of aliasing the shared mmap pages"
    )
    # A C++-side push must appear LIVE in the already-wrapped numpy array.
    bridge.push_snapshot(_make_snapshot(12345))
    assert int(view[0, SEQ_COL]) == 12345, "zero-copy view did not see live push"


def test_overflow_is_backpressure_not_overwrite(bridge):
    """Push past capacity with no consumer. Documented behaviour = back-pressure:
    push() returns False when full and the items already resident are untouched."""
    cap = bridge.capacity()
    view = np.asarray(bridge.latest_view())

    results = [bridge.push_snapshot(_make_snapshot(i)) for i in range(cap + 50)]

    # Exactly `cap` pushes succeed; everything after is rejected (False).
    assert results[:cap] == [True] * cap, "ring accepted fewer than capacity items"
    assert all(r is False for r in results[cap:]), (
        "ring did NOT back-pressure: it accepted a push past capacity "
        "(would imply silent overwrite of unread data)"
    )
    assert bridge.size() == cap, "size() exceeded capacity — overflow corruption"

    # The first `cap` slots must still hold the ORIGINAL ordered items 0..cap-1,
    # proving no oldest-dropped overwrite occurred.
    for i in range(cap):
        _verify_snapshot(view[i].copy(), expected_seq=i)


def test_high_rate_spsc_integrity_and_ordering(bridge):
    """>= 10k snapshots/sec producer thread vs a zero-copy consumer thread.

    The consumer reads each item at slot (head & mask) directly from the shared
    view, validates checksum + monotonic order, then advances head via drain(1).
    Back-pressure on the producer guarantees a slot is never overwritten while
    the consumer still owes a read of it (SPSC safety)."""
    cap = bridge.capacity()
    mask = cap - 1
    view = np.asarray(bridge.latest_view())
    total = 80_000

    produced = {"n": 0}
    errors: list[str] = []
    consume_latencies: list[float] = []

    def producer():
        seq = 0
        while seq < total:
            snap = _make_snapshot(seq)
            if bridge.push_snapshot(snap):
                seq += 1
            # else: ring full -> spin; consumer will free a slot (back-pressure)
        produced["n"] = seq

    def consumer():
        head = 0
        while head < total:
            if bridge.size() == 0:
                continue
            slot = head & mask
            t0 = time.perf_counter()
            row = view[slot].copy()  # snapshot the slot (defensive copy for check)
            t1 = time.perf_counter()
            try:
                _verify_snapshot(row, expected_seq=head)
            except AssertionError as e:
                errors.append(str(e))
                return
            consume_latencies.append(t1 - t0)
            head += 1
            bridge.drain(1)  # advance C++ head so producer may reuse the slot

    t_start = time.perf_counter()
    pt = threading.Thread(target=producer)
    ct = threading.Thread(target=consumer)
    ct.start()
    pt.start()
    pt.join(timeout=60)
    ct.join(timeout=60)
    elapsed = time.perf_counter() - t_start

    assert not pt.is_alive() and not ct.is_alive(), "producer/consumer deadlocked"
    assert not errors, f"data integrity / ordering failure: {errors[:3]}"
    assert produced["n"] == total

    rate = total / elapsed if elapsed > 0 else float("inf")
    assert rate >= 10_000, f"throughput {rate:,.0f}/s below the 10k/s red-team floor"

    lat = np.array(consume_latencies)
    # Sanity: we actually measured most reads.
    assert lat.size > total * 0.5
    p50 = np.percentile(lat, 50) * 1e9
    p99 = np.percentile(lat, 99) * 1e9
    print(
        f"\n[stress] {total:,} snaps, cap={cap}, {rate:,.0f}/s, "
        f"read-latency p50={p50:,.0f}ns p99={p99:,.0f}ns"
    )
    # A zero-copy single-row read must be cheap (well under 1ms even on a noisy
    # CI box). This catches an accidental copy-the-whole-buffer regression.
    assert p99 < 1_000_000, f"per-snapshot read p99 {p99:,.0f}ns implies a hidden copy"


def test_single_process_benchmark_meets_floor(bridge):
    """The C++ self-draining benchmark must clear the 10k/s floor by orders of
    magnitude (documents the hot-path push cost)."""
    stats = bridge.benchmark(500_000)
    assert stats["pushed"] > 0
    assert stats["pushes_per_sec"] >= 10_000, stats
    print(f"\n[bench] {stats['pushes_per_sec']:,.0f} pushes/sec")
