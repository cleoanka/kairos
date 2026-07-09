"""Red-team race / torn-read test for the SPSC lock-free ring.

ATTACK SIMULATED
----------------
We try to provoke a torn read or a lost/duplicated/reordered snapshot on the
std::atomic head_/tail_ pair by interleaving rapid push (producer thread) and
pop-advance (consumer thread) right at the empty/full boundaries, where a weak
or mis-ordered acquire/release would surface.

Detection:
* Every snapshot carries a STRICTLY MONOTONIC sequence id (col 0) and a checksum
  over its whole 64-double payload (col 1). The consumer reads the slot through
  the zero-copy view at (head & mask) and asserts:
    - sequence ids are received in exact 0,1,2,... order (no gap, no dupe, no
      reorder)  -> proves head/tail visibility is consistent;
    - the checksum matches  -> proves no torn (partially-written) slot was read.
* We run many short rounds with adversarial buffer occupancy (kept near-empty so
  head and tail are constantly racing at the same cache lines) to maximise the
  window for a memory-ordering bug.

The SPSC contract means exactly one producer + one consumer thread. We honour
that (the ring is documented as undefined for multi-producer), but we stress the
two-thread interleaving as hard as possible. Skips if lob_bridge is absent.
"""
from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BUILD_DIR = os.path.join(_REPO_ROOT, "build")
if _BUILD_DIR not in sys.path:
    sys.path.insert(0, _BUILD_DIR)

try:
    import lob_bridge as L  # type: ignore
    _IMPORT_ERR = None
except Exception as exc:  # pragma: no cover
    L = None
    _IMPORT_ERR = exc

pytestmark = pytest.mark.skipif(
    L is None,
    reason=f"lob_bridge not importable (run scripts/build_cpp.sh): {_IMPORT_ERR!r}",
)

N_DOUBLES = L.N_DOUBLES if L is not None else 66  # track the bridge's contract width
SEQ_COL = 0
CHK_COL = 1


def _make(seq: int) -> np.ndarray:
    a = (np.arange(N_DOUBLES, dtype=np.float64) + 1.0) * ((seq % 9973) + 1.0)
    a[SEQ_COL] = float(seq)
    a[CHK_COL] = 0.0
    a[CHK_COL] = float(np.sum(a))
    return a


def _check(row: np.ndarray, seq: int) -> None:
    assert int(row[SEQ_COL]) == seq, f"seq desync: want {seq} got {int(row[SEQ_COL])}"
    chk = row[CHK_COL]
    tmp = row.copy()
    tmp[CHK_COL] = 0.0
    assert np.isclose(chk, float(np.sum(tmp))), f"TORN READ at seq {seq}"


@pytest.fixture
def bridge():
    name = f"aura_rt_race_{os.getpid()}_{int(time.time()*1e6) % 1_000_000}"
    b = L.create(name, 64)  # small ring -> head/tail collide often (max contention)
    try:
        yield b
    finally:
        b.detach()


def test_no_torn_reads_under_interleaved_push_pop(bridge):
    """Hammer push/pop across two threads; assert monotonic seq + intact checksum
    for every single snapshot. A torn read or reorder fails immediately."""
    cap = bridge.capacity()
    mask = cap - 1
    view = np.asarray(bridge.latest_view())
    total = 60_000

    errors: list[str] = []
    done = threading.Event()

    def producer():
        seq = 0
        while seq < total:
            if bridge.push_snapshot(_make(seq)):
                seq += 1
            # full -> back off to the consumer (keeps occupancy low / racy)

    def consumer():
        head = 0
        while head < total:
            if bridge.size() == 0:
                if done.is_set():
                    break
                continue
            row = view[head & mask].copy()
            try:
                _check(row, head)
            except AssertionError as e:
                errors.append(str(e))
                return
            head += 1
            bridge.drain(1)
        # confirm we actually consumed everything
        if head != total:
            errors.append(f"consumer stalled at {head}/{total}")

    pt = threading.Thread(target=producer)
    ct = threading.Thread(target=consumer)
    ct.start()
    pt.start()
    pt.join(timeout=120)
    done.set()
    ct.join(timeout=120)

    assert not pt.is_alive() and not ct.is_alive(), "deadlock under contention"
    assert not errors, f"RACE DETECTED: {errors[:3]}"


def test_size_never_exceeds_capacity_under_race(bridge):
    """While the producer hammers, size() (tail - head) must never observe a
    value > capacity. A broken full-check or wrap would surface here."""
    cap = bridge.capacity()
    total = 60_000
    violations: list[int] = []
    stop = threading.Event()

    def producer():
        seq = 0
        while seq < total:
            if bridge.push_snapshot(_make(seq)):
                seq += 1
        stop.set()

    def watcher():
        # Poll size() concurrently; it must always be a sane 0..cap.
        while not stop.is_set():
            s = bridge.size()
            if s > cap:
                violations.append(s)
                return

    def consumer():
        head = 0
        while head < total:
            n = bridge.drain(256)
            head += n

    pt = threading.Thread(target=producer)
    wt = threading.Thread(target=watcher)
    ct = threading.Thread(target=consumer)
    ct.start(); wt.start(); pt.start()
    pt.join(timeout=120); ct.join(timeout=120)
    stop.set(); wt.join(timeout=120)

    assert not violations, f"size() exceeded capacity {cap}: saw {violations[:3]}"


def test_no_lost_or_duplicated_snapshots(bridge):
    """End-to-end count + coverage: every seq in [0,total) is consumed exactly
    once, in order. Combines with the checksum test to rule out dup/drop."""
    cap = bridge.capacity()
    mask = cap - 1
    view = np.asarray(bridge.latest_view())
    total = 60_000
    seen_max = {"v": -1}
    fail: list[str] = []

    def producer():
        seq = 0
        while seq < total:
            if bridge.push_snapshot(_make(seq)):
                seq += 1

    def consumer():
        head = 0
        while head < total:
            if bridge.size() == 0:
                continue
            seq = int(view[head & mask][SEQ_COL])
            if seq != head:
                fail.append(f"gap/dupe: expected {head} got {seq}")
                return
            seen_max["v"] = seq
            head += 1
            bridge.drain(1)

    pt = threading.Thread(target=producer)
    ct = threading.Thread(target=consumer)
    ct.start(); pt.start()
    pt.join(timeout=120); ct.join(timeout=120)

    assert not fail, fail[:3]
    assert seen_max["v"] == total - 1, f"did not consume all: last={seen_max['v']}"
