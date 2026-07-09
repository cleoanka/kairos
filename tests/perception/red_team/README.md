# Red Team — adversarial verification of the C++<->Python zero-copy bridge

These tests independently try to **break** `lob_bridge` (the mmap `MAP_SHARED`
SPSC lock-free ring from `src/cpp/lockfree_ring.hpp` + `src/cpp/shm_lob.hpp`,
exposed via `src/bindings/lob_bridge.cpp`). They do not trust the build summary;
they re-derive the ring's real behaviour and assert it.

If `build/lob_bridge*.so` is missing/unimportable, **every test SKIPs** with a
clear message (it never silently passes), so the regression gate stays honest.

## Attacks simulated

### `test_bridge_stress.py`
- **Zero-copy aliasing**: asserts `np.asarray(latest_view()).ctypes.data ==
  base_addr()` and that a live C++ push appears in the already-wrapped numpy
  array (proves the view is the shared mmap pages, not a copy).
- **Buffer overflow**: pushes `capacity + 50` items with no consumer and asserts
  the **documented behaviour = back-pressure**: exactly `capacity` pushes
  succeed, the rest return `False`, `size()` never exceeds `capacity`, and the
  resident items are **not** overwritten (no silent oldest-dropped). This is the
  real behaviour of `push()` in `lockfree_ring.hpp`
  (`if (next - head > capacity_) return false;`).
- **High-rate SPSC integrity + ordering**: a producer thread pushes 200k
  snapshots at well over 10k/s while a consumer thread reads each slot
  `(head & mask)` through the zero-copy view, validates a per-snapshot checksum
  (over all 64 doubles) and strict monotonic sequence ids, then advances `head`
  with `drain(1)`. Back-pressure guarantees the producer can't lap an unread
  slot.
- **Read latency**: measures per-snapshot zero-copy read latency (p50/p99) and
  asserts p99 < 1ms — a hidden whole-buffer copy regression would blow this.
- **Throughput floor**: the C++ `benchmark()` must clear 10k pushes/sec by a
  wide margin.

### `test_bridge_race.py`
- **Torn-read / reorder**: 300k push/pop interleaved across a producer and a
  consumer thread on a deliberately tiny (64-slot) ring so `head_`/`tail_`
  collide constantly. Every snapshot must arrive with an intact checksum and in
  exact sequence order — a weak `acquire`/`release` or a partial slot write
  would fail immediately.
- **size() bound under race**: a watcher thread polls `size()` concurrently with
  a hammering producer; it must always observe `0 <= size <= capacity`.
- **No lost/duplicated snapshots**: end-to-end coverage check that every
  sequence id in `[0, total)` is consumed exactly once, in order.

## SPSC contract

The ring is **single-producer / single-consumer** by design. We honour that
(exactly one producer thread + one consumer thread); we do **not** test
multi-producer, which the implementation documents as undefined.

## Run

```sh
# build the extension first if needed:
scripts/build_cpp.sh

# run the red-team suite:
.venv/bin/python -m pytest tests/red_team -q

# see the latency / throughput prints:
.venv/bin/python -m pytest tests/red_team -q -s
```

Skips (rather than failures) mean the `.so` isn't built. Failures mean a real
bridge defect — read the assertion message; it names the seq id and the
corruption/ordering/overflow class.
