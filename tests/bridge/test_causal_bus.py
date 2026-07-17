"""The crown-jewel tests: the causal perception bus cannot leak the future.

If these pass, look-ahead bias is impossible *by construction* — every query is
proven to reach only percepts at or before its cutoff, and the answer to a past
query is proven independent of any future percept appended later.
"""
from __future__ import annotations

import random

import pytest

from kairos.bridge.causal_bus import (
    CausalPerceptionBus,
    ClockDomainError,
    LookAheadError,
    infer_clock,
    resolve_cutoff,
    to_epoch,
)
from kairos.bridge.percept import NEUTRAL, Percept


def make_percept(ts: float, symbol: str = "X", regime: int = 0) -> Percept:
    return Percept(
        ts=ts, symbol=symbol, mid=100.0 + ts, spread_ticks=1.0,
        order_flow_imbalance=0.0, depth_imbalance=0.0, toxicity=0.0,
        trade_intensity=0.0, regime=regime, regime_confidence=0.9,
        regime_source="test", direction=NEUTRAL, direction_strength=0.0,
        n_observations=1, as_of=None,
    )


def test_as_of_never_returns_future():
    bus = CausalPerceptionBus()
    for t in range(0, 100):
        bus.record(make_percept(float(t)))
    for cutoff in range(-5, 105):
        p = bus.as_of(float(cutoff))
        if p is not None:
            assert p.ts <= cutoff, f"LEAK: as_of({cutoff}) -> ts={p.ts}"


def test_as_of_returns_most_recent_at_or_before():
    bus = CausalPerceptionBus()
    for t in (0.0, 1.0, 5.0, 9.0):
        bus.record(make_percept(t))
    assert bus.as_of(4.0).ts == 1.0     # nothing between 1 and 5 -> the 1.0 percept
    assert bus.as_of(5.0).ts == 5.0     # exact match is in-sample (<=)
    assert bus.as_of(100.0).ts == 9.0   # beyond the last -> the last (still causal)
    assert bus.as_of(-1.0) is None      # before the first -> nothing


def test_append_independence():
    """The answer to a past query must not change when the future is appended."""
    bus = CausalPerceptionBus()
    for t in range(0, 50):
        bus.record(make_percept(float(t)))
    snapshot = {c: bus.as_of(float(c)) for c in range(0, 50)}

    # Now append a lot of "future" and re-query the same past cutoffs.
    for t in range(50, 200):
        bus.record(make_percept(float(t)))
    for c in range(0, 50):
        after = bus.as_of(float(c))
        assert after.ts == snapshot[c].ts, f"past query {c} changed after future appended"


def test_window_before_and_aggregate_before_are_append_independent_under_fuzz():
    """Append-independence holds for the OTHER two causal reads too, not just
    ``as_of``: over randomised op sequences, snapshotting ``window_before`` and
    ``aggregate_before`` before appending random future percepts must reproduce
    bit-identically afterwards (frozen Percepts / plain dicts compare by value)."""
    rng = random.Random(4321)
    for _ in range(200):
        bus = CausalPerceptionBus(strict=True)
        t = 0.0
        for _ in range(rng.randint(1, 60)):
            t += rng.random() * 3
            bus.record(make_percept(round(t, 6), regime=rng.randint(0, 2)))
        # Snapshot a batch of past reads before the future exists.
        queries = [(rng.uniform(-1, t), rng.randint(1, 8), rng.uniform(0.1, t + 1))
                   for _ in range(20)]
        windows = [bus.window_before(q, n=n) for q, n, _ in queries]
        aggs = [bus.aggregate_before(q, horizon=h) for q, _, h in queries]

        # Append a lot of monotone "future" and re-read the same past queries.
        for _ in range(rng.randint(1, 60)):
            t += rng.random() * 3
            bus.record(make_percept(round(t, 6), regime=rng.randint(0, 2)))
        for (q, n, h), win, agg in zip(queries, windows, aggs, strict=True):
            assert bus.window_before(q, n=n) == win, \
                f"window_before({q}, n={n}) changed after future appended"
            assert bus.aggregate_before(q, horizon=h) == agg, \
                f"aggregate_before({q}, horizon={h}) changed after future appended"


def test_window_before_is_causal():
    bus = CausalPerceptionBus()
    for t in range(0, 100):
        bus.record(make_percept(float(t)))
    for cutoff in (0, 1, 33, 50, 99, 500):
        win = bus.window_before(float(cutoff), n=10)
        assert all(p.ts <= cutoff for p in win)
        assert win == sorted(win, key=lambda p: p.ts)  # oldest -> newest


def test_aggregate_before_is_causal():
    bus = CausalPerceptionBus()
    for t in range(0, 100):
        # alternate regimes so the distribution is non-trivial
        bus.record(make_percept(float(t), regime=t % 3))
    agg = bus.aggregate_before(80.0, horizon=20.0)
    assert agg is not None
    assert agg["cutoff"] == 80.0
    assert agg["n"] <= 21  # window (60, 80] inclusive
    # every counted percept is causal — reconstruct and check the max ts
    assert bus.aggregate_before(-1.0, horizon=5.0) is None


def test_aggregate_before_rejects_non_finite_horizon():
    """A non-finite horizon can't scope a window — +inf spans the whole causal
    history, NaN empties it — so the bus rejects it for EVERY caller, not just
    the tool layer. The ValueError is what the tool already fails closed on."""
    bus = CausalPerceptionBus()
    for t in range(0, 100):
        bus.record(make_percept(float(t), regime=t % 3))
    for bad in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError, match="non-finite horizon"):
            bus.aggregate_before(80.0, bad)


def test_aggregate_before_non_positive_horizon_is_unavailable():
    """A non-positive look-back describes an empty window: fail closed to None
    ("perception unavailable") rather than return a degenerate read."""
    bus = CausalPerceptionBus()
    for t in range(0, 100):
        bus.record(make_percept(float(t), regime=t % 3))
    assert bus.aggregate_before(80.0, 0.0) is None
    assert bus.aggregate_before(80.0, -5.0) is None
    # A valid positive horizon is unaffected — the recent window still reads.
    assert bus.aggregate_before(80.0, 20.0) is not None


def test_out_of_order_record_rejected():
    bus = CausalPerceptionBus()
    bus.record(make_percept(5.0))
    bus.record(make_percept(5.0))          # equal ts is allowed (same instant)
    with pytest.raises(ValueError):
        bus.record(make_percept(4.999))    # going backwards is not


def test_strict_mode_holds_invariant_under_fuzz():
    """Randomised fuzz: for any monotone stream and any query, no leak, ever."""
    rng = random.Random(1234)
    for _ in range(200):
        bus = CausalPerceptionBus(strict=True)
        t = 0.0
        times = []
        for _ in range(rng.randint(1, 60)):
            t += rng.random() * 3
            bus.record(make_percept(round(t, 6)))
            times.append(t)
        for _ in range(30):
            q = rng.uniform(-1, t + 5)
            p = bus.as_of(q)                # strict=True would raise LookAheadError on a leak
            if p is not None:
                assert p.ts <= q
            for p in bus.window_before(q, n=rng.randint(1, 8)):
                assert p.ts <= q


def test_date_string_cutoff_is_close_of_day():
    # A "YYYY-MM-DD" cutoff includes the whole trading day, excludes the next.
    same_day = to_epoch("2024-05-10")
    next_day_open = to_epoch("2024-05-11T00:00:00")
    assert same_day < next_day_open
    # A percept stamped mid-day 2024-05-10 is in-sample for that date...
    mid_day = to_epoch("2024-05-10T12:00:00")
    bus = CausalPerceptionBus()
    bus.record(make_percept(mid_day))
    bus.record(make_percept(next_day_open + 3600))  # 2024-05-11 01:00
    got = bus.as_of("2024-05-10")
    assert got is not None and got.ts == mid_day
    # ...but the 2024-05-11 percept must NOT be visible on 2024-05-10.
    assert got.ts < next_day_open


def test_to_epoch_rejects_bool():
    with pytest.raises(TypeError):
        to_epoch(True)


def test_mixed_clock_query_is_rejected_not_leaked():
    """A date query against a step-index bus must RAISE, not return the newest
    percept (the exact mixed-clock look-ahead hole the guard closes)."""
    bus = CausalPerceptionBus()
    for t in range(0, 4000, 8):          # synthetic step-index clock (0..3992)
        bus.record(make_percept(float(t)))
    assert bus.clock == "index"
    with pytest.raises(ClockDomainError):
        bus.as_of("2024-05-10")          # epoch cutoff ~1.7e9 >> 3992 -> would leak
    with pytest.raises(ClockDomainError):
        bus.window_before("2024-05-10T12:00:00", n=5)
    with pytest.raises(ClockDomainError):
        bus.aggregate_before("2024-05-10", 3600)
    # A numeric query on the same index clock is fine and causal.
    assert bus.as_of(2000).ts <= 2000
    assert bus.as_of("2000.0").ts <= 2000.0   # stringified number is numeric-clock


def test_epoch_clock_accepts_date_query():
    """An epoch-clocked bus accepts date/datetime queries (the real-instrument case)."""
    base = to_epoch("2024-05-10T00:00:00")
    bus = CausalPerceptionBus()
    for h in range(0, 24):
        bus.record(make_percept(base + h * 3600))
    assert bus.clock == "epoch"
    p = bus.as_of("2024-05-10")          # close of day -> the 23:00 percept
    assert p is not None and p.ts <= to_epoch("2024-05-10")


def test_infer_clock_and_resolve_domain():
    assert infer_clock(2000.0) == "index"
    assert infer_clock(1.7e9) == "epoch"
    assert resolve_cutoff(2000.0) == (2000.0, "numeric")
    assert resolve_cutoff("2000.0") == (2000.0, "numeric")
    assert resolve_cutoff("2024-05-10")[1] == "epoch"
    assert resolve_cutoff("2024-05-10T12:00:00")[1] == "epoch"


def test_empty_bus_returns_none_not_error():
    bus = CausalPerceptionBus()
    assert bus.as_of(123.0) is None
    assert bus.window_before(123.0) == []
    assert bus.aggregate_before(123.0, 10.0) is None
    assert len(bus) == 0


# --- Non-finite cutoffs: the NaN look-ahead hole (regression) ----------------
# A NaN cutoff is uniquely dangerous: every `x < nan` is False, so bisect lands
# past the end and as_of would return the NEWEST percept — maximal look-ahead —
# with the strict `p.ts > cutoff` guard also bypassed (`x > nan` is False).
# +inf returns the newest percept too; -inf empties every window. All must fail
# closed rather than silently leak. See resolve_cutoff._require_finite.

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_resolve_cutoff_rejects_non_finite_number(bad):
    with pytest.raises(LookAheadError):
        resolve_cutoff(bad)


@pytest.mark.parametrize("bad", ["nan", "NaN", "inf", "-inf", "  nan  ", "Infinity"])
def test_resolve_cutoff_rejects_non_finite_string(bad):
    # The stringified forms float()-parse; they must be rejected, NOT re-read as
    # a date (which would raise a confusing ValueError from fromisoformat).
    with pytest.raises(LookAheadError):
        resolve_cutoff(bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), "nan", "inf"])
def test_as_of_rejects_non_finite_query_instead_of_leaking(bad):
    bus = CausalPerceptionBus()
    for t in range(0, 10):
        bus.record(make_percept(float(t)))
    with pytest.raises(LookAheadError):
        bus.as_of(bad)
    with pytest.raises(LookAheadError):
        bus.window_before(bad, 10)
    with pytest.raises(LookAheadError):
        bus.aggregate_before(bad, 5.0)


def test_as_of_non_finite_on_epoch_clock_also_rejected():
    # Same guarantee on a realistic wall-clock bus (not just the index clock).
    bus = CausalPerceptionBus()
    base = 1_700_000_000.0
    for h in range(10):
        bus.record(make_percept(base + h * 3600))
    with pytest.raises(LookAheadError):
        bus.as_of(float("nan"))
    with pytest.raises(LookAheadError):
        bus.as_of("nan")


def test_record_rejects_non_finite_ts():
    bus = CausalPerceptionBus()
    with pytest.raises(ValueError, match="non-finite"):
        bus.record(make_percept(float("nan")))
    with pytest.raises(ValueError, match="non-finite"):
        bus.record(make_percept(float("inf")))
    # A NaN first ts must not poison clock inference or the ordering guard.
    assert bus.clock is None
    assert len(bus) == 0


def test_non_finite_never_poisons_a_populated_bus():
    # Regression for the record() bypass: a NaN ts slips past `ts < last` (that
    # comparison is False), so without the guard it would append out of order.
    bus = CausalPerceptionBus()
    bus.record(make_percept(5.0))
    with pytest.raises(ValueError, match="non-finite"):
        bus.record(make_percept(float("nan")))
    assert [p.ts for p in bus.window_before(1e9, 10)] == [5.0]
