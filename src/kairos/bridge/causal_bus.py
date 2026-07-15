"""The Causal Perception Bus — the anti-look-ahead heart of Kairos.

A naive integration would let a System-2 agent, reasoning "as of 2024-05-10",
call a data vendor that quietly returns revised fundamentals, forward-adjusted
prices, or "latest" news — leaking the future into a backtest and inflating
every result. TradingAgents, like most agentic backtests, is exposed to this.

Kairos closes the hole *structurally*. Every microstructure reading is a
timestamped :class:`~kairos.bridge.percept.Percept` recorded on an append-only,
monotonically-ordered bus. The **only** way System-2 reads perception is
:meth:`CausalPerceptionBus.as_of`, which performs a ``bisect`` and can reach
**only** percepts whose timestamp is ``<= cutoff``. Two invariants make this a
guarantee rather than a convention:

1. **No future access.** Every percept returned by any query has ``ts <= cutoff``.
2. **Append-independence.** Recording later percepts never changes the answer to
   an earlier ``as_of`` — the past is immutable once observed.

Both invariants are asserted at runtime (cheaply) and exhaustively property-
tested (``tests/bridge/test_causal_bus.py``). If perception can only ever be read
through this bus, look-ahead bias is impossible by construction, not by review.
"""
from __future__ import annotations

import bisect
import math
import re
from datetime import date, datetime, timezone

from .percept import Percept

# A bare calendar day, no time component. Detected explicitly BEFORE trying
# ``datetime.fromisoformat`` — which on Python 3.11+ parses "2024-05-10" as
# *midnight* (start of day), silently turning a close-of-day cutoff into a
# start-of-day one and reintroducing intraday look-ahead.
_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class LookAheadError(AssertionError):
    """Raised if a query would ever surface a percept from the future.

    Reaching this exception means an invariant of the bus was violated — a bug
    in the bus itself, never a mere data gap (an empty causal history simply
    returns ``None``).
    """


class ClockDomainError(LookAheadError):
    """Raised when a query's clock domain disagrees with the bus's clock.

    A date/datetime cutoff resolves to a UNIX epoch (~1.7e9), but a synthetic /
    replay bus is often clocked on a monotonic *step index* (0, 1, 2, …).
    Silently comparing the two would make ``as_of("2024-05-10")`` on an
    index-clocked bus return the *newest* percept — maximal look-ahead. We refuse
    such a query loudly instead, so a mismatched clock can never leak the future.
    """


# A recorded ts at or beyond this is treated as a wall-clock UNIX epoch; below
# it, as a monotonic step index. 1e7 seconds ≈ 116 days past 1970 — far above any
# plausible index count and far below any real market epoch (~1.7e9).
_EPOCH_THRESHOLD = 1e7


def _require_finite(value: float, original: object) -> float:
    """Reject a non-finite (NaN / ±inf) cutoff value.

    A NaN cutoff is uniquely dangerous on a ``bisect``-backed bus: every
    ``x < nan`` comparison is ``False``, so ``bisect_right`` lands at the end of
    the array and ``as_of`` returns the **newest** percept — maximal look-ahead,
    with the strict ``p.ts > cutoff`` guard also bypassed (``x > nan`` is
    ``False``). ``+inf`` returns the newest percept too; ``-inf`` silently
    empties every window. None of these is a legitimate point-in-time query, so
    we fail closed. Raises :class:`LookAheadError` (an ``AssertionError``, *not*
    a ``ValueError``) deliberately: the numeric-string branch of
    :func:`resolve_cutoff` swallows ``ValueError`` to fall through to date
    parsing, and a non-finite value must never be re-interpreted as a date.
    """
    if not math.isfinite(value):
        raise LookAheadError(
            f"non-finite cutoff {original!r} (resolved to {value}) is not a valid "
            "point-in-time query: it would bypass the causal boundary and surface "
            "the newest percept (look-ahead)."
        )
    return value


def resolve_cutoff(query: float | int | str | date | datetime) -> tuple[float, str]:
    """Resolve a query time to ``(value, domain)``.

    ``domain`` is ``"epoch"`` for a wall-clock date/datetime (which fixes the
    value on the UNIX-epoch clock) or ``"numeric"`` for a raw number / numeric
    string (which is clock-agnostic — it means "this exact value on whatever
    clock the bus already uses").

    * ``float``/``int`` or a numeric string (``"2000.0"``) — ``numeric``.
    * ``"YYYY-MM-DD"`` — the *close* of that calendar day (UTC): the whole
      trading day is in-sample. This is the strict cutoff a daily reasoning run
      ("trade on 2024-05-10") maps to. Domain ``epoch``.
    * ISO datetime / ``datetime`` / ``date`` — a UTC epoch (naive → UTC). Domain
      ``epoch``.
    """
    if isinstance(query, bool):  # guard: bool is an int subclass
        raise TypeError("query time cannot be a bool")
    if isinstance(query, (int, float)):
        return _require_finite(float(query), query), "numeric"
    if isinstance(query, datetime):
        dt = query if query.tzinfo else query.replace(tzinfo=timezone.utc)
        return dt.timestamp(), "epoch"
    if isinstance(query, date):
        dt = datetime(query.year, query.month, query.day, 23, 59, 59, 999999, tzinfo=timezone.utc)
        return dt.timestamp(), "epoch"
    if isinstance(query, str):
        s = query.strip()
        try:                       # a stringified number is a numeric-clock value
            return _require_finite(float(s), query), "numeric"
        except ValueError:
            pass
        if _DATE_ONLY.match(s):    # "YYYY-MM-DD" -> close of that day, UTC
            d = date.fromisoformat(s)
            return (datetime(d.year, d.month, d.day, 23, 59, 59, 999999,
                             tzinfo=timezone.utc).timestamp(), "epoch")
        dt = datetime.fromisoformat(s)  # has a time component
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp(), "epoch"
    raise TypeError(f"unsupported query time: {query!r}")


def to_epoch(query: float | int | str | date | datetime) -> float:
    """Back-compat: just the resolved value (see :func:`resolve_cutoff`)."""
    return resolve_cutoff(query)[0]


def infer_clock(ts: float) -> str:
    """Classify a recorded timestamp as an ``"epoch"`` or ``"index"`` clock."""
    return "epoch" if abs(ts) >= _EPOCH_THRESHOLD else "index"


class CausalPerceptionBus:
    """An append-only, strictly-causal store of Percepts for one symbol space."""

    def __init__(self, *, strict: bool = True, clock: str | None = None):
        self._percepts: list[Percept] = []
        self._ts: list[float] = []          # parallel, sorted, for bisect
        self._strict = strict               # runtime-assert the causal invariant
        self._clock = clock                 # "epoch" | "index" | None (infer on first record)

    @property
    def clock(self) -> str | None:
        """The bus's clock domain — ``"epoch"`` (UNIX seconds) or ``"index"``
        (monotonic step). Inferred from the first recorded ts if not set."""
        return self._clock

    def _cutoff(self, query) -> float:
        """Resolve a query to a numeric cutoff, refusing a cross-clock query."""
        value, domain = resolve_cutoff(query)
        if domain == "epoch" and self._clock == "index":
            raise ClockDomainError(
                f"date/datetime query ({query!r} -> epoch {value:.0f}) against an "
                "index-clocked bus. A wall-clock cutoff on step-index percepts would "
                "return the newest percept (look-ahead). Query with the same clock the "
                "percepts use, or build the bus with real epoch timestamps."
            )
        return value

    # --- Ingestion (write side) ---------------------------------------------
    def record(self, percept: Percept) -> None:
        """Append a percept. Timestamps must be non-decreasing — a real feed
        never delivers an older book after a newer one."""
        ts = float(percept.ts)
        if not math.isfinite(ts):
            raise ValueError(
                f"non-finite percept ts={ts}. A NaN/inf timestamp poisons the bus: "
                "it defeats the out-of-order guard and the clock inference, and every "
                "subsequent causal query would silently mis-order."
            )
        if self._ts and ts < self._ts[-1]:
            raise ValueError(
                f"out-of-order percept: ts={ts} precedes last recorded ts={self._ts[-1]}. "
                "The causal bus models a forward-only stream."
            )
        if self._clock is None:
            self._clock = infer_clock(ts)
        self._percepts.append(percept)
        self._ts.append(ts)

    def extend(self, percepts) -> None:
        for p in percepts:
            self.record(p)

    # --- Retrieval (read side) — the causal boundary ------------------------
    def as_of(self, query: float | int | str | date | datetime) -> Percept | None:
        """The most recent percept at or before ``query``. ``None`` if the
        causal history is empty at that cutoff. Never returns the future."""
        cutoff = self._cutoff(query)
        idx = bisect.bisect_right(self._ts, cutoff) - 1
        if idx < 0:
            return None
        p = self._percepts[idx]
        if self._strict and p.ts > cutoff:
            raise LookAheadError(f"as_of returned ts={p.ts} > cutoff={cutoff}")
        return p

    def window_before(self, query, n: int = 32) -> list[Percept]:
        """Up to ``n`` most recent percepts at or before ``query`` (oldest→newest)."""
        if n <= 0:
            return []
        cutoff = self._cutoff(query)
        hi = bisect.bisect_right(self._ts, cutoff)
        lo = max(0, hi - n)
        out = self._percepts[lo:hi]
        if self._strict and out and out[-1].ts > cutoff:
            raise LookAheadError(f"window_before leaked ts={out[-1].ts} > cutoff={cutoff}")
        return out

    def aggregate_before(self, query, horizon: float) -> dict | None:
        """Summarise percepts in ``(cutoff - horizon, cutoff]`` — a causal
        rolling read of the recent regime distribution and mean flow."""
        cutoff = self._cutoff(query)
        hi = bisect.bisect_right(self._ts, cutoff)
        lo = bisect.bisect_right(self._ts, cutoff - horizon)
        window = self._percepts[lo:hi]
        if not window:
            return None
        from collections import Counter
        regimes = Counter(p.regime_name for p in window)
        n = len(window)
        return {
            "n": n,
            "cutoff": cutoff,
            "regime_distribution": dict(regimes),
            "dominant_regime": regimes.most_common(1)[0][0],
            "mean_order_flow_imbalance": sum(p.order_flow_imbalance for p in window) / n,
            "mean_depth_imbalance": sum(p.depth_imbalance for p in window) / n,
            "mean_toxicity": sum(p.toxicity for p in window) / n,
            "toxic_fraction": sum(p.is_toxic for p in window) / n,
        }

    # --- Introspection -------------------------------------------------------
    @property
    def latest(self) -> Percept | None:
        return self._percepts[-1] if self._percepts else None

    @property
    def first_ts(self) -> float | None:
        return self._ts[0] if self._ts else None

    @property
    def last_ts(self) -> float | None:
        return self._ts[-1] if self._ts else None

    def __len__(self) -> int:
        return len(self._percepts)

    def __repr__(self) -> str:
        span = f"[{self.first_ts}, {self.last_ts}]" if self._ts else "[]"
        return f"CausalPerceptionBus(n={len(self)}, ts_span={span}, strict={self._strict})"


def build_causal_bus(df, symbol: str, *, window: int = 64, step: int = 1,
                     regime_backend=None, cfg=None, timestamps=None) -> CausalPerceptionBus:
    """Replay a raw LOB DataFrame into a causal bus.

    Emits one :class:`Percept` every ``step`` rows, each aggregated over a
    trailing ``window`` of rows (so each percept sees only its own past). This
    is how a recorded / synthetic session becomes the strictly-causal perception
    history that System-2 reads during a backtest.

    ``timestamps`` optionally overrides per-percept wall-clock (an iterable of
    ISO strings aligned to the row where each percept is emitted); otherwise the
    percept ``ts`` is taken from the row's ``ts`` column.
    """
    from .microstructure import percept_from_window

    bus = CausalPerceptionBus()
    n = len(df)
    for end in range(1, n + 1, step):
        lo = max(0, end - window)
        win = df.iloc[lo:end]
        ts = float(win.iloc[-1]["ts"])
        as_of = None
        if timestamps is not None and (end - 1) < len(timestamps):
            as_of = timestamps[end - 1]
        bus.record(percept_from_window(win, symbol, ts, regime_backend=regime_backend,
                                       cfg=cfg, as_of=as_of))
    return bus
