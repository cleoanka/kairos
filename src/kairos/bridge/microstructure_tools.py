"""LangChain tools that expose the causal perception bus to System-2 agents.

These are the *only* way a reasoning agent reads microstructure, and they read
it exclusively through :meth:`CausalPerceptionBus.as_of` — so an agent reasoning
"as of ``curr_date``" can never be handed a percept from after that date. If no
causal percept exists at the cutoff, the tools say so plainly rather than
falling back to the latest (which would leak the future).

Requires the ``[reasoning]`` extra (langchain-core). Imported only by the
reasoning wiring, never by the bridge core.
"""
from __future__ import annotations

import math
import threading
from typing import Annotated

from langchain_core.tools import tool

from .causal_bus import CausalPerceptionBus, LookAheadError

# Active bus registry, set by the trading graph before a run (mirrors how the
# reasoning dataflows read a process-global config via ``set_config``). The lock
# guards every mutation so two overlapping same-symbol runs can't interleave a
# set against another's clear; ``clear`` also compares object identity so a run's
# ``finally`` only ever removes the bus *it* registered — it can never pop a
# concurrent run's live bus and silently blind that run's Microstructure Analyst.
_BUSES: dict[str, CausalPerceptionBus] = {}
_BUSES_LOCK = threading.Lock()
_DEFAULT_KEY = "*"


def set_perception_bus(bus: CausalPerceptionBus, symbol: str | None = None) -> None:
    """Register the causal bus a run's microstructure tools should read."""
    with _BUSES_LOCK:
        _BUSES[symbol.upper() if symbol else _DEFAULT_KEY] = bus


def clear_perception_bus(
    symbol: str | None = None, bus: CausalPerceptionBus | None = None
) -> None:
    """Unregister buses. With ``symbol`` given, only pop that key — and only if
    ``bus`` (when supplied) is still the exact object registered there, so a late
    ``finally`` never clobbers a concurrent same-symbol run's bus."""
    with _BUSES_LOCK:
        if symbol is None:
            _BUSES.clear()
            return
        key = symbol.upper()
        if bus is None or _BUSES.get(key) is bus:
            _BUSES.pop(key, None)


def _bus_for(symbol: str) -> CausalPerceptionBus | None:
    with _BUSES_LOCK:
        return _BUSES.get(symbol.upper()) or _BUSES.get(_DEFAULT_KEY)


_UNAVAILABLE = (
    "Microstructure perception is unavailable for {symbol} as of {curr_date} "
    "(no System-1 percept at or before this cutoff). Reason from the other "
    "analysts and do NOT assume a regime."
)


@tool
def get_microstructure_regime(
    symbol: Annotated[str, "ticker/instrument symbol, e.g. NVDA or BTCUSDT"],
    curr_date: Annotated[str, "the current decision date/time, YYYY-mm-dd (treated as 'now')"],
) -> str:
    """Point-in-time System-1 microstructure read from the limit order book.

    Returns the strictly-causal regime (RANGE / TREND / TOXIC), a BULL/BEAR
    direction, order-flow and resting-depth imbalance, and liquidity toxicity as
    of ``curr_date``. This is fast subsymbolic perception of *how liquidity is
    behaving right now* — use it to ground your view in market microstructure,
    not price history. Treat a TOXIC regime as a strong stand-aside signal."""
    bus = _bus_for(symbol)
    if bus is None:
        return _UNAVAILABLE.format(symbol=symbol, curr_date=curr_date)
    try:
        p = bus.as_of(curr_date)
    except (LookAheadError, ValueError, TypeError):
        # An unresolvable or non-finite cutoff (a malformed date, "nan", inf)
        # must fail closed: report no perception rather than raise into the
        # agent — and never leak or fabricate a regime.
        return _UNAVAILABLE.format(symbol=symbol, curr_date=curr_date)
    if p is None:
        return _UNAVAILABLE.format(symbol=symbol, curr_date=curr_date)
    return p.to_prompt()


# On an index-clocked bus (the default synthetic/replay path) "seconds" are
# meaningless: percepts are stamped with a monotonic step index, so a 3600-"second"
# horizon spans the entire history. We interpret the horizon in whatever unit the
# bus is clocked in and derive an honest default per clock, so "recent" stays
# recent and the header never lies about its unit.
_INDEX_LOOKBACK = 64.0  # step-index units ≈ a couple of dozen recorded percepts


@tool
def get_order_flow_state(
    symbol: Annotated[str, "ticker/instrument symbol"],
    curr_date: Annotated[str, "the current decision date/time, YYYY-mm-dd"],
    lookback: Annotated[
        float | None,
        "causal look-back horizon for the rolling regime read, in the bus's own "
        "clock unit (seconds on a real epoch-timestamped bus, monotonic step-index "
        "units on the synthetic/replay bus). Leave unset for a sensible per-clock "
        "default.",
    ] = None,
) -> str:
    """Rolling causal summary of recent order flow and regime stability.

    Aggregates every percept in the ``(cutoff - lookback, cutoff]`` window into a
    regime distribution, a toxic-fraction, and mean order-flow / depth imbalance.
    Use it to judge whether the current regime is stable or flickering, and how
    persistent the recent flow has been — all strictly before ``curr_date``.

    The horizon is measured in the bus's clock unit: seconds on a real
    epoch-timestamped bus, step-index units on the synthetic/replay bus (where
    "seconds" are meaningless). The header states the unit used so the read is
    never mislabelled."""
    bus = _bus_for(symbol)
    if bus is None:
        return _UNAVAILABLE.format(symbol=symbol, curr_date=curr_date)
    # An index clock counts monotonic steps, not seconds; default and label the
    # horizon in that unit so the window really is "recent" and the header honest.
    is_index = bus.clock == "index"
    unit = "steps" if is_index else "s"
    horizon = lookback if lookback is not None else (_INDEX_LOOKBACK if is_index else 3600.0)
    # Defence-in-depth: a non-finite or non-positive horizon can't scope a window
    # (inf spans the whole history, ≤0 empties it); fail closed to "unavailable"
    # rather than reason over a silently mis-scoped read.
    if not math.isfinite(horizon) or horizon <= 0:
        return _UNAVAILABLE.format(symbol=symbol, curr_date=curr_date)
    try:
        agg = bus.aggregate_before(curr_date, horizon)
    except (LookAheadError, ValueError, TypeError):
        return _UNAVAILABLE.format(symbol=symbol, curr_date=curr_date)
    if agg is None:
        return _UNAVAILABLE.format(symbol=symbol, curr_date=curr_date)
    dist = ", ".join(f"{k}={v}" for k, v in agg["regime_distribution"].items())
    return (
        f"ORDER-FLOW STATE [{symbol} @ {curr_date}, last {horizon:.0f}{unit}]\n"
        f"  Percepts observed  : {agg['n']} (strictly causal)\n"
        f"  Dominant regime    : {agg['dominant_regime']}\n"
        f"  Regime distribution: {dist}\n"
        f"  Toxic fraction     : {agg['toxic_fraction']:.0%}\n"
        f"  Mean order-flow imb: {agg['mean_order_flow_imbalance']:+.3f}\n"
        f"  Mean depth imbal.  : {agg['mean_depth_imbalance']:+.3f}\n"
        f"  Mean toxicity      : {agg['mean_toxicity']:.3f}"
    )


MICROSTRUCTURE_TOOLS = [get_microstructure_regime, get_order_flow_state]
