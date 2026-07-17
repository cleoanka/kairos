"""Red-team: the execution path touches money. Corrupt prices (one-sided / NaN /
inf books), extreme inventory, and broken marks must never produce NaN PnL,
crossed quotes, or an unhalted runaway."""
from __future__ import annotations

import math

import pandas as pd

from kairos.perception.execution.risk import RiskGate
from kairos.perception.execution.simulator import run_backtest
from kairos.perception.schema import Regime
from kairos.perception.strategy.avellaneda import AvellanedaStoikovMaker
from kairos.perception.strategy.maker import MakerStrategy

_COLS = ["mid", "bid_px_0", "ask_px_0", "trade_buy", "trade_sell", "regime"]


def _df(rows):
    return pd.DataFrame(rows, columns=_COLS)


def _row(mid=100.0, bp=1.0, ap=1.0, tb=0.0, ts=0.0, reg=Regime.RANGE):
    return {"mid": mid, "bid_px_0": bp, "ask_px_0": ap,
            "trade_buy": tb, "trade_sell": ts, "regime": int(reg)}


def test_makers_refuse_non_finite_book():
    for m in (MakerStrategy(), AvellanedaStoikovMaker()):
        for bb, ba in [(float("nan"), 101.0), (99.0, float("inf")), (float("inf"), float("inf"))]:
            q = m.decide(int(Regime.RANGE), bb, ba, 0.0)
            assert q.bid_px is None and q.ask_px is None


def test_as_maker_refuses_overflowing_mid_from_finite_book():
    # Both operands are finite but their sum overflows the mid to +/-inf; the
    # corrupt-book guard must catch the computed mid, not just its inputs — a
    # non-finite quote must never leak past the stand-aside intent, and the
    # overflowing mid must not poison the volatility deque.
    m = AvellanedaStoikovMaker()
    for bb, ba in [(1e308, 1e308), (-1e308, -1e308)]:
        q = m.decide(int(Regime.RANGE), bb, ba, 0.0)
        assert q.bid_px is None and q.ask_px is None
    assert all(math.isfinite(x) for x in m._mids)  # deque never saw a non-finite mid
    # A finite-mid book still quotes (guard does not over-fire).
    q = m.decide(int(Regime.RANGE), 99.0, 101.0, 0.0)
    assert q.two_sided


def test_as_maker_refuses_overflowing_volatility_from_finite_mids():
    # Each book's mid is individually finite, but the alternating +/-1e200 swing
    # makes the volatility deque's std() overflow to inf; with inventory==0 the
    # reservation r = mid - 0*gamma*inf = nan would leak past the mid-only guard.
    # The volatility term must be guarded so no NaN price ever leaves decide().
    m = AvellanedaStoikovMaker()
    q = None
    for i in range(12):
        s = 1e200 if i % 2 == 0 else -1e200
        q = m.decide(int(Regime.RANGE), s, s, 0.0)      # mid == s, individually finite
        assert q.bid_px is None or math.isfinite(q.bid_px)
        assert q.ask_px is None or math.isfinite(q.ask_px)
    assert q.bid_px is None and q.ask_px is None         # stood aside, not a NaN quote
    # A normal book still quotes (guard does not over-fire on finite volatility).
    q = AvellanedaStoikovMaker().decide(int(Regime.RANGE), 99.0, 101.0, 0.0)
    assert q.two_sided


def test_as_extreme_inventory_quotes_finite_and_uncrossed():
    q = AvellanedaStoikovMaker().decide(int(Regime.RANGE), 99.0, 101.0, inventory=1e9)
    if q.bid_px is not None and q.ask_px is not None:
        assert math.isfinite(q.bid_px) and math.isfinite(q.ask_px)
        assert q.ask_px >= q.bid_px


def test_backtest_skips_corrupt_rows_and_pnl_stays_finite():
    df = _df([_row(mid=float("nan"), tb=1.0), _row(mid=100.0, tb=2.0), _row(bp=float("inf"))])
    r = run_backtest(df, MakerStrategy(), RiskGate())
    assert math.isfinite(r["final_pnl"])


def test_backtest_inf_trade_volume_stays_finite():
    r = run_backtest(_df([_row(tb=0.0), _row(tb=float("inf"))]), MakerStrategy(), RiskGate())
    assert math.isfinite(r["final_pnl"])


def test_empty_backtest_is_zero():
    r = run_backtest(_df([]), MakerStrategy(), RiskGate())
    assert r["final_pnl"] == 0.0 and r["fills"] == 0


def test_risk_gate_halts_on_non_finite_pnl():
    g = RiskGate(max_drawdown=1e9)
    assert g.update(10.0) is True
    assert g.update(float("nan")) is False and g.halted
