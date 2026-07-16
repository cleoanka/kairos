"""Tests for the maker decision layer, risk gate, and paper fill simulator
(Component E). All offline — no orders, no network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kairos.perception.execution.risk import RiskGate
from kairos.perception.execution.simulator import run_backtest
from kairos.perception.schema import Regime
from kairos.perception.strategy.avellaneda import AvellanedaStoikovMaker
from kairos.perception.strategy.maker import Maker, MakerStrategy, Quote


# --- maker decisions ---------------------------------------------------------
def test_maker_pulls_in_toxic():
    q = MakerStrategy().decide(int(Regime.TOXIC), 99.0, 101.0, 0.0)
    assert q.bid_px is None and q.ask_px is None and not q.two_sided


def test_maker_two_sided_in_range():
    q = MakerStrategy().decide(int(Regime.RANGE), 99.0, 101.0, 0.0)
    assert q.two_sided and q.bid_sz > 0 and q.ask_sz > 0
    assert 99.0 <= q.bid_px <= q.ask_px <= 101.0  # quotes inside-or-at the spread


def test_maker_skews_against_long_inventory():
    q = MakerStrategy(max_inventory=20).decide(int(Regime.RANGE), 99.0, 101.0, inventory=10.0)
    assert q.bid_sz < q.ask_sz  # long -> quote less bid, more ask (mean-revert)


def test_maker_trend_is_thinner_than_range():
    rng = MakerStrategy().decide(int(Regime.RANGE), 99.0, 101.0, 0.0)
    trn = MakerStrategy().decide(int(Regime.TREND), 99.0, 101.0, 0.0)
    assert (trn.bid_sz + trn.ask_sz) < (rng.bid_sz + rng.ask_sz)


def test_maker_trend_is_reduce_only():
    s = MakerStrategy()
    assert s.decide(int(Regime.TREND), 99.0, 101.0, 0.0).bid_sz == 0.0  # flat -> no exposure
    lng = s.decide(int(Regime.TREND), 99.0, 101.0, 5.0)
    assert lng.ask_sz > 0 and lng.bid_sz == 0.0   # long  -> sell-only (flatten)
    sht = s.decide(int(Regime.TREND), 99.0, 101.0, -5.0)
    assert sht.bid_sz > 0 and sht.ask_sz == 0.0   # short -> buy-only (flatten)


@pytest.mark.parametrize("maker", [MakerStrategy(), AvellanedaStoikovMaker()])
@pytest.mark.parametrize("best_bid, best_ask", [
    (99.0, 101.0),    # normal book
    (100.0, 100.0),   # locked book
    (101.0, 99.0),    # crossed book — negative/corrupt offset
])
@pytest.mark.parametrize("inventory", [-15.0, 0.0, 15.0])
@pytest.mark.parametrize("regime", [int(Regime.RANGE), int(Regime.TREND)])
def test_maker_never_crosses(maker, best_bid, best_ask, inventory, regime):
    # Structural defense: for ANY finite book (including crossed inputs) the two
    # legs must not cross — bid_px <= ask_px whenever both sides are quoted.
    q = maker.decide(regime, best_bid, best_ask, inventory)
    assert q.bid_px is None or q.ask_px is None or q.bid_px <= q.ask_px


# --- Avellaneda-Stoikov maker ------------------------------------------------
def test_as_maker_pulls_in_toxic():
    q = AvellanedaStoikovMaker().decide(int(Regime.TOXIC), 99.0, 101.0, 0.0)
    assert not q.two_sided


def test_as_positive_spread():
    q = AvellanedaStoikovMaker().decide(int(Regime.RANGE), 99.0, 101.0, 0.0)
    assert q.ask_px > q.bid_px


def test_as_reservation_skews_down_when_long():
    m = AvellanedaStoikovMaker(gamma=0.1, default_sigma=2.0)
    flat = m.decide(int(Regime.RANGE), 99.0, 101.0, 0.0)
    long = m.decide(int(Regime.RANGE), 99.0, 101.0, 10.0)
    # Long inventory leans the quotes down (sell-biased): the ask becomes more
    # aggressive; the bid never becomes *more* aggressive. (Both may clamp to the
    # touch when the spread is wide, so only the ask is guaranteed to move.)
    assert long.ask_px < flat.ask_px
    assert long.bid_px <= flat.bid_px


def test_as_trend_reduce_only():
    lng = AvellanedaStoikovMaker().decide(int(Regime.TREND), 99.0, 101.0, 5.0)
    assert lng.ask_sz > 0 and lng.bid_sz == 0.0


def test_as_maker_has_no_inert_tick_param():
    # The A-S maker quotes purely from the reservation price/half-spread clamped
    # to the touch — it never snaps to a grid — so it must not carry an inert
    # `tick` field whose name would mislead readers into thinking it does.
    with pytest.raises(TypeError):
        AvellanedaStoikovMaker(tick=1.0)
    assert not hasattr(AvellanedaStoikovMaker(), "tick")


def test_makers_satisfy_maker_protocol():
    # run_backtest is typed against the structural Maker contract; every maker it
    # is fed must satisfy it (baseline skew-maker and the A-S sibling alike).
    assert isinstance(MakerStrategy(), Maker)
    assert isinstance(AvellanedaStoikovMaker(), Maker)


# --- risk gate ---------------------------------------------------------------
def test_risk_gate_halts_on_drawdown():
    g = RiskGate(max_drawdown=5.0)
    assert g.update(10.0) is True      # peak = 10
    assert g.update(6.0) is True       # drawdown 4 < 5
    assert g.update(4.0) is False      # drawdown 6 > 5 -> halt
    assert g.halted


def test_risk_clamp_blocks_overlimit_buy():
    g = RiskGate(max_inventory=5.0)
    q = Quote(99.0, 1.0, 101.0, 1.0)
    g.clamp(q, inventory=5.0)          # already max long
    assert q.bid_sz == 0.0 and q.ask_sz == 1.0


# --- fill simulation / backtest ----------------------------------------------
def test_fill_on_aggressive_buy_reduces_inventory_and_adds_cash():
    # row0 posts a resting quote (RANGE, no trades); row1's aggressive buys lift it.
    df = pd.DataFrame([
        {"mid": 100.0, "bid_px_0": 1.0, "ask_px_0": 1.0,
         "trade_buy": 0.0, "trade_sell": 0.0, "regime": int(Regime.RANGE)},
        {"mid": 100.0, "bid_px_0": 1.0, "ask_px_0": 1.0,
         "trade_buy": 2.0, "trade_sell": 0.0, "regime": int(Regime.RANGE)},
    ])
    rep = run_backtest(df, MakerStrategy(base_size=1.0))
    assert rep["fills"] == 1
    assert rep["final_inventory"] < 0          # lifted ask -> we sold -> short
    assert rep["final_pnl"] == 1.0             # sold 1 @101, marked at mid 100


def test_no_fills_in_toxic_step():
    df = pd.DataFrame([{"mid": 100.0, "bid_px_0": 1.0, "ask_px_0": 1.0,
                        "trade_buy": 5.0, "trade_sell": 5.0, "regime": int(Regime.TOXIC)}])
    rep = run_backtest(df)
    assert rep["fills"] == 0 and rep["final_inventory"] == 0.0


def test_backtest_runs_over_synthetic_and_is_bounded():
    from kairos.perception.synthetic.generate import generate
    df = generate(n_steps=900, seed=4)
    rep = run_backtest(df, MakerStrategy(base_size=1.0, max_inventory=20.0))
    assert rep["steps"] == len(df)
    assert np.isfinite(rep["final_pnl"])
    assert rep["max_abs_inventory"] <= 22.0     # bounded by skew + clamp (+ one fill)
    assert set(rep["regime_exposure"]) <= {"RANGE", "TREND", "TOXIC"}
