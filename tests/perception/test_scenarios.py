"""Tests for the synthetic market scenarios and the finding that the makers earn
when the market is *makeable* (a stable RANGE market) — proving the toxic-market
loss is a property of the market, not a broken strategy. Offline; no orders."""
from __future__ import annotations

from kairos.perception.execution.risk import RiskGate
from kairos.perception.execution.simulator import run_backtest
from kairos.perception.schema import Regime
from kairos.perception.strategy.avellaneda import AvellanedaStoikovMaker
from kairos.perception.strategy.maker import MakerStrategy
from kairos.perception.synthetic.generate import SCENARIOS, generate


def test_scenarios_registered():
    assert {"toxic", "calm", "range"} <= set(SCENARIOS)


def test_range_scenario_is_all_range_stable_mid():
    df = generate(n_steps=800, seed=1, scenario="range")
    assert set(df["regime"].unique()) == {int(Regime.RANGE)}
    assert float(df["mid"].std()) < 5.0   # stable mid (no trend drift)


def test_calm_scenario_has_no_toxic():
    df = generate(n_steps=900, seed=1, scenario="calm")
    regimes = set(df["regime"].unique())
    assert int(Regime.TOXIC) not in regimes
    assert {int(Regime.RANGE), int(Regime.TREND)} <= regimes


def test_baseline_maker_earns_in_benign_range_market():
    df = generate(n_steps=3000, seed=7, scenario="range")
    rep = run_backtest(df, MakerStrategy(), RiskGate())
    assert rep["final_pnl"] > 0          # earns the spread when the market is makeable
    assert rep["fills"] > 100


def test_as_maker_competitive_and_profitable_in_range():
    df = generate(n_steps=2500, seed=7, scenario="range")
    rep = run_backtest(df, AvellanedaStoikovMaker(), RiskGate())
    assert rep["fills"] > 0              # competitive clamp -> it actually trades
    assert rep["final_pnl"] > 0          # and earns in a benign market
