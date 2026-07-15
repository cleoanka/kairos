"""End-to-end: the closed cognitive loop runs, is causal, and is deterministic."""
from __future__ import annotations

import pytest

from kairos.bridge.percept import BEAR, BULL, NEUTRAL
from kairos.loop import LoopConfig, deterministic_policy, run_cognitive_loop
from kairos.loop.cognitive_loop import _reflect  # noqa: F401 (import smoke)


@pytest.mark.parametrize("scenario", ["toxic", "calm", "range"])
def test_loop_runs_end_to_end(scenario):
    res = run_cognitive_loop(LoopConfig(scenario=scenario, n_steps=2000, seed=7))
    assert res.percept is not None
    assert res.decision.action in ("BUY", "HOLD", "SELL")
    assert "final_pnl" in res.execution
    assert set(res.baselines) == {"stand_aside", "naive_long", "pure_market_making"}
    assert isinstance(res.reflection, str) and res.symbol in res.reflection


def test_loop_decision_is_causal():
    """The stance is formed only from the in-sample percept (ts <= decision_ts)."""
    res = run_cognitive_loop(LoopConfig(scenario="toxic", n_steps=2000, seed=7))
    assert res.percept.ts <= res.decision_ts


def test_loop_is_deterministic():
    a = run_cognitive_loop(LoopConfig(scenario="toxic", n_steps=2000, seed=7)).to_dict()
    b = run_cognitive_loop(LoopConfig(scenario="toxic", n_steps=2000, seed=7)).to_dict()
    assert a["decision"] == b["decision"]
    assert a["execution"]["final_pnl"] == b["execution"]["final_pnl"]


def test_stand_aside_is_a_genuinely_flat_baseline():
    """stand_aside does nothing and risks nothing: no fills, zero PnL/inventory,
    never halted — a real null reference, not a HOLD/0.0 self-comparison."""
    res = run_cognitive_loop(LoopConfig(scenario="range", n_steps=2000, seed=7))
    sa = res.baselines["stand_aside"]
    assert sa == {"final_pnl": 0.0, "fills": 0, "final_inventory": 0.0, "halted": False}


def test_stand_aside_differs_from_pure_market_making():
    """The two baselines must be distinct: pure_market_making actively quotes
    two-sided (generally taking fills), stand_aside stays flat."""
    res = run_cognitive_loop(LoopConfig(scenario="range", n_steps=2000, seed=7))
    sa, pmm = res.baselines["stand_aside"], res.baselines["pure_market_making"]
    assert pmm["fills"] > 0                       # the maker actually traded
    assert (sa["fills"], sa["final_pnl"]) != (pmm["fills"], pmm["final_pnl"])


def test_deterministic_policy_stands_aside_on_toxic():
    from kairos.bridge.percept import Percept

    toxic = Percept(ts=1.0, symbol="X", mid=100.0, spread_ticks=1.0,
                    order_flow_imbalance=0.9, depth_imbalance=0.0, toxicity=0.9,
                    trade_intensity=0.5, regime=2, regime_confidence=1.0,
                    regime_source="test", direction=NEUTRAL, direction_strength=0.0,
                    n_observations=1)
    d = deterministic_policy(toxic)
    assert d.action == "HOLD" and d.conviction == 0.0


def test_deterministic_policy_follows_direction():
    from kairos.bridge.percept import Percept

    bull = Percept(ts=1.0, symbol="X", mid=100.0, spread_ticks=1.0,
                   order_flow_imbalance=0.6, depth_imbalance=0.4, toxicity=0.0,
                   trade_intensity=0.5, regime=1, regime_confidence=0.9,
                   direction=BULL, regime_source="t", direction_strength=0.8,
                   n_observations=1)
    assert deterministic_policy(bull).action == "BUY"
    bear = Percept(ts=1.0, symbol="X", mid=100.0, spread_ticks=1.0,
                   order_flow_imbalance=-0.6, depth_imbalance=-0.4, toxicity=0.0,
                   trade_intensity=0.5, regime=1, regime_confidence=0.9,
                   direction=BEAR, regime_source="t", direction_strength=0.8,
                   n_observations=1)
    assert deterministic_policy(bear).action == "SELL"
