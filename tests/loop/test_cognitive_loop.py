"""End-to-end: the closed cognitive loop runs, is causal, and is deterministic."""
from __future__ import annotations

import pytest

import kairos.bridge.execution_link as el
from kairos.bridge import Decision
from kairos.bridge.percept import BEAR, BULL, NEUTRAL
from kairos.loop import LoopConfig, cognitive_loop as cl, deterministic_policy, run_cognitive_loop
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


def test_perceived_regime_array_computed_once_per_run(monkeypatch):
    """The perceived-regime array is a pure function of the forward window, but the
    decision run and both baselines re-derive it — so it must be shared, not
    recomputed 3x (which triples the dominant learned-backend inference cost)."""
    calls = {"n": 0, "arrays": []}
    real = el.perceive_regimes

    def counting(df, **kw):
        out = real(df, **kw)
        calls["n"] += 1
        calls["arrays"].append(out)
        return out

    monkeypatch.setattr(el, "perceive_regimes", counting)
    run_cognitive_loop(LoopConfig(scenario="range", n_steps=2000, seed=7))
    assert calls["n"] == 1                                    # was 3 before sharing
    # and the wrapper is restored after the run (no leaked memoization).
    assert el.perceive_regimes is counting


def test_sharing_perception_keeps_headline_numbers_identical():
    """Sharing the perceived-regime array must be bit-for-bit behaviour-preserving:
    the decision, PnL and every baseline are unchanged versus recomputing it 3x."""
    with_sharing = run_cognitive_loop(LoopConfig(scenario="range", n_steps=2000, seed=7)).to_dict()

    # Force the pre-fix path: neutralise the memoization so execute recomputes.
    original = cl._shared_perception

    import contextlib

    @contextlib.contextmanager
    def _no_sharing():
        yield

    cl._shared_perception = _no_sharing
    try:
        without_sharing = run_cognitive_loop(
            LoopConfig(scenario="range", n_steps=2000, seed=7)).to_dict()
    finally:
        cl._shared_perception = original

    assert with_sharing["decision"] == without_sharing["decision"]
    assert with_sharing["execution"] == without_sharing["execution"]
    assert with_sharing["baselines"] == without_sharing["baselines"]


def test_max_conviction_clamps_llm_decision(monkeypatch):
    """max_conviction must cap conviction in LLM mode too, not only the
    deterministic policy — otherwise a de-risking cap is silently a no-op."""
    monkeypatch.setattr(
        cl, "_llm_decision",
        lambda *a, **k: Decision("BUY", 0.9, rationale="mock", source="llm"))
    res = run_cognitive_loop(LoopConfig(scenario="range", n_steps=2000, seed=7,
                                        mode="llm", max_conviction=0.25))
    assert res.decision.action == "BUY"
    assert res.decision.conviction == pytest.approx(0.25)
    assert res.decision.bias == pytest.approx(0.25)          # bias follows the cap


def test_max_conviction_below_cap_left_untouched_in_llm_mode(monkeypatch):
    """A conviction already under the cap is passed through verbatim."""
    monkeypatch.setattr(
        cl, "_llm_decision",
        lambda *a, **k: Decision("SELL", 0.1, rationale="mock", source="llm"))
    res = run_cognitive_loop(LoopConfig(scenario="range", n_steps=2000, seed=7,
                                        mode="llm", max_conviction=0.5))
    assert res.decision.conviction == pytest.approx(0.1)


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
