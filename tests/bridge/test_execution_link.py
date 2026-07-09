"""The neuro-symbolic execution link: parsing, sizing, and the System-1 veto."""
from __future__ import annotations

import numpy as np

from kairos.bridge.execution_link import (
    Decision,
    DirectionalMaker,
    ExecutionLink,
    parse_decision,
    perceive_regimes,
)
from kairos.bridge.microstructure import MicrostructureConfig
from kairos.perception.schema import Regime
from kairos.perception.synthetic.generate import generate


def test_parse_final_transaction_proposal():
    d = parse_decision("blah blah FINAL TRANSACTION PROPOSAL: **BUY** because reasons")
    assert d.action == "BUY" and d.conviction > 0
    assert parse_decision("... FINAL TRANSACTION PROPOSAL: **SELL**").action == "SELL"
    assert parse_decision("FINAL TRANSACTION PROPOSAL: **HOLD**").action == "HOLD"
    assert parse_decision("FINAL TRANSACTION PROPOSAL: **HOLD**").conviction == 0.0


def test_parse_conviction_scales():
    assert abs(parse_decision("BUY, conviction 4/5").conviction - 0.8) < 1e-9
    assert abs(parse_decision("BUY, confidence 70%").conviction - 0.7) < 1e-9
    assert abs(parse_decision("BUY, rating 8/10").conviction - 0.8) < 1e-9


def test_parse_conviction_spaced_denominator():
    # "9 / 10" (spaces) must read 0.9, not fall through to 0.09.
    assert abs(parse_decision("BUY, confidence 9 / 10").conviction - 0.9) < 1e-9
    assert abs(parse_decision("BUY, confidence 4 / 10").conviction - 0.4) < 1e-9


def test_parse_conviction_bare_and_outof_scales():
    assert abs(parse_decision("BUY, rating 7").conviction - 0.7) < 1e-9         # bare 1-10
    assert abs(parse_decision("BUY, conviction is 8 out of 10").conviction - 0.8) < 1e-9
    assert abs(parse_decision("BUY, conviction 0.9").conviction - 0.9) < 1e-9   # already [0,1]


def test_parse_ignores_implausible_rating_magnitude():
    # A price-like number after the keyword must not pin conviction.
    d = parse_decision("BUY, confidence 4200 shares", default_conviction=0.6)
    assert d.conviction == 0.6


def test_parse_recommendation_scoped_action_beats_trailing_caveat():
    # "recommend BUY ... do not SELL yet" -> BUY (not the trailing SELL token).
    d = parse_decision("We recommend BUY now; do not SELL yet.")
    assert d.action == "BUY"


def test_decision_bias_sign():
    assert Decision("BUY", 0.5).bias == 0.5
    assert Decision("SELL", 0.5).bias == -0.5
    assert Decision("HOLD", 0.9).bias == 0.0


def test_directional_maker_skews_sizes_but_respects_regime_veto():
    # In a RANGE book the base maker quotes two-sided; the BUY bias should grow
    # the bid and shrink the ask.
    m = DirectionalMaker(bias=0.5, base_size=1.0)
    q = m.decide(int(Regime.RANGE), best_bid=99.0, best_ask=101.0, inventory=0.0)
    if q.bid_sz and q.ask_sz:
        assert q.bid_sz > q.ask_sz
    # In TOXIC the base maker stands aside — bias must not override the veto.
    q_toxic = m.decide(int(Regime.TOXIC), best_bid=99.0, best_ask=101.0, inventory=0.0)
    assert q_toxic.bid_sz == 0.0 and q_toxic.ask_sz == 0.0


def test_perceive_regimes_is_per_row_and_causal():
    df = generate(n_steps=500, seed=5, scenario="toxic")
    regimes = perceive_regimes(df, window=32)
    assert len(regimes) == len(df)
    assert set(np.unique(regimes)).issubset({int(r) for r in Regime})


def test_execution_reports_toxic_veto_fraction():
    df = generate(n_steps=1500, seed=7, scenario="toxic")
    fwd = df.iloc[750:].reset_index(drop=True)
    link = ExecutionLink()
    rep = link.execute(Decision("BUY", 0.8), fwd)
    ns = rep["neuro_symbolic"]
    assert 0.0 <= ns["toxic_veto_fraction"] <= 1.0
    assert ns["action"] == "BUY"
    assert rep["steps"] == len(fwd)
    assert np.isfinite(rep["final_pnl"])


def test_toxic_only_market_forces_standaside_pnl_flat():
    """If every step is perceived TOXIC, the maker never quotes: no fills."""
    df = generate(n_steps=800, seed=1, scenario="calm")
    cfg = MicrostructureConfig(toxic_threshold=-1.0)  # tox > -1 always -> every window TOXIC
    link = ExecutionLink()
    rep = link.execute(Decision("BUY", 1.0), df, cfg=cfg)
    assert rep["neuro_symbolic"]["toxic_veto_fraction"] == 1.0
    assert rep["fills"] == 0            # System-1 vetoed every quote
    assert rep["final_inventory"] == 0.0
