"""Tests for the live-streaming server (no network).

The important one: the incremental ``_LiveMaker`` used on the live path must be a
*faithful* mirror of the tested batch ``run_backtest`` — same fills, same PnL.
"""
from __future__ import annotations

from kairos.perception.execution.risk import RiskGate
from kairos.perception.execution.simulator import run_backtest
from kairos.perception.strategy.avellaneda import AvellanedaStoikovMaker
from kairos.perception.synthetic.generate import generate


def test_live_maker_matches_batch_backtest():
    df = generate(n_steps=400, seed=3)
    bt = run_backtest(df, AvellanedaStoikovMaker(), RiskGate())

    from kairos.perception.stream import _LiveMaker
    lm = _LiveMaker()
    pnl = inv = None
    keys = ("mid", "bid_px_0", "ask_px_0", "trade_buy", "trade_sell")
    for i in range(len(df)):
        r = df.iloc[i]
        pnl, inv = lm.step({k: float(r[k]) for k in keys}, int(r["regime"]))

    # incremental == batch, to floating-point tolerance
    assert abs(pnl - bt["pnl_curve"][-1]) < 1e-6
    assert abs(inv - bt["inv_curve"][-1]) < 1e-6


def test_set_symbol_whitelist():
    from kairos.perception.stream import SYMBOLS, _Stream
    s = _Stream(predictor=None, metrics={}, model_src="test")
    s.want = "BTCUSDT"
    s.set_symbol("ETHUSDT")
    assert s.want == "ETHUSDT"
    s.set_symbol("BTCUSDT; rm -rf /")      # not whitelisted → ignored
    assert s.want == "ETHUSDT"
    assert set(SYMBOLS) >= {"BTCUSDT", "ETHUSDT"}


def test_load_predictor_degrades_gracefully():
    # never raises; returns (predictor_or_None, label, is_real)
    from kairos.perception.stream import _load_predictor
    pred, src, is_real = _load_predictor()
    assert isinstance(src, str) and isinstance(is_real, bool)
