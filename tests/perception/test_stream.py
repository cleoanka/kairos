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


def test_inference_failure_fails_safe_to_toxic():
    # A broken live inference must NOT silently pick RANGE (regime 0, the least
    # cautious): it must fail SAFE to TOXIC — mirroring the bridge's non-finite
    # rule — so the maker keeps its stand-aside veto on an unreadable book.
    from kairos.perception.ingest.orderbook import LiveOrderBook
    from kairos.perception.schema import Regime
    from kairos.perception.stream import _Stream

    class _BrokenPredictor:
        model = None
        stats = None

        def predict_features(self, X):
            raise RuntimeError("model exploded")

    book = LiveOrderBook(tick_size=0.1)
    book.apply_depth([(100.0 - i * 0.1, 2.0 + i) for i in range(10)],
                     [(100.1 + i * 0.1, 2.0 + i) for i in range(10)], is_snapshot=True)
    assert book.ready()

    s = _Stream(predictor=_BrokenPredictor(), metrics={}, model_src="test")
    s._process(book.snapshot(), s.want)

    assert s.latest is not None
    assert s.latest["rp"] == int(Regime.TOXIC)   # not RANGE (0)
