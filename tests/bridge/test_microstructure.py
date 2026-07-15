"""System-1 perception: signals and heuristic regime are correct and bounded."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from kairos.bridge.microstructure import (
    LearnedRegime,
    MicrostructureConfig,
    heuristic_regime,
    percept_from_window,
    raw_signals,
)
from kairos.bridge.percept import BEAR, BULL
from kairos.perception.models.embedder import NumpyEncoder
from kairos.perception.schema import Regime
from kairos.perception.synthetic.generate import generate


def _window(scenario: str, n: int = 400):
    return generate(n_steps=n, seed=3, scenario=scenario)


def test_signals_are_finite_and_bounded():
    cfg = MicrostructureConfig()
    for scenario in ("toxic", "calm", "range"):
        df = _window(scenario)
        sig = raw_signals(df.tail(64), cfg)
        assert -1.0 <= sig["order_flow_imbalance"] <= 1.0
        assert -1.0 <= sig["depth_imbalance"] <= 1.0
        assert 0.0 <= sig["toxicity"] <= 1.0
        assert 0.0 <= sig["trade_intensity"] <= 1.0
        assert np.isfinite(sig["mid"])


def test_range_scenario_reads_as_range():
    df = _window("range", 800)
    p = percept_from_window(df.tail(64), "X", ts=float(df.iloc[-1]["ts"]))
    # A benign two-sided market should not read as TOXIC.
    assert not p.is_toxic


def test_corrupt_book_forces_toxic_standaside():
    cfg = MicrostructureConfig()
    df = _window("calm").copy()
    df.iloc[-1, df.columns.get_loc("mid")] = np.nan  # corrupt the current book
    sig = raw_signals(df.tail(32), cfg)
    regime, conf, source = heuristic_regime(sig, cfg)
    assert regime == int(Regime.TOXIC)
    p = percept_from_window(df.tail(32), "X", ts=float(df.iloc[-2]["ts"]))
    assert p.is_toxic and p.direction_strength == 0.0  # no direction in TOXIC


def test_corrupt_size_forces_toxic_standaside():
    # A finite mid with NaN depth/size yields a NaN toxicity; it must fail SAFE
    # to TOXIC, not silently read as RANGE and drop the safety veto.
    cfg = MicrostructureConfig()
    df = _window("calm").copy()
    df.iloc[-1, df.columns.get_loc("bid_sz_0")] = np.nan  # mid stays finite
    sig = raw_signals(df.tail(32), cfg)
    assert sig["corrupt"] and np.isfinite(sig["toxicity"]) and sig["toxicity"] == 1.0
    regime, _, _ = heuristic_regime(sig, cfg)
    assert regime == int(Regime.TOXIC)
    p = percept_from_window(df.tail(32), "X", ts=float(df.iloc[-2]["ts"]))
    assert p.is_toxic and p.direction_strength == 0.0


class _NaNEncoder(NumpyEncoder):
    """NumpyEncoder whose forward pass emits a non-finite embedding (no weights)."""

    def __init__(self):
        pass

    def encode(self, X):
        return np.full((np.asarray(X).shape[0], 4), np.nan, dtype=np.float32)


def test_learned_regime_fails_safe_on_nonfinite_feature():
    # Corrupt size -> non-finite featurized row -> TOXIC before the encoder runs.
    df = _window("calm", 64).copy()
    df.iloc[-1, df.columns.get_loc("bid_sz_0")] = np.nan
    predictor = SimpleNamespace(model=_NaNEncoder(), stats={"mu": 0.0, "sd": 1.0})
    assert LearnedRegime(predictor)(df, {}) == (int(Regime.TOXIC), 1.0, "learned")


def test_learned_regime_fails_safe_on_nonfinite_embedding():
    # Finite features but a non-finite embedding -> TOXIC (never a confident wrong read).
    df = _window("calm", 64)
    predictor = SimpleNamespace(model=_NaNEncoder(), stats={"mu": 0.0, "sd": 1.0})
    assert LearnedRegime(predictor)(df, {}) == (int(Regime.TOXIC), 1.0, "learned")


def test_toxic_regime_has_no_direction():
    cfg = MicrostructureConfig(toxic_threshold=0.0)  # force TOXIC
    df = _window("toxic")
    p = percept_from_window(df.tail(64), "X", ts=float(df.iloc[-1]["ts"]), cfg=cfg)
    assert p.is_toxic
    assert p.direction_strength == 0.0


def test_direction_follows_order_flow_sign():
    cfg = MicrostructureConfig()
    # Construct a strongly one-sided flow window from calm data.
    df = _window("calm").copy().tail(64).reset_index(drop=True)
    df["trade_buy"] = 100.0
    df["trade_sell"] = 0.0
    p = percept_from_window(df, "X", ts=float(df.iloc[-1]["ts"]), cfg=cfg)
    if not p.is_toxic:
        assert p.direction == BULL
    df["trade_buy"] = 0.0
    df["trade_sell"] = 100.0
    p2 = percept_from_window(df, "X", ts=float(df.iloc[-1]["ts"]), cfg=cfg)
    if not p2.is_toxic:
        assert p2.direction == BEAR


def test_percept_prompt_is_deterministic_and_priceforecast_free():
    df = _window("calm")
    p = percept_from_window(df.tail(64), "NVDA", ts=1.0)
    text = p.to_prompt()
    assert p.to_prompt() == text  # deterministic
    assert "NVDA" in text and "Regime" in text
    # It reports state, not a price forecast.
    assert "target" not in text.lower() and "forecast" not in text.lower()
