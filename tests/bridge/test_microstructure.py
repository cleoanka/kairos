"""System-1 perception: signals and heuristic regime are correct and bounded."""
from __future__ import annotations

import math
import struct
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
from kairos.perception.schema import N_LEVELS, Regime
from kairos.perception.synthetic.generate import generate


def _window(scenario: str, n: int = 400):
    return generate(n_steps=n, seed=3, scenario=scenario)


def _raw_signals_scalar(window, cfg):
    """Reference scalar implementation of :func:`raw_signals` (pre-numpy path).

    Kept verbatim as the equivalence oracle: the numpy-first kernel must stay
    bit-for-bit identical to this per-column pandas computation.
    """
    n = len(window)
    last = window.iloc[-1]
    mid = float(last["mid"])
    best_bid_off = float(last["bid_px_0"])
    best_ask_off = float(last["ask_px_0"])
    spread_ticks = best_bid_off + best_ask_off
    bid_depth = float(sum(last[f"bid_sz_{i}"] for i in range(N_LEVELS)))
    ask_depth = float(sum(last[f"ask_sz_{i}"] for i in range(N_LEVELS)))
    rest = bid_depth + ask_depth
    depth_imbalance = (bid_depth - ask_depth) / rest if rest > 1e-9 else 0.0
    buy = float(window["trade_buy"].sum())
    sell = float(window["trade_sell"].sum())
    flow = buy + sell
    ofi = (buy - sell) / flow if flow > 1e-9 else 0.0
    cxl = float(sum(window[f"bid_cxl_{i}"].sum() + window[f"ask_cxl_{i}"].sum()
                    for i in range(N_LEVELS)))
    avg_cxl = cxl / max(n, 1)
    toxicity = avg_cxl / (avg_cxl + rest + 1e-9)
    trades = float(window["trade_n"].sum())
    trade_intensity = 1.0 - math.exp(-trades / (max(n, 1) * cfg.intensity_scale))
    blend = cfg.ofi_weight * ofi + (1.0 - cfg.ofi_weight) * depth_imbalance
    if not all(map(math.isfinite, (mid, spread_ticks, toxicity, blend))):
        return {"mid": mid, "spread_ticks": spread_ticks, "order_flow_imbalance": 0.0,
                "depth_imbalance": 0.0, "toxicity": 1.0, "trade_intensity": 0.0,
                "blend": 0.0, "corrupt": True}
    return {
        "mid": mid, "spread_ticks": max(spread_ticks, 0.0),
        "order_flow_imbalance": float(np.clip(ofi, -1.0, 1.0)),
        "depth_imbalance": float(np.clip(depth_imbalance, -1.0, 1.0)),
        "toxicity": float(np.clip(toxicity, 0.0, 1.0)),
        "trade_intensity": float(np.clip(trade_intensity, 0.0, 1.0)),
        "blend": float(np.clip(blend, -1.0, 1.0)), "corrupt": False,
    }


def _bits(x) -> bytes:
    """Raw IEEE-754 bytes so NaN compares equal to NaN and every ULP counts."""
    return struct.pack("<d", float(x))


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


def test_raw_signals_pins_exact_values():
    # Golden values guard the numpy-first kernel: any drift in the arithmetic
    # (summation order, clipping, corrupt guard) trips these exact assertions.
    cfg = MicrostructureConfig()
    expected = {
        "toxic": {"mid": 10241.0, "spread_ticks": 2.0,
                  "order_flow_imbalance": 0.9961463240574414,
                  "depth_imbalance": -0.06768078764862867,
                  "toxicity": 0.4004591020790868,
                  "trade_intensity": 0.10009740563654379,
                  "blend": 0.5706154793750133, "corrupt": False},
        "range": {"mid": 10000.0, "spread_ticks": 2.0,
                  "order_flow_imbalance": -0.3221317428020152,
                  "depth_imbalance": 0.0011284154476404445,
                  "toxicity": 0.0, "trade_intensity": 0.05506640613514391,
                  "blend": -0.19282767950215293, "corrupt": False},
    }
    for scenario, want in expected.items():
        sig = raw_signals(_window(scenario).tail(64), cfg)
        assert sig.keys() == want.keys()
        for key, value in want.items():
            if isinstance(value, bool):
                assert sig[key] is value, (scenario, key)
            else:
                assert _bits(sig[key]) == _bits(value), (scenario, key)  # bit-exact


def test_raw_signals_is_bit_identical_to_scalar_reference():
    # The numpy-first path must reproduce the pandas per-column path byte-for-byte
    # over a full trailing-window percept stream, corrupt rows included.
    cfg = MicrostructureConfig()
    for scenario in ("toxic", "calm", "range"):
        df = generate(n_steps=1200, seed=7, scenario=scenario)
        for i in range(len(df)):
            win = df.iloc[max(0, i - 63):i + 1]
            fast, ref = raw_signals(win, cfg), _raw_signals_scalar(win, cfg)
            assert fast.keys() == ref.keys()
            for key in ref:
                if isinstance(ref[key], bool):
                    assert fast[key] is ref[key], (scenario, i, key)
                else:
                    assert _bits(fast[key]) == _bits(ref[key]), (scenario, i, key)
