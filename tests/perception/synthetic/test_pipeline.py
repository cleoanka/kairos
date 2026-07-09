"""Fast regression tests for the synthetic LOB pipeline (Component A) and the
data contract. These run without MLX so they stay quick in the regression gate.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from kairos.perception.schema import (
    FEATURE_DIM,
    RAW_SNAPSHOT_DOUBLES,
    Regime,
    column_names,
    featurize,
    featurize_from_raw,
)
from kairos.perception.synthetic.engine import OrderBook
from kairos.perception.synthetic.generate import generate

ROOT = Path(__file__).resolve().parents[3]  # tests/perception/synthetic/ -> repo root


def test_generator_emits_all_three_regimes():
    df = generate(n_steps=900, seed=1)
    assert list(df.columns) == column_names()
    present = set(df["regime"].unique())
    assert present == {int(Regime.RANGE), int(Regime.TREND), int(Regime.TOXIC)}


def test_featurize_shape_and_finite():
    df = generate(n_steps=300, seed=2)
    X, names = featurize(df)
    assert X.shape == (len(df), FEATURE_DIM)
    assert len(names) == FEATURE_DIM
    assert np.isfinite(X).all()
    assert "regime" not in names  # Rule 4: label never leaks into features


def test_orderbook_market_order_consumes_liquidity():
    b = OrderBook(mid_tick=1000)
    b.add_limit("ask", 1001, 10.0)
    b.add_limit("ask", 1002, 10.0)
    filled = b.market_order("buy", 15.0)
    assert filled == 15.0
    assert b.best_ask() == 1002 and b.asks[1002] == 5.0


def test_cancel_flow_is_recorded():
    b = OrderBook(mid_tick=1000)
    b.add_limit("bid", 998, 50.0)
    removed = b.cancel("bid", 998, 30.0)
    assert removed == 30.0
    snap = b.snapshot(ts=0.0, regime=int(Regime.TOXIC))
    assert snap.bid_cxl.sum() == 30.0


def test_toxic_has_far_more_cancel_flow_than_range():
    df = generate(n_steps=1500, seed=3)
    from kairos.perception.schema import block_slices
    X, _ = featurize(df)
    sl = block_slices()
    cxl = X[:, sl["bid_cxl"]].sum(1) + X[:, sl["ask_cxl"]].sum(1)
    toxic = cxl[df["regime"].to_numpy() == int(Regime.TOXIC)].mean()
    rang = cxl[df["regime"].to_numpy() == int(Regime.RANGE)].mean()
    assert toxic > 5 * (rang + 1e-6)  # spoofing must dominate cancel-flow


def test_raw_contract_width_matches_columns():
    assert len(column_names()) == RAW_SNAPSHOT_DOUBLES


def test_featurize_from_raw_matches_dataframe_path():
    """The zero-copy bridge path (raw array) must equal the parquet path exactly,
    so the native and offline pipelines are interchangeable."""
    df = generate(n_steps=400, seed=5)
    raw = df.loc[:, column_names()].to_numpy(dtype=np.float64)  # mimics the (n,66) view
    X_df, _ = featurize(df)
    X_raw, _ = featurize_from_raw(raw)
    assert X_raw.shape == (len(df), FEATURE_DIM)
    np.testing.assert_array_equal(X_df, X_raw)


def test_featurize_from_raw_rejects_wrong_width():
    import pytest
    with pytest.raises(ValueError):
        featurize_from_raw(np.zeros((3, RAW_SNAPSHOT_DOUBLES - 1), dtype=np.float64))


def test_soul_check_passes():
    rc = subprocess.call([sys.executable, str(ROOT / "scripts" / "soul_check.py"), "--quiet"])
    assert rc == 0
