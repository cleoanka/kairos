"""Invariant tests for the data contract (schema.py) — the foundation every other
module depends on. These guard against *silent* regressions: e.g. a typo in
mirror_spec would quietly break the VICReg direction-invariance (and degrade
clustering) without any other test noticing."""
from __future__ import annotations

import numpy as np

from kairos.perception.schema import (
    COLUMN_INDEX,
    FEATURE_DIM,
    RAW_SNAPSHOT_DOUBLES,
    block_slices,
    column_names,
    feature_weights,
    featurize,
    level_column_groups,
    mirror_spec,
)
from kairos.perception.synthetic.generate import generate


def test_mirror_is_an_involution():
    """Applying the side-swap twice must be the identity — VICReg's invariance
    term relies on mirror(mirror(x)) == x."""
    idx, sign = mirror_spec()
    e = np.arange(FEATURE_DIM)
    assert (e[idx][idx] == e).all()          # index permutation is its own inverse
    assert (sign[idx] * sign == 1).all()     # signs cancel on the second application


def test_mirror_swaps_sides_and_negates_signed_trade():
    idx, sign = mirror_spec()
    sl = block_slices()
    for a, b in (("bid_px", "ask_px"), ("bid_sz", "ask_sz"), ("bid_cxl", "ask_cxl")):
        assert (idx[sl[a]] == np.arange(sl[b].start, sl[b].stop)).all()
    assert sign[sl["trade"].start] == -1.0   # signed trade flow negated
    # the trade magnitude features are NOT negated
    assert sign[sl["trade"].start + 1] == 1.0 and sign[sl["trade"].start + 2] == 1.0


def test_column_index_bijective_and_width():
    assert RAW_SNAPSHOT_DOUBLES == len(column_names()) == 66
    assert sorted(COLUMN_INDEX.values()) == list(range(66))
    assert len(set(COLUMN_INDEX)) == 66


def test_feature_weights_shape_and_positive():
    w = feature_weights()
    assert w.shape == (FEATURE_DIM,) and (w > 0).all()


def test_level_groups_cover_depth_and_cancel_columns():
    flat = [c for g in level_column_groups() for c in g]
    assert len(flat) == 60 == len(set(flat))   # px+sz+cxl, both sides, 10 levels
    assert max(flat) < FEATURE_DIM


def test_featurize_excludes_regime_and_is_finite():
    df = generate(n_steps=200, seed=2)
    X, names = featurize(df)
    assert X.shape == (len(df), FEATURE_DIM)
    assert "regime" not in names and np.isfinite(X).all()   # Rule 4 + sanity


def test_mirror_applied_to_features_matches_spec_on_random_input():
    """A reflected feature row equals the row with bid/ask blocks swapped and the
    signed-trade feature negated — checked directly on random finite input."""
    idx, sign = mirror_spec()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, FEATURE_DIM)).astype(np.float32)
    sl = block_slices()
    xm = x[:, idx] * sign
    assert np.allclose(xm[:, sl["bid_px"]], x[:, sl["ask_px"]])
    assert np.allclose(xm[:, sl["trade"].start], -x[:, sl["trade"].start])
