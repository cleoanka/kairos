"""Tests for the real-market pipeline (signature-based regime naming + the
label-free build). No network: a synthetic stream stands in for recorded Bybit
data (the build is source-agnostic). The output dir is monkeypatched so the real
model under artifacts/real/ is never clobbered."""
from __future__ import annotations

import numpy as np
import pytest

from kairos.perception.regime.naming import name_clusters_by_signature
from kairos.perception.schema import FEATURE_DIM, Regime, block_slices, featurize
from kairos.perception.synthetic.generate import generate


def test_name_clusters_by_signature():
    sl = block_slices()
    pred = np.array([0] * 10 + [1] * 10 + [2] * 10)
    X = np.zeros((30, FEATURE_DIM), dtype=np.float32)
    X[pred == 0, sl["bid_cxl"]] = 1.0              # cluster 0 = most cancel-flow
    X[pred == 1, sl["trade"].start + 1] = 1.0      # cluster 1 = most trade volume
    # cluster 2 = quiet
    c2r = name_clusters_by_signature(X, pred, k=3)
    assert c2r[0] == int(Regime.TOXIC)
    assert c2r[1] == int(Regime.TREND)
    assert c2r[2] == int(Regime.RANGE)


@pytest.mark.mlx
def test_build_real_model_round_trip(tmp_path, monkeypatch):
    pytest.importorskip("mlx.core")  # build_real_model trains the encoder (MLX)
    import kairos.perception.real as real
    monkeypatch.setattr(real, "REAL_DIR", tmp_path)

    df = generate(n_steps=800, seed=2)             # stands in for recorded data
    r = real.build_real_model(df, epochs=10)
    assert r["n"] == len(df)
    assert set(r["regime_dist"]) <= {"RANGE", "TREND", "TOXIC"}
    for f in ("lob_encoder.safetensors", "latents.npz", "regime_model.npz"):
        assert (tmp_path / f).exists()

    # the saved real model labels new books
    from kairos.perception.regime.predict import RegimePredictor
    p = RegimePredictor.load(str(tmp_path / "lob_encoder.safetensors"),
                             str(tmp_path / "latents.npz"),
                             str(tmp_path / "regime_model.npz"))
    X, _ = featurize(df)
    pred = p.predict_features(X)
    assert pred.shape == (len(df),)
    assert {int(v) for v in pred} <= {0, 1, 2}
