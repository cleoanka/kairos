"""Test the live regime predictor (Component D/F).

Skips if no fitted regime model is present (artifacts are gitignored / regenerable
via `lob_core --mode synthetic`). When present, it checks the frozen predictor
labels a FRESH unseen stream accurately — the live inference API the system uses
to label new books in real time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_W = Path("artifacts/lob_encoder.safetensors")
_RM = Path("artifacts/regime_model.npz")

pytestmark = pytest.mark.skipif(
    not (_W.exists() and _RM.exists()),
    reason="no fitted regime model (run `lob_core --mode synthetic` first)")


def test_regime_predictor_labels_fresh_stream_accurately():
    from kairos.perception.regime.predict import RegimePredictor
    from kairos.perception.schema import column_names
    from kairos.perception.synthetic.generate import generate

    p = RegimePredictor.load()
    df = generate(n_steps=2000, seed=77)
    pred = p.predict(df)
    true = df["regime"].to_numpy()
    assert pred.shape == true.shape
    assert float((pred == true).mean()) >= 0.85   # accurate live labelling

    # predict_from_raw (the zero-copy live path) matches the DataFrame path.
    raw = df.loc[:, column_names()].to_numpy(np.float64)
    np.testing.assert_array_equal(p.predict_from_raw(raw), pred)


def test_predictor_fail_safe_on_degenerate_features():
    """A corrupt (non-finite) observation must yield the conservative TOXIC label
    (strategy stands aside), never a confident garbage regime."""
    from kairos.perception.regime.predict import RegimePredictor
    from kairos.perception.schema import FEATURE_DIM, Regime

    p = RegimePredictor.load()
    bad = np.array([[np.nan] * FEATURE_DIM, [np.inf] * FEATURE_DIM], dtype=np.float32)
    out = p.predict_features(bad)
    assert (out == int(Regime.TOXIC)).all()


def test_reject_mixed_provenance_run_id(tmp_path):
    # A crash mid-persist can leave new latents beside old centroids; the shared
    # run_id must let the loader reject that mixed model instead of loading it.
    from kairos.perception.regime.predict import MixedModelError, _reject_mixed_provenance

    lat = tmp_path / "latents.npz"
    np.savez(lat, mu=np.zeros(2), sd=np.ones(2), run_id="RUN_A")

    # Same run -> accepted.
    np.savez(tmp_path / "rm_ok.npz", z_mean=np.zeros(2), run_id="RUN_A")
    _reject_mixed_provenance(str(lat), np.load(tmp_path / "rm_ok.npz"))

    # Different run -> rejected (mixed provenance).
    np.savez(tmp_path / "rm_bad.npz", z_mean=np.zeros(2), run_id="RUN_B")
    with pytest.raises(MixedModelError, match="mixed-provenance"):
        _reject_mixed_provenance(str(lat), np.load(tmp_path / "rm_bad.npz"))

    # Legacy artifact without a run_id -> accepted (backward-compatible).
    np.savez(tmp_path / "rm_legacy.npz", z_mean=np.zeros(2))
    _reject_mixed_provenance(str(lat), np.load(tmp_path / "rm_legacy.npz"))
