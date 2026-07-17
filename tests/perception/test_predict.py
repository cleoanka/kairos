"""Test the live regime predictor (Component D/F).

Skips if no fitted regime model is present (artifacts are gitignored / regenerable
via `lob_core --mode synthetic`). When present, it checks the frozen predictor
labels a FRESH unseen stream accurately — the live inference API the system uses
to label new books in real time.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

_W = Path("artifacts/lob_encoder.safetensors")
_RM = Path("artifacts/regime_model.npz")


def _write_safetensors(path: Path, run_id: str | None) -> None:
    """Minimal dependency-free ``.safetensors`` writer (one tiny tensor, optional
    ``__metadata__.run_id``) so the provenance test needs no MLX to stamp weights."""
    data = np.zeros(1, dtype=np.float32).tobytes()
    header: dict = {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, len(data)]}}
    if run_id is not None:
        header["__metadata__"] = {"run_id": run_id}
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + data)

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


def test_reject_mixed_provenance_includes_encoder_weights(tmp_path):
    # real.py swaps the encoder .safetensors FIRST, so a crash before the latents
    # swap leaves a NEW encoder beside OLD (but mutually-agreeing) npz files. The
    # two-npz guard passes that mixed set; stamping run_id into the safetensors
    # metadata lets the loader reject it.
    from kairos.perception.regime.predict import MixedModelError, _reject_mixed_provenance

    lat = tmp_path / "latents.npz"
    np.savez(lat, mu=np.zeros(2), sd=np.ones(2), run_id="RUN_A")
    np.savez(tmp_path / "rm.npz", z_mean=np.zeros(2), run_id="RUN_A")
    rm = np.load(tmp_path / "rm.npz")

    # Encoder from the SAME run -> accepted (no over-fix: matching stamps pass).
    w_ok = tmp_path / "enc_ok.safetensors"
    _write_safetensors(w_ok, "RUN_A")
    _reject_mixed_provenance(str(lat), rm, str(w_ok))

    # Encoder from a DIFFERENT run (crash committed new weights) -> rejected, even
    # though the two npz files still agree — this is the specific gap kai8-01 closes.
    w_bad = tmp_path / "enc_bad.safetensors"
    _write_safetensors(w_bad, "RUN_B")
    with pytest.raises(MixedModelError, match="mixed-provenance"):
        _reject_mixed_provenance(str(lat), rm, str(w_bad))

    # Legacy encoder with no stamp -> accepted (backward-compatible, no under-fix
    # panic on old artifacts; nothing to compare).
    w_legacy = tmp_path / "enc_legacy.safetensors"
    _write_safetensors(w_legacy, None)
    _reject_mixed_provenance(str(lat), rm, str(w_legacy))

    # Weights arg omitted entirely -> behaves exactly like the two-npz guard.
    _reject_mixed_provenance(str(lat), rm)


def test_read_weights_run_id_roundtrips_and_tolerates_legacy(tmp_path):
    from kairos.perception.models.embedder import read_weights_run_id

    stamped = tmp_path / "stamped.safetensors"
    _write_safetensors(stamped, "RUN_XYZ")
    assert read_weights_run_id(str(stamped)) == "RUN_XYZ"

    legacy = tmp_path / "legacy.safetensors"
    _write_safetensors(legacy, None)
    assert read_weights_run_id(str(legacy)) is None


def test_latents_ts_stored_as_float64(tmp_path):
    """kai8-06: epoch timestamps must survive with sub-second resolution. float32
    has a 128s ULP near a modern epoch, so two snapshots <128s apart collapse to
    an identical stored ts; float64 keeps them distinct."""
    pytest.importorskip("mlx.core")  # embedder.main trains the encoder (MLX)
    import kairos.perception.models.embedder as embedder
    from kairos.perception.schema import column_names
    from kairos.perception.synthetic.generate import generate

    df = generate(n_steps=400, seed=3)
    # Retime onto a modern epoch, 1s apart: float32 would quantise these away.
    base = 1_700_000_000.0
    df["ts"] = base + np.arange(len(df), dtype=np.float64)
    data = tmp_path / "synthetic.parquet"
    df.loc[:, column_names()].to_parquet(data, engine="pyarrow", index=False)

    latents = tmp_path / "latents.npz"
    weights = tmp_path / "lob_encoder.safetensors"
    assert embedder.main([
        "--data", str(data), "--epochs", "2",
        "--latents", str(latents), "--weights", str(weights),
    ]) == 0

    npz = np.load(latents)
    assert npz["ts"].dtype == np.float64
    # All 400 one-second-apart timestamps are preserved distinctly (float32 would
    # collapse this window to ~3 unique values).
    assert len(np.unique(npz["ts"])) == len(df)
    np.testing.assert_array_equal(npz["ts"], df["ts"].to_numpy(np.float64))
    # The encoder weights are stamped with the same run_id as the latents.
    assert embedder.read_weights_run_id(str(weights)) == str(npz["run_id"])
