"""Out-of-sample generalisation test for the regime embedder.

Skips if no trained encoder is present (artifacts are gitignored / regenerable via
`lob_core --mode synthetic`). When present, it freezes the encoder and checks the
embedding still separates regimes on a FRESH unseen synthetic stream — i.e. it
generalises rather than memorising the in-sample report's training snapshots.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_W = Path("artifacts/lob_encoder.safetensors")
_L = Path("artifacts/latents.npz")

pytestmark = pytest.mark.skipif(
    not (_W.exists() and _L.exists()),
    reason="no trained encoder (run `lob_core --mode synthetic` first)")


def test_embedding_generalises_out_of_sample():
    from kairos.perception.regime.evaluate import evaluate_oos
    r = evaluate_oos(seed=99, n_steps=2000)
    assert r["oos_toxic_vs_trend_separated"], "spoofing vs trend not separated out-of-sample"
    assert r["oos_adjusted_rand_index"] >= 0.7, r
    assert r["oos_linear_separability_5cv"] >= 0.9, r
