"""Inference hot-path benchmark test: batching must amortise the MLX dispatch
overhead to well under the spec's 0.1 ms target. Skips if no fitted model."""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not (Path("artifacts/regime_model.npz").exists()
         and Path("artifacts/lob_encoder.safetensors").exists()),
    reason="no fitted regime model (run `lob_core --mode synthetic` first)")


def test_batched_inference_amortises_dispatch():
    from kairos.perception.bench import run_bench
    r = run_bench(n=512, seed=5)
    assert r["single_snapshot_us"] > 0
    per_snap = r["batches"][256]["per_snap_us"]
    # batching is far cheaper per snapshot than one-at-a-time …
    assert per_snap < r["single_snapshot_us"]
    # … and beats the 0.1 ms (100 µs) target with large margin.
    assert per_snap < 100.0
