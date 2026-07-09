#!/usr/bin/env python3
"""Quick synthetic-pipeline smoke test (Regression Gate, Part V §3).

Generates a short LOB stream and asserts the core invariants without needing
MLX, so it can run anywhere as a fast pre-commit check. Exits non-zero on
failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "python"))

import numpy as np  # noqa: E402

from kairos.perception.schema import FEATURE_DIM, Regime, block_slices, featurize  # noqa: E402
from kairos.perception.synthetic.generate import generate  # noqa: E402


def main() -> int:
    df = generate(n_steps=1200, seed=11)
    regimes = set(df["regime"].unique())
    assert regimes == {int(Regime.RANGE), int(Regime.TREND), int(Regime.TOXIC)}, regimes

    X, names = featurize(df)
    assert X.shape == (len(df), FEATURE_DIM), X.shape
    assert np.isfinite(X).all(), "non-finite features"
    assert "regime" not in names, "Rule 4: label leaked into features"

    sl = block_slices()
    cxl = X[:, sl["bid_cxl"]].sum(1) + X[:, sl["ask_cxl"]].sum(1)
    reg = df["regime"].to_numpy()
    toxic = cxl[reg == int(Regime.TOXIC)].mean()
    rang = cxl[reg == int(Regime.RANGE)].mean()
    assert toxic > 5 * (rang + 1e-6), f"spoofing cancel-flow not dominant: {toxic} vs {rang}"

    print(f"✓ synthetic smoke OK — {len(df):,} snapshots, "
          f"regimes={sorted(regimes)}, toxic/range cancel-flow ratio={toxic/(rang+1e-6):.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
