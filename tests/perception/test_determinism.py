"""Training must be deterministic so the headline metrics are reproducible.

Guards the `mx.random.seed(seed)` fix (cycle #24): the numpy RNG alone is not
enough — MLX weight initialisation must be seeded too, or a fresh train can
converge to a different (sometimes sub-target) result. Removing the MLX seed would
make this test fail in the fast gate, not just in the slow `make reproduce`.
"""
from __future__ import annotations

import numpy as np
import pytest

from kairos.perception.models.embedder import embed, train
from kairos.perception.schema import featurize
from kairos.perception.synthetic.generate import generate

# Training is the MLX (Apple-Silicon) path; skip cleanly where MLX is absent.
pytest.importorskip("mlx.core")
pytestmark = pytest.mark.mlx


def test_training_is_bit_deterministic():
    df = generate(n_steps=800, seed=3)
    X, _ = featurize(df)
    m1, s1, _ = train(X, epochs=10, seed=0)
    m2, s2, _ = train(X, epochs=10, seed=0)
    z1, z2 = embed(m1, X, s1), embed(m2, X, s2)
    np.testing.assert_array_equal(z1, z2)   # bit-identical -> fully reproducible
