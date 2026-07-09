"""Native end-to-end zero-copy pipeline test (Milestone #2).

    generator snapshot -> C++ ring.push -> zero-copy latest_view()
        -> schema.featurize_from_raw() -> MLX encode

Asserts the shared view byte-matches the pushed rows (true zero-copy, no copy,
no corruption) and that featurizing the view is identical to featurizing the
source DataFrame — so the native and offline paths are interchangeable. Skips
cleanly if the C++ bridge has not been built (`scripts/build_cpp.sh`).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "build"), os.path.join(_ROOT, "src", "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import lob_bridge as L  # type: ignore
    _IMPORT_ERR = None
except Exception as exc:  # pragma: no cover - only on an unbuilt bridge
    L = None
    _IMPORT_ERR = exc

from kairos.perception.schema import (  # noqa: E402
    RAW_SNAPSHOT_DOUBLES,
    column_names,
    featurize,
    featurize_from_raw,
)
from kairos.perception.synthetic.generate import generate  # noqa: E402

pytestmark = pytest.mark.skipif(
    L is None, reason=f"lob_bridge not built (run scripts/build_cpp.sh): {_IMPORT_ERR!r}")


def _raw_rows(n: int):
    df = generate(n_steps=n, seed=21)
    raw = df.loc[:, column_names()].to_numpy(dtype=np.float64)
    return df, raw


def test_bridge_contract_width_matches_schema():
    """The C++ POD width must equal the Python contract width, or the zero-copy
    view would misalign the columns."""
    assert L.N_DOUBLES == RAW_SNAPSHOT_DOUBLES == 66


def test_native_zero_copy_featurize_matches_offline():
    df, raw = _raw_rows(96)
    cap = 256
    b = L.create(f"aura_e2e_{os.getpid()}", cap)
    try:
        view = np.asarray(b.latest_view())
        assert view.dtype == np.float64
        assert view.shape == (cap, RAW_SNAPSHOT_DOUBLES)
        assert view.ctypes.data == b.base_addr(), "latest_view() is not a zero-copy alias"

        for i in range(len(raw)):
            assert b.push_snapshot(raw[i]), "ring rejected a push under capacity"

        # The shared view must byte-match the pushed rows (no copy / no torn write).
        np.testing.assert_array_equal(view[:len(raw)], raw)

        # Featurizing the zero-copy view == featurizing the source DataFrame.
        X_view, _ = featurize_from_raw(view[:len(raw)])
        X_df, _ = featurize(df)
        np.testing.assert_array_equal(X_view, X_df)
    finally:
        b.detach()


def test_native_view_feeds_mlx_encoder():
    """The featurized zero-copy view drives the MLX encoder (full chain runs)."""
    mx = pytest.importorskip("mlx.core")
    from kairos.perception.models.embedder import LobAutoEncoder

    df, raw = _raw_rows(48)
    b = L.create(f"aura_mlx_{os.getpid()}", 64)
    try:
        view = np.asarray(b.latest_view())
        for i in range(len(raw)):
            b.push_snapshot(raw[i])
        X, _ = featurize_from_raw(view[:len(raw)])
        enc = LobAutoEncoder()
        z = enc.encode(mx.array(X))
        mx.eval(z)
        assert tuple(z.shape) == (len(raw), 16)
        assert bool(np.isfinite(np.array(z)).all())
    finally:
        b.detach()
