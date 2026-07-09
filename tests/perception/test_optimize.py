"""Test for the Avellaneda–Stoikov parameter grid search (offline, no orders)."""
from __future__ import annotations

import math

from kairos.perception.execution.optimize import grid_search
from kairos.perception.synthetic.generate import generate


def test_grid_search_returns_valid_best():
    df = generate(n_steps=900, seed=8)
    r = grid_search(df, gammas=(0.05, 0.2), ks=(0.5, 1.5), min_fills=10)
    assert r["n_configs"] == 4
    assert len(r["grid"]) == 4
    for row in r["grid"]:
        assert row["gamma"] in (0.05, 0.2) and row["k"] in (0.5, 1.5)
        assert math.isfinite(row["pnl"]) and row["fills"] >= 0
    # best is the max-PnL config in the grid
    assert r["best"]["pnl"] == max(g["pnl"] for g in r["grid"])
    # wider quotes (k=0.5) trade no more than tighter ones (k=1.5) at fixed gamma
    by = {(g["gamma"], g["k"]): g for g in r["grid"]}
    assert by[(0.2, 0.5)]["fills"] <= by[(0.2, 1.5)]["fills"]
