"""The MCTS ring-variant benchmark (Part IV Milestone 1) builds and runs, and
selects a champion. Skips if the native benchmark has not been built."""
from __future__ import annotations

import os
import subprocess

import pytest

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "build", "bench_rings")

pytestmark = pytest.mark.skipif(
    not os.path.exists(_BIN), reason="bench_rings not built (run scripts/build_cpp.sh)")


def test_ring_bench_runs_and_picks_champion():
    out = subprocess.run([_BIN, "200000"], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "CHAMPION" in out.stdout
    # all four synthesised variants reported
    assert out.stdout.count("M items/s") >= 4
