"""Tests for the web dashboard data builder (Component: web UI). The dashboard is
a monitor (no orders); these check the engine produces a valid bundle."""
from __future__ import annotations

import json

from kairos.perception.web.build import WEB_DIR, build_dashboard_data, write_bundle


def test_index_html_ships_with_the_package():
    assert (WEB_DIR / "index.html").exists()


def test_dashboard_data_structure():
    d = build_dashboard_data(n=60, seed=4)
    assert d["snapshots"]
    s0 = d["snapshots"][0]
    for key in ("mid", "bpx", "bsz", "apx", "asz", "cxl", "tf", "rt", "rp", "pnl", "inv"):
        assert key in s0
    assert len(s0["bsz"]) == 10 and len(s0["apx"]) == 10
    assert set(d["regime_names"]) == {"0", "1", "2"}
    assert "bridge_read_ns" in d["metrics"]


def test_write_bundle_creates_servable_files(tmp_path):
    out = write_bundle(tmp_path, n=40, seed=4)
    assert (out / "index.html").exists()
    data = json.loads((out / "dashboard_data.json").read_text())
    assert len(data["snapshots"]) == 40
