"""Tests for the web dashboard data builder (Component: web UI). The dashboard is
a monitor (no orders); these check the engine produces a valid bundle."""
from __future__ import annotations

import json

import pytest

from kairos.perception.schema import N_LEVELS
from kairos.perception.synthetic.generate import generate
from kairos.perception.web import build as web_build
from kairos.perception.web.build import WEB_DIR, build_dashboard_data, write_bundle


def _reference_snap_fields(df):
    """The original per-row (``df.iloc[i]`` + label-lookup) snapshot fields, kept
    verbatim so the vectorized builder stays bit-identical to it."""
    out = []
    for i in range(len(df)):
        row = df.iloc[i]
        out.append({
            "mid": round(float(row["mid"]), 2),
            "bpx": [round(float(row[f"bid_px_{k}"]), 1) for k in range(N_LEVELS)],
            "bsz": [round(float(row[f"bid_sz_{k}"]), 1) for k in range(N_LEVELS)],
            "apx": [round(float(row[f"ask_px_{k}"]), 1) for k in range(N_LEVELS)],
            "asz": [round(float(row[f"ask_sz_{k}"]), 1) for k in range(N_LEVELS)],
            "cxl": round(sum(float(row[f"bid_cxl_{k}"]) + float(row[f"ask_cxl_{k}"])
                             for k in range(N_LEVELS)), 1),
            "tf": round(float(row["trade_buy"] - row["trade_sell"]), 1),
        })
    return out


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


@pytest.mark.parametrize("port", [99999, 70000, 0, -1])
def test_serve_rejects_out_of_range_port_cleanly(port, monkeypatch, capsys):
    """An out-of-range port must yield the same clean message + non-zero exit as
    a port-in-use, not an uncaught OverflowError — and must not waste the build."""
    def _boom(*a, **k):
        raise AssertionError("write_bundle ran before the port was validated")
    monkeypatch.setattr(web_build, "write_bundle", _boom)
    rc = web_build.serve(port=port, open_browser=False)
    assert rc == 1
    assert "could not start server" in capsys.readouterr().out


@pytest.mark.parametrize("port", [1, 8000, 65535])
def test_serve_accepts_in_range_port(port, monkeypatch):
    """The guard must not reject valid ports: an in-range port passes validation
    and proceeds to build + bind (bind stubbed so no socket is opened)."""
    monkeypatch.setattr(web_build, "write_bundle", lambda *a, **k: None)

    class _FakeServer:
        def __init__(self, *a, **k):
            pass  # a successful "bind"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def serve_forever(self):
            raise KeyboardInterrupt  # stop the loop cleanly → return 0

    monkeypatch.setattr(web_build.socketserver, "TCPServer", _FakeServer)
    assert web_build.serve(port=port, open_browser=False) == 0


def test_snapshots_bit_identical_to_per_row_reference():
    """The vectorized (numpy-first) snapshot fields must match the original
    per-row ``df.iloc`` reference exactly — same values, same JSON."""
    for seed in (0, 4, 17):
        for n in (7, 60, 233):
            d = build_dashboard_data(n=n, seed=seed)
            ref = _reference_snap_fields(generate(n_steps=n, seed=seed))
            got = [{k: s[k] for k in ("mid", "bpx", "bsz", "apx", "asz", "cxl", "tf")}
                   for s in d["snapshots"]]
            assert json.dumps(got) == json.dumps(ref), f"seed={seed} n={n}"
