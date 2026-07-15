"""`kairos reason` and the source-only audit commands fail cleanly, never a raw error.

Three CLI-completeness contracts: (a) `reason` validates its date up front, so a
malformed or future date exits 2 with a clear message before any LLM/API-key
setup — never an unrelated 'API key not set' error; (b) `reason --json` mirrors
`loop --json`, emitting the decision as machine-readable JSON; (c) `soul-check`
and `reproduce` shell out to scripts/ that are absent from an installed wheel, so
a missing script prints an actionable 'source checkout only' message and exits 2
rather than a raw file-not-found from subprocess.
"""
from __future__ import annotations

import datetime as _dt
import json

import pytest

from kairos import cli


# --- (a) date is validated at parse time, before any reasoning setup ---
@pytest.mark.parametrize("bad", ["2026-13-99", "not-a-date", "2026/07/15", "20260715"])
def test_reason_bad_date_rejected(capsys, bad):
    """A malformed date exits 2 with a clear message, before LLM/API-key setup."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["reason", "AAPL", bad])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "invalid date" in err or "YYYY-MM-DD" in err
    assert "Traceback" not in err


def test_reason_future_date_rejected(capsys):
    """A date in the future is rejected up front, matching the interactive path."""
    future = (_dt.date.today() + _dt.timedelta(days=365)).isoformat()
    with pytest.raises(SystemExit) as exc:
        cli.main(["reason", "AAPL", future])
    assert exc.value.code == 2
    assert "future" in capsys.readouterr().err


def test_valid_date_passthrough():
    """A well-formed, non-future date passes the validator unchanged."""
    assert cli._valid_date("2020-01-02") == "2020-01-02"


# --- (b) --json mirrors loop, without needing any API keys ---
def test_reason_json_output(capsys, monkeypatch):
    """--json emits the decision as JSON; no API keys required (graph is stubbed)."""
    class _FakeGraph:
        def __init__(self, *a, **k):
            pass

        def propagate(self, ticker, date, asset_type="stock"):
            return {}, "BUY"

    monkeypatch.setattr("kairos.reasoning.graph.trading_graph.TradingAgentsGraph", _FakeGraph)
    rc = cli.main(["reason", "AAPL", "2020-01-02", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"ticker": "AAPL", "date": "2020-01-02",
                       "asset_type": "stock", "decision": "BUY"}


def test_reason_plain_output_unchanged(capsys, monkeypatch):
    """Without --json the decision still prints verbatim (no regression)."""
    class _FakeGraph:
        def __init__(self, *a, **k):
            pass

        def propagate(self, ticker, date, asset_type="stock"):
            return {}, "SELL"

    monkeypatch.setattr("kairos.reasoning.graph.trading_graph.TradingAgentsGraph", _FakeGraph)
    rc = cli.main(["reason", "AAPL", "2020-01-02"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "SELL"


# --- (c) soul-check/reproduce degrade cleanly when the script is not packaged ---
@pytest.mark.parametrize("command", ["soul-check", "reproduce"])
def test_source_only_command_missing_script_is_friendly(capsys, monkeypatch, command):
    """A missing scripts/ target (pip-installed wheel) exits 2 with a clear message."""
    monkeypatch.setattr("pathlib.Path.is_file", lambda self: False)
    rc = cli.main([command])
    assert rc == 2
    err = capsys.readouterr().err
    assert "source checkout" in err
    assert command in err
    assert "Traceback" not in err
