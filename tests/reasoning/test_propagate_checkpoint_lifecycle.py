"""propagate()'s manual checkpointer enter/exit must never leak the sqlite3
connection when compile()/checkpoint_step() raises AFTER the saver's __enter__.

The setup (get_checkpointer().__enter__ + compile + checkpoint_step) runs inside
the try whose finally drives __exit__, so a failing step still closes the
connection instead of orphaning it.
"""
import sqlite3
import tempfile
import types

import pytest

from kairos.reasoning.graph import trading_graph
from kairos.reasoning.graph.checkpointer import get_checkpointer
from kairos.reasoning.graph.trading_graph import TradingAgentsGraph


class _RecordingCtx:
    """Wraps the real get_checkpointer CM to record whether __exit__ ran."""

    def __init__(self, cm):
        self._cm = cm
        self.saver = None
        self.exited = False

    def __enter__(self):
        self.saver = self._cm.__enter__()
        return self.saver

    def __exit__(self, *exc):
        self.exited = True
        return self._cm.__exit__(*exc)


def _make_stub(tmpdir, ctx_holder):
    """Minimal object exposing only what propagate() touches."""
    stub = types.SimpleNamespace()
    stub.ticker = None
    stub._checkpointer_ctx = None
    stub.config = {"data_cache_dir": tmpdir, "checkpoint_enabled": True}
    stub._resolve_pending_entries = lambda company: None
    # compile() must succeed so the failure comes from checkpoint_step (after
    # __enter__ opened the connection); either raising is covered by the fix.
    stub.workflow = types.SimpleNamespace(compile=lambda **kw: object())

    def _get_checkpointer(data_dir, ticker):
        ctx = _RecordingCtx(get_checkpointer(data_dir, ticker))
        ctx_holder.append(ctx)
        return ctx

    stub._get_checkpointer_factory = _get_checkpointer
    return stub


@pytest.mark.unit
def test_failing_checkpoint_step_still_closes_saver(monkeypatch):
    """checkpoint_step raising after __enter__ must not leak the connection."""
    tmpdir = tempfile.mkdtemp()
    ctx_holder: list[_RecordingCtx] = []
    stub = _make_stub(tmpdir, ctx_holder)

    monkeypatch.setattr(
        trading_graph, "get_checkpointer", stub._get_checkpointer_factory
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("corrupt/locked checkpoint DB")

    monkeypatch.setattr(trading_graph, "checkpoint_step", _boom)

    with pytest.raises(RuntimeError, match="corrupt/locked checkpoint DB"):
        TradingAgentsGraph.propagate(stub, "TEST", "2026-04-20")

    # The saver was opened, then the finally tore it down: __exit__ ran and the
    # sqlite3 connection is closed (not orphaned), and the ctx handle is cleared.
    assert len(ctx_holder) == 1
    ctx = ctx_holder[0]
    assert ctx.exited, "__exit__ never ran — the sqlite3 connection leaked"
    with pytest.raises(sqlite3.ProgrammingError):
        # Operating on a closed connection raises ProgrammingError.
        ctx.saver.conn.execute("SELECT 1")
    assert stub._checkpointer_ctx is None


@pytest.mark.unit
def test_failing_compile_still_closes_saver(monkeypatch):
    """compile() raising after __enter__ must not leak the connection either."""
    tmpdir = tempfile.mkdtemp()
    ctx_holder: list[_RecordingCtx] = []
    stub = _make_stub(tmpdir, ctx_holder)

    def _boom_compile(**kwargs):
        if "checkpointer" in kwargs:
            raise RuntimeError("cannot compile with checkpointer")
        return object()

    stub.workflow = types.SimpleNamespace(compile=_boom_compile)

    monkeypatch.setattr(
        trading_graph, "get_checkpointer", stub._get_checkpointer_factory
    )

    with pytest.raises(RuntimeError, match="cannot compile with checkpointer"):
        TradingAgentsGraph.propagate(stub, "TEST", "2026-04-20")

    assert len(ctx_holder) == 1
    ctx = ctx_holder[0]
    assert ctx.exited, "__exit__ never ran — the sqlite3 connection leaked"
    with pytest.raises(sqlite3.ProgrammingError):
        ctx.saver.conn.execute("SELECT 1")
    assert stub._checkpointer_ctx is None
