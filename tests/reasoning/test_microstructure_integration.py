"""Kairos bridge ↔ TradingAgents graph integration (needs the [reasoning] extra).

Proves the Microstructure Analyst is wired into the LangGraph correctly, that its
tools read the causal perception bus point-in-time, and that the report reaches
the downstream researchers/trader.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("langgraph")

from kairos.bridge import build_causal_bus
from kairos.bridge.microstructure_tools import (
    clear_perception_bus,
    get_microstructure_regime,
    get_order_flow_state,
    set_perception_bus,
)
from kairos.perception.synthetic.generate import generate


@pytest.fixture
def bus():
    df = generate(n_steps=1500, seed=4, scenario="toxic")
    b = build_causal_bus(df, "BTCUSDT", window=64, step=8)
    set_perception_bus(b, symbol="BTCUSDT")
    yield b
    clear_perception_bus()


def test_graph_compiles_with_microstructure_analyst():
    client = MagicMock()
    client.get_llm.return_value = MagicMock()
    with patch("kairos.reasoning.graph.trading_graph.create_llm_client", return_value=client):
        from kairos.reasoning.graph.trading_graph import TradingAgentsGraph
        ta = TradingAgentsGraph(selected_analysts=("microstructure", "market"))
    assert "microstructure" in ta.tool_nodes
    nodes = set(ta.graph.get_graph().nodes)
    for n in ("Microstructure Analyst", "tools_microstructure", "Msg Clear Microstructure"):
        assert n in nodes, f"missing graph node {n!r}"


def test_microstructure_tool_reads_bus_causally(bus):
    # A numeric (index-clock) query returns a real percept whose ts <= cutoff.
    out = get_microstructure_regime.invoke({"symbol": "BTCUSDT", "curr_date": "700.0"})
    assert "MICROSTRUCTURE PERCEPT [BTCUSDT" in out
    assert "Regime" in out
    # A cutoff before any data yields the explicit unavailable message, never a leak.
    early = get_microstructure_regime.invoke({"symbol": "BTCUSDT", "curr_date": "-1"})
    assert "unavailable" in early.lower()


def test_order_flow_tool_summarizes_recent_window(bus):
    out = get_order_flow_state.invoke(
        {"symbol": "BTCUSDT", "curr_date": "1000.0", "lookback": 500.0}
    )
    assert "ORDER-FLOW STATE" in out
    assert "Dominant regime" in out


def test_order_flow_labels_horizon_in_bus_clock_unit(bus):
    # The integration bus is index-clocked (ts = step index), so "seconds" are
    # meaningless: the header must state the unit as steps, never a false "…s".
    assert bus.clock == "index"
    out = get_order_flow_state.invoke(
        {"symbol": "BTCUSDT", "curr_date": "1000.0", "lookback": 40.0}
    )
    assert "last 40steps]" in out
    assert "40s]" not in out


def test_order_flow_default_horizon_is_recent_on_index_clock(bus):
    # Regression for the silent full-history window: on an index clock the default
    # horizon must scope a *recent* slice, not sweep every percept at-or-before the
    # cutoff (which is what a 3600-"second" horizon did on index steps).
    default = get_order_flow_state.invoke({"symbol": "BTCUSDT", "curr_date": "1000.0"})
    whole = get_order_flow_state.invoke(
        {"symbol": "BTCUSDT", "curr_date": "1000.0", "lookback": 1e9}
    )
    def observed(text):
        line = next(ln for ln in text.splitlines() if "Percepts observed" in ln)
        return int(line.split(":")[1].split("(")[0])
    n_default, n_whole = observed(default), observed(whole)
    assert 0 < n_default < n_whole, (n_default, n_whole)


@pytest.mark.parametrize("bad_horizon", [float("inf"), float("nan"), 0.0, -10.0])
def test_order_flow_fails_closed_on_bad_horizon(bus, bad_horizon):
    # A non-finite or non-positive lookback can't scope a window (inf spans the
    # whole history, ≤0 empties it): fail closed to "unavailable", never silently
    # mis-scope the rolling read the analyst reasons over.
    out = get_order_flow_state.invoke(
        {"symbol": "BTCUSDT", "curr_date": "1000.0", "lookback": bad_horizon}
    )
    assert "unavailable" in out.lower()


def test_clear_perception_bus_is_identity_safe():
    # Two overlapping same-symbol runs: run A's clear must not evict run B's live
    # bus. clear_perception_bus only pops the exact object it was handed.
    from kairos.bridge.microstructure_tools import _bus_for, clear_perception_bus

    clear_perception_bus()
    df = generate(n_steps=400, seed=1, scenario="toxic")
    bus_a = build_causal_bus(df, "BTCUSDT", window=64, step=8)
    bus_b = build_causal_bus(df, "BTCUSDT", window=64, step=8)
    set_perception_bus(bus_a, symbol="BTCUSDT")
    set_perception_bus(bus_b, symbol="BTCUSDT")  # B is now the live bus
    clear_perception_bus("BTCUSDT", bus_a)       # A's stale finally
    assert _bus_for("BTCUSDT") is bus_b, "B's live bus was clobbered by A's clear"
    clear_perception_bus("BTCUSDT", bus_b)
    assert _bus_for("BTCUSDT") is None
    clear_perception_bus()


def test_propagate_finally_honors_bus_identity_guard():
    # The wired-up guard: a run's real propagate() finally must pass the bus it
    # registered to clear_perception_bus, so it can never evict a *concurrent*
    # same-symbol run's live bus. The isolated helper test above already proves the
    # guard; this drives the actual production finally path the call site uses.
    from kairos.bridge.microstructure_tools import _bus_for, clear_perception_bus

    clear_perception_bus()
    df = generate(n_steps=400, seed=1, scenario="toxic")
    bus_a = build_causal_bus(df, "BTCUSDT", window=64, step=8)
    bus_b = build_causal_bus(df, "BTCUSDT", window=64, step=8)

    client = MagicMock()
    client.get_llm.return_value = MagicMock()
    with patch("kairos.reasoning.graph.trading_graph.create_llm_client", return_value=client):
        from kairos.reasoning.graph.trading_graph import TradingAgentsGraph

        ta = TradingAgentsGraph(selected_analysts=("microstructure", "market"))

    # Run A propagates with bus_a but, mid-flight, run B registers bus_b as the
    # live bus for the same symbol. When A's _run_graph returns, A's finally fires.
    def _hand_off_to_run_b(*_args, **_kwargs):
        set_perception_bus(bus_b, symbol="BTCUSDT")  # run B is now the live bus
        return ({}, "HOLD")

    with (
        patch.object(ta, "_run_graph", side_effect=_hand_off_to_run_b),
        patch.object(ta, "_resolve_pending_entries"),
    ):
        ta.propagate("BTCUSDT", "2026-01-10", asset_type="crypto", perception_bus=bus_a)

    # With bus=perception_bus wired at the call site, A's finally only pops its own
    # bus — B's live bus survives and its Microstructure Analyst keeps its percepts.
    assert _bus_for("BTCUSDT") is bus_b, "run A's finally clobbered run B's live bus"
    clear_perception_bus()


def test_tool_without_registered_bus_is_safe():
    clear_perception_bus()
    out = get_microstructure_regime.invoke({"symbol": "ZZZZ", "curr_date": "1.0"})
    assert "unavailable" in out.lower()


@pytest.mark.parametrize("bad_date", ["nan", "inf", "-inf", "not-a-date"])
def test_tool_fails_closed_on_unresolvable_cutoff(bus, bad_date):
    # An LLM (or a corrupt date field) handing "nan"/"inf"/garbage as curr_date
    # must NOT leak the newest percept and must NOT raise into the agent — it
    # degrades to the explicit unavailable message.
    out = get_microstructure_regime.invoke({"symbol": "BTCUSDT", "curr_date": bad_date})
    assert "unavailable" in out.lower()
    agg = get_order_flow_state.invoke({"symbol": "BTCUSDT", "curr_date": bad_date})
    assert "unavailable" in agg.lower()


def test_initial_state_seeds_microstructure_report():
    from kairos.reasoning.graph.propagation import Propagator
    state = Propagator().create_initial_state("BTCUSDT", "700.0", asset_type="crypto")
    assert state["microstructure_report"] == ""


def test_researchers_consume_microstructure_report():
    # The bull/bear prompts must reference the microstructure report so System-1
    # actually reaches the debate (regression guard for the integration).
    from pathlib import Path
    agents = Path(__file__).resolve().parents[2] / "src" / "kairos" / "reasoning" / "agents"
    for rel in ("researchers/bull_researcher.py", "researchers/bear_researcher.py", "trader/trader.py"):
        src = (agents / rel).read_text()
        assert "microstructure_report" in src, f"{rel} does not consume microstructure_report"
