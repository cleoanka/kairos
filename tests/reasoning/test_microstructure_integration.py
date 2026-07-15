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
        {"symbol": "BTCUSDT", "curr_date": "1000.0", "lookback_seconds": 500.0}
    )
    assert "ORDER-FLOW STATE" in out
    assert "Dominant regime" in out


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
