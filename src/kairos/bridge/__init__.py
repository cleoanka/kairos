"""The Kairos bridge — the causal seam between System-1 and System-2.

This package's *core* (percept, causal bus, microstructure computation,
execution link) has no LLM/langchain dependency and runs everywhere. The
LangChain-facing pieces (``microstructure_tools``, ``microstructure_analyst``)
live in submodules imported only by the reasoning wiring, so ``import
kairos.bridge`` stays lightweight.
"""
from __future__ import annotations

from .causal_bus import (
    CausalPerceptionBus,
    ClockDomainError,
    LookAheadError,
    build_causal_bus,
    resolve_cutoff,
    to_epoch,
)
from .execution_link import (
    Decision,
    DirectionalMaker,
    ExecutionLink,
    parse_decision,
    perceive_regimes,
)
from .microstructure import (
    LearnedRegime,
    MicrostructureConfig,
    heuristic_regime,
    percept_from_window,
    raw_signals,
)
from .percept import BEAR, BULL, NEUTRAL, Percept

__all__ = [
    "Percept", "BULL", "BEAR", "NEUTRAL",
    "CausalPerceptionBus", "LookAheadError", "ClockDomainError",
    "build_causal_bus", "to_epoch", "resolve_cutoff",
    "MicrostructureConfig", "raw_signals", "heuristic_regime", "percept_from_window",
    "LearnedRegime",
    "Decision", "parse_decision", "ExecutionLink", "DirectionalMaker", "perceive_regimes",
]
