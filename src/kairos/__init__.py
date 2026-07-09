"""Kairos — the AURA dual-process trading brain.

A strictly-causal fusion of two cognitive systems:

* **System 1 — Perception** (``kairos.perception``): a self-supervised,
  label-free limit-order-book micro-structure engine (LOB-Core). Fast,
  subsymbolic reading of market regime, order flow and liquidity toxicity.
* **System 2 — Reasoning** (``kairos.reasoning``): a multi-agent LLM trading
  firm (TradingAgents). Slow, symbolic, deliberative debate over a decision.

joined by the **Bridge** (``kairos.bridge``): a Causal Perception Bus that lets
System-2 read System-1 only through a point-in-time cutoff — closing the
look-ahead hole by construction — plus a neuro-symbolic execution link where
System-2 sets the stance and System-1 executes it and can veto it.

    perceive → reason → act → reflect   (``kairos.loop.cognitive_loop``)
"""
from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
