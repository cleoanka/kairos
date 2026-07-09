"""The neuro-symbolic execution link — System-2 decides, System-1 executes.

The two systems have complementary authority, and the ordering is deliberate:

* **System 2** (the multi-agent debate) sets the *stance*: BUY / HOLD / SELL and
  a conviction in [0, 1]. It is smart but slow, and it cannot see the book.
* **System 1** (microstructure perception + the market maker) *executes* that
  stance passively, sizing the directional lean by conviction — and retains a
  **hard veto**: in a TOXIC (spoofed / phantom-liquidity) regime it stands
  aside no matter how convinced System-2 is. A reflexive, fast safety reflex
  overrides deliberation, exactly as in a dual-process mind.

Crucially the regime that drives execution is the *perceived* regime computed
causally per step — never the ground-truth label — so this path is a faithful
shadow of live trading. No REST, no network, no real orders (Constitution).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from kairos.perception.execution.risk import RiskGate
from kairos.perception.execution.simulator import run_backtest
from kairos.perception.schema import Regime
from kairos.perception.strategy.avellaneda import AvellanedaStoikovMaker
from kairos.perception.strategy.maker import Quote

from .microstructure import MicrostructureConfig, heuristic_regime, raw_signals

_ACTION_RE = re.compile(r"FINAL\s+TRANSACTION\s+PROPOSAL:\s*\**\s*(BUY|HOLD|SELL)", re.IGNORECASE)
_BARE_RE = re.compile(r"\b(BUY|HOLD|SELL)\b")
# A recommendation-scoped action: the token nearest a conclusion word, so
# "we recommend BUY; do not SELL yet" reads BUY, not the trailing caveat.
_RECO_RE = re.compile(
    r"(?:recommend(?:ation)?|decision|stance|proposal|verdict|conclusion|rating)\W{0,24}?\b(BUY|HOLD|SELL)\b",
    re.IGNORECASE)
# Conviction: capture the scale in named groups (never a fragile substring test),
# and require a rating-like magnitude so a stray "confidence: 4200" can't latch.
_RATING_RE = re.compile(
    r"(?:rating|conviction|confidence)\D{0,14}?(\d+(?:\.\d+)?)\s*"
    r"(?:/\s*(?P<den>5|10|100)|(?P<pct>%)|(?:out\s+of|of)\s+(?P<den2>5|10|100))?",
    re.IGNORECASE)


@dataclass(frozen=True)
class Decision:
    """A System-2 stance handed to System-1 for execution."""
    action: str            # BUY | HOLD | SELL
    conviction: float      # [0, 1]
    rationale: str = ""
    source: str = "system2"

    @property
    def bias(self) -> float:
        """Signed directional lean in [-1, 1]: +conviction to accumulate, - to shed."""
        s = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}.get(self.action.upper(), 0.0)
        return s * float(np.clip(self.conviction, 0.0, 1.0))


def parse_decision(text: str, *, default_conviction: float = 0.6) -> Decision:
    """Extract a :class:`Decision` from a free-text System-2 verdict.

    Prefers the explicit ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` marker
    the TradingAgents pipeline emits; falls back to the last bare BUY/HOLD/SELL
    token. Conviction is read from an optional rating/confidence phrase (``x/5``,
    ``x/10`` or ``x%``), else defaults (0 for HOLD)."""
    text = text or ""
    # Action, in priority order: explicit marker > recommendation-scoped token >
    # last bare token (weakest — prone to trailing caveats).
    m = _ACTION_RE.search(text)
    if m:
        action = m.group(1).upper()
    elif (r := _RECO_RE.search(text)):
        action = r.group(1).upper()
    else:
        hits = _BARE_RE.findall(text.upper())
        action = hits[-1] if hits else "HOLD"

    conviction = 0.0 if action == "HOLD" else default_conviction
    rr = _RATING_RE.search(text)
    if rr and action != "HOLD":
        val = float(rr.group(1))
        den = rr.group("den") or rr.group("den2")
        if den:
            conviction = val / float(den)
        elif rr.group("pct"):
            conviction = val / 100.0
        elif val <= 1.0:
            conviction = val                     # already a [0,1] probability
        elif val <= 5.0:
            conviction = val / 5.0               # bare 1-5 scale
        elif val <= 10.0:
            conviction = val / 10.0              # bare 1-10 scale (very common)
        else:
            conviction = default_conviction      # implausible magnitude — ignore it
    return Decision(action=action, conviction=float(np.clip(conviction, 0.0, 1.0)),
                    rationale=text[:280], source="system2")


class DirectionalMaker(AvellanedaStoikovMaker):
    """Avellaneda–Stoikov maker skewed by a System-2 directional bias.

    The base maker's regime overlay is preserved intact (TOXIC → no quote, TREND
    → reduce-only), so System-1's vetoes always win; the bias only re-sizes the
    two legs when the maker is otherwise willing to quote."""

    def __init__(self, bias: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.bias = float(np.clip(bias, -1.0, 1.0))

    def decide(self, regime: int, best_bid: float, best_ask: float, inventory: float) -> Quote:
        q = super().decide(regime, best_bid, best_ask, inventory)
        # Bias is a RANGE-only directional skew. In TREND (reduce-only) and TOXIC
        # (no quote) System-1's regime overlay governs — a stance lean there would
        # only shrink the inventory-flattening leg, never legitimately add edge.
        if int(regime) == int(Regime.RANGE):
            if q.bid_sz:
                q.bid_sz = max(0.0, q.bid_sz * (1.0 + self.bias))   # BUY: lean into bids
            if q.ask_sz:
                q.ask_sz = max(0.0, q.ask_sz * (1.0 - self.bias))   # SELL: lean into asks
        # Never post more than the remaining inventory headroom, so a single
        # bias-inflated fill cannot overshoot the cap before the risk gate reacts.
        if q.bid_sz:
            q.bid_sz = min(q.bid_sz, max(0.0, self.max_inv - inventory))
        if q.ask_sz:
            q.ask_sz = min(q.ask_sz, max(0.0, self.max_inv + inventory))
        return q


def perceive_regimes(df, *, window: int = 64, regime_backend=None,
                     cfg: MicrostructureConfig | None = None) -> np.ndarray:
    """Per-row *perceived* regime (int), each from a trailing causal window.

    This is what execution acts on — the same System-1 read the reasoning layer
    saw, never the ground-truth ``regime`` column."""
    cfg = cfg or MicrostructureConfig()
    out = np.empty(len(df), dtype=np.int64)
    for i in range(len(df)):
        lo = max(0, i - window + 1)
        win = df.iloc[lo:i + 1]
        if regime_backend is None:
            regime, _, _ = heuristic_regime(raw_signals(win, cfg), cfg)
        else:
            regime, _, _ = regime_backend(win, raw_signals(win, cfg))
        out[i] = int(regime)
    return out


@dataclass
class ExecutionLink:
    """Runs a System-2 decision through System-1 execution over a forward window."""
    base_size: float = 1.0
    gamma: float = 0.02
    k: float = 0.5
    max_inventory: float = 20.0
    max_drawdown: float = 1e9
    window: int = 64

    def execute(self, decision: Decision, forward_df, *, regime_backend=None,
                cfg: MicrostructureConfig | None = None) -> dict:
        """Execute ``decision`` over ``forward_df`` (the book *after* the decision
        time). Returns the backtest report plus neuro-symbolic annotations."""
        perceived = perceive_regimes(forward_df, window=self.window,
                                     regime_backend=regime_backend, cfg=cfg)
        df2 = forward_df.copy()
        df2["regime"] = perceived   # execution acts on the PERCEIVED regime

        maker = DirectionalMaker(bias=decision.bias, base_size=self.base_size,
                                 gamma=self.gamma, k=self.k,
                                 max_inventory=self.max_inventory)
        risk = RiskGate(max_inventory=self.max_inventory, max_drawdown=self.max_drawdown)
        report = run_backtest(df2, strategy=maker, risk=risk)

        toxic_frac = float(np.mean(perceived == int(Regime.TOXIC))) if len(perceived) else 0.0
        report["neuro_symbolic"] = {
            "action": decision.action,
            "conviction": round(decision.conviction, 3),
            "applied_bias": round(decision.bias, 3),
            "toxic_veto_fraction": round(toxic_frac, 3),
            "system1_vetoed": toxic_frac > 0.5,   # majority-toxic window → System-1 dominated
            "decision_source": decision.source,
        }
        return report
