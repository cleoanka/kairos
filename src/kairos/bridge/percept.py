"""The Percept — a point-in-time System-1 reading of market microstructure.

A :class:`Percept` is the atom that crosses the boundary between the two
cognitive systems of Kairos:

* **System 1** (``kairos.perception``, LOB-Core) *produces* Percepts — fast,
  subsymbolic, self-supervised readings of the limit-order-book: regime,
  order-flow, resting-depth pressure and liquidity toxicity.
* **System 2** (``kairos.reasoning``, TradingAgents) *consumes* Percepts —
  slow, symbolic, deliberative multi-agent reasoning.

Every field is computable from information available **at or before** the
percept's timestamp. There is no field that peeks at a future price. That is
what lets the :class:`~kairos.bridge.causal_bus.CausalPerceptionBus` guarantee,
*by construction*, that the reasoning layer can never see the future — closing
the look-ahead hole that a naive point-in-time vendor lookup leaves open.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from kairos.perception.schema import REGIME_NAMES, Regime

# Direction labels for the fast BULL/BEAR read (a sign, not a price forecast).
BULL = "BULL"
BEAR = "BEAR"
NEUTRAL = "NEUTRAL"


@dataclass(frozen=True, slots=True)
class Percept:
    """An immutable, point-in-time microstructure state.

    All magnitudes are dimensionless / normalised so they read the same across
    instruments and price scales.
    """

    ts: float                       # causal timestamp (epoch seconds or monotonic index)
    symbol: str

    mid: float                      # mid price at ts
    spread_ticks: float             # best-ask minus best-bid, in ticks (>= 0)

    order_flow_imbalance: float     # (buy-sell)/(buy+sell) aggressive flow, [-1, 1]
    depth_imbalance: float          # (bid-ask) resting size pressure, [-1, 1]
    toxicity: float                 # cancel / phantom-liquidity ratio, [0, 1]
    trade_intensity: float          # normalised recent trade activity, [0, 1]

    regime: int                     # Regime enum value (RANGE/TREND/TOXIC)
    regime_confidence: float        # [0, 1]
    regime_source: str              # "learned" (MLX/numpy encoder) | "heuristic"

    direction: str                  # BULL | BEAR | NEUTRAL
    direction_strength: float       # [0, 1]

    n_observations: int             # raw snapshots aggregated into this percept
    as_of: str | None = None        # ISO-8601 wall-clock of ts, when known

    # --- Views ---------------------------------------------------------------
    @property
    def regime_name(self) -> str:
        return REGIME_NAMES.get(int(self.regime), "UNKNOWN")

    @property
    def is_toxic(self) -> bool:
        """TOXIC regime = illusory (spoofed) liquidity. System-1's hard veto."""
        return int(self.regime) == int(Regime.TOXIC)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["regime_name"] = self.regime_name
        return d

    def to_prompt(self) -> str:
        """A compact, deterministic, LLM-facing summary of the percept.

        Deliberately free of price *forecasts*: it reports the *state* of the
        book (regime, flow, depth, toxicity), never a predicted return — the
        System-2 agents form the directional view themselves.
        """
        arrow = {BULL: "▲", BEAR: "▼", NEUTRAL: "◆"}.get(self.direction, "◆")
        stamp = self.as_of or f"t={self.ts:.3f}"
        return (
            f"MICROSTRUCTURE PERCEPT [{self.symbol} @ {stamp}]\n"
            f"  Regime            : {self.regime_name} "
            f"(confidence {self.regime_confidence:.0%}, source={self.regime_source})\n"
            f"  Direction         : {self.direction} {arrow} "
            f"(strength {self.direction_strength:.0%})\n"
            f"  Order-flow imbal. : {self.order_flow_imbalance:+.3f}  "
            f"(aggressive buy vs sell, [-1,1])\n"
            f"  Resting-depth imb.: {self.depth_imbalance:+.3f}  "
            f"(bid vs ask resting size, [-1,1])\n"
            f"  Liquidity toxicity: {self.toxicity:.3f}  "
            f"(cancel / phantom-liquidity ratio, [0,1])\n"
            f"  Trade intensity   : {self.trade_intensity:.3f}  ([0,1])\n"
            f"  Mid / spread      : {self.mid:.4f} / {self.spread_ticks:.2f} ticks\n"
            f"  Aggregated over   : {self.n_observations} book snapshots (strictly causal)"
        )
