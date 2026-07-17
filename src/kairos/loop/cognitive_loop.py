"""The closed dual-process loop: perceive → reason → act → reflect.

This is the whole thesis of Kairos running as one function:

1. **Perceive** — replay a session into a strictly-causal
   :class:`~kairos.bridge.causal_bus.CausalPerceptionBus` (System-1).
2. **Reason** — at the decision instant, form a stance (BUY/HOLD/SELL +
   conviction). Two backends:
     * ``deterministic`` — a transparent System-2 *stand-in* policy over the
       causal percept, so the entire loop runs with no LLM, no keys, anywhere.
     * ``llm`` — the real multi-agent debate (TradingAgents), with the causal
       bus attached so its Microstructure Analyst reads System-1 percepts
       point-in-time.
3. **Act** — execute the stance over the *forward* window via the neuro-symbolic
   :class:`~kairos.bridge.execution_link.ExecutionLink`, where System-1 keeps a
   hard veto in TOXIC regimes.
4. **Reflect** — score the result against honest baselines (stand-aside, naive
   always-long, pure market-making) so the value of each system is measurable.

The decision is formed from the in-sample half only; execution runs on the
forward half. Look-ahead is impossible by construction, so the reported PnL is a
faithful causal shadow of live trading — never an inflated backtest.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace

from kairos.bridge import (
    Decision,
    ExecutionLink,
    MicrostructureConfig,
    Percept,
    build_causal_bus,
)


@dataclass
class LoopConfig:
    symbol: str = "BTCUSDT"
    scenario: str = "toxic"           # synthetic market scenario (see perception.synthetic)
    n_steps: int = 4000
    seed: int = 7
    perception_window: int = 64       # trailing rows aggregated into each percept
    perception_step: int = 8          # emit a percept every N rows
    decision_fraction: float = 0.5    # point through the session where the stance is set
    mode: str = "deterministic"       # "deterministic" | "llm"
    max_conviction: float = 1.0
    micro: MicrostructureConfig = field(default_factory=MicrostructureConfig)
    regime_backend: object | None = None   # None=heuristic; or a bridge.LearnedRegime


@dataclass
class LoopResult:
    symbol: str
    decision_ts: float
    percept: Percept
    decision: Decision
    execution: dict
    baselines: dict
    reflection: str

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "decision_ts": self.decision_ts,
            "percept": self.percept.to_dict(),
            "decision": {
                "action": self.decision.action,
                "conviction": self.decision.conviction,
                "bias": self.decision.bias,
                "source": self.decision.source,
            },
            "execution": {k: v for k, v in self.execution.items()
                          if k not in ("pnl_curve", "inv_curve")},
            "baselines": self.baselines,
            "reflection": self.reflection,
        }


def deterministic_policy(percept: Percept, *, max_conviction: float = 1.0) -> Decision:
    """A transparent System-2 stand-in: map a causal percept to a stance.

    This is deliberately simple and auditable — it exists so the loop is fully
    runnable without an LLM, and so the LLM path has an honest baseline to beat.
    The real System-2 is the multi-agent debate; this is its shadow.
    """
    if percept.is_toxic:
        # System-1 already says the book is phantom — stand aside.
        return Decision("HOLD", 0.0, rationale="TOXIC regime — stand aside",
                        source="deterministic-policy")
    conviction = min(max_conviction, percept.direction_strength * percept.regime_confidence)
    if percept.direction == "BULL" and conviction > 0.05:
        return Decision("BUY", conviction,
                        rationale=f"{percept.regime_name}/BULL", source="deterministic-policy")
    if percept.direction == "BEAR" and conviction > 0.05:
        return Decision("SELL", conviction,
                        rationale=f"{percept.regime_name}/BEAR", source="deterministic-policy")
    return Decision("HOLD", 0.0, rationale="no directional edge",
                    source="deterministic-policy")


def _llm_decision(bus, symbol: str, decision_time: str, config: dict | None,
                  selected_analysts=("microstructure",)) -> Decision:
    """Run the real multi-agent debate with the causal bus attached, and parse
    its verdict. Imported lazily — only the ``llm`` mode needs the reasoning extra.

    ``decision_time`` must be on the SAME clock the percepts use: an ISO date for
    an epoch-clocked bus, or the numeric step (as a string) for a synthetic
    index-clocked bus — the causal bus refuses a cross-clock query. The loop
    defaults to the bus-backed Microstructure Analyst only, so the demo needs an
    LLM key but no external market-data vendor (which would require real dates)."""
    from kairos.bridge import parse_decision
    from kairos.reasoning.default_config import DEFAULT_CONFIG
    from kairos.reasoning.graph.trading_graph import TradingAgentsGraph

    cfg = (config or DEFAULT_CONFIG).copy()
    ta = TradingAgentsGraph(selected_analysts=tuple(selected_analysts), config=cfg)
    _, decision_text = ta.propagate(
        symbol, decision_time, asset_type="crypto", perception_bus=bus
    )
    return parse_decision(str(decision_text))


def _baselines(forward_df, link: ExecutionLink, regime_backend, cfg) -> dict:
    """Score honest reference strategies over the same forward window.

    ``stand_aside`` is the genuinely flat null strategy — it posts no quotes,
    takes no fills and risks nothing, so it is a real reference for the edge a
    strategy adds rather than a self-comparison. (A HOLD/0.0 decision still runs
    the two-sided market maker through System-1, which is exactly what
    ``pure_market_making`` measures — hence the two must not be conflated.)"""
    out = {"stand_aside": {"final_pnl": 0.0, "fills": 0,
                           "final_inventory": 0.0, "halted": False}}
    for name, dec in {
        "naive_long": Decision("BUY", 1.0, source="baseline"),
        "pure_market_making": Decision("HOLD", 0.0, source="baseline"),
    }.items():
        rep = link.execute(dec, forward_df, regime_backend=regime_backend, cfg=cfg)
        out[name] = {"final_pnl": rep["final_pnl"], "fills": rep["fills"],
                     "final_inventory": rep["final_inventory"], "halted": rep["halted"]}
    return out


@contextmanager
def _shared_perception():
    """Memoize :func:`perceive_regimes` for the duration of one loop run.

    ``ExecutionLink.execute`` re-derives the *perceived* regime array from
    ``(forward_df, window, regime_backend, cfg)`` on every call, and one run
    calls it three times over the SAME forward window (the real decision plus
    the ``naive_long`` / ``pure_market_making`` baselines). Since
    ``perceive_regimes`` is a pure function of exactly those args, the three
    arrays are bit-identical — so under a learned backend the dominant per-row
    inference cost is paid 3× for nothing. We wrap the module-level function
    with a per-run identity cache (keyed on ``id``/``window``, which is sound
    because all four inputs are the same live objects across the three calls)
    and restore it afterwards, so behaviour and headline numbers are unchanged.
    """
    import kairos.bridge.execution_link as el

    original = el.perceive_regimes
    cache: dict = {}

    def memoized(df, *, window=64, regime_backend=None, cfg=None):
        key = (id(df), window, id(regime_backend), id(cfg))
        out = cache.get(key)
        if out is None:
            out = original(df, window=window, regime_backend=regime_backend, cfg=cfg)
            cache[key] = out
        return out

    el.perceive_regimes = memoized
    try:
        yield
    finally:
        # Restore unconditionally — even if the body raised — so a failed run
        # never leaves the memoized wrapper installed for the next run. Guard
        # the swap-back so we only unwind our OWN wrapper: if something nested
        # installed a newer one we leave it in place (strict LIFO), never
        # clobbering it with our stale `original`.
        if el.perceive_regimes is memoized:
            el.perceive_regimes = original


def run_cognitive_loop(cfg: LoopConfig | None = None, *, df=None,
                       reasoning_config: dict | None = None) -> LoopResult:
    """Run one full perceive→reason→act→reflect cycle and return the result."""
    cfg = cfg or LoopConfig()

    # 1) PERCEIVE — a session and its strictly-causal perception history.
    if df is None:
        from kairos.perception.synthetic.generate import generate
        df = generate(n_steps=cfg.n_steps, seed=cfg.seed, scenario=cfg.scenario)

    k = max(cfg.perception_window, int(len(df) * cfg.decision_fraction))
    in_sample = df.iloc[:k].reset_index(drop=True)
    forward = df.iloc[k:].reset_index(drop=True)
    if len(forward) == 0:
        raise ValueError("decision_fraction leaves no forward window to execute over")

    bus = build_causal_bus(in_sample, cfg.symbol, window=cfg.perception_window,
                           step=cfg.perception_step, regime_backend=cfg.regime_backend,
                           cfg=cfg.micro)
    decision_ts = float(in_sample.iloc[-1]["ts"])
    percept = bus.as_of(decision_ts)
    assert percept is not None and percept.ts <= decision_ts, "causal invariant violated"

    # 2) REASON — set the stance (deterministic stand-in or real LLM debate).
    if cfg.mode == "llm":
        # Query on the bus's own clock: a real ISO date if percepts are epoch-
        # stamped, else the numeric step (the causal bus rejects a cross-clock query).
        decision_time = percept.as_of if percept.as_of else str(decision_ts)
        decision = _llm_decision(bus, cfg.symbol, decision_time, reasoning_config)
        # Honour max_conviction here too — the LLM's parsed conviction is a raw
        # verdict, not a risk-capped one (deterministic_policy already clamps).
        if decision.conviction > cfg.max_conviction:
            decision = replace(decision, conviction=cfg.max_conviction)
    else:
        decision = deterministic_policy(percept, max_conviction=cfg.max_conviction)

    # 3) ACT — execute over the forward window (System-1 retains the TOXIC veto).
    # The perceived-regime array is shared across the decision run and both
    # baselines (all read the same forward window) so it is computed only once.
    link = ExecutionLink(window=cfg.perception_window)
    with _shared_perception():
        execution = link.execute(decision, forward, regime_backend=cfg.regime_backend, cfg=cfg.micro)
        baselines = _baselines(forward, link, cfg.regime_backend, cfg.micro)

    # 4) REFLECT — an honest, quantitative post-mortem.
    reflection = _reflect(cfg, percept, decision, execution, baselines)
    return LoopResult(cfg.symbol, decision_ts, percept, decision, execution, baselines, reflection)


def _reflect(cfg, percept, decision, execution, baselines) -> str:
    ns = execution["neuro_symbolic"]
    pnl = execution["final_pnl"]
    edge = pnl - baselines["stand_aside"]["final_pnl"]
    lines = [
        f"Kairos cognitive loop — {cfg.symbol} (scenario={cfg.scenario}, mode={cfg.mode})",
        f"  System-1 percept : {percept.regime_name} / {percept.direction} "
        f"(conf {percept.regime_confidence:.0%}, tox {percept.toxicity:.2f})",
        f"  System-2 stance  : {decision.action} @ conviction {decision.conviction:.2f} "
        f"(source={decision.source})",
        f"  Execution        : PnL {pnl:+.1f}, {execution['fills']} fills, "
        f"inv {execution['final_inventory']:+.2f}, halted={execution['halted']}",
        f"  System-1 veto    : {ns['toxic_veto_fraction']:.0%} of the window perceived TOXIC "
        f"(dominated={ns['system1_vetoed']})",
        f"  Edge vs stand-aside: {edge:+.1f}",
        "  Baselines        : " + ", ".join(
            f"{k}={v['final_pnl']:+.0f}" for k, v in baselines.items()),
    ]
    if percept.is_toxic and decision.action == "HOLD":
        lines.append("  → Dual-process win: System-1 flagged phantom liquidity; System-2 stood aside.")
    return "\n".join(lines)
