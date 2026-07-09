# Changelog

All notable changes to Kairos are documented here. This project fuses two
prior systems — **LOB-Core** (System-1 perception) and **TradingAgents**
(System-2 reasoning) — into a single strictly-causal dual-process trading brain.

## [1.0.0] — 2026-07-08 — "Genesis: the dual-process fusion"

The first release: two independent systems become one cognitive loop.

### Added — the Bridge (the novel core, `kairos.bridge`)
- **`Percept`** — an immutable, point-in-time System-1 read of market
  microstructure (regime, order-flow imbalance, resting-depth imbalance,
  toxicity, BULL/BEAR direction, confidence). Every field is computable from
  data at or before its timestamp.
- **`CausalPerceptionBus`** — an append-only, monotonic store whose only read
  path (`as_of`) is a `bisect` that can reach **only** percepts at or before the
  cutoff. Two invariants — *no future access* and *append-independence* — make
  look-ahead impossible **by construction**, closing the hole a naive
  point-in-time vendor lookup leaves open.
- **Clock-domain guard** — a date/datetime query against a step-index-clocked
  bus (or vice versa) raises `ClockDomainError` instead of silently returning
  the newest percept. Explicit guard against the Python-3.11 `fromisoformat`
  midnight bug so `"YYYY-MM-DD"` always maps to the *close* of that day.
- **Microstructure Analyst** — a new opt-in member of the TradingAgents analyst
  team whose LangChain tools read the causal bus point-in-time; wired into the
  LangGraph (state, execution plan, conditional logic, setup, tool node, and
  `propagate(perception_bus=...)`). Its report is consumed by the bull/bear
  researchers and the trader.
- **`ExecutionLink`** — the neuro-symbolic control link: System-2 sets a stance
  (BUY/HOLD/SELL + conviction), a bias-skewed Avellaneda–Stoikov `DirectionalMaker`
  executes it, and System-1 retains a **hard veto** — TOXIC regime ⇒ stand aside.
  Execution acts on the *perceived* regime per row, never the ground truth.
- **`cognitive_loop`** — the closed perceive → reason → act → reflect loop, with
  a deterministic System-2 stand-in (no LLM/keys, runs anywhere) and a real-LLM
  mode. Decision from the in-sample half, execution on the forward half.

### Added — packaging & rigor
- Unified `kairos` CLI: `loop`, `perceive`, `reason`, `web`, `soul-check`,
  `reproduce`.
- Layered dependency extras: core (always), `[reasoning]`, `[mlx]`, `[native]`.
- Portable numpy inference backend for the System-1 encoder (MLX only needed to
  *train*), with a dependency-free `.safetensors` reader.
- **Scoped Constitution** (`scripts/soul_check.py`): System-1 forbids classic TA;
  System-2 may reason about it. New **Rule 5 (causality)** forbids the
  reasoning-facing bridge from touching non-causal accessors.
- Honest reproducibility gate (`scripts/reproduce.py`), CI (core + reasoning
  jobs), and docs: `ARCHITECTURE`, `PHILOSOPHY`, `CAUSALITY`, `HOW_IT_WORKS`.
- Property-tested causal contract; graph-compilation and tool-causality
  integration tests.

### Changed — from the two source projects
- `lob_core` → `kairos.perception`; `tradingagents` → `kairos.reasoning`;
  `cli` → `kairos.reasoning_cli`.
- MLX import in the perception embedder is now optional (numpy inference
  fallback), so the whole perception package imports on any platform.

### Honesty note
The dual-process loop's measured edge is on **benign (range) markets**, where it
captures spread and beats naive-always-long across seeds. On trending/toxic
markets a market maker suffers adverse selection — a property of the market, not
a bug — and naive directional exposure can win. `reproduce.py` claims only what
is robustly true; it does not claim profit in every regime.

### Attribution
Bundles TradingAgents (© Tauric Research, Apache-2.0) under `kairos.reasoning`.
See `NOTICE` and `LICENSE-APACHE-2.0.txt`.
