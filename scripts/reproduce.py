#!/usr/bin/env python3
"""End-to-end reproducibility gate for Kairos.

Verifies the project's headline claims from a clean slate. Every check that does
NOT require MLX or API keys runs everywhere (CI, Linux, any host); the
MLX-dependent System-1 training claims run only when MLX is installed and are
otherwise reported as SKIPPED (honestly, not silently passed).

Headline claims checked
-----------------------
  1. Constitution is green (scoped soul_check).
  2. Core test suite is green (bridge + loop + perception, no MLX/native).
  3. CAUSALITY: a fresh causal bus never leaks the future (invariant re-proven).
  4. The cognitive loop is bit-deterministic across two identical runs.
  5. System-1 VETO: a forced-TOXIC market yields zero fills (stand-aside).
  6. Regime-adaptive edge: on a BENIGN (range) market the dual-process loop
     captures spread and beats naive-always-long across seeds. (We deliberately
     do NOT claim it wins on trending/toxic markets — there a market maker
     suffers adverse selection, which is a property of the market, not a bug.)
  7. Adaptivity: the loop's stance tracks the regime — it is not always long.
  8. [MLX only] System-1 self-supervised pipeline separates regimes (ARI >= 0.9).

Exit 0 iff every APPLICABLE check passes.
"""
from __future__ import annotations

import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SRC = str(ROOT / "src")


def _run(cmd) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    print("[1/8] Constitution (soul_check)…")
    rc, _ = _run([PY, "scripts/soul_check.py", "--quiet"])
    checks.append(("Constitution green", rc == 0, ""))

    print("[2/8] Core test suite (bridge + loop + perception)…")
    rc, out = _run([PY, "-m", "pytest", "tests/bridge", "tests/loop", "tests/perception",
                    "-q", "-m", "not mlx and not native and not reasoning"])
    tail = out.strip().splitlines()[-1] if out.strip() else ""
    checks.append(("Core tests green", rc == 0, tail))

    # In-process checks (need src on the path).
    sys.path.insert(0, SRC)

    print("[3/8] Causality — as_of never leaks the future…")
    from kairos.bridge import build_causal_bus
    from kairos.perception.synthetic.generate import generate
    df = generate(n_steps=3000, seed=11, scenario="toxic")
    bus = build_causal_bus(df, "BTCUSDT", window=64, step=8)
    leak = False
    for q in (df["ts"].iloc[len(df) // 3], df["ts"].iloc[len(df) // 2], df["ts"].iloc[-1] * 2):
        p = bus.as_of(float(q))
        if p is not None and p.ts > float(q):
            leak = True
    checks.append(("No look-ahead leak", not leak, f"probed 3 cutoffs over {len(bus)} percepts"))

    print("[4/8] Cognitive loop is deterministic…")
    from kairos.loop import LoopConfig, run_cognitive_loop
    a = run_cognitive_loop(LoopConfig(scenario="toxic", n_steps=3000, seed=11)).to_dict()
    b = run_cognitive_loop(LoopConfig(scenario="toxic", n_steps=3000, seed=11)).to_dict()
    det = a["decision"] == b["decision"] and a["execution"]["final_pnl"] == b["execution"]["final_pnl"]
    checks.append(("Loop deterministic", det, f"pnl={a['execution']['final_pnl']}"))

    print("[5/8] System-1 veto — forced-TOXIC market stands aside…")
    from kairos.bridge import Decision, ExecutionLink, MicrostructureConfig
    rep = ExecutionLink().execute(Decision("BUY", 1.0), df,
                                  cfg=MicrostructureConfig(toxic_threshold=-1.0))
    veto = rep["fills"] == 0 and rep["neuro_symbolic"]["toxic_veto_fraction"] == 1.0
    checks.append(("System-1 veto works", veto, f"fills={rep['fills']}"))

    print("[6/8] Regime-adaptive edge on the benign (range) market…")
    wins = 0
    seeds = [1, 3, 7, 11, 21]
    for s in seeds:
        r = run_cognitive_loop(LoopConfig(scenario="range", n_steps=4000, seed=s))
        if r.execution["final_pnl"] > r.baselines["naive_long"]["final_pnl"]:
            wins += 1
    checks.append(("Beats naive-long on RANGE", wins == len(seeds), f"{wins}/{len(seeds)} seeds"))

    print("[7/8] Adaptivity — stance tracks the regime (not always long)…")
    actions = set()
    for scen in ("range", "calm", "toxic"):
        for s in (1, 3, 7, 11, 21):
            actions.add(run_cognitive_loop(LoopConfig(scenario=scen, n_steps=3000, seed=s)).decision.action)
    checks.append(("Stance is regime-adaptive", len(actions) >= 2,
                   f"observed stances: {sorted(actions)}"))

    print("[8/8] System-1 self-supervised regime separation (MLX)…")
    if find_spec("mlx") is None:
        checks.append(("Regime ARI >= 0.9 (MLX)", None, "SKIPPED — MLX not installed"))
    else:
        rc, out = _run([PY, "-m", "pytest", "tests/perception", "-q", "-m", "mlx"])
        checks.append(("Regime pipeline (MLX)", rc == 0, out.strip().splitlines()[-1] if out.strip() else ""))

    # --- Verdict ---------------------------------------------------------------
    print("\n" + "=" * 66)
    print(" KAIROS — reproducibility verdict")
    print("=" * 66)
    failed = 0
    for name, ok, note in checks:
        if ok is None:
            mark = "\033[33m− SKIP\033[0m"
        elif ok:
            mark = "\033[32m✓ PASS\033[0m"
        else:
            mark = "\033[31m✗ FAIL\033[0m"
            failed += 1
        print(f"  {mark}  {name:<32}  {note}")
    print("=" * 66)
    if failed:
        print(f"\033[31m✗ {failed} headline claim(s) failed.\033[0m")
        return 1
    print("\033[32m✓ every applicable headline claim reproduced.\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
