"""Grid-search the Avellaneda–Stoikov maker parameters on the paper backtest.

Risk-aversion γ and order-flow intensity k trade off spread width against fill
rate: larger γ / smaller k widen the quotes (fewer, safer fills). This sweeps a
grid and reports both the PnL-maximising config and the best *active* config
(one that actually trades) — on a toxic market the unconstrained optimum is often
"quote so wide you barely trade", which the active filter exposes. Offline; no orders.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from ..strategy.avellaneda import AvellanedaStoikovMaker
from .risk import RiskGate
from .simulator import run_backtest

DEFAULT_GAMMAS = (0.02, 0.05, 0.1, 0.2, 0.5)
DEFAULT_KS = (0.5, 1.0, 1.5, 3.0)


def grid_search(df, gammas=DEFAULT_GAMMAS, ks=DEFAULT_KS, base_size: float = 1.0,
                max_inventory: float = 20.0, min_fills: int = 100) -> dict:
    results = []
    for g, k in itertools.product(gammas, ks):
        strat = AvellanedaStoikovMaker(base_size=base_size, gamma=g, k=k,
                                       max_inventory=max_inventory)
        rep = run_backtest(df, strat, RiskGate(max_inventory=max_inventory))
        results.append({"gamma": g, "k": k, "pnl": rep["final_pnl"],
                        "fills": rep["fills"], "max_abs_inventory": rep["max_abs_inventory"]})
    best = max(results, key=lambda r: r["pnl"])
    active = [r for r in results if r["fills"] >= min_fills]
    best_active = max(active, key=lambda r: r["pnl"]) if active else None
    return {"grid": results, "best": best, "best_active": best_active,
            "n_configs": len(results)}


def main(argv: list[str] | None = None) -> int:
    from ..synthetic.generate import SCENARIOS

    ap = argparse.ArgumentParser(description="Grid-search Avellaneda-Stoikov maker params")
    ap.add_argument("--scenario", choices=list(SCENARIOS), default=None,
                    help="tune on a freshly generated market scenario (toxic/calm/range)")
    ap.add_argument("--data", default="data/synthetic.parquet")
    ap.add_argument("--out", type=Path, default=Path("artifacts/maker_grid.json"))
    args = ap.parse_args(argv)

    import pandas as pd
    if args.scenario:
        from ..synthetic.generate import generate
        df = generate(n_steps=6000, seed=7, scenario=args.scenario)
        print(f"(tuning on a fresh '{args.scenario}' market)")
    elif Path(args.data).exists():
        df = pd.read_parquet(args.data)
    else:
        print(f"no dataset at {args.data} — run `lob_core --mode synthetic` first.")
        return 1
    print("=" * 64)
    print(" LOB-Core — Avellaneda-Stoikov parameter grid (paper, NO orders)")
    print("=" * 64)
    r = grid_search(df)
    for row in sorted(r["grid"], key=lambda x: -x["pnl"])[:8]:
        print(f"  gamma={row['gamma']:.3f}  k={row['k']:.2f}   pnl={row['pnl']:11.1f}  "
              f"fills={row['fills']:5d}  max|inv|={row['max_abs_inventory']:6.2f}")
    b, ba = r["best"], r["best_active"]
    print(f"\nbest PnL overall : gamma={b['gamma']} k={b['k']}  pnl={b['pnl']}  fills={b['fills']}")
    if ba:
        print(f"best PnL active  : gamma={ba['gamma']} k={ba['k']}  pnl={ba['pnl']}  "
              f"fills={ba['fills']}  max|inv|={ba['max_abs_inventory']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
