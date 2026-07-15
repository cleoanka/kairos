"""Paper fill simulator / backtester (Component E).

Drives the maker strategy over a recorded LOB stream and simulates *passive*
fills against each step's aggressive trade flow: aggressive buys (that lift asks)
fill our resting ask (we sell); aggressive sells fill our resting bid (we buy).
Tracks inventory, cash, mark-to-market PnL, and enforces the risk gate. NO real
orders are ever sent — this is the offline shadow of the live execution engine.

Constitution Rule 3: no REST / no network here.
"""
from __future__ import annotations

import math

import numpy as np

from ..schema import REGIME_NAMES, Regime
from ..strategy.maker import Maker, MakerStrategy, Quote
from .risk import RiskGate


def run_backtest(df, strategy: Maker | None = None,
                 risk: RiskGate | None = None) -> dict:
    strategy = strategy or MakerStrategy()
    risk = risk or RiskGate()

    inv = 0.0
    cash = 0.0
    fills = 0
    pnl_curve: list[float] = []
    inv_curve: list[float] = []
    exposure = {r.value: 0 for r in Regime}

    q = Quote(None, 0.0, None, 0.0)   # quote resting from the PREVIOUS step (causal)
    last_mid = 0.0
    for row in df.itertuples(index=False):
        mid = float(row.mid)
        best_bid = mid - float(row.bid_px_0)
        best_ask = mid + float(row.ask_px_0)
        # Corrupt snapshot (one-sided / NaN / inf book) — never mark or quote on
        # non-finite prices; carry inventory forward at the last good mid.
        if not (math.isfinite(mid) and math.isfinite(best_bid) and math.isfinite(best_ask)):
            continue
        last_mid = mid
        regime = int(row.regime)
        exposure[regime] = exposure.get(regime, 0) + 1

        # (1) Fill the resting quote (posted at the prior step) against THIS step's
        #     aggressive flow — no look-ahead: the quote predates these trades.
        # A resting quote is only hit if it is competitive — at or improving the
        # touch. A quote resting behind the best price is not reached by the flow.
        tb, ts = float(row.trade_buy), float(row.trade_sell)
        tb = tb if math.isfinite(tb) else 0.0
        ts = ts if math.isfinite(ts) else 0.0
        if (q.ask_px is not None and q.ask_sz > 0 and tb > 0
                and q.ask_px <= best_ask + 1e-9):            # we sell into buys
            f = min(q.ask_sz, tb)
            inv -= f
            cash += f * q.ask_px
            fills += 1
        if (q.bid_px is not None and q.bid_sz > 0 and ts > 0
                and q.bid_px >= best_bid - 1e-9):            # we buy from sells
            f = min(q.bid_sz, ts)
            inv += f
            cash -= f * q.bid_px
            fills += 1

        # (2) Mark to market, run the risk gate, and post the quote that will rest
        #     into the NEXT step.
        pnl = cash + inv * mid
        if not risk.update(pnl):
            q = Quote(None, 0.0, None, 0.0)      # kill-switch tripped — stop quoting
        else:
            q = risk.clamp(strategy.decide(regime, best_bid, best_ask, inv), inv)

        pnl_curve.append(cash + inv * mid)
        inv_curve.append(inv)

    final_pnl = cash + inv * last_mid
    return {
        "steps": int(len(df)),
        "fills": fills,
        "final_pnl": round(final_pnl, 3),
        "final_inventory": round(inv, 3),
        "max_abs_inventory": round(float(np.max(np.abs(inv_curve))) if inv_curve else 0.0, 3),
        "halted": bool(risk.halted),
        "regime_exposure": {REGIME_NAMES[r]: c for r, c in exposure.items() if r in REGIME_NAMES},
        "pnl_curve": pnl_curve,
        "inv_curve": inv_curve,
    }
