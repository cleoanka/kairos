"""Synthetic Chaos Generator (Component A).

Three statistical agents trade against a shared order book to manufacture an
L2 LOB stream with *known* micro-structure regimes:

* MarketMaker — rests a symmetric two-sided ladder around mid and replenishes
  it; supplies genuine liquidity. Dominates the RANGE regime.
* Retail      — sends market orders. Balanced & small in calm markets; in a
  TREND segment it turns one-sided and aggressive, walking the book so mid
  drifts. Produces directional trade flow.
* Spoofer     — floods large phantom limit orders a few levels deep, then
  cancels them almost immediately, over and over. Huge displayed depth + huge
  cancel-flow but little real trading. Defines the TOXIC regime.

The active regime is recorded per snapshot purely as ground truth for
*evaluation* (Constitution Rule 4 — it never enters a model loss).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import N_LEVELS, Regime, column_names, snapshot_to_row
from .engine import OrderBook

MM_DEPTH = N_LEVELS                 # ladder depth the maker maintains
MM_TARGET_SIZE = 5.0                # resting size per maker level
SPOOF_SIZE = (200.0, 420.0)         # phantom order size range
SPOOF_LEVELS = (2, 6)               # how deep the spoofer plants phantom walls


class MarketMaker:
    """Maintains a symmetric ladder and re-centres it on the current mid."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def seed_book(self, book: OrderBook, mid: int) -> None:
        for i in range(1, MM_DEPTH + 1):
            book.add_limit("bid", mid - i, MM_TARGET_SIZE)
            book.add_limit("ask", mid + i, MM_TARGET_SIZE)

    def step(self, book: OrderBook) -> None:
        mid = int(round(book.mid_tick()))
        # Top each near level back up to target (replenish consumed liquidity).
        for i in range(1, MM_DEPTH + 1):
            for side, px in (("bid", mid - i), ("ask", mid + i)):
                cur = (book.bids if side == "bid" else book.asks).get(px, 0.0)
                deficit = MM_TARGET_SIZE - cur
                if deficit > 0:
                    jitter = 1.0 + 0.15 * self.rng.standard_normal()
                    book.add_limit(side, px, max(0.0, deficit * jitter))
        # Retire stale levels that drifted far from mid (light, symmetric cancel).
        for side, book_side in (("bid", book.bids), ("ask", book.asks)):
            for px in [p for p in book_side if abs(p - mid) > MM_DEPTH + 2]:
                book.cancel(side, px, book_side[px])


class Retail:
    """Market-order flow; one-sided & aggressive during TREND segments."""

    def __init__(self, rng: np.random.Generator, trend_size: tuple[float, float] = (6.0, 16.0)):
        self.rng = rng
        self.trend_size = trend_size

    def step(self, book: OrderBook, regime: int, direction: int) -> None:
        if regime == Regime.TREND:
            n = self.rng.integers(1, 4)
            side = "buy" if direction > 0 else "sell"
            for _ in range(n):
                book.market_order(side, float(self.rng.uniform(*self.trend_size)))
        elif regime == Regime.RANGE:
            if self.rng.random() < 0.5:
                side = "buy" if self.rng.random() < 0.5 else "sell"
                book.market_order(side, float(self.rng.uniform(1, 4)))
        else:  # TOXIC — thin, hesitant real trading amid the phantom walls
            if self.rng.random() < 0.2:
                side = "buy" if self.rng.random() < 0.5 else "sell"
                book.market_order(side, float(self.rng.uniform(1, 3)))


class Spoofer:
    """Plants phantom walls and cancels them — the toxic-liquidity illusion."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self._pending: list[tuple[str, int, float]] = []

    def step(self, book: OrderBook) -> None:
        # First, cancel last round's phantom orders (the spoof reveal).
        for side, px, sz in self._pending:
            book.cancel(side, px, sz)
        self._pending.clear()
        # Plant new phantom walls, biased to one side to fake imbalance.
        mid = int(round(book.mid_tick()))
        heavy = "bid" if self.rng.random() < 0.5 else "ask"
        for side in ("bid", "ask"):
            n_walls = self.rng.integers(2, 4) if side == heavy else 1
            for _ in range(n_walls):
                lvl = int(self.rng.integers(*SPOOF_LEVELS))
                px = mid - lvl if side == "bid" else mid + lvl
                sz = float(self.rng.uniform(*SPOOF_SIZE))
                book.add_limit(side, px, sz)
                self._pending.append((side, px, sz))


# Market scenarios. "toxic" (default) is the original adversarial mix used for
# regime-separation training; "calm" is a benign, makeable market (RANGE-heavy,
# gentle trends, NO spoofing) to show the maker earns when the market allows it.
SCENARIOS: dict[str, dict] = {
    "toxic": {
        "cycle": [(Regime.RANGE, 0), (Regime.TREND, +1),
                  (Regime.TOXIC, 0), (Regime.TREND, -1)],
        "trend_size": (6.0, 16.0),
    },
    "calm": {
        "cycle": [(Regime.RANGE, 0), (Regime.TREND, +1),
                  (Regime.RANGE, 0), (Regime.TREND, -1)],
        "trend_size": (2.0, 6.0),
    },
    "range": {   # benign, makeable market: balanced two-sided flow, stable mid
        "cycle": [(Regime.RANGE, 0)],
        "trend_size": (2.0, 6.0),
    },
}


def _regime_schedule(rng: np.random.Generator, n_steps: int, cycle: list):
    """Yield (regime, direction) per step, in contiguous segments."""
    plan = []
    k = 0
    while len(plan) < n_steps:
        regime, direction = cycle[k % len(cycle)]
        seg = int(rng.integers(120, 260))
        plan.extend([(int(regime), direction)] * seg)
        k += 1
    return plan[:n_steps]


def generate(n_steps: int = 6000, seed: int = 7, warmup: int = 50,
             scenario: str = "toxic") -> pd.DataFrame:
    cfg = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    book = OrderBook(mid_tick=10_000)
    mm = MarketMaker(rng)
    retail = Retail(rng, trend_size=cfg["trend_size"])
    spoofer = Spoofer(rng)
    mm.seed_book(book, 10_000)

    # Warm up so the book is healthy before we start recording.
    for _ in range(warmup):
        mm.step(book)

    schedule = _regime_schedule(rng, n_steps, cfg["cycle"])
    rows = []
    for t, (regime, direction) in enumerate(schedule):
        mm.step(book)
        if regime == Regime.TOXIC:
            spoofer.step(book)
        retail.step(book, regime, direction)
        snap = book.snapshot(ts=float(t), regime=regime)
        rows.append(snapshot_to_row(snap))

    df = pd.DataFrame(rows, columns=column_names())
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LOB-Core synthetic chaos generator")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="toxic")
    ap.add_argument("--out", type=Path, default=Path("data/synthetic.parquet"))
    args = ap.parse_args(argv)

    df = generate(n_steps=args.steps, seed=args.seed, scenario=args.scenario)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, engine="pyarrow", index=False)

    counts = df["regime"].value_counts().sort_index()
    dist = ", ".join(f"{Regime(int(r)).name}={c}" for r, c in counts.items())
    print(f"wrote {len(df):,} snapshots -> {args.out}  ({dist})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
